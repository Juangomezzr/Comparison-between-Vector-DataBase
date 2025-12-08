import time
import json
import os
import csv
from datetime import datetime

import numpy as np
import chromadb
import psycopg2

# ---------- CONFIG ----------

DIM = 128              # vector dimension
N_VECTORS = 1000       # adjust as needed (small/medium/large)
CHROMA_PATH = "./chroma-data"
CHROMA_COLLECTION = "items_128d"
PG_TABLE = "items"
METRICS_FILE = "load_metrics.csv"

# Chroma client
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

# PostgreSQL connection
pg_conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="postgres",
    password="postgres",
    dbname="vecdb"
)
pg_conn.autocommit = True
pg_cur = pg_conn.cursor()


# ---------- UTILS ----------

def get_folder_size_bytes(path: str) -> int:
    """
    Recursively compute folder size in bytes.
    If folder doesn't exist yet, return 0.
    """
    if not os.path.exists(path):
        return 0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def get_pg_table_size_bytes(table_name: str) -> int:
    """
    Get total PostgreSQL table size (including indexes) in bytes.
    """
    pg_cur.execute(
        "SELECT pg_total_relation_size(%s);",
        (table_name,)
    )
    res = pg_cur.fetchone()
    if res and res[0] is not None:
        return int(res[0])
    return 0


def append_metrics_row(row: dict):
    """
    Append a row of metrics to METRICS_FILE (CSV).
    If file doesn't exist, write header first.
    """
    file_exists = os.path.exists(METRICS_FILE)
    fieldnames = list(row.keys())

    with open(METRICS_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------- DATA GENERATION ----------

def generate_vectors(n, dim):
    vectors = np.random.rand(n, dim).astype("float32")
    categories = np.random.choice(["A", "B", "C"], size=n)
    return vectors, categories


# ---------- LOAD INTO CHROMA ----------

def load_into_chroma(collection_name, vectors, categories):
    collection = chroma_client.get_or_create_collection(name=collection_name)

    ids = [f"{collection_name}_{i}" for i in range(len(vectors))]
    metadatas = [{"category": c} for c in categories]

    t0 = time.time()
    collection.add(
        ids=ids,
        embeddings=vectors.tolist(),
        metadatas=metadatas
    )
    t1 = time.time()

    load_time = t1 - t0
    return load_time


# ---------- LOAD INTO PGVECTOR ----------

def load_into_pg(table_name, vectors, categories):
    t0 = time.time()

    for v, c in zip(vectors, categories):
        meta = {"category": c}
        # Convertimos el vector de numpy.float32 -> lista de float de Python
        pg_cur.execute(
            f"""
            INSERT INTO {table_name} (metadata, embedding)
            VALUES (%s, %s)
            """,
            (json.dumps(meta), [float(x) for x in v])
            # o: (json.dumps(meta), v.astype(float).tolist())
        )

    t1 = time.time()
    load_time = t1 - t0
    return load_time

# ---------- MAIN ----------

if __name__ == "__main__":
    print(f"=== DATA LOAD EXPERIMENT ===")
    print(f"Vectors: {N_VECTORS} | Dimension: {DIM}")

    # Before sizes (baseline)
    chroma_size_before = get_folder_size_bytes(CHROMA_PATH)
    pg_size_before = get_pg_table_size_bytes(PG_TABLE)

    print("Generating vectors...")
    vectors, cats = generate_vectors(N_VECTORS, DIM)

    # ---- Chroma load ----
    print("Loading into Chroma...")
    chroma_load_time = load_into_chroma(CHROMA_COLLECTION, vectors, cats)
    chroma_size_after = get_folder_size_bytes(CHROMA_PATH)

    # ---- pgvector load ----
    print("Loading into PostgreSQL (pgvector)...")
    pg_load_time = load_into_pg(PG_TABLE, vectors, cats)
    pg_size_after = get_pg_table_size_bytes(PG_TABLE)

    # Size deltas
    chroma_delta_bytes = chroma_size_after - chroma_size_before
    pg_delta_bytes = pg_size_after - pg_size_before

    # Convert to MB
    chroma_size_mb = chroma_size_after / (1024 * 1024)
    pg_size_mb = pg_size_after / (1024 * 1024)
    chroma_delta_mb = chroma_delta_bytes / (1024 * 1024)
    pg_delta_mb = pg_delta_bytes / (1024 * 1024)

    # ---- Print summary ----
    print("\n=== SUMMARY ===")
    print(f"Chroma:")
    print(f"  Load time: {chroma_load_time:.4f} s")
    print(f"  Total storage after: {chroma_size_mb:.2f} MB")
    print(f"  Storage added in this run: {chroma_delta_mb:.2f} MB")

    print(f"\nPostgreSQL (pgvector, table '{PG_TABLE}'):")
    print(f"  Load time: {pg_load_time:.4f} s")
    print(f"  Total storage after: {pg_size_mb:.2f} MB")
    print(f"  Storage added in this run: {pg_delta_mb:.2f} MB")

    # ---- Save metrics to CSV ----
    metrics_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_vectors": N_VECTORS,
        "dimension": DIM,
        "collection_name": CHROMA_COLLECTION,
        "table_name": PG_TABLE,
        "chroma_load_time_sec": f"{chroma_load_time:.6f}",
        "chroma_total_size_mb": f"{chroma_size_mb:.4f}",
        "chroma_delta_size_mb": f"{chroma_delta_mb:.4f}",
        "pg_load_time_sec": f"{pg_load_time:.6f}",
        "pg_total_size_mb": f"{pg_size_mb:.4f}",
        "pg_delta_size_mb": f"{pg_delta_mb:.4f}",
    }

    append_metrics_row(metrics_row)

    print(f"\nMetrics saved to: {METRICS_FILE}")
    print("Done!")
