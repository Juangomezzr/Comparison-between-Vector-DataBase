import time
import statistics
import csv
from datetime import datetime
import os
import psutil

import numpy as np
import chromadb
import psycopg2


# ======================================================
# CONFIGURACIÓN DEL BENCHMARK
# ======================================================

DIM = 128
TOP_K = 10
N_QUERIES = 100

CHROMA_COLLECTION = "items_128d"
PG_TABLE = "items"

SUMMARY_CSV = "query_metrics_summary_large.csv"
DETAIL_CSV = "query_metrics_detail_large.csv"

DATASET_LABEL = "large_300000"   # Cambiar a SMALL / MEDIUM / LARGE en tus pruebas


# ======================================================
# CONEXIONES
# ======================================================

chroma_client = chromadb.HttpClient(host="localhost", port=8000)

def get_pg_conn():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="postgres",
        dbname="vecdb"
    )
    conn.autocommit = True
    return conn


# ======================================================
# UTILIDADES
# ======================================================

def generate_query_vectors(n, dim):
    return np.random.rand(n, dim).astype("float32")


def summarize_latencies(latencies):
    avg = statistics.mean(latencies)

    if len(latencies) >= 20:
        p95 = statistics.quantiles(latencies, n=20)[18]
    else:
        p95 = max(latencies)

    if len(latencies) >= 100:
        p99 = statistics.quantiles(latencies, n=100)[98]
    else:
        p99 = max(latencies)

    return avg, p95, p99


def append_row(csv_file, row):
    exists = os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_detail(db, index, latency, use_filter):
    append_row(DETAIL_CSV, {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_label": DATASET_LABEL,
        "db": db,
        "query_index": index,
        "filter": "category=A" if use_filter else "none",
        "latency_ms": f"{latency*1000:.4f}"
    })


def summary_row(db, use_filter, avg, p95, p99, throughput,
                cpu_avg, cpu_max, cpu_proc_avg, cpu_proc_max,
                mem_avg, mem_max):

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_label": DATASET_LABEL,
        "db": db,
        "filter": "category=A" if use_filter else "none",
        "n_queries": N_QUERIES,
        "dim": DIM,
        "top_k": TOP_K,
        "mean_ms": f"{avg*1000:.4f}",
        "p95_ms": f"{p95*1000:.4f}",
        "p99_ms": f"{p99*1000:.4f}",
        "throughput_qps": f"{throughput:.4f}",
        "cpu_total_avg": f"{cpu_avg:.2f}",
        "cpu_total_max": f"{cpu_max:.2f}",
        "cpu_proc_avg": f"{cpu_proc_avg:.2f}",
        "cpu_proc_max": f"{cpu_proc_max:.2f}",
        "mem_avg_mb": f"{mem_avg:.2f}",
        "mem_max_mb": f"{mem_max:.2f}"
    }


# ======================================================
# BENCHMARK: CHROMA
# ======================================================

def benchmark_chroma(queries, use_filter=False):

    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)

    cpu_total_load = []
    cpu_proc_load = []
    mem_proc_load = []

    proc = psutil.Process(os.getpid())
    latencies = []

    # Warm-up
    for q in queries[:5]:
        kwargs = {"query_embeddings": [q.tolist()], "n_results": TOP_K}
        if use_filter:
            kwargs["where"] = {"category": "A"}
        collection.query(**kwargs)

    t_global0 = time.time()

    # Queries reales
    for idx, q in enumerate(queries):

        # CPU / MEM BEFORE QUERY
        cpu_total_load.append(psutil.cpu_percent(interval=None))
        cpu_proc_load.append(proc.cpu_percent(interval=None))
        mem_proc_load.append(proc.memory_info().rss / (1024 * 1024))

        kwargs = {"query_embeddings": [q.tolist()], "n_results": TOP_K}
        if use_filter:
            kwargs["where"] = {"category": "A"}

        t0 = time.time()
        _ = collection.query(**kwargs)
        t1 = time.time()

        lat = t1 - t0
        latencies.append(lat)

        save_detail("chroma", idx, lat, use_filter)

    t_global1 = time.time()

    throughput = len(queries) / (t_global1 - t_global0)

    return (
        latencies,
        throughput,
        statistics.mean(cpu_total_load),
        max(cpu_total_load),
        statistics.mean(cpu_proc_load),
        max(cpu_proc_load),
        statistics.mean(mem_proc_load),
        max(mem_proc_load)
    )


# ======================================================
# BENCHMARK: PGVECTOR
# ======================================================

def benchmark_pg(queries, use_filter=False):

    conn = get_pg_conn()
    cur = conn.cursor()

    cpu_total_load = []
    cpu_proc_load = []
    mem_proc_load = []

    proc = psutil.Process(os.getpid())
    latencies = []

    # Warm-up
    for q in queries[:5]:
        vec = "[" + ",".join(str(float(x)) for x in q) + "]"
        cur.execute(
            f"SELECT id FROM {PG_TABLE} ORDER BY embedding <-> %s::vector LIMIT %s",
            (vec, TOP_K)
        )
        cur.fetchall()

    t_global0 = time.time()

    for idx, q in enumerate(queries):

        cpu_total_load.append(psutil.cpu_percent(interval=None))
        cpu_proc_load.append(proc.cpu_percent(interval=None))
        mem_proc_load.append(proc.memory_info().rss / (1024*1024))

        vec = "[" + ",".join(str(float(x)) for x in q) + "]"

        t0 = time.time()

        if use_filter:
            cur.execute(
                f"""
                SELECT id FROM {PG_TABLE}
                WHERE metadata->>'category' = 'A'
                ORDER BY embedding <-> %s::vector
                LIMIT %s
                """,
                (vec, TOP_K)
            )
        else:
            cur.execute(
                f"""
                SELECT id FROM {PG_TABLE}
                ORDER BY embedding <-> %s::vector
                LIMIT %s
                """,
                (vec, TOP_K)
            )

        cur.fetchall()
        t1 = time.time()

        lat = t1 - t0
        latencies.append(lat)

        save_detail("pgvector", idx, lat, use_filter)

    t_global1 = time.time()
    throughput = len(queries) / (t_global1 - t_global0)

    cur.close()
    conn.close()

    return (
        latencies,
        throughput,
        statistics.mean(cpu_total_load),
        max(cpu_total_load),
        statistics.mean(cpu_proc_load),
        max(cpu_proc_load),
        statistics.mean(mem_proc_load),
        max(mem_proc_load)
    )


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    print(f"\n=== QUERY BENCHMARK ({DATASET_LABEL}) ===")

    queries = generate_query_vectors(N_QUERIES, DIM)

    # --------------------------------------------------
    # WITHOUT FILTER
    # --------------------------------------------------
    print("\n--- WITHOUT FILTER ---")

    (
        lat_c, thr_c, cpu_c, cpu_c_max,
        cpu_pc, cpu_pc_max, mem_c, mem_c_max
    ) = benchmark_chroma(queries, False)

    avg_c, p95_c, p99_c = summarize_latencies(lat_c)

    (
        lat_p, thr_p, cpu_p, cpu_p_max,
        cpu_pp, cpu_pp_max, mem_p, mem_p_max
    ) = benchmark_pg(queries, False)

    avg_p, p95_p, p99_p = summarize_latencies(lat_p)

    append_row(SUMMARY_CSV, summary_row(
        "chroma", False, avg_c, p95_c, p99_c,
        thr_c, cpu_c, cpu_c_max, cpu_pc, cpu_pc_max,
        mem_c, mem_c_max
    ))

    append_row(SUMMARY_CSV, summary_row(
        "pgvector", False, avg_p, p95_p, p99_p,
        thr_p, cpu_p, cpu_p_max, cpu_pp, cpu_pp_max,
        mem_p, mem_p_max
    ))


    # --------------------------------------------------
    # WITH FILTER
    # --------------------------------------------------
    print("\n--- WITH FILTER (category=A) ---")

    (
        lat_cf, thr_cf, cpu_cf, cpu_cf_max,
        cpu_pcf, cpu_pcf_max, mem_cf, mem_cf_max
    ) = benchmark_chroma(queries, True)

    avg_cf, p95_cf, p99_cf = summarize_latencies(lat_cf)

    (
        lat_pf, thr_pf, cpu_pf, cpu_pf_max,
        cpu_ppf, cpu_ppf_max, mem_pf, mem_pf_max
    ) = benchmark_pg(queries, True)

    avg_pf, p95_pf, p99_pf = summarize_latencies(lat_pf)

    append_row(SUMMARY_CSV, summary_row(
        "chroma", True, avg_cf, p95_cf, p99_cf,
        thr_cf, cpu_cf, cpu_cf_max, cpu_pcf, cpu_pcf_max,
        mem_cf, mem_cf_max
    ))

    append_row(SUMMARY_CSV, summary_row(
        "pgvector", True, avg_pf, p95_pf, p99_pf,
        thr_pf, cpu_pf, cpu_pf_max, cpu_ppf, cpu_ppf_max,
        mem_pf, mem_pf_max
    ))

    print("\nBenchmark complete!")
    print(f"Summary saved in {SUMMARY_CSV}")
    print(f"Detail saved in {DETAIL_CSV}")
