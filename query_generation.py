import numpy as np

DIM = 128           # misma dimensión que tus embeddings
N_QUERIES = 100     # cuántas queries quieres generar
CATEGORIES = ["A", "B", "C"]

def generate_queries(n_queries, dim, with_filters=False):
    """
    Devuelve una lista de queries.
    Cada query es un dict con:
      - 'vector': el vector de consulta (lista de floats)
      - 'filter': None o un filtro tipo {"category": "A"}
    """
    queries = []
    # vectores aleatorios del mismo espacio que tus datos
    q_vectors = np.random.rand(n_queries, dim).astype("float32")

    for i in range(n_queries):
        q_vec = q_vectors[i].tolist()

        if with_filters:
            # escogemos una categoría aleatoria como filtro
            cat = np.random.choice(CATEGORIES)
            q_filter = {"category": cat}
        else:
            q_filter = None

        queries.append({
            "vector": q_vec,
            "filter": q_filter
        })

    return queries


if __name__ == "__main__":
    # Ejemplo: generar 100 queries sin filtro
    queries_no_filter = generate_queries(N_QUERIES, DIM, with_filters=False)
    print("Example query without filter:", queries_no_filter[0])

    # Ejemplo: generar 100 queries con filtro por category
    queries_with_filter = generate_queries(N_QUERIES, DIM, with_filters=True)
    print("Example query with filter:", queries_with_filter[0])
