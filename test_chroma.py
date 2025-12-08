import chromadb

# Connect to Chroma server
client = chromadb.HttpClient(host="localhost", port=8000)

# Create or open a collection
collection = client.get_or_create_collection("test_collection")

# Insert test vectors
collection.add(
    ids=["1", "2"],
    embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    metadatas=[{"type": "example"}, {"type": "example"}]
)

# List collections
print("Collections:", client.list_collections())

# Query
result = collection.query(
    query_embeddings=[[1.0, 2.0, 3.0]],
    n_results=1
)

print("Query Result:", result)
