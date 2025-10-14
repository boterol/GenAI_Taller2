
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient

# === Cargar documentos ===
reader = SimpleDirectoryReader(input_dir="./data")
documents = reader.load_data()

# === Inicializar Qdrant (contenedor Docker) ===
qdrant_client = QdrantClient(url="http://localhost:6333")

# Crear vector stores separados por agente
agents_collections = {
    "devoluciones": QdrantVectorStore(collection_name="devoluciones", client=qdrant_client),
    "pedidos": QdrantVectorStore(collection_name="pedidos", client=qdrant_client),
    "faq": QdrantVectorStore(collection_name="faq", client=qdrant_client),
}

# Crear StorageContext para cada agente
storage_contexts = {
    agent: StorageContext.from_defaults(vector_store=vector_store)
    for agent, vector_store in agents_collections.items()
}

# === Configurar Ollama y embeddings locales ===
print("Initializing Ollama and embeddings...")
llm = Ollama(model="mistral", request_timeout=100)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Crear índices para cada agente ===
print("Creating indexes for all agents...")
indexes = {}
for agent, storage_context in storage_contexts.items():
    print(f"Creating index for {agent}...")
    indexes[agent] = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# === Función para seleccionar agente ===
def choose_agent():
    print("\nSelect an agent:")
    print("1 - Devoluciones")
    print("2 - Pedidos")
    print("3 - Preguntas y Respuestas")
    print("0 - Exit")
    choice = input("Enter your choice: ").strip()
    mapping = {"1": "devoluciones", "2": "pedidos", "3": "faq", "0": "exit"}
    return mapping.get(choice, None)

# === Bucle principal de chat ===
while True:
    agent_choice = choose_agent()
    if agent_choice == "exit":
        print("Exiting...")
        break
    elif agent_choice is None:
        print("Invalid choice. Try again.")
        continue

    query_engine = indexes[agent_choice].as_query_engine()
    print(f"\n--- Using agent: {agent_choice} ---")
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() == "back":
            print("Returning to agent menu...")
            break
        elif user_input.lower().strip() == "exit":
            exit()
        response = query_engine.query(user_input)
        print(response)
