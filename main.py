import os
import csv
import json
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient
import pdfplumber
import tiktoken

# === Load .env ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No se encontró la variable de entorno OPENAI_API_KEY")

# Función para dividir texto en chunks de N tokens
def chunk_text(text: str, chunk_size: int = 512, model_name: str = "gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

# === Load documents con chunks de 512 tokens ===
def load_pdf(file_path, chunk_size=512):
    documents = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for chunk in chunk_text(text, chunk_size):
                    documents.append(Document(text=chunk))
    return documents

pdf_docs = load_pdf("./data/devoluciones/devoluciones.pdf", chunk_size=512)




class CSVReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        documents = []
        with open(self.file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (
                    f"El id del pedido es {row['id']}, Comprado por {row['nombre_comprador']}, "
                    f"El Producto comprado es {row['nombre_producto']}, Cantidad: {row['cantidad']}, "
                )
                # Toda la info del pedido en metadata
                metadata = {
                    "pedido_id": row["id"],
                    "nombre_comprador": row["nombre_comprador"],
                    "email": row["email"],
                    "nombre_producto": row["nombre_producto"],
                    "cantidad": row["cantidad"],
                    "precio": row["precio"],
                    "total": row["total"],
                    "estado": row["estado"],
                    "fecha_pedido": row["fecha_pedido"],
                    "fecha_entrega": row["fecha_entrega"]
                }
                documents.append(Document(text=text, metadata=metadata))
        return documents

csv_docs = CSVReader("./data/pedidos/pedidos.csv").load_data()

class JSONReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        documents = []
        with open(self.file_path, encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                text = f"Pregunta: {item['question']} -> Respuesta: {item['answer']}"
                documents.append(Document(text=text))
        return documents

json_docs = JSONReader("./data/faq/faq.json").load_data()

# === Initialize Qdrant ===
qdrant_client = QdrantClient(url="http://localhost:6333")
agents_collections = {
    "devoluciones": QdrantVectorStore(collection_name="devoluciones", client=qdrant_client),
    "pedidos": QdrantVectorStore(collection_name="pedidos", client=qdrant_client),
    "faq": QdrantVectorStore(collection_name="faq", client=qdrant_client),
}

storage_contexts = {
    agent: StorageContext.from_defaults(vector_store=vector_store)
    for agent, vector_store in agents_collections.items()
}

# === Set up OpenAI LLM and embeddings ===
llm = OpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0)
Settings.llm = llm
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# === Load system prompts for each agent ===
def load_system_prompt(agent_name: str) -> str:
    prompt_file = f"./prompts/{agent_name}.txt"
    if not os.path.exists(prompt_file):
        return ""  # si no existe, usar prompt vacío
    with open(prompt_file, encoding="utf-8") as f:
        return f.read()

system_prompts = {
    "devoluciones": load_system_prompt("devoluciones"),
    "pedidos": load_system_prompt("pedidos"),
    "faq": load_system_prompt("faq"),
}

# === Create indices ===
indexes = {}
for agent_name, docs in [("devoluciones", pdf_docs),
                         ("pedidos", csv_docs),
                         ("faq", json_docs)]:
    indexes[agent_name] = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_contexts[agent_name],
        embed_model=embed_model
    )

# === Agent selection function ===
def choose_agent():
    print("\nSelect an agent:")
    print("1 - Devoluciones")
    print("2 - Pedidos")
    print("3 - Preguntas y Respuestas")
    print("0 - Exit")
    choice = input("Enter your choice: ").strip()
    mapping = {"1": "devoluciones", "2": "pedidos", "3": "faq", "0": "exit"}
    return mapping.get(choice, None)

# === Main chat loop with per-agent system prompt ===
count=0
while True:
    agent_choice = choose_agent()
    if agent_choice == "exit":
        print("Exiting...")
        break
    elif agent_choice is None:
        print("Invalid choice. Try again.")
        continue

    # Use retriever explicitly with agent-specific system prompt
    query_engine = indexes[agent_choice].as_query_engine(
        retriever_mode="embedding",
        response_mode="compact",
        system_prompt=system_prompts[agent_choice]
    )

    print(f"\n--- Using agent: {agent_choice} ---")
    while True:
        if agent_choice=='pedidos' and count==0: 
            print("para preguntar sobre un pedido debe brindar toda la info posible. Id, nombre, producto, fecha de compra, etc...")
            print("estos  se debe a un modelo de embedding gratis y ligero que impide el retreival de calidad.")
            print()
            print("Idealmente se usaria un sistema de autenticacion donde se verifique si el pedido traido del RAG")
            print("Pertenece al usuario autenticado para no mostrar informacion de pedidos ajenos")
        user_input = input("You: ")
        if user_input.lower().strip() == "back":
            print("Returning to agent menu...")
            break
        elif user_input.lower().strip() == "exit":
            exit()
        
        # Filtrado especial para pedidos por ID exacto
        if agent_choice == "pedidos":
            retriever = indexes["pedidos"].as_retriever()
            results = retriever.retrieve(user_input)  # busca por embedding del ID
            if results:
                # Mostrar toda la metadata como respuesta
                pedido_info = results[0].metadata
                response_text = "\n".join([f"{k}: {v}" for k, v in pedido_info.items()])
                print(response_text)
                continue
            else:
                print(f"No se dispone de información sobre el pedido {user_input}")
                continue

        response = query_engine.query(user_input)
        print(response)
        count+=1
