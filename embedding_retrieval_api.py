import logging
import time
from threading import Lock
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for API ---
class AddRequest(BaseModel):
    collection_name: str
    texts: list[str]

class AddResponse(BaseModel):
    message: str
    documents_received: int
    chunks_added: int

class RetrievalRequest(BaseModel):
    collection_name: str
    query: str
    k: int = 3

class Doc(BaseModel):
    page_content: str
    metadata: dict

class RetrievalResponse(BaseModel):
    matches: list[Doc]
    retrieval_time_seconds: float

# --- Thread-Safe Retriever and Indexer Class ---
class EmbeddingService:
    def __init__(self):
        self.lock = Lock()
        self.ollama_url = "http://13.59.72.219:11434"
        self.qdrant_url = "http://13.59.72.219:6333"

        logger.info("Initializing Ollama embeddings client...")
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=self.ollama_url
        )
        logger.info("✅ Ollama embeddings client ready.")

        logger.info("Initializing Qdrant client...")
        self.qdrant_client = QdrantClient(url=self.qdrant_url)
        logger.info("✅ Qdrant client ready.")

        logger.info("Initializing text splitter...")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # The size of each text chunk in characters
            chunk_overlap=200, # The number of characters to overlap between chunks
            length_function=len,
        )
        logger.info("✅ Text splitter ready.")

    def add(self, collection_name: str, texts: list[str]) -> int:
        """
        Splits, embeds, and stores texts in a Qdrant collection.
        If the collection doesn't exist, it will be created automatically.
        """
        with self.lock:
            try:
                logger.info(f"Received {len(texts)} documents to split and add to collection '{collection_name}'")

                # Split the documents into smaller chunks suitable for embedding.
                # create_documents handles a list of texts correctly.
                documents = self.text_splitter.create_documents(texts)

                if not documents:
                    logger.warning("No text chunks were generated after splitting. Nothing to add.")
                    return 0

                logger.info(f"Split documents into {len(documents)} chunks. Now embedding and adding to Qdrant...")

                # Instantiate the Qdrant class with the existing client.
                qdrant_store = Qdrant(
                    client=self.qdrant_client,
                    collection_name=collection_name,
                    embeddings=self.embeddings,
                )

                # Now, add the documents to the store.
                # This will create the collection automatically if it doesn't exist.
                qdrant_store.add_documents(documents)

                logger.info("✅ Chunks added successfully.")
                return len(documents)
            except Exception as e:
                logger.error(f"❌ Failed to add documents to Qdrant: {e}")
                raise RuntimeError(f"Qdrant add operation failed: {e}")

    def search(self, collection_name: str, query: str, k: int) -> list[dict]:
        """Performs a similarity search in a specified Qdrant collection."""
        with self.lock:
            try:
                logger.info(f"Connecting to Qdrant collection: '{collection_name}'")
                # Instantiate the Qdrant store directly with the client for searching
                qdrant_store = Qdrant(
                    client=self.qdrant_client,
                    collection_name=collection_name,
                    embeddings=self.embeddings,
                )
                logger.info(f"Performing similarity search for query: '{query}'")
                found_docs = qdrant_store.similarity_search(query=query, k=k)
                return [doc.dict() for doc in found_docs]
            except Exception as e:
                logger.error(f"❌ Failed to search in Qdrant: {e}")
                raise RuntimeError(f"Qdrant search failed (collection might not exist): {e}")

# --- FastAPI App ---
app = FastAPI(
    title="Embedding and Retrieval API",
    description="An API to add and search for text embeddings in a Qdrant vector store.",
    version="1.1.0"
)

service: EmbeddingService = None

@app.on_event("startup")
async def startup_event():
    global service
    service = EmbeddingService()

# --- API Endpoints ---

@app.post("/add", response_model=AddResponse)
async def add_documents(request: AddRequest):
    """
    Embed a list of texts and add them to a Qdrant collection.
    If the collection doesn't exist, it will be created.
    """
    if not service:
        raise HTTPException(status_code=503, detail="Embedding service not loaded.")
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided to add.")

    try:
        start_time = time.time()
        num_chunks_added = service.add(collection_name=request.collection_name, texts=request.texts)
        add_time = time.time() - start_time
        logger.info(f"✅ Add operation completed in {add_time:.4f}s.")

        return AddResponse(
            message=f"Successfully added {num_chunks_added} chunks from {len(request.texts)} documents.",
            documents_received=len(request.texts),
            chunks_added=num_chunks_added,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Add operation failed: {e}")

@app.post("/search", response_model=RetrievalResponse)
async def search_embeddings(request: RetrievalRequest):
    """
    Get the top 'k' most similar documents from a Qdrant collection for a given query.
    """
    if not service:
        raise HTTPException(status_code=503, detail="Embedding service not loaded.")

    try:
        start_time = time.time()
        matches = service.search(
            collection_name=request.collection_name,
            query=request.query,
            k=request.k
        )
        retrieval_time = time.time() - start_time
        logger.info(f"✅ Search successful. Found {len(matches)} matches in {retrieval_time:.4f}s.")

        return RetrievalResponse(
            matches=[Doc(**doc) for doc in matches],
            retrieval_time_seconds=round(retrieval_time, 4)
        )
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "service_loaded": service is not None}
