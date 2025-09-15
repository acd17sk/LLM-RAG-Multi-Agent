import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

def clear_vector_store(CHROMA_PATH, COLLECTION_NAME):
    """
    Deletes the ChromaDB collection from the persistent storage.
    """
    try:
        print("Initializing ChromaDB client to clear the store...")
        client = chromadb.PersistentClient(path=CHROMA_PATH)

        print(f"Attempting to delete collection: '{COLLECTION_NAME}'")
        client.delete_collection(name=COLLECTION_NAME)

        print("Vector store collection has been successfully cleared.")
    except ValueError:
        # This error is often raised if the collection doesn't exist
        print(f"Collection '{COLLECTION_NAME}' not found or already deleted. No action taken.")
    except Exception as e:
        print(f"An error occurred while trying to clear the vector store: {e}")



def create_and_populate_vector_store(chunks: List[Dict[str, Any]], CHROMA_PATH: str, COLLECTION_NAME: str, embedding_model: SentenceTransformer):
    """
    Creates embeddings for the chunks and stores them in ChromaDB.

    Args:
        chunks: A list of chunk dictionaries from the chunking step.
    """
    print("Initializing ChromaDB client...")
    # Using a persistent client to save the DB to disk
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    print(f"Getting or creating ChromaDB collection: '{COLLECTION_NAME}'")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Prepare data for ChromaDB
    documents = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print(f"Generating embeddings for {len(documents)} documents...")
    # Generate embeddings in batches for efficiency
    embeddings = embedding_model.encode(documents, show_progress_bar=True)

    print("Adding documents to the collection in batches...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))

        collection.add(
            embeddings=embeddings[i:batch_end].tolist(), # ChromaDB expects lists
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end]
        )
        print(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")

    print("Vector store creation and population complete.")
    print(f"Total documents in collection: {collection.count()}")