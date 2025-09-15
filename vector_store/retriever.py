import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


def query_vector_store(
    query: str,
    embedding_model,
    CHROMA_PATH: str,
    COLLECTION_NAME: str,
    fetch_k: int = 15
) -> List[Dict[str, Any]]:
    """
    Queries the vector store for a single query and returns the results.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k
    )

    # Format the results into a consistent list of dictionaries
    formatted_results = []
    for i, doc_id in enumerate(results['ids'][0]):
        formatted_results.append({
            "id": doc_id,
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "score": results['distances'][0][i]
        })
        
    return formatted_results


def retrieve_unique_docs(
    subqueries: List[str],
    embedding_model,
    CHROMA_PATH: str,
    COLLECTION_NAME: str,
    fetch_k: int = 15
) -> List[Dict[str, Any]]:
    """
    Orchestrates fetching docs for a list of subqueries using the helper
    function and returns a deduplicated list.
    """
    unique_docs = {} # Use a dictionary to automatically handle duplicates
    print(f"\n--- Retrieving docs for {len(subqueries)} queries ---")
    
    for query in subqueries:
        print(f"  - Querying: '{query}'")
        # **THE CHANGE**: Use the dedicated, single-query function here
        results = query_vector_store(
            query, 
            embedding_model, 
            CHROMA_PATH=CHROMA_PATH,
            COLLECTION_NAME=COLLECTION_NAME,
            fetch_k=fetch_k
        )
        
        for doc in results:
            if doc['id'] not in unique_docs:
                unique_docs[doc['id']] = doc
    
    unique_doc_list = list(unique_docs.values())
    print(f"Retrieved {len(unique_doc_list)} unique documents in total.")
    return unique_doc_list


def rerank_docs(
    docs: List[Dict[str, Any]],
    original_query: str,
    reranker
) -> List[Dict[str, Any]]:
    """
    Reranks a list of documents against the original user query.
    """
    rerank_pairs = [[original_query, doc['text']] for doc in docs]
    
    print(f"\nReranking the top {len(rerank_pairs)} unique documents...")
    scores = reranker.predict(rerank_pairs, batch_size=16, show_progress_bar=True)
    
    # Add the score to each document
    for i, doc in enumerate(docs):
        doc['score'] = scores[i]
        
    # Sort by the new score
    docs.sort(key=lambda x: x.get('score', 0), reverse=True)
    return docs

