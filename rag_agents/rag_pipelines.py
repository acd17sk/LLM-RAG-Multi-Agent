from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from vector_store.retriever import retrieve_unique_docs
from llm.local_llm import LocalLLM
from vector_store.retriever import rerank_docs
from IPython.display import display, Markdown

import json



def run_multi_query_rag_agent(
    user_query: str,
    subqueries: List[str],
    embedding_model: SentenceTransformer,
    CHROMA_PATH: str,
    COLLECTION_NAME: str,
    reranker: CrossEncoder = None,
    final_n_results: int = 5,
    initial_fetch: int = 15
) -> str:
    """
    Orchestrates the multi-query retrieval process according to the corrected logic.
    """
    print("\n--- 3. Retrieval Agent: Retrieving Context ---")

    # Step 1: Retrieve a unique set of documents for all subqueries
    retrieved_docs = retrieve_unique_docs(
        subqueries, 
        embedding_model,  
        CHROMA_PATH=CHROMA_PATH, 
        COLLECTION_NAME=COLLECTION_NAME,
        fetch_k=initial_fetch
        )
    
    # Step 2: Pre-Reranker Selection - Select the top initial_fetch from the unique set
    if not retrieved_docs:
        return "Could not find any relevant documents."

    if len(subqueries) >1:
        print(f"\n--- Pre-selecting top {initial_fetch} docs from {len(retrieved_docs)} unique docs ---")

        # Get embeddings for the original query and all unique documents
        original_query_embedding = embedding_model.encode(user_query)
        doc_embeddings = embedding_model.encode([doc['text'] for doc in retrieved_docs])
        
        # Calculate cosine similarity between the original query and each document
        # Note: For normalized embeddings, dot product is equivalent to cosine similarity
        similarities = np.dot(doc_embeddings, original_query_embedding.T)
        
        # Add the similarity score to each document
        for i, doc in enumerate(retrieved_docs):
            doc['pre_score'] = similarities[i]
            
        # Sort by this pre-score to find the best candidates
        retrieved_docs.sort(key=lambda x: x['pre_score'], reverse=True)
    
    # Select the top 15 documents to pass to the reranker
    docs_for_reranking = retrieved_docs[:initial_fetch]
    
    # Step 3: Conditionally rerank the selected initial_fetch documents
    if reranker:
        final_docs = rerank_docs(docs_for_reranking, user_query, reranker)
    else:
        print("\nReranker not provided. Skipping the reranking step.")
        final_docs = docs_for_reranking
        
    # Step 4: Select the final top N documents and format the context
    top_docs = final_docs[:final_n_results]
    
    context = "\n\n---\n\n".join([
        f"[Source: {chunk['metadata']['source']}, Page: {chunk['metadata']['page']}] {chunk['text']}"
        for chunk in top_docs
    ])
    
    print("Context retrieved successfully.")
    print("-"*50)
    print(context)
    print("-"*50)
    return context


def run_query_decomposition_agent(user_query: str, llm_client: LocalLLM) -> List[str]:
    """
    Uses the local LLM to break a complex query into smaller, searchable subqueries.
    """
    print("\n--- 2. Query Decomposition Agent: Generating Subqueries ---")
    
    prompt = f"""\
You are an expert at query decomposition. Your task is to break down a user's question into 2-4 smaller, self-contained questions that can be used to search a vector database.
- If the question is simple, return only the original question in the list.
- If the question is complex or comparative, break it down effectively for more accurate semantic retrieval.

User Question: "{user_query}"
"""
    
    # Always include the original query as a baseline
    subqueries = [user_query]
    
    generated_queries = llm_client.decompose_query(prompt, user_query)
    for q in generated_queries:
        if q.lower() not in [sq.lower() for sq in subqueries]:
            subqueries.append(q)
            
    print(f"Generated {len(subqueries)} queries: {json.dumps(subqueries, indent=2, ensure_ascii=False)}")
    return subqueries



def run_orchestrator_agent(user_query: str, llm_client: LocalLLM):
    """
    Uses the local LLM to make a structured routing decision.
    """
    print("--- 1. Orchestrator Agent: Routing Query ---")
    prompt = f"""\
Analyze the user's query and decide the next action.
- If the query is about medical device regulations, FDA, WHO, design controls, etc., choose 'SEARCH' and refine the query.
- If the query is a greeting or chit-chat, choose 'ANSWER_DIRECTLY'.

User Query: "{user_query}"
"""
    decision = llm_client.decide(prompt)
    # print(decision)
    print(f"Orchestrator Decision: action='{decision['action']}'")
    return decision



def run_response_agent(user_query: str, context: str, llm_client: LocalLLM, max_new_tokens: int, temp: float = 0.1) -> str:
    """Uses the local LLM to generate a final answer."""
    print("\n--- 4. Response Agent: Generating Final Answer ---")
    # The system prompt is now simpler and more direct.
    sys_prompt = """\
You are a precise regulatory assistant.
- Your task is to answer the user's question based ONLY on the provided context.
- When you use information from the context, you MUST copy its corresponding citation tag, like `[Source: document.pdf, Page X]`, immediately after the information.
- If the context is insufficient, state that clearly.
/no_think
"""

    # This prompt provides a very clear "few-shot" example, which is the most
    # effective way to teach a small model the desired output format.
    prompt = f"""\
Context:
{context}
User Question:
{user_query}

Do not forget to use the citation for every information used in your answer from the context.
Output:
"""


    final_answer = llm_client.generate(sys_prompt, prompt, max_new_tokens=max_new_tokens, temp=temp)
    return final_answer


def get_rag_response(
    user_query: str,
    local_llm: Any,
    embedding_model: Any,
    reranker: Any,
    CHROMA_PATH: str,
    COLLECTION_NAME: str,
    final_n_results: int = 5,
    initial_fetch: int = 15,
    max_new_tokens: int = 512,
    rag_temp: float = 0.05,
    direct_temp: float = 0.3
):
    """
    Executes the full RAG pipeline based on an orchestrator's decision.

    This function routes a user query to either a full retrieval-augmented
    generation pipeline or a direct answer from the LLM, and then
    displays the final answer.

    Args:
        user_query (str): The input question from the user.
        local_llm (Any): The initialized local LLM client.
        embedding_model (Any): The initialized sentence embedding model.
        reranker (Any): The initialized cross-encoder reranker model.
        CHROMA_PATH (str): The file path to the ChromaDB database.
        COLLECTION_NAME (str): The name of the collection within ChromaDB.
        final_n_results (int, optional): The number of documents to use for the final context. Defaults to 5.
        initial_fetch (int, optional): The number of documents to initially retrieve. Defaults to 15.
        max_new_tokens (int, optional): The maximum number of tokens for the generated response. Defaults to 512.
        rag_temp (float, optional): The temperature for the RAG-based response generation. Defaults to 0.05.
        direct_temp (float, optional): The temperature for the direct LLM response. Defaults to 0.3.
    """
    # 1. Orchestrator decides the action
    orchestrator_decision = run_orchestrator_agent(user_query, local_llm)

    # 2. Conditionally run the query deconstruction and retrieval pipelines
    if orchestrator_decision.get("action") == "SEARCH":
        print("--- Action: SEARCH ---")
        subqueries = run_query_decomposition_agent(user_query, local_llm)

        # 3. Conditionally run the Retrieval pipeline
        retrieved_context = run_multi_query_rag_agent(
            user_query=user_query,
            subqueries=subqueries,
            embedding_model=embedding_model,
            reranker=reranker,
            CHROMA_PATH=CHROMA_PATH,
            COLLECTION_NAME=COLLECTION_NAME,
            final_n_results=final_n_results,
            initial_fetch=initial_fetch
        )

        # 4a. Run RAG model output
        final_answer = run_response_agent(
            user_query,
            retrieved_context,
            local_llm,
            max_new_tokens=max_new_tokens,
            temp=rag_temp
        )

    else:
        print("--- Action: ANSWER_DIRECTLY ---")
        # 4b. Run direct model output
        final_answer = local_llm.generate(
            None,
            user_query,
            max_new_tokens=max_new_tokens,
            temp=direct_temp
        )

    # Print the final result
    print("\n" + "="*50)
    print("✅ Final Answer")
    print("="*50)
    display(Markdown(final_answer))
