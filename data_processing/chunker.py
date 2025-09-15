from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_data(data: List[Dict[str, Any]], chunk_size: int = 1024, chunk_overlap: int = 256) -> List[Dict[str, Any]]:
    """
    Chunks the text content of documents from the parsing step.

    Args:
        data: The list of page data dictionaries from the parse_pdfs function.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of dictionaries, where each dictionary represents a chunk
        with its text content and original metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = []
    current_header = ""
    print(f"Starting contextual chunking on {len(data)} blocks...")

    for block in data:
        if block['type'] == 'header':
            # Update the current header
            current_header = block['text']
        elif block['type'] == 'paragraph':
            # This is a paragraph, so combine it with the current header
            # We create a combined text that gives context to the paragraph
            contextual_text = f"{current_header}\n\n{block['text']}"

            # Now, split this combined text into chunks
            # This handles cases where a single paragraph is still too long
            page_chunks = text_splitter.create_documents(
                texts=[contextual_text],
                metadatas=[block['metadata']] # Use the paragraph's metadata
            )

            # Convert to our desired dictionary format
            for chunk_doc in page_chunks:
                chunk_dict = {
                    "text": chunk_doc.page_content,
                    "metadata": chunk_doc.metadata
                }
                chunks.append(chunk_dict)

    print(f"Contextual chunking complete. Created {len(chunks)} chunks.")
    return chunks
