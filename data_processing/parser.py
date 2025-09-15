import fitz
import os
import numpy as np
from typing import List, Dict, Any


def is_header(block: Dict[str, Any], avg_font_size: float) -> bool:
    """Heuristic to determine if a text block is a header."""
    # A block is a header if its font size is larger than average, it's bold,
    # and the line is relatively short.
    try:
        # Assumes the first line of a block determines its properties
        line = block['lines'][0]
        span = line['spans'][0]
        is_bold = "bold" in span['font'].lower()
        # Heuristic: Header if font size > avg + 1 OR if it's bold
        if span['size'] > avg_font_size + 1 or is_bold:
            # Further check: headers are usually short
            if len(span['text'].strip()) < 100 and not span['text'].strip().endswith('.'):
                return True
    except (IndexError, KeyError):
        return False
    return False

def get_avg_font_size(doc: fitz.Document) -> float:
    """Calculates the average font size across the first few pages."""
    font_sizes = []
    # Analyze first 5 pages or all if fewer
    num_pages_to_scan = min(len(doc), 5)
    for page_num in range(num_pages_to_scan):
        page = doc[page_num]
        blocks = page.get_text("dict")['blocks']
        for block in blocks:
            if block['type'] == 0:  # Text block
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        font_sizes.append(span['size'])
    return np.mean(font_sizes) if font_sizes else 12.0 # Default size

def parse_pdfs_structured(pdf_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Parses PDFs and extracts structured text blocks (headers, paragraphs).

    Args:
        pdf_paths: A list of file paths to the PDF documents.

    Returns:
        A list of dictionaries, each representing a structured text block.
    """
    structured_data = []
    print(f"Starting structured parsing for {len(pdf_paths)} PDF file(s)...")

    for pdf_path in pdf_paths:
        try:
            file_name = os.path.basename(pdf_path)
            doc = fitz.open(pdf_path)

            # Calculate the document's average font size for better header detection
            avg_font_size = get_avg_font_size(doc)

            print(f"Processing '{file_name}' with avg font size {avg_font_size:.2f}...")

            for page_num, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block['type'] == 0:  # This is a text block
                        block_text = ""
                        for line in block.get('lines', []):
                            for span in line.get('spans', []):
                                block_text += span['text']
                            block_text += "\n"

                        block_text = block_text.strip()
                        if len(block_text) < 20: # Filter out very short, likely irrelevant text
                            continue

                        # Classify the block
                        if is_header(block, avg_font_size):
                            block_type = "header"
                        else:
                            block_type = "paragraph"

                        structured_data.append({
                            "type": block_type,
                            "text": block_text,
                            "metadata": {
                                "source": file_name,
                                "page": page_num + 1
                            }
                        })
            doc.close()
        except Exception as e:
            print(f"Could not process file {pdf_path}. Reason: {e}")

    print(f"Structured parsing complete. Extracted {len(structured_data)} blocks.")
    return structured_data


def parse_pdfs_in_folder(pdf_folder):

    # Ensure the directory exists before proceeding
    if not os.path.isdir(pdf_folder):
        print(f"Error: The directory '{pdf_folder}' was not found.")
        print("Please create it and place your PDF files inside.")
        return None
    else:
        # List of PDF files to be processed
        # This will automatically find all .pdf files in the specified folder
        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

        if not pdf_files:
            print(f"No PDF files found in the '{pdf_folder}' directory.")
            return None

        else:
            # Call the parsing function
            pages_data = parse_pdfs_structured(pdf_files)

            if pages_data:
                print("\n--- Sample of Structured Data ---")
                # Print a few blocks to see the classification
                for item in pages_data[20:25]: # Look at a sample slice
                    print(f"Type: {item['type']}")
                    print(f"Source: {item['metadata']['source']}, Page: {item['metadata']['page']}")
                    print(f"Text: {item['text'][:150]}...\n")
                print("---------------------------------")
            return pages_data

