import pdfplumber
import json
import os
import tiktoken

# ----------------------
# 1. Extract text from PDF
# ----------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts and concatenates all text from a PDF file.
    """
    full_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:  # Some pages may be empty or just images
                    full_text.append(text)
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")
    
    return "\n".join(full_text)

# ----------------------
# 2. Token-based chunking with overlap
# ----------------------
def tokenize_and_chunk(text: str, chunk_size=500, overlap=50, model="cl100k_base"):
    """
    Splits text into token-based chunks with overlap using tiktoken.
    
    Args:
        text (str): Input text.
        chunk_size (int): Number of tokens per chunk.
        overlap (int): Number of overlapping tokens between chunks.
        model (str): Encoding model (cl100k_base works for OpenAI embeddings).
    
    Returns:
        list[str]: List of text chunks.
    """
    enc = tiktoken.get_encoding(model)
    tokens = enc.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i+chunk_size]
        chunks.append(enc.decode(chunk_tokens))
    
    return chunks

# ----------------------
# 3. JSONify pipeline
# ----------------------
def jsonify(paths: list, chunk_size: int, overlap: int, output_filename: str = "fitness_books.json"):
    """
    Convert PDFs into JSON file with author and text chunks (token-based).
    
    Args:
        paths (list): List of PDF paths.
        chunk_size (int): Max number of tokens per chunk.
        overlap (int): Overlap between chunks.
        output_filename (str): Name of the JSON file to save.
    """
    data = []
    
    for pdf_path in paths:
        # Use the filename as the "author" field
        author_name = os.path.basename(pdf_path).replace(".pdf", "")
        
        # Extract all text
        full_text = extract_text_from_pdf(pdf_path)
        
        # Tokenize + chunk with overlap
        chunks = tokenize_and_chunk(full_text, chunk_size=chunk_size, overlap=overlap)
        
        # Build JSON objects
        for chunk in chunks:
            data.append({
                "author": author_name,
                "text": chunk
            })
    
    # Save JSON to the back-end folder
    output_path = r"C:\Users\adamh\Desktop\fit_bot\back-end" + "\\" + output_filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ JSON saved at {output_path} with {len(data)} chunks.")


# ----------------------
# 4. Example usage
# ----------------------
pdf1 = r"C:\Users\adamh\Desktop\fit_bot\back-end\fitness_books\Beyond Bigger Leaner Stronger - Michael Matthews.pdf"
pdf2 = r"C:\Users\adamh\Desktop\fit_bot\back-end\fitness_books\The Lean Muscle Diet PDF.pdf"
pdf3 = r"C:\Users\adamh\Desktop\fit_bot\back-end\fitness_books\Science_fitness_book.pdf"

jsonify([pdf1, pdf2, pdf3], chunk_size=500, overlap=50, output_filename="fitness_books.json")
