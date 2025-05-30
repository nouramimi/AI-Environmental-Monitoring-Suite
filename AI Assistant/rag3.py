import ollama
import os
import fitz  # PyMuPDF for PDF processing
from sentence_transformers import SentenceTransformer
import faiss  # FAISS for vector storage
import numpy as np

# Constants
PDF_PATH = "data/100-ways-you-can-improve-the-environment.pdf"  # Fixed PDF path
MODEL = "minicpm-v"  # Change to the actual model you're using

# Initialize SentenceTransformer model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_pdf_text(pdf_path):
    # Open the PDF file using fitz
    document = fitz.open(pdf_path)  # Correct way to open the PDF file
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text("text")  # Extracts the text content from the page
    return text

# Function to generate query embeddings
def generate_query_embedding(query):
    """Generate an embedding for the query string."""
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    return query_embedding

# Function to split text into smaller chunks
def chunk_text(text, chunk_size=500):
    """Split text into chunks of a specified size."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# Function to build embeddings for text chunks
def build_embeddings(chunks):
    embeddings = embedder.encode(chunks, convert_to_tensor=True)
    return embeddings

def store_embeddings_in_faiss(embeddings):
    # Ensure embeddings are on the CPU
    embeddings_cpu = embeddings.cpu().detach().numpy()
    
    # Create a FAISS index (using L2 distance metric)
    dimension = embeddings_cpu.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)
    
    # Convert to a contiguous numpy array if needed
    embeddings_np = np.ascontiguousarray(embeddings_cpu, dtype='float32')
    
    # Add embeddings to the FAISS index
    index.add(embeddings_np)
    return index


def search_relevant_context(query, index, chunks):
    """Find the most relevant context for the query using FAISS."""
    query_embedding = generate_query_embedding(query)  # Generate the query embedding
    
    # Ensure query embedding is on CPU and in the correct format
    query_embedding_cpu = query_embedding.cpu().detach().numpy().reshape(1, -1)
    
    # Perform the search in the FAISS index
    D, I = index.search(query_embedding_cpu, k=1)  # Find top-1 most similar chunk
    
    # Retrieve the chunk based on the index I
    relevant_chunk = chunks[I[0][0]]  # Ensure chunks is a list of document chunks
    
    return relevant_chunk


# Build the FAISS index once at the start
def build_and_store_index(pdf_path):
    # Extract and chunk the PDF text
    pdf_text = extract_pdf_text(pdf_path)
    chunks = chunk_text(pdf_text)
    
    # Generate embeddings for each chunk
    embeddings = build_embeddings(chunks)
    
    # Store embeddings in FAISS
    index = store_embeddings_in_faiss(embeddings)
    
    return index, chunks

# Function to build the prompt dynamically
def build_prompt_with_context(query: str, index, chunks, image_path: str = None) -> dict:
    """
    Build a RAG prompt by combining the query with the most relevant context from the PDF.
    """
    # Retrieve the most relevant context for the query
    relevant_context = search_relevant_context(query, index, chunks)
    
    messages = [
        {
            "role": "user",
            "content": (
                "You are an expert AI assistant that will answer the following query strictly based on the context provided. "
                "Do not use any external knowledge or make assumptions beyond the provided content. "
                "You will receive a query along with context extracted from a PDF, and optionally, an image for reference. "
                "Your answer should strictly reference and use the provided context. "
                "If they ask you what you are here for or what you do,The structure of your response should be:\n\n"
                "1. Provide a summary of the relevant context from the PDF (if applicable).\n"
                "2. Answer the user's query directly, based solely on the context.\n"
                "3. If an image is provided, integrate its content only if necessary to support the answer.\n\n"
                "Keep the response concise and focused on the key points from the context."
                "If they chat with you normal conversation like hello how are you etc.. answer but if they ask you something outside the context just say: I am only here to help reduce pollution and save the world. "
                "Never say based on this context, the user should not know that you have a context pdf."
            )
        },
        {
            "role": "system",
            "content": f"Relevant context from the PDF: {relevant_context}"
        }
    ]

    if image_path:
        messages.append({
            "role": "system",
            "content": f"Optional image file provided: {os.path.basename(image_path)}"
        })

    messages.append({
        "role": "user",
        "content": query
    })

    return {
        "model": MODEL,
        "messages": messages
    }

# Main function
def main():
    print("Welcome to the AI Chat Assistant with PDF and optional image support!")
    print(f"Using fixed PDF file: {PDF_PATH}")
    print("You can optionally provide an image for additional context.")
    print("Type 'exit' to quit the chat.")

    if not os.path.exists(PDF_PATH):
        print(f"Error: The mandatory PDF file '{PDF_PATH}' was not found. Please check the file path.")
        return

    # Build and store the index for the PDF
    index, chunks = build_and_store_index(PDF_PATH)
    print("Index built and ready for searching relevant context!")

    # Optional image input
    image_path = None
    add_image = input("Do you want to upload an optional image for context? (yes/no): ").strip().lower()
    if add_image == "yes":
        image_path = input("Enter the path to your image file: ").strip()
        if os.path.exists(image_path):
            print("Image file successfully loaded!")
        else:
            print("Error: File not found. Continuing without an image.")

    print("\nInteractive Chat Started! Ask your questions below.")
    
    while True:
        query = input("\nYour Query: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        prompt = build_prompt_with_context(query, index, chunks, image_path)
        try:
            response = ollama.chat(**prompt)
            print("\nAI Response:")
            print(response['message']['content'])
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")

if __name__ == "__main__":
    main()
