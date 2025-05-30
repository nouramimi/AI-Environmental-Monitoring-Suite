import time
import numpy as np
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import ollama

# File Path
PDF_PATH = "data/100-ways-you-can-improve-the-environment.pdf"

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === 1. Extract Text from PDF ===
def extract_pdf_text(pdf_path):
    document = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in document])
    return text.strip()

# === 2. Split Text into Meaningful Chunks ===
def chunk_text(text, chunk_size=3):
    """Breaks text into sentence-based chunks for better retrieval accuracy."""
    sentences = text.split(". ")
    chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

# === 3. Build FAISS Index ===
def build_embeddings(chunks):
    return embedder.encode(chunks, convert_to_tensor=True)

def store_embeddings_in_faiss(embeddings):
    embeddings_np = embeddings.cpu().detach().numpy().astype("float32")
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index

# === 4. Improved Retrieval Function ===
def search_relevant_context(query, index, chunks):
    query_embedding = embedder.encode(query, convert_to_tensor=True).cpu().detach().numpy().reshape(1, -1)
    D, I = index.search(query_embedding, k=1)  

    if I[0][0] < len(chunks):  
        return chunks[I[0][0]]  
    return "No relevant context found."

# === 5. Evaluate Retrieval ===
def evaluate_retrieval(index, queries, chunks):
    recall_scores, mrr_scores = [], []
    
    for query, ground_truth in queries.items():
        start_time = time.time()
        relevant_chunk = search_relevant_context(query, index, chunks)
        retrieval_latency = time.time() - start_time
        
        recall = int(ground_truth.lower() in relevant_chunk.lower())
        recall_scores.append(recall)

        rank = 1 if ground_truth.lower() in relevant_chunk.lower() else 0
        mrr_scores.append(rank)

        print(f"\nQuery: {query}")
        print(f"Retrieved Context: {relevant_chunk[:200]}...")
        print(f"Retrieval Latency: {retrieval_latency:.4f} sec")

    print(f"\nMean Recall@1: {np.mean(recall_scores):.2f}")
    print(f"Mean Reciprocal Rank (MRR): {np.mean(mrr_scores):.2f}")

# === 6. Evaluate LLM Response (Fix BLEU Scoring) ===
def evaluate_generation(queries, index, chunks):
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    bleu_scores, rouge1_scores, rougeL_scores = [], [], []
    smooth = SmoothingFunction().method1  # Apply smoothing to avoid BLEU=0

    for query, ground_truth in queries.items():
        start_time = time.time()
        relevant_context = search_relevant_context(query, index, chunks)

        response = ollama.chat(
            model="minicpm-v",
            messages=[
                {"role": "system", "content": f"Relevant context: {relevant_context}"},
                {"role": "user", "content": query}
            ]
        )

        llm_latency = time.time() - start_time
        generated_text = response['message']['content']

        # BLEU Score (with smoothing)
        reference = ground_truth.split()
        candidate = generated_text.split()
        bleu = sentence_bleu([reference], candidate, smoothing_function=smooth)
        bleu_scores.append(bleu)

        # ROUGE Score
        scores = rouge.score(ground_truth, generated_text)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

        print(f"\nQuery: {query}")
        print(f"Generated Response: {generated_text[:200]}...")
        print(f"LLM Response Latency: {llm_latency:.4f} sec")

    print(f"\nAverage BLEU Score: {np.mean(bleu_scores):.2f}")
    print(f"Average ROUGE-1: {np.mean(rouge1_scores):.2f}")
    print(f"Average ROUGE-L: {np.mean(rougeL_scores):.2f}")

# === 7. Fix FAISS Memory Usage Calculation ===
def evaluate_memory(index):
    """FAISS does not have a direct function to check memory size, so we estimate."""
    vector_size = index.d
    num_vectors = index.ntotal
    estimated_memory = vector_size * num_vectors * 4 / (1024 * 1024)  # Convert bytes to MB
    print(f"\nEstimated FAISS Index Memory Usage: {estimated_memory:.2f} MB")

# === Run Evaluation ===
if __name__ == "__main__":
    print("\n=== Building RAG Index ===")
    pdf_text = extract_pdf_text(PDF_PATH)
    chunks = chunk_text(pdf_text)
    index = store_embeddings_in_faiss(build_embeddings(chunks))

    queries = {
        "How can I conserve energy at home?": "Turn off unneeded lights and use energy-efficient appliances.",
        "What are some water-saving tips?": "Fix leaks, install low-flow shower heads, and avoid wasteful habits."
    }

    print("\n=== Evaluating Retrieval ===")
    evaluate_retrieval(index, queries, chunks)

    print("\n=== Evaluating LLM Response ===")
    evaluate_generation(queries, index, chunks)

    print("\n=== Evaluating Memory Usage ===")
    evaluate_memory(index)
