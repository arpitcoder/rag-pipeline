from ingestion import extract_text_from_pdf, chunk_documents
from embeddings import create_document_embeddings, create_faiss_index
from querying import retrieve_relevant_chunks, generate_answer
from sentence_transformers import SentenceTransformer

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose another model if needed

def rag_pipeline(pdf_paths, query):
    """
    The main pipeline function that processes PDFs, retrieves relevant chunks,
    and generates an answer based on a query.
    
    Args:
        pdf_paths (list): List of PDF file paths to process.
        query (str): The user's query.
    
    Returns:
        str: The generated answer.
    """
    
    # Step 1: Extract text from PDFs
    document_texts = [extract_text_from_pdf(pdf) for pdf in pdf_paths]
    
    # Step 2: Split documents into chunks
    document_chunks = chunk_documents(document_texts)
    
    # Step 3: Create document embeddings
    embeddings = create_document_embeddings(document_chunks)
    
    # Step 4: Create FAISS index
    index = create_faiss_index(embeddings)
    
    # Step 5: Retrieve relevant chunks based on the query
    relevant_chunks = retrieve_relevant_chunks(query, index, document_chunks, model)
    
    # Step 6: Generate answer using the retrieved chunks
    answer = generate_answer(query, relevant_chunks)
    
    return answer


if __name__ == "__main__":
    # Example usage: Specify your PDF files and the query you want to ask
    pdf_files = ["Arpit_Nigam_DevSecOps_Architect (1).pdf"]  # Replace with actual file paths
    query = "What is the main purpose of the project described in these documents?"
    
    # Run the RAG pipeline
    result = rag_pipeline(pdf_files, query)
    
    # Output the answer
    print("Answer:", result)
