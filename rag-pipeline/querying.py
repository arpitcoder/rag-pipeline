# querying.py

import openai

# Set your OpenAI API key
openai.api_key = ""   

def retrieve_relevant_chunks(query, index, document_chunks, model):
    query_embedding = model.encode([query])
    _, indices = index.search(query_embedding, k=3)  # Retrieve top 3 chunks
    return [document_chunks[i] for i in indices[0]]

def generate_answer(query, retrieved_docs):
    """
    Generates an answer from the retrieved document chunks using OpenAI's new API structure.

    Args:
        query (str): The user's query.
        retrieved_docs (list): The list of relevant document chunks.

    Returns:
        str: The generated answer from GPT.
    """
    # Combine the relevant chunks into a context string
    context = "\n".join(retrieved_docs)
    
    # Create a message for the new OpenAI Chat API
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]

    try:
        # Call OpenAI's new Chat API to generate an answer
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Change to the newer model
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )

        # Extract and return the answer from the response
        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        # Handle any errors from the API request
        print(f"Error generating answer: {e}")
        return "I'm sorry, there was an error generating an answer."
