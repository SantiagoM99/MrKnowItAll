import sys
import re
from ollama import chat
from rich.console import Console

try:
    from app import get_embeddings, console, collection
except ImportError:
    print("Error importing from app.py. Ensure that app.py is in the same directory.")
    sys.exit(1)


def main():
    """
    Main function to interact with the Ollama chat model.
    """
    console.rule("[bold magenta]Chat with Mr. Know it all[/bold magenta]")
    console.print("[bold green]Escribe 'salir' para terminar el programa[/bold green]")

    while True:
        user_input = console.input("[bold cyan]Tu pregunta:[/] ")
        if user_input.lower() == "salir":
            break

        answer = chat_agent(user_input)
        console.print(f"[bold green]Respuesta:[/] {answer}")


def chat_agent(query: str) -> str:
    """
    Function to interact with the chat model and get a response.

    1. Embed the query using the get_embeddings function.
    2. Query the ChromaDB collection for similar embeddings.
    3. Get the response from the chat model using the query and the retrieved documents.
    """

    system_message = """
        Eres un asistente académico confiable de una universidad colombiana. Respondes únicamente en español, utilizando exclusivamente la información proporcionada en los documentos recuperados.

        Reglas estrictas que debes seguir:
        1. No inventes, completes ni supongas información que no esté en los textos entregados por el sistema de recuperación.
        2. Si no encuentras información relacionada directamente con la pregunta, responde: 
        "Lo siento, no tengo información suficiente para responder esa pregunta."
        3. No hables en inglés bajo ninguna circunstancia.
        4. Cuando sea posible, incluye al final de la respuesta entre paréntesis el **tema** y la **fuente** correspondiente, si esta aparece en el texto recuperado.
        5. Redacta tus respuestas con claridad y coherencia, en forma de explicaciones informativas, no como preguntas y respuestas.

        Tu objetivo es ofrecer información confiable, clara y bien escrita, sin desviarte de las fuentes proporcionadas.
    """

    # Get the embeddings for the query
    query_embedding = get_embeddings(query)
    if query_embedding is None:
        return "Error al obtener los embeddings para la consulta."
    
    # Query the ChromaDB collection for similar embeddings
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

    except Exception as e:
        return "Error al consultar la colección."
    
    if not results or "documents" not in results or not results["documents"]:
        return "No se encontraron documentos relevantes."
    
    documents = results["documents"][0]
    context = " ".join(documents)
    if not context.strip():
        return "No se encontraron documentos relevantes."
    
    # Get the response from the chat model
    return ollama_chat(system_message, query, context)
    
def ollama_chat(system_message: str, query: str, context: str) -> str:
    """
    Function to get a response from the chat model using the system message, query, and context.
    """
    try:
        response = chat(
            model="deepseek-r1:7b",
            messages=[
                {"role": "system", "content": f"{system_message} \n\n Context: {context}"},
                {"role": "user", "content": query}
            ],
            stream=False
        )

        # Eliminar el bloque <think>...</think> si existe
        cleaned_response = re.sub(r"<think>.*?</think>", "", response["message"]["content"], flags=re.DOTALL).strip()
        
        return cleaned_response
    except Exception as e:
        return f"Error al obtener la respuesta del modelo: {e}"
    
if __name__ == "__main__":
    main()
    
    

