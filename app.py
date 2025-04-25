import os
import json
import time
import sys
import csv
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from chromadb import Client
from chromadb.config import Settings

PERSISTENT_DIRECTORY = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "mrknowitall"

console = Console()


# Set up ChromaDB client

settings = Settings(persist_directory=PERSISTENT_DIRECTORY, is_persistent=True)
client = Client(settings)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(text: str) -> list:
    """
    Get the embeddings for a given text using the SentenceTransformer model.
    """
    try:
        embeddings = model.encode(text).tolist()
        return embeddings
    except Exception as e:
        console.print(f"[red]Error getting embeddings: {e}[/red]")
        return None


# Track processed files
PROCESSED_FILES_PATH = os.path.join(os.path.dirname(__file__), "processed_files.json")


def load_processed_files():
    """
    Returns a dict with { file_id: {modified:str, vectors: [vectors_ids], name: str}, ...}
    """
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, "r") as f:
            return json.load(f)
    else:
        return {}


def save_processed_files(processed_files):
    """
    Save the processed files to a JSON file.
    """
    with open(PROCESSED_FILES_PATH, "w") as f:
        json.dump(processed_files, f, indent=2)
        console.print(f"[green]Processed files saved to {PROCESSED_FILES_PATH}[/green]")


# Get local files
def read_local_file(directory: str) -> list:
    """
    Read a file in the given directory and return its content as a string.
    """
    try:
        with open(directory, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        console.print(f"[red]Error reading file {directory}: {e}[/red]")
        return ""


# Split text into chunks
import csv


def load_entries_from_csv(file_path: str) -> list[dict]:
    """
    Load entries from a CSV file into a list of dictionaries.
    Assumes headers: name, topic, source, content.
    """
    try:
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            return [row for row in reader]
    except Exception as e:
        console.print(f"[red]Error reading CSV: {e}[/red]")
        return []


def format_entries(entries: list[dict]) -> list[tuple[str, str]]:
    """
    Convert structured QA entries into formatted text and return with unique IDs.

    Returns:
    - List of tuples (id, text).
    """
    formatted = []

    for i, entry in enumerate(entries):
        name = entry.get("name", "").strip()
        topic = entry.get("topic", "").strip()
        source = entry.get("source", "").strip()
        content = entry.get("content", "").strip()

        if name and content:
            sentence = f"{name}: {content} ({topic}, {source})"
            uid = f"{hash(sentence)}_{i}"  # unique ID
            formatted.append((uid, sentence))

    return formatted


def embed_csv_and_store(file_path: str):
    """
    Process a CSV file: format, embed, and store in ChromaDB if not already processed.
    """
    filename = os.path.basename(file_path)
    last_modified = os.path.getmtime(file_path)
    modified_str = datetime.fromtimestamp(last_modified).isoformat()
    file_id = f"{filename}_{modified_str}"

    console.print(f"[blue]Processing {filename}...[/blue]")

    content = read_local_file(file_path)
    if not content:
        console.print(f"[red]No content found in {filename}. Skipping...[/red]")
        return

    entries = load_entries_from_csv(file_path)
    formatted_entries = format_entries(entries)
    console.print(
        f"[blue]Loaded {len(formatted_entries)} entries from {filename}...[/blue]"
    )

    processed_files = load_processed_files()
    inserted_ids = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"Embedding {filename}", total=len(formatted_entries))

        for uid, text in formatted_entries:
            embedding = get_embeddings(text)
            if embedding:
                metadata = {
                    "file_name": filename,
                    "chunk_index": uid.split("_")[-1],
                    "text": text[:100],
                }
                collection.add(
                    ids=[uid],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[text],
                )
                inserted_ids.append(uid)
            progress.update(task, advance=1)

    if inserted_ids:
        processed_files[file_id] = {
            "modified": modified_str,
            "vectors": inserted_ids,
            "name": filename,
        }
        save_processed_files(processed_files)
        console.print(
            f"[green]Successfully embedded and stored {len(inserted_ids)} entries from {filename}[/green]"
        )


def delete_vectors(file_name: str):
    """
    Delete vectors from ChromaDB based on the file name.
    """
    processed_files = load_processed_files()
    file_data = processed_files.get(file_name, {})

    try:
        collection.delete(where={"file_name": file_name})
        console.print(f"[green]Deleted vectors for {file_name}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Error deleting vectors for {file_name}: {e}[/red]")
        return False


# Poll and Update
def list_local_files(directory: str = "documents/") -> list[str]:
    """
    List all files in the given directory.
    """
    folder_path = os.path.join(os.path.dirname(__file__), directory)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        console.print(
            f"[yellow]Directory {folder_path} does not exist. Creating it...[/yellow]"
        )

    files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".csv"):
            files.append(
                {
                    "path": file_path,
                    "name": file_name,
                    "modified": os.path.getmtime(file_path),
                }
            )
    return files


def update_files():
    """
    Check for new or modified files and process them.
    """
    console.print(f"[blue]Updated started at {datetime.now().isoformat()}...[/blue]")
    processed_files = load_processed_files()

    try:
        current_files = list_local_files()

        for file_name in list(processed_files.keys()):
            file_path = os.path.join(os.path.dirname(__file__), "documents", file_name)
            if not os.path.exists(file_path):
                console.print(
                    f"[red]File {file_name} no longer exists. Deleting from processed files...[/red]"
                )
                del processed_files[file_name]
                save_processed_files(processed_files)

        for file in current_files:
            existing = processed_files.get(file["name"])
            if not existing or file["modified"] > existing["modified"]:
                console.print(f"[blue]Processing {file['name']}...[/blue]")
                delete_vectors(file["name"])
                embed_csv_and_store(file["path"])

    except Exception as e:
        console.print(f"[red]Error during update: {e}[/red]")


# Main loop function
def wait_or_pull(interval: int = 3600):
    """
    Main loop to continuously check for updates and process files.
    """
    start_time = time.time()
    while time.time() - start_time < interval:
        user_input = (
            input("Press 'q' to quit or type pull to run update instantly: ")
            .strip()
            .lower()
        )
        if user_input == "q":
            console.print("[yellow]Exiting...[/yellow]")
            sys.exit(0)
        elif user_input == "pull":
            return
        else:
            console.print("[red]Invalid input. Please try again.[/red]")
        time.sleep(1)  # Sleep for a short duration to avoid busy waiting


if __name__ == "__main__":

    while True:
        console.print("[green]Starting the script...[/green]")
        update_files()
        wait_or_pull(interval=3600)  # Run the main loop for 1 hour (3600 seconds)
        console.print("[green]Script finished.[/green]")
