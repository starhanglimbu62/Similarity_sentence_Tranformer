from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from pypdf import PdfReader

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
PDF_PATH = os.getenv("PDF_PATH", "Book.pdf")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


def initialize_client():
    if not AZURE_API_KEY or not AZURE_ENDPOINT or not EMBEDDING_MODEL:
        raise ValueError(
            "Please set AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_EMBEDDING_DEPLOYMENT in .env"
        )

    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version="2024-02-01"
    )


def load_pdf(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if not path.lower().endswith(".pdf"):
        raise ValueError("Unsupported file type. Please provide a PDF file.")

    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


def create_embeddings(client, chunks):
    embeddings = []
    print(f"Creating embeddings for {len(chunks)} chunks...")

    for idx, chunk in enumerate(chunks, start=1):
        print(f"  Embedding chunk {idx}/{len(chunks)}")
        response = client.embeddings.create(input=chunk, model=EMBEDDING_MODEL)
        embeddings.append(response.data[0].embedding)

    return embeddings


def main():
    client = initialize_client()
    text = load_pdf(PDF_PATH)

    if not text:
        raise ValueError("The PDF is empty or no text could be extracted.")

    chunks = split_into_chunks(text)
    embeddings = create_embeddings(client, chunks)

    print("Data ingestion pipeline completed successfully.")
    print(f"Generated {len(embeddings)} embedding vectors.")


if __name__ == "__main__":
    main()
