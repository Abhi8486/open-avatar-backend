import os
import sys
import argparse
from dotenv import load_dotenv

# Ensure the backend/src directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from handlers.llm.openai_compatible.rag_hybrid import setup_database, ingest_file

def main():
    parser = argparse.ArgumentParser(description="Ingest a book (PDF/TXT) into the Tutor Avatar PostgreSQL RAG database.")
    parser.add_argument("--file", type=str, required=True, help="Path to the PDF or TXT file to ingest.")
    parser.add_argument("--tenant", type=str, default="default", help="Tenant ID (must match what the LLM queries).")
    parser.add_argument("--course", type=str, default="default", help="Course ID.")
    
    args = parser.parse_args()
    
    # Load environment variables (e.g. DATABASE_URL)
    load_dotenv()
    
    if not os.path.exists(args.file):
        print(f"Error: File not found -> {args.file}")
        sys.exit(1)
    
    print("1. Initializing and setting up PostgreSQL vector database schema...")
    setup_database()
    
    print(f"2. Ingesting '{args.file}' for tenant '{args.tenant}', course '{args.course}'...")
    print("   (This will chunk the text, compute embeddings using HuggingFace all-MiniLM-L6-v2, and store them in the DB.)")
    
    ingest_file(args.tenant, args.course, args.file, api_key="")
    
    print("\n✅ Ingestion complete! The avatar knowledge base is now updated.")

if __name__ == "__main__":
    main()
