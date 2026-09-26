import os
import json
import time
from loguru import logger
import psycopg2
from psycopg2 import pool
from openai import OpenAI

# Connection pool setup
DB_DSN = os.getenv("DATABASE_URL", "postgresql://localhost:5432/tutor_db")
_db_pool = None

def get_db_pool():
    global _db_pool
    if _db_pool is None:
        try:
            _db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, DB_DSN)
            logger.info("Initialized PostgreSQL connection pool for RAG")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
    return _db_pool

def setup_database():
    """
    Milestone A: Database Schema & Indexing
    Creates the table and indexes required for Hybrid RAG.
    """
    p = get_db_pool()
    if not p: return
    conn = p.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS course_knowledge (
                    id BIGSERIAL PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    course_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(1536) NOT NULL,
                    fts_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                );
            """)
            # HNSW Index for fast vector similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS course_knowledge_embedding_idx 
                ON course_knowledge USING hnsw (embedding vector_cosine_ops);
            """)
            # GIN Index for fast full-text lexical search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS course_knowledge_fts_idx 
                ON course_knowledge USING gin (fts_vector);
            """)
        conn.commit()
        logger.info("PostgreSQL RAG schema setup successfully.")
    finally:
        p.putconn(conn)

def chunk_text(text: str, chunk_size: int = 300) -> list:
    """Basic word-based semantic chunker."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def ingest_document(tenant_id: str, course_id: str, text: str, api_key: str):
    """
    Milestone B: Knowledge Base Ingestion Pipeline
    Chunks text, gets embeddings, and bulk inserts into PostgreSQL.
    """
    client = OpenAI(api_key=api_key)
    chunks = chunk_text(text)
    if not chunks:
        return
        
    logger.info(f"Ingesting document for tenant {tenant_id}, {len(chunks)} chunks.")
    
    # Batch generate embeddings
    emb_res = client.embeddings.create(input=chunks, model="text-embedding-3-small")
    embeddings = [d.embedding for d in emb_res.data]
    
    p = get_db_pool()
    if not p: return
    conn = p.getconn()
    try:
        with conn.cursor() as cur:
            for chunk, emb in zip(chunks, embeddings):
                cur.execute("""
                    INSERT INTO course_knowledge (tenant_id, course_id, content, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (tenant_id, course_id, chunk, json.dumps(emb)))
        conn.commit()
    finally:
        p.putconn(conn)

def ingest_file(tenant_id: str, course_id: str, file_path: str, api_key: str):
    """
    Helper function to automatically read and ingest either a .txt or .pdf file.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    text = ""
    if file_path.lower().endswith(".pdf"):
        try:
            import pypdf
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except ImportError:
            logger.error("pypdf is not installed. Run: pip install pypdf")
            return
        except Exception as e:
            logger.error(f"Failed to read PDF: {e}")
            return
    else:
        # Fallback to standard text reading
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    ingest_document(tenant_id, course_id, text, api_key)

def hybrid_rrf_search(tenant_id: str, query: str, query_embedding: list, limit: int = 3) -> list:
    """
    Milestone C: The Hybrid RRF Query Function
    Performs parallel semantic + lexical searches and merges them using RRF.
    """
    query_vector_str = json.dumps(query_embedding)
    
    sql = """
    WITH semantic_search AS (
        SELECT id, content,
               ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
        FROM course_knowledge
        WHERE tenant_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT 20
    ),
    keyword_search AS (
        SELECT id, content,
               ROW_NUMBER() OVER (ORDER BY ts_rank(fts_vector, plainto_tsquery('english', %s)) DESC) AS rank
        FROM course_knowledge
        WHERE tenant_id = %s AND fts_vector @@ plainto_tsquery('english', %s)
        ORDER BY ts_rank(fts_vector, plainto_tsquery('english', %s)) DESC
        LIMIT 20
    )
    SELECT
        COALESCE(s.id, k.id) AS id,
        COALESCE(s.content, k.content) AS content,
        COALESCE(1.0 / (60 + s.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) AS rrf_score
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.id = k.id
    ORDER BY rrf_score DESC, id ASC
    LIMIT %s;
    """
    
    p = get_db_pool()
    if not p:
        return []
        
    conn = p.getconn()
    try:
        start_time = time.time()
        with conn.cursor() as cur:
            # Bind parameters strictly to prevent SQL injection and data leakage
            cur.execute(sql, (
                query_vector_str, tenant_id, query_vector_str,
                query, tenant_id, query, query,
                limit
            ))
            results = cur.fetchall()
            
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Hybrid RRF PostgreSQL Query executed in {elapsed_ms:.2f}ms")
        
        # Return just the text chunks
        return [row[1] for row in results]
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return []
    finally:
        p.putconn(conn)
        