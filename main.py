import os
import json
import time
import hashlib
import secrets
import tempfile
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------
# ENV
# --------------------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "models/gemini-embedding-001"
)

USERS_FILE = "users.json"

app = FastAPI(title="RAGMate API")

# Setup CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in proc
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# AUTH
# --------------------------------------------------

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200_000,
    ).hex()

def create_user(username, password):
    users = load_users()
    if username in users:
        return False
    salt = secrets.token_hex(16)
    users[username] = {
        "salt": salt,
        "password_hash": hash_password(password, salt),
        "created_at": time.time(),
    }
    save_users(users)
    return True

def verify_user(username, password):
    users = load_users()
    rec = users.get(username)
    if not rec:
        return False
    return secrets.compare_digest(
        hash_password(password, rec["salt"]),
        rec["password_hash"],
    )

# --------------------------------------------------
# APP LOGIC
# --------------------------------------------------

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

def get_vectorstore(user_id: str):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    namespace = f"user_{user_id}"
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
        namespace=namespace,
    )

def split_docs(pages, user_id):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        if chunk.metadata is None:
            chunk.metadata = {}
        chunk.metadata["user_id"] = user_id
    return chunks

# --------------------------------------------------
# PROMPT & CHAIN
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a helpful document QA assistant with conversation memory.

Rules:
- Answer questions based on the provided context from the user's documents.
- Use the conversation history to maintain context and provide coherent responses.
- If referring to something mentioned earlier in the conversation, acknowledge it naturally.
- If the answer is not in the documents, say: "I apologize, but I could not find that specific information in your uploaded documents."
- Be conversational and remember what was discussed previously.
"""

PROMPT_WITH_MEMORY = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("system", "Context from documents:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def execute_chat(user_id: str, question: str, history: List[Dict[str, str]]):
    store = get_vectorstore(user_id)
    retriever = store.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"user_id": user_id},
        }
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )

    chat_history = []
    for msg in history:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: chat_history,
        }
        | PROMPT_WITH_MEMORY
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)

# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------

class AuthRequest(BaseModel):
    username: str
    password: str

@app.post("/api/auth/login")
def login(req: AuthRequest):
    if verify_user(req.username, req.password):
        return {"success": True, "username": req.username}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/auth/signup")
def signup(req: AuthRequest):
    if create_user(req.username, req.password):
        return {"success": True, "message": "Account created. Please log in."}
    raise HTTPException(status_code=400, detail="User already exists.")

@app.post("/api/docs/upload")
async def upload_document(username: str = Form(...), file: UploadFile = File(...)):
    file_bytes = await file.read()
    suffix = os.path.splitext(file.filename)[-1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(file_bytes)
        tmp_path = f.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        chunks = split_docs(pages, username)
        
        store = get_vectorstore(username)
        store.add_documents(chunks)
        
        return {"success": True, "chunks_indexed": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)


class ChatRequest(BaseModel):
    username: str
    question: str
    history: List[Dict[str, str]]

@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        answer = execute_chat(req.username, req.question, req.history)
        return {"success": True, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
