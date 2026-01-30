# rag_chatbot_with_memory.py
# Secure multi-user RAG chatbot WITH conversational memory
# Streamlit + Gemini + Pinecone
# Using modern LangChain approach (manual message history management)

import os
import json
import time
import hashlib
import secrets
import tempfile

import streamlit as st
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

# Conversation memory settings
MAX_HISTORY_MESSAGES = 10  # Keep last 10 messages (5 exchanges)


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
# VECTOR STORE
# --------------------------------------------------


def get_vectorstore(embeddings, user_id):
    """
    Get vectorstore with namespace isolation for the user.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    namespace = f"user_{user_id}"

    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
    )


# --------------------------------------------------
# INGESTION
# --------------------------------------------------


def split_docs(pages, user_id):
    """
    Properly adds metadata AFTER splitting to ensure all chunks have user_id.
    """
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


def ingest_pdf(file_bytes, filename, embeddings, user_id):
    suffix = os.path.splitext(filename)[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(file_bytes)
        tmp_path = f.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        chunks = split_docs(pages, user_id)

        store = get_vectorstore(embeddings, user_id)
        store.add_documents(chunks)

        if chunks:
            sample_metadata = chunks[0].metadata
            if "user_id" not in sample_metadata:
                st.error("WARNING: Metadata not properly attached!")
                return 0

        return len(chunks)

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# --------------------------------------------------
# CONVERSATION MEMORY HELPERS
# --------------------------------------------------


def get_chat_history():
    """
    Get chat history from session state and convert to LangChain message format.
    Returns the last MAX_HISTORY_MESSAGES messages.
    """
    messages = st.session_state.get("messages", [])

    # Keep only last N messages for context window management
    recent_messages = messages[-MAX_HISTORY_MESSAGES:] if len(messages) > MAX_HISTORY_MESSAGES else messages

    # Convert to LangChain message objects
    chat_history = []
    for msg in recent_messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    return chat_history


def format_chat_history(chat_history):
    """
    Format chat history for display in the prompt.
    This is an alternative if you want string-based history instead of message objects.
    """
    if not chat_history:
        return ""

    formatted = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")

    return "\n".join(formatted)


# --------------------------------------------------
# PROMPT WITH MEMORY
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

# MODERN APPROACH: Using MessagesPlaceholder for chat history
PROMPT_WITH_MEMORY = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("system", "Context from documents:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),  # This is the key for memory!
        ("human", "{question}"),
    ]
)


# --------------------------------------------------
# CHAIN WITH MEMORY
# --------------------------------------------------


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_chain_with_memory(embeddings, user_id):
    """
    Build RAG chain with conversation memory support.
    """
    store = get_vectorstore(embeddings, user_id)

    retriever = store.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"user_id": user_id},
        }
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        api_key=GOOGLE_API_KEY,
        temperature=0.3,  # Slightly higher for more natural conversation
        streaming=True,
    )

    # Chain with memory support
    chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: get_chat_history(),  # Inject chat history
            }
            | PROMPT_WITH_MEMORY
            | llm
            | StrOutputParser()
    )

    return chain


# --------------------------------------------------
# STREAMLIT STATE
# --------------------------------------------------


def init_state():
    defaults = {
        "stage": "auth",
        "username": None,
        "messages": [],  # This stores our conversation history
        "chain": None,
        "emb": None,
    }

    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def ensure_embeddings():
    if st.session_state.emb is None:
        st.session_state.emb = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )


# --------------------------------------------------
# AUTH VIEW
# --------------------------------------------------


def auth_view():
    st.title("🔒 Secure RAG Chatbot with Memory")

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if verify_user(u.strip(), p):
                st.session_state.username = u.strip()
                st.session_state.stage = "docs"
                st.rerun()
            else:
                st.error("Invalid credentials")

    with signup_tab:
        u = st.text_input("New Username", key="signup_user")
        p = st.text_input("New Password", type="password", key="signup_pass")

        if st.button("Create Account"):
            if create_user(u.strip(), p):
                st.success("Account created. Please log in.")
            else:
                st.warning("User already exists.")


# --------------------------------------------------
# DOC UPLOAD VIEW
# --------------------------------------------------


def docs_view():
    st.subheader(f"Welcome, {st.session_state.username} 👋")

    ensure_embeddings()

    uploaded_file = st.file_uploader(
        "Upload PDF to index",
        type=["pdf"],
    )

    if uploaded_file:
        with st.spinner("Indexing document..."):
            n = ingest_pdf(
                uploaded_file.read(),
                uploaded_file.name,
                st.session_state.emb,
                st.session_state.username,
            )

            if n > 0:
                st.success(f"✅ Indexed {n} chunks for user '{st.session_state.username}'")

                st.session_state.chain = build_chain_with_memory(
                    st.session_state.emb,
                    st.session_state.username,
                )

                st.session_state.stage = "chat"
                st.rerun()
            else:
                st.error("Failed to index document. Please try again.")

    if st.button("Go to Chat"):
        st.session_state.chain = build_chain_with_memory(
            st.session_state.emb,
            st.session_state.username,
        )
        st.session_state.stage = "chat"
        st.rerun()

    if st.button("Logout"):
        st.session_state.username = None
        st.session_state.stage = "auth"
        st.session_state.messages = []
        st.session_state.chain = None
        st.rerun()


# --------------------------------------------------
# CHAT VIEW WITH MEMORY
# --------------------------------------------------


def chat_view():
    st.subheader(f"💬 Chat - {st.session_state.username}")

    # Display conversation history count
    msg_count = len(st.session_state.messages)
    st.caption(f"💾 Conversation: {msg_count // 2} exchanges | Remembering last {MAX_HISTORY_MESSAGES // 2} exchanges")

    # Display chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat input
    query = st.chat_input("Ask about your documents...")

    if query:
        # Add user message to history
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        # Generate assistant response with memory
        with st.chat_message("assistant"):
            placeholder = st.empty()
            text = ""

            try:
                # The chain automatically uses chat history via get_chat_history()
                for token in st.session_state.chain.stream(query):
                    if token:
                        text += token
                        placeholder.markdown(text)
            except Exception as e:
                text = f"Error processing query: {str(e)}"
                placeholder.markdown(text)

        # Add assistant response to history
        st.session_state.messages.append(
            {"role": "assistant", "content": text}
        )

    # Sidebar controls
    with st.sidebar:
        st.write(f"**User:** {st.session_state.username}")
        st.write(f"**Messages:** {len(st.session_state.messages)}")

        st.divider()

        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat history cleared!")
            st.rerun()

        if st.button("📄 Back to Upload", use_container_width=True):
            st.session_state.stage = "docs"
            st.rerun()

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.username = None
            st.session_state.stage = "auth"
            st.session_state.messages = []
            st.session_state.chain = None
            st.rerun()

        st.divider()

        # Memory settings display
        st.caption("Memory Settings")
        st.write(f"Max history: {MAX_HISTORY_MESSAGES // 2} exchanges")

        # Show current memory usage
        if st.session_state.messages:
            current_history = get_chat_history()
            st.write(f"Active memory: {len(current_history)} messages")


# --------------------------------------------------
# MAIN
# --------------------------------------------------


def main():
    st.set_page_config(
        page_title="Secure RAG Chatbot with Memory",
        page_icon="🧠",
        layout="centered"
    )

    init_state()

    if st.session_state.username is None:
        st.session_state.stage = "auth"

    if st.session_state.stage == "auth":
        auth_view()
    elif st.session_state.stage == "docs":
        docs_view()
    else:
        chat_view()


if __name__ == "__main__":
    main()