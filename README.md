# RAGMate

A production-ready, secure Retrieval-Augmented Generation (RAG) application built with **React (Vite)**, **FastAPI**, **LangChain**, and **Google Gemini**. This chatbot allows multiple users to create private accounts, upload PDF documents, and perform Q&A with conversational memory, ensuring strict data isolation between users.

## 🚀 Features

* **🔐 User Authentication:** Secure sign-up and login system with password hashing (SHA-256 + Salt).
* **📂 Multi-User Data Isolation:** Users can only access and query their own uploaded documents.
* **🧠 Conversational Memory:** Remembers context from previous messages for a natural chat experience.
* **📄 PDF Ingestion:** Upload and index PDF documents using PyPDFLoader and RecursiveCharacterTextSplitter.
* **⚡ AI Powered:** Utilizes **Google Gemini 2.5 Flash** for both embeddings and text generation.
* **🗄️ Vector Database:** Stores document embeddings efficiently using **Pinecone**.
* **✨ Modern UI:** A beautiful and dynamic React frontend built with Vite.

## 🛠️ Tech Stack

* **Frontend:** React, Vite, CSS
* **Backend API:** FastAPI (Python 3.13)
* **Orchestration:** LangChain (LCEL)
* **LLM & Embeddings:** Google Gemini (via `langchain-google-genai`)
* **Vector Store:** Pinecone
* **Containerization:** Docker (Optional)

## 📋 Prerequisites

Before running the application, ensure you have the following:

1.  **Python 3.13+** installed.
2.  **Node.js (18+)** installed for the frontend.
3.  **Google AI Studio API Key** (for Gemini models).
4.  **Pinecone API Key** (for vector storage).

## ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AryanMaher3/Secure-Multi-User-RAG-Chatbot-for-Document-Q-A.git
    cd Secure-Multi-User-RAG-Chatbot-for-Document-Q-A
    ```

2.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your credentials:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    PINECONE_API_KEY=your_pinecone_api_key_here
    PINECONE_INDEX_NAME=rag-document-analyzer
    LLM_MODEL=gemini-2.5-flash
    ```

### Running the Backend (FastAPI)
Open a terminal and run:
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.

### Running the Frontend (React)
Open a **second** terminal, navigate to the `frontend` folder, and run:
```bash
cd frontend
npm install
npm run dev
```
**Access the App:** Open your browser and navigate to the local URL provided by Vite (usually `http://localhost:5173`).

### 🐳 Docker (Optional)
If you want to run the application using Docker, you can build and run the provided `Dockerfile`.

## 💡 Workflow

* **Login/Signup:** Create a new account or log in.
* **Upload:** Upload a PDF document. The app will chunk, embed, and index it securely.
* **Chat:** Ask questions about your document. The bot will answer based *only* on your uploaded context.

## 📂 Project Structure
* `frontend/`: The React (Vite) user interface.
* `main.py`: The FastAPI application handling authentication and RAG workflows.
* `rag_chatbot.py`: Legacy Streamlit application (if applicable).
* `requirements.txt`: Python dependencies.
