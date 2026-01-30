# Secure Multi-User RAG Chatbot for Document Q&A

A production-ready, secure Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain**, and **Google Gemini**. This chatbot allows multiple users to create private accounts, upload PDF documents, and perform Q&A with conversational memory, ensuring strict data isolation between users.

## 🚀 Features

* **🔐 User Authentication:** Secure sign-up and login system with password hashing (SHA-256 + Salt).
* **📂 Multi-User Data Isolation:** Users can only access and query their own uploaded documents.
* **🧠 Conversational Memory:** Remembers context from previous messages for a natural chat experience.
* **📄 PDF Ingestion:** Upload and index PDF documents using PyPDFLoader and RecursiveCharacterTextSplitter.
* **⚡ AI Powered:** Utilizes **Google Gemini 2.5 Flash** for both embeddings and text generation.
* **🗄️ Vector Database:** Stores document embeddings efficiently using **Pinecone**.

## 🛠️ Tech Stack

* **Language:** Python 3.13
* **Frontend:** Streamlit
* **Orchestration:** LangChain (LCEL)
* **LLM & Embeddings:** Google Gemini (via `langchain-google-genai`)
* **Vector Store:** Pinecone
* **Dependency Management:** Pipenv

## 📋 Prerequisites

Before running the application, ensure you have the following:

1.  **Python 3.13+** installed.
2.  **Google AI Studio API Key** (for Gemini models).
3.  **Pinecone API Key** (for vector storage).

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AryanMaher3/Secure-Multi-User-RAG-Chatbot-for-Document-Q-A.git](https://github.com/AryanMaher3/Secure-Multi-User-RAG-Chatbot-for-Document-Q-A.git)
    cd Secure-Multi-User-RAG-Chatbot-for-Document-Q-A
    ```

2.  **Install dependencies using Pipenv:**
    ```bash
    pipenv install
    pipenv shell
    ```
    *(Alternatively, you can use `pip install -r requirements.txt` if you generate one).*

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your credentials:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    PINECONE_API_KEY=your_pinecone_api_key_here
    PINECONE_INDEX_NAME=rag-document-analyzer
    LLM_MODEL=gemini-2.5-flash
    ```

## 🚀 Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run rag_chatbot.py
    ```

2.  **Access the App:**
    Open your browser and navigate to `http://localhost:8501`.

3.  **Workflow:**
    * **Login/Signup:** Create a new account or log in.
    * **Upload:** Upload a PDF document. The app will chunk, embed, and index it securely.
    * **Chat:** Ask questions about your document. The bot will answer based *only* on your uploaded context.

## 📂 Project Structure
