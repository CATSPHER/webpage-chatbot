# Webpage Chatbot

Ask questions about any webpage — paste a URL, and chat with its content using a local RAG (Retrieval-Augmented Generation) pipeline powered by open-source LLMs via Hugging Face.

## How it works

1. Loads the target webpage using Playwright (handles JavaScript-rendered and moderately bot-protected sites)
2. Splits the page content into chunks
3. Embeds chunks and stores them in a FAISS vector store
4. Retrieves the most relevant chunks for each question
5. Answers using Llama-3.1-8B-Instruct via Hugging Face's Inference API

## Setup

1. Clone the repo and create a virtual environment.
2. Install dependencies:
3. Create a `.env` file with your Hugging Face token
