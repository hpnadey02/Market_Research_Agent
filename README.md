# Market Research Insight Application

## Overview
This is a production-ready **RAG-based Market Research Insight Application**. It automates the process of gathering commodity news, analyzing supply/demand drivers, and generating professional analyst reports.

## Features
- **Multi-Source Ingestion**: Fetches news from NewsAPI and scrapes SteelOrbis.
- **Advanced RAG**: Hybrid semantic chunking, FAISS vector search, and query expansion.
- **Multi-LLM Support**: Works with Google Gemini, GROQ, or Local HuggingFace models.
- **Analyst-Grade Output**: Generates Supply, Demand, and Executive Summary sections.
- **Fail-Safe Architecture**: Robust error handling, retry logic, and validation.

## Prerequisites
- **Python 3.10+**
- **API Keys** (optional but recommended):
  - NewsAPI Key (free tier is fine)
  - Google Gemini API Key OR GROQ API Key (for best results)

## Installation

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you have a GPU, ensure you have the correct PyTorch version installed.*

2.  **Environment Setup**
    - Copy `.env.example` to `.env`:
    ```bash
    copy .env.example .env
    ```
    - Open `.env` and fill in your API keys.
    - Select your `LLM_PROVIDER` (default is `gemini`).

3.  **Local Model (Optional)**
    - If you want to use a local model, download a HuggingFace model (like Mistral-7B) into `model/` folder.
    - Or just use `gemini` or `groq` which are faster and easier.

## How to Run

1.  **Start the Application**
    ```bash
    streamlit run app.py
    ```

2.  **Access UI**
    - The browser will open automatically at `http://localhost:8501`.

3.  **Generate Insight**
    - Enter Metal (e.g., "Steel"), Country (e.g., "India"), Price, and % Change.
    - Click **Generate Insight**.
    - Watch the progress bar as the system ingests news, embeds it, and writes the report.

## Project Structure
- `src/`: Core logic modules (retriever, pipeline, cleaning, etc.)
- `data/`: Stores downloaded news corpus, FAISS index, and cached embeddings.
- `logs/`: Application logs for debugging.
- `app.py`: Main Streamlit dashboard.

## Troubleshooting
- **Input Validation Error**: Ensure Price and % Change are numbers.
- **No News Found**: Try a broader Metal/Country name.
- **LLM Error**: Check your API key in `.env`.
