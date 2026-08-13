# TailorTalk Saree Search Agent

## Overview
TailorTalk is an AI-powered visual search agent designed to find visually similar sarees from a dataset of images. It accepts an image (via upload or URL) or natural language queries ("Find similar sarees to this one") and returns the closest visual matches along with similarity scores and metadata.

## Features
- **Visual Similarity Search**: Uses state-of-the-art embedding models to find visually similar sarees.
- **Visual Reranking**: Enhances raw embeddings with fine-grained color histogram matching (HSV space) to prioritize matching fabric colors and patterns.
- **AI Agent Interface**: A LangChain-powered conversational agent that intelligently routes user queries to the deterministic search pipeline.
- **Streamlit UI**: A clean, professional, and interactive chat-based UI.
- **Robust Ingestion**: Fault-tolerant image downloader with caching and metadata alignment.

## Architecture
1. **Query Processing**: The user uploads an image and/or provides a natural language query via Streamlit.
2. **AI Agent**: The LangChain agent understands the intent and executes the `search_similar_sarees` tool with the provided image.
3. **Embedding Generation**: The query image is passed through `openai/clip-vit-base-patch32` to generate a normalized 512-dimensional embedding.
4. **Vector Database**: FAISS (IndexFlatIP) performs an Inner-Product search (equivalent to Cosine Similarity for normalized vectors) to retrieve the top 20 candidate matches.
5. **Visual Reranking**: The top 20 candidates are reranked using a weighted combination of their embedding similarity (alpha=0.6) and Color Histogram Correlation (beta=0.4). To keep the system lightweight for deployment, HSV color histograms are **precomputed** and injected into the FAISS metadata, completely eliminating the need to store the 1,000+ raw images on the deployment server.
6. **Result Presentation**: The final top 5 results are returned by the tool and presented visually in the Streamlit app.

## Dataset
The application processes a CSV file containing product metadata and image URLs.
- The downloader securely fetched 1,070 valid images (discarding 4 broken URLs).
- Metadata (SKU, Name, Price, Link, Image URL, precomputed HSV Histograms) is indexed alongside the vectors for rich, fast results.

## Environment Variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```
*(Required for the conversational AI agent. The visual search engine itself runs entirely locally on CPU/GPU).*

## Installation
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

## Build Index
Before running the application locally, you must download the images and build the FAISS index.

1. **Download Images**:
```bash
python src/image_downloader.py
```
*(Downloads the full dataset of 1,070 images)*

2. **Generate Embeddings and FAISS Index**:
```bash
python build_index.py
```
*(Generates 512-D CLIP embeddings and precomputes HSV color histograms for metadata storage)*

## Run Application
Start the Streamlit interface:
```bash
streamlit run app.py
```

## Testing
An end-to-end testing script verifies the embedding generation, FAISS retrieval, and visual reranker on a local subset.
```bash
python test_end_to_end.py
```

A search evaluation script rigorously tests the final FAISS index by querying 10 random images across the 1,070 corpus (excluding self-matches) to manually inspect and verify search quality:
```bash
python evaluate_search.py
```

## Deployment
This application is ready for deployment on **Streamlit Community Cloud**.
- `requirements.txt` includes all necessary dependencies, including headless OpenCV (`opencv-python-headless`) to avoid Linux GUI dependencies.
- The `clip-vit-base-patch32` model is lightweight enough to run within Streamlit Cloud's 1GB memory limit.
- Ensure `.env` is **not** committed to version control, and configure secrets directly in the Streamlit Cloud dashboard.

## Assumptions & Trade-offs
- **Model Choice**: `openai/clip-vit-base-patch32` was chosen as it balances excellent zero-shot visual features with low memory footprint, crucial for Streamlit deployment.
- **Reranking**: While texture (LBP) could be added, HSV color histograms were prioritized as sarees are highly color-dependent, providing a fast, lightweight, and effective reranking signal.
- **Vector DB**: FAISS FlatIP was chosen over HNSW due to the small dataset size (~1000 items), where exhaustive search is instantaneous and memory overhead is minimal.
