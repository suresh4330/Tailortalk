import json
from langchain_core.tools import tool
from typing import Optional
import os

# We will rely on a global or singleton VisualSearcher initialized in the app
# Alternatively, we pass the path directly and the tool handles the search.
# For simplicity and performance, the app will inject the searcher into the tool context.

# Global reference to be set by the main app
_SEARCHER = None

def set_searcher(searcher):
    global _SEARCHER
    _SEARCHER = searcher

@tool
def search_similar_sarees(image_source: str, top_k: int = 5) -> str:
    """
    Search for visually similar sarees in the TailorTalk dataset based on an image.
    
    Args:
        image_source (str): The local file path or URL of the query image.
        top_k (int): The number of top similar sarees to return. Defaults to 5.
        
    Returns:
        str: A JSON string containing the search results with similarity scores and metadata.
             The LLM should present these results to the user.
    """
    global _SEARCHER
    if _SEARCHER is None:
        return json.dumps({"error": "Searcher not initialized. Please configure the system."})
        
    if not os.path.exists(image_source):
        # In a real app, we might want to handle URLs here by downloading them temporarily.
        # We will assume image_source is a local path (downloaded by Streamlit).
        return json.dumps({"error": f"Image source not found: {image_source}"})
        
    print(f"\n[Tool Execution] Searching similar sarees for: {image_source} (top_k={top_k})")
    
    try:
        results = _SEARCHER.search(image_source, top_k=top_k)
        
        # Clean up output for the LLM
        clean_results = []
        for r in results:
            clean_results.append({
                "image_id": r["image_id"],
                "image_path": r["image_path"],
                "similarity_score": round(r["similarity_score"], 4),
                "name": r["metadata"].get("Name", "Unknown Saree"),
                "price": r["metadata"].get("Discounted Price", "N/A"),
                "link": r["metadata"].get("Website Link", "")
            })
            
        return json.dumps({"results": clean_results})
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})
