import os
import time
from src.embeddings import EmbeddingGenerator
from src.vector_store import SareeVectorStore
from src.search import VisualSearcher

def run_tests():
    print("=== End-to-End Testing ===")
    
    # 1. Initialize Components
    print("Loading models and index...")
    t0 = time.time()
    emb_gen = EmbeddingGenerator()
    vector_store = SareeVectorStore(dimension=512)
    vector_store.load('embeddings/faiss.index', 'embeddings/index_meta.json')
    searcher = VisualSearcher(emb_gen, vector_store)
    print(f"Loaded in {time.time() - t0:.2f}s")
    
    # Get a valid test image (use the first one from metadata)
    import pandas as pd
    df = pd.read_csv('data/metadata.csv')
    df_success = df[df['download_status'] == 'Success']
    
    if len(df_success) < 2:
        print("Not enough images to test.")
        return
        
    test_image_1 = df_success.iloc[0]['local_image_path']
    test_image_2 = df_success.iloc[1]['local_image_path']
    
    print("\n--- Test 1: Valid Image Search ---")
    results = searcher.search(test_image_1, top_k=3)
    assert len(results) == 3, "Should return 3 results"
    # The first result should be the image itself (score ~1.0)
    print(f"Query: {test_image_1}")
    print(f"Top Match: {results[0]['image_path']} (Score: {results[0]['similarity_score']:.3f})")
    assert results[0]['similarity_score'] > 0.8, "Self-similarity should be very high."
    print("Test 1 Passed.")
    
    print("\n--- Test 2: Another Valid Image ---")
    results2 = searcher.search(test_image_2, top_k=2)
    print(f"Query: {test_image_2}")
    print(f"Top Match: {results2[0]['image_path']} (Score: {results2[0]['similarity_score']:.3f})")
    assert results2[0]['similarity_score'] > 0.8, "Self-similarity should be very high."
    print("Test 2 Passed.")
    
    print("\n--- Test 3: Invalid Image Path ---")
    results_invalid = searcher.search("non_existent_image.jpg")
    assert len(results_invalid) == 0, "Should return empty list for invalid path."
    print("Test 3 Passed.")
    
    print("\nAll Tests Passed Successfully!")

if __name__ == "__main__":
    run_tests()
