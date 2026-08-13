import os
import random
import pandas as pd
from src.embeddings import EmbeddingGenerator
from src.vector_store import SareeVectorStore
from src.search import VisualSearcher
import json

def run_evaluation():
    print("=== Visual Search Quality Evaluation ===")
    
    # 1. Initialize Components
    print("Loading models and FAISS index...")
    emb_gen = EmbeddingGenerator()
    vector_store = SareeVectorStore(dimension=512)
    
    if not os.path.exists('embeddings/faiss.index'):
        print("FAISS index not found. Run build_index.py first.")
        return
        
    vector_store.load('embeddings/faiss.index', 'embeddings/index_meta.json')
    searcher = VisualSearcher(emb_gen, vector_store, alpha=0.7, beta=0.3)
    
    # 2. Select 10 random query images
    df = pd.read_csv('data/metadata.csv')
    df_success = df[df['download_status'] == 'Success']
    
    if len(df_success) < 10:
        print(f"Not enough successful downloads to evaluate (Found {len(df_success)}).")
        return
        
    # Sample 10 random images for evaluation
    queries = df_success.sample(n=10, random_state=42).to_dict(orient='records')
    
    eval_results = []
    
    print("\nStarting evaluation queries...")
    for i, query in enumerate(queries):
        query_path = query['local_image_path']
        query_sku = query['SKU']
        print(f"\n[{i+1}/10] Query Image: {query_sku} | {query['Name']}")
        
        # We fetch top 6 to ensure we can exclude the query itself and keep top 5
        # The searcher uses top_k for final return, but internally fetches faiss_k=20
        # Let's request top_k=10 to filter out self.
        raw_results = searcher.search(query_path, top_k=10, faiss_k=30)
        
        # Filter out the query image itself
        filtered_results = [r for r in raw_results if r['image_id'] != query_sku][:5]
        
        for j, res in enumerate(filtered_results):
            print(f"   Match {j+1}: {res['image_id']} | Score: {res['similarity_score']:.3f} | {res['metadata'].get('Name', '')}")
            
        eval_results.append({
            'query_sku': query_sku,
            'query_path': query_path,
            'query_name': query['Name'],
            'top_matches': [
                {
                    'sku': r['image_id'],
                    'score': r['similarity_score'],
                    'name': r['metadata'].get('Name', ''),
                    'path': r['image_path']
                } for r in filtered_results
            ]
        })
        
    # Save results to a log file
    os.makedirs('eval_logs', exist_ok=True)
    with open('eval_logs/search_evaluation.json', 'w') as f:
        json.dump(eval_results, f, indent=4)
        
    print("\nEvaluation complete. Results saved to eval_logs/search_evaluation.json.")
    print("Please inspect the visual similarity of the logged results.")

if __name__ == "__main__":
    run_evaluation()
