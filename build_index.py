import os
import pandas as pd
from src.embeddings import EmbeddingGenerator
from src.vector_store import SareeVectorStore
from src.search import calculate_color_histogram

def main():
    metadata_path = 'data/metadata.csv'
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        print("Please run the image downloader first.")
        return
        
    print("Loading dataset metadata...")
    df = pd.read_csv(metadata_path)
    
    # Filter only successfully downloaded images
    df_success = df[df['download_status'] == 'Success']
    print(f"Found {len(df_success)} successfully downloaded images.")
    
    if len(df_success) == 0:
        print("No images to process.")
        return
        
    image_paths = df_success['local_image_path'].tolist()
    metadata_list = df_success.to_dict(orient='records')
    # Precompute color histograms and update metadata
    print("Precomputing HSV color histograms...")
    for meta, path in zip(metadata_list, image_paths):
        hist = calculate_color_histogram(path)
        meta['hsv_hist'] = hist.tolist() if hist is not None else None
        
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.generate_batch(image_paths)
    
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Add to Vector Store
    vector_store = SareeVectorStore(dimension=embeddings.shape[1])
    vector_store.add_embeddings(embeddings, metadata_list)
    
    # Save the index
    index_path = 'embeddings/faiss.index'
    meta_json_path = 'embeddings/index_meta.json'
    vector_store.save(index_path, meta_json_path)
    print(f"Vector store saved to {index_path} and {meta_json_path}")
    
    # Test Search with the first image
    print("\nTesting Search on the first image...")
    query_emb = embeddings[0]
    results = vector_store.search(query_emb, top_k=3)
    print(f"Query Image: {metadata_list[0]['SKU']}")
    for i, res in enumerate(results):
        print(f"{i+1}. SKU: {res['image_id']} - Score: {res['similarity_score']:.4f}")
        
if __name__ == "__main__":
    main()
