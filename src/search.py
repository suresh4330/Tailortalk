import cv2
import numpy as np
from PIL import Image

def calculate_color_histogram(image_path):
    """
    Calculates a normalized 3D color histogram in HSV color space.
    HSV is better for color comparisons than RGB.
    """
    try:
        # Load image with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram (Hue: 8 bins, Saturation: 12 bins, Value: 3 bins)
        # We emphasize Hue and Saturation for sarees.
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 12, 3], 
                            [0, 180, 0, 256, 0, 256])
                            
        # Normalize the histogram
        cv2.normalize(hist, hist)
        return hist.flatten()
    except Exception as e:
        print(f"Error calculating histogram for {image_path}: {e}")
        return None

def compare_histograms(hist1, hist2):
    """
    Compares two histograms using Correlation.
    Returns a score between 0 (no correlation) and 1 (perfect correlation).
    """
    if hist1 is None or hist2 is None:
        return 0.0
    # Use cv2.HISTCMP_CORREL
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # Ensure it's bounded between 0 and 1
    return max(0.0, min(1.0, score))

class VisualSearcher:
    def __init__(self, embedding_generator, vector_store, alpha=0.6, beta=0.4):
        """
        alpha: weight for embedding similarity
        beta: weight for color histogram similarity
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.alpha = alpha
        self.beta = beta
        
    def search(self, query_image_path, top_k=5, faiss_k=20):
        """
        Executes the reranking pipeline:
        1. FAISS top `faiss_k` using embeddings
        2. Visual reranking using color histograms
        3. Returns final top `top_k`
        """
        # 1. Generate Query Embedding
        query_emb = self.embedding_generator.generate_single(query_image_path)
        if query_emb is None:
            return []
            
        # 2. Query FAISS
        faiss_results = self.vector_store.search(query_emb, top_k=faiss_k)
        
        # 3. Calculate Query Histogram
        query_hist = calculate_color_histogram(query_image_path)
        
        # 4. Rerank
        for res in faiss_results:
            emb_score = res['similarity_score']
            result_path = res['metadata'].get('local_image_path')
            
            if query_hist is not None:
                # Load precomputed histogram from metadata
                res_hist_list = res['metadata'].get('hsv_hist')
                if res_hist_list:
                    res_hist = np.array(res_hist_list, dtype=np.float32)
                    hist_score = compare_histograms(query_hist, res_hist)
                
            # Transparent scoring
            final_score = (self.alpha * emb_score) + (self.beta * hist_score)
            
            res['rerank_details'] = {
                'embedding_score': float(emb_score),
                'histogram_score': float(hist_score)
            }
            res['similarity_score'] = float(final_score)
            
        # 5. Sort by new similarity score descending
        reranked_results = sorted(faiss_results, key=lambda x: x['similarity_score'], reverse=True)
        
        return reranked_results[:top_k]
