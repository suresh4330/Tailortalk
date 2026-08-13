import json
import os
import cv2
import numpy as np

def calculate_color_histogram(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 12, 3], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().tolist()
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    meta_path = 'embeddings/index_meta.json'
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    for i, meta in enumerate(metadata):
        path = meta.get('local_image_path')
        if path and os.path.exists(path):
            hist = calculate_color_histogram(path)
            meta['hsv_hist'] = hist
        else:
            meta['hsv_hist'] = None
            
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} images...")
            
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)
    print("Added histograms to metadata!")

if __name__ == "__main__":
    main()
