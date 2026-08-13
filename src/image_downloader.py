import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_image(url, sku, output_dir, timeout=10):
    """
    Downloads an image from a URL, verifies it, and saves it.
    Returns a tuple (sku, file_path, status, error_message).
    """
    if pd.isna(url) or not url:
        return sku, None, "Failed", "Empty URL"
        
    try:
        # Create stable filename based on SKU
        # Assuming format is mostly JPEG or WEBP based on inspect output
        ext = url.split('.')[-1]
        if len(ext) > 4 or '?' in ext:
            ext = "jpg" # default fallback
        
        file_path = os.path.join(output_dir, f"{sku}.{ext}")
        
        # If it already exists and is valid, skip
        if os.path.exists(file_path):
            try:
                with Image.open(file_path) as img:
                    img.verify()
                return sku, file_path, "Success", "Already exists"
            except Exception:
                pass # Re-download if invalid
                
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        # Verify it's a valid image before saving
        image = Image.open(BytesIO(response.content))
        image.verify() # Checks if it's broken
        
        # Save it
        with open(file_path, 'wb') as f:
            f.write(response.content)
            
        return sku, file_path, "Success", ""
        
    except requests.exceptions.RequestException as e:
        return sku, None, "Failed", f"Network error: {str(e)}"
    except Exception as e:
        return sku, None, "Failed", f"Image error: {str(e)}"

def process_dataset(csv_path, output_image_dir, metadata_path, limit=None, max_workers=5):
    """
    Reads the CSV, downloads images, and saves the metadata.
    """
    print(f"Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if limit:
        df = df.head(limit)
        print(f"Limiting to first {limit} rows for testing.")
        
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sku = {
            executor.submit(download_image, row['image_url'], row['SKU'], output_image_dir): row 
            for _, row in df.iterrows()
        }
        
        for future in tqdm(as_completed(future_to_sku), total=len(future_to_sku), desc="Downloading images"):
            row = future_to_sku[future]
            sku, file_path, status, error = future.result()
            
            row_dict = row.to_dict()
            row_dict['local_image_path'] = file_path
            row_dict['download_status'] = status
            row_dict['download_error'] = error
            results.append(row_dict)
            
    # Save metadata
    results_df = pd.DataFrame(results)
    
    # Sort to keep original order roughly or by SKU
    results_df = results_df.sort_values('SKU')
    results_df.to_csv(metadata_path, index=False)
    
    success_count = (results_df['download_status'] == 'Success').sum()
    failed_count = (results_df['download_status'] == 'Failed').sum()
    
    print(f"\nDownload Summary:")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed to download: {failed_count}")
    print(f"Metadata saved to: {metadata_path}")
    
    if failed_count > 0:
        print("\nFailed examples:")
        print(results_df[results_df['download_status'] == 'Failed'][['SKU', 'image_url', 'download_error']].head())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images to download')
    args = parser.parse_args()
    
    CSV_PATH = '../byrappa_tejas_31july.csv'
    OUTPUT_DIR = '../data/images'
    METADATA_PATH = '../data/metadata.csv'
    
    # Adjust paths if running from root
    if os.path.exists('byrappa_tejas_31july.csv'):
        CSV_PATH = 'byrappa_tejas_31july.csv'
        OUTPUT_DIR = 'data/images'
        METADATA_PATH = 'data/metadata.csv'
        
    process_dataset(CSV_PATH, OUTPUT_DIR, METADATA_PATH, limit=args.limit)
