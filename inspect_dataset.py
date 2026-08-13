import pandas as pd
import json

def inspect_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        
        stats = {
            "num_rows": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_urls": int(df.duplicated(subset=['image_url']).sum()) if 'image_url' in df.columns else "N/A",
            "sample_data": df.head(3).to_dict(orient='records')
        }
        
        with open('dataset_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
        print("Inspection successful. Results saved to dataset_stats.json")
    except Exception as e:
        print(f"Error inspecting CSV: {e}")

if __name__ == "__main__":
    inspect_csv('byrappa_tejas_31july.csv')
