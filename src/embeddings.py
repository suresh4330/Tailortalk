import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from tqdm import tqdm

class EmbeddingGenerator:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} on {self.device}...")
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
    def generate_single(self, image_path_or_obj):
        """Generates normalized embedding for a single image."""
        try:
            if isinstance(image_path_or_obj, str):
                image = Image.open(image_path_or_obj).convert("RGB")
            else:
                image = image_path_or_obj.convert("RGB")
                
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.model(pixel_values=inputs['pixel_values']).image_embeds
            
            # Normalize for cosine similarity
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            return features.cpu().numpy()[0]
        except Exception as e:
            print(f"Error generating embedding for {image_path_or_obj}: {e}")
            return None

    def generate_batch(self, image_paths, batch_size=32):
        """Generates normalized embeddings for a list of image paths."""
        embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Generating embeddings"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_indices = []
            
            for j, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    valid_indices.append(i + j)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    
            if not batch_images:
                continue
                
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.model(pixel_values=inputs['pixel_values']).image_embeds
                
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            embeddings.extend(features.cpu().numpy())
            
        return np.array(embeddings)
