import pickle
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
full_data = pd.read_csv('amazon.csv')

# Keep only necessary columns 
data = full_data[['product_name', 'img_link']].dropna().head(10)

# Load ResNet-50 Model (Pretrained)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last classification layer
model = model.to(device)
model.eval()

# Load Sentence-BERT model
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# Image Preprocessing Function
def preprocess_image(image_url):
    try:
        image_url = image_url.strip()
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return transform(image).unsqueeze(0).to(device)
    except requests.exceptions.RequestException as req_err:
        print(f"Request error: {req_err} for URL: {image_url}")
    except Exception as e:
        print(f"Error loading image {image_url}: {e}")
    
    return None

# Extract Features for All Images
image_features = []
text_features = []
valid_data = []  # Store valid product details

for _, row in data.iterrows():
    img_tensor = preprocess_image(row['img_link'])
    if img_tensor is not None:
        with torch.no_grad():
            features = model(img_tensor).squeeze().flatten().cpu().numpy()
            image_features.append(features)
        
        # Compute Sentence-BERT text embeddings
        text_embedding = text_model.encode(row['product_name'])
        text_features.append(text_embedding)

        # Store product details
        valid_data.append({
            'product_name': row['product_name'],
            'img_link': row['img_link'],
        })

# Convert to NumPy Arrays
image_features = np.array(image_features)
text_features = np.array(text_features)

# Save Extracted Features
with open('image_text_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'image_features': image_features, 'text_features': text_features, 'data': valid_data}, f)

# Function to Get Recommendations
def get_recommendations_by_image(image_url, n_recommendations=4):
    """
    Get product recommendations based on combined image + text similarity.
    """
    try:
        # Load stored data
        with open('image_text_model.pkl', 'rb') as f:
            saved_data = pickle.load(f)

        model = saved_data['model']
        image_features = saved_data['image_features']
        text_features = saved_data['text_features']
        valid_data = saved_data['data']

        # Extract Features for Input Image
        img_tensor = preprocess_image(image_url)
        if img_tensor is None:
            return []

        with torch.no_grad():
            query_image_features = model(img_tensor).squeeze().flatten().cpu().numpy()

        # Compute Image Similarity
        image_similarities = cosine_similarity(query_image_features.reshape(1, -1), image_features)[0]
        top_image_indices = np.argsort(image_similarities)[::-1][1:n_recommendations+5]  # Get more results to refine with text similarity

        # Compute Text Similarity on Filtered Items
        query_text = "Unknown Product"  # Since no product name is given, we match against top images
        query_text_embedding = text_model.encode(query_text)

        text_similarities = cosine_similarity(query_text_embedding.reshape(1, -1), text_features[top_image_indices])[0]

        # Combine Similarities (Weighted Sum)
        combined_scores = 0.6 * image_similarities[top_image_indices] + 0.4 * text_similarities
        final_indices = np.argsort(combined_scores)[::-1][:n_recommendations]

        # Prepare Recommendations
        recommendations = []
        for idx in final_indices:
            recommended_product = valid_data[top_image_indices[idx]]

            recommendations.append({
                'product_name': recommended_product['product_name'],
                'img_link': recommended_product['img_link'],
                'similarity': combined_scores[idx],
                'rating': recommended_product.get('rating', 'N/A'),
                'category': recommended_product.get('category', ''),
                'product_link': recommended_product.get('product_link', ''),
                'rating_count': recommended_product.get('rating_count', 'N/A')
                
            })

        return recommendations
    except Exception as e:
        print(f"Error: {e}")
        return []

# Example Usage
image_url = "https://m.media-amazon.com/images/I/315GdnF+LcL._SY300_SX300_.jpg"
recommendations = get_recommendations_by_image(image_url)

# Print recommendations
for rec in recommendations:
    print(f"Product: {rec['product_name']} - Similarity: {rec['similarity']:.4f}")
    print(f"Image Link: {rec['img_link']}\n")
