import pickle
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Load data
full_data = pd.read_csv('amazon.csv')

# Drop unnecessary columns
columns_to_drop = [
    'category', 'discounted_price', 'actual_price',
    'discount_percentage', 'about_product', 'user_name', 'review_id',
    'review_title', 'review_content', 'product_link',
    'rating_count', 'user_id', 'rating'
]
data = full_data.drop(columns=columns_to_drop).dropna()

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for product names
product_names = data['product_name'].tolist()
embeddings = model.encode(product_names, convert_to_tensor=True)

# Save embeddings and model
with open('sbert_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'embeddings': embeddings, 'data': data}, f)


def get_recommendations_by_name(product_name, n_recommendations=16):
    """
    Get product recommendations based on product name similarity using Sentence-BERT.
    """
    try:
        # Load trained model and embeddings
        with open('sbert_model.pkl', 'rb') as f:
            saved_data = pickle.load(f)
        
        model = saved_data['model']
        embeddings = saved_data['embeddings']
        data = saved_data['data']
        
        # Compute embedding for the input product name
        query_embedding = model.encode(product_name, convert_to_tensor=True).unsqueeze(0)
        
        # Compute similarity with all product embeddings
        similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        
        # Get top recommendations (excluding itself)
        top_indices = torch.argsort(similarities, descending=True)[1:n_recommendations+1]
        
        recommendations = []
        for idx in top_indices:
            recommended_product = data.iloc[idx.item()]
            
            recommendation_info = {
                'product_name': recommended_product['product_name'],
                'img_link': recommended_product.get('img_link', ''),
                'category': recommended_product.get('category', ''),
                'product_link': recommended_product.get('product_link', ''),
                'rating_count': recommended_product.get('rating_count', 'N/A'),
                'rating': recommended_product.get('rating', 'N/A'),
                'similarity': similarities[idx].item()
            }
            recommendations.append(recommendation_info)
        
        return recommendations
    except Exception as e:
        print(f"Error: {e}")
        return []

# Example Usage
product_name = "Wireless Bluetooth Earbuds"
recommendations = get_recommendations_by_name(product_name)

# Print recommendations
for rec in recommendations:
    print(f"{rec['product_name']} ({rec['category']}) - Similarity: {rec['similarity']}")
    print(f"Product Link: {rec['product_link']}")
    print("\n")
