import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors











# Load data
full_data = pd.read_csv('amazon.csv')

# Drop unnecessary columns
columns_to_drop = [
    'product_name', 'discounted_price', 'actual_price',
    'discount_percentage', 'about_product', 'user_name', 'review_id',
    'review_title', 'review_content', 'img_link', 'product_link',
    'rating_count', 'user_id', 'rating'
]
data = full_data.drop(columns=columns_to_drop).dropna()
data = data[data['category'] != '|']  # Remove invalid category rows

# Encode 'product_id'
product_id_encoder = LabelEncoder()
data['product_id'] = product_id_encoder.fit_transform(data['product_id'])

# Handle multi-category encoding
#data['category'] = data['category'].apply(lambda x: x.split('|'))  # Convert 'category' column to list
mlb = MultiLabelBinarizer()
category_encoded = mlb.fit_transform(data['category'])
category_df = pd.DataFrame(category_encoded, columns=mlb.classes_)

# Merge new category encoding and drop original
data = data.drop(columns=['category']).reset_index(drop=True)
data = pd.concat([data, category_df], axis=1)

# Split dataset
Train, Test = train_test_split(data, test_size=0.2, random_state=42)

# Train Nearest Neighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(Train.drop(columns=['product_id']))  # Exclude 'product_id' during training

# Save model
pickle.dump(model, open('model.pkl', 'wb'))


def get_recommendations_with_details(model=model, data=data, full_data=full_data, product_id_encoder=product_id_encoder, mlb=mlb, n_recommendations=5, product_id=10):
    """
    Get product recommendations prioritizing the same category first, then fallback to nearest neighbors.
    """
    try:
        
        product_index_list = data.index[data['product_id'] == product_id].tolist()
        if not product_index_list:
            print(f"Product ID {product_id} not found in dataset.")
            return []
        product_index = product_index_list[0]

        # Get category encoding of the target product
        product_categories = data.iloc[product_index][mlb.classes_].values

        # Compute nearest neighbors
        distances, indices = model.kneighbors(data.iloc[product_index].drop('product_id').values.reshape(1, -1))



        recommendations = []
        same_category_count = 0
        other_category_count = 0
        max_same_category = n_recommendations
        max_other_category = n_recommendations - max_same_category

        for j in range(1, len(indices[0])):
            if len(recommendations) >= n_recommendations:
                break
            recommended_index = indices[0][j]
            recommended_product_encoded = data.iloc[recommended_index]['product_id']
            recommended_original_id = product_id_encoder.inverse_transform([recommended_product_encoded])[0]
            recommended_product_details = full_data[full_data['product_id'] == recommended_original_id].iloc[0]
            
            recommended_categories = data.iloc[recommended_index][mlb.classes_].values
            same_category = (product_categories * recommended_categories).sum() > 0

            if same_category and same_category_count < max_same_category:
                same_category_count += 1
                
            elif not same_category and other_category_count < max_other_category:
                other_category_count += 1
                


        
            
            recommendation_info = {
                'product_id': recommended_original_id,
                'product_name': recommended_product_details['product_name'],
                'category': recommended_product_details['category'],
                'rating': recommended_product_details.get('rating', 'N/A'),
                'rating_count': recommended_product_details.get('rating_count', 'N/A'),
                'img_link': recommended_product_details.get('img_link', ''),
                'product_link': recommended_product_details.get('product_link', ''),
                'distance': distances[0][j]
            }
            recommendations.append(recommendation_info)

        return recommendations
    except Exception as e:
        print(f"Error: {e}")
        return []

# Example Usage
product_id = 21
recommendations = get_recommendations_with_details(product_id=product_id)

# Print recommendations
for rec in recommendations:
    print(f"{rec['product_name']} ({rec['category']}) - Distance: {rec['distance']}")
    print(f"Product Link: {rec['product_link']}")
    print("\n")
