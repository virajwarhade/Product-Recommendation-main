import streamlit as st
import pandas as pd

from main import get_recommendations_with_details
from sentence_bert import get_recommendations_by_name
from images_recomend import get_recommendations_by_image

from sklearn.preprocessing import LabelEncoder  # Assuming you need to import LabelEncoder



# GENERAL SETTINGS
PAGE_TITLE = "Product recomendation system"
PAGE_ICON = ":wave:"
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON,layout="wide")


import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return 1 or more
print(torch.cuda.get_device_name(0))  # Should show 'NVIDIA GeForce GTX 1650'








col1, col2, col3 = st.columns(3)
# Load the CSV dataset
@st.cache_data    # Caching for faster reloading
def load_data():
    data = pd.read_csv('amazon.csv')
    return data

def display_selected_product_details(selected_product_details):
    """
    Display the product details of the selected product
    """
    st.header("Selected Product Details")
    st.write(f"Product ID: {selected_product_details['product_id']}")
    st.write(f"Product Name: {selected_product_details['product_name']}")
    st.write(f"Category: {selected_product_details['category']}")
    st.write(f"Rating: {selected_product_details['rating']} ({selected_product_details['rating_count']})")

def display_selected_product_image(selected_product_details):
    st.image(selected_product_details['img_link'], use_container_width=True)
    st.write(f"Product Link: [Link]({selected_product_details['product_link']})")











def display_recommendations(recommendations,type):

    with st.container(border=True):


        #st.markdown("<div style='border:2px #3d3d3d solid; width:full;margin-top:10px;'></div>", unsafe_allow_html=True)

        st.header(type)

        num_columns = 4  # Number of columns in the grid
        columns = st.columns(num_columns)

        for index, recommendation in enumerate(recommendations):
            with columns[index % num_columns]:  # Distribute items across columns
                with st.container():
                    # Use st.image, st.markdown, and st.caption to create the card
                    st.image(recommendation['img_link'], use_container_width =False)
                    st.markdown(f"**{recommendation['product_name']}**", help="Click below to view product")
                    st.caption(f"🔢 Rank: {index + 1} | ⭐ {recommendation['rating']} ({recommendation['rating_count']} reviews)")
                    st.link_button("🔗 View Product", recommendation['product_link'])

                    st.markdown(f"**Category: {recommendation['category']}**")

        
        #st.markdown("<div style='border:2px #3d3d3d solid; width:full;margin-top:10px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:60px;margin-bottom:60px;'></div>", unsafe_allow_html=True)





#with st.container(border=True):
#    st.header("📌 Recommended by product name (sentence bert)")
#
#    num_columns = 4  # Number of columns in the grid
#    columns = st.columns(num_columns)
#
#    for index, recommendation in enumerate(recommendations):
#        with columns[index % num_columns]:  # Distribute items across columns
#            with st.container():
#                # Use st.image, st.markdown, and st.caption to create the card
#                st.image(recommendation['img_link'], use_container_width =False)
#                st.markdown(f"**{recommendation['product_name']}**", help="Click below to view product")
#                st.caption(f"🔢 Rank: {index + 1} | ⭐ {recommendation['rating']} ({recommendation['rating_count']} reviews)")
#                st.link_button("🔗 View Product", recommendation['product_link'])
#
#                st.markdown(f"**Category: {recommendation['category']}**")
#
#st.markdown("<div style='margin-top:10px;margin-bottom:10px;'></div>", unsafe_allow_html=True)
#
#with st.container(border=True):
#    st.header("📌 Recommended by description (sentence bert)")
#
#    num_columns = 4  # Number of columns in the grid
#    columns = st.columns(num_columns)
#
#    for index, recommendation in enumerate(recommendations):
#        with columns[index % num_columns]:  # Distribute items across columns
#            with st.container():
#                # Use st.image, st.markdown, and st.caption to create the card
#                st.image(recommendation['img_link'], use_container_width =False)
#                st.markdown(f"**{recommendation['product_name']}**", help="Click below to view product")
#                st.caption(f"🔢 Rank: {index + 1} | ⭐ {recommendation['rating']} ({recommendation['rating_count']} reviews)")
#                st.link_button("🔗 View Product", recommendation['product_link'])
#
#                st.markdown(f"**Category: {recommendation['category']}**")
#    
#st.markdown("<div style='margin-top:10px;margin-bottom:10px;'></div>", unsafe_allow_html=True)




















col0, col02, col03 = st.columns([1, 2, 1])  # Middle column is wider


def main():

    with col02:
            

        st.title('Product recomendation system')

        # Load the dataset
        data = load_data()





        st.title('Filter Category')
        # Filter products by category
        selected_category = st.selectbox('Select a category:', sorted(data['category'].unique()))
        filtered_data = data[data['category'] == selected_category]

        st.title("Product Recommendations")

        # Create an input field for the user to enter the product ID    
        product_id_encoder = LabelEncoder()
        selected_product_name = st.selectbox('Select a product:', sorted(filtered_data['product_name'].values))

        # Check if a product is selected
        if selected_product_name != 'Select a product':
            # Get the product ID for the selected product name
            selected_product_id = filtered_data[filtered_data['product_name'] == selected_product_name]['product_id'].values[0]
            product_id_encoder.fit(data['product_id'])
            product_id = product_id_encoder.transform([selected_product_id])[0]

            selected_product_details = filtered_data[filtered_data['product_id'] == selected_product_id].iloc[0]
            details_col, image_col = st.columns(2)
            # Display product details in the left column
            with details_col:
                display_selected_product_details(selected_product_details)

            # Display product image and link in the right column
            with image_col:
                display_selected_product_image(selected_product_details)

    # Button to trigger the recommendations
    if st.button("Get Recommendations"):
        # Call the recommendation function with default parameter values
        recommendations_category = get_recommendations_with_details(product_id=product_id)


        recommendations_product_name = get_recommendations_by_name(product_name=selected_product_details['product_name'])


        recommendations_image = get_recommendations_by_image(image_url=selected_product_details['img_link'])


        #Recommended by product name (KNN)
        display_recommendations(recommendations_product_name,"📌 Recommended by product name (sentence bert)")


        #Recommended by Image (CNN)
        display_recommendations(recommendations_image,"📌 Recommended by Image ")



        #Recommended by category (KNN)
        display_recommendations(recommendations_category,"📌 Recommendation of categorys (KNN)")





















        

if __name__ == '__main__':
    main()
