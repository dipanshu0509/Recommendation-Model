import streamlit as st
import pickle
import pandas as pd
import scipy.sparse

# Load the trained recommendation model
with open("recommendation_model.pkl", "rb") as file:
    preds_matrix = pickle.load(file)

data = pd.read_csv("formatted.csv")  # Ensure it has 'user_id', 'Item_ID', and 'rating'

# Convert sparse matrix to DataFrame
if isinstance(preds_matrix, scipy.sparse.csr_matrix):
    preds_matrix = pd.DataFrame(preds_matrix.toarray())

# Ensure preds_matrix has proper indexing
preds_matrix.index.name = "user_id"
preds_matrix.columns = data['Item_ID'].unique()  # Ensure columns match Item_IDs

# Load ratings dataset (Ensure you have this CSV in the same directory)

# Function to get top N recommendations for a user
def get_recommendations(user_id, preds_matrix, n=5):
    if user_id not in preds_matrix.index:
        return ["User ID not found. Please try another."]
    
    # Get top N product predictions
    recommendations = preds_matrix.loc[user_id].nlargest(n)
    recommended_products = recommendations.index.astype(str).tolist()
    
    return recommended_products

# Streamlit UI
st.title("E-Commerce Product Recommendation System")
st.write("Enter a User ID to get personalized product recommendations.")

user_id = st.text_input("Enter User ID:")

if st.button("Get Recommendations"):
    if user_id:
        try:
            user_id = int(user_id)  # Convert input to integer
            recommendations = get_recommendations(user_id, preds_matrix)
            
            if recommendations and "User ID not found" not in recommendations:
                st.write("### Top Recommended Products:")
                for prod in recommendations:
                    st.write(f"- {prod}")
            else:
                st.warning("User ID not found. Please try another.")
        except ValueError:
            st.error("Invalid User ID. Please enter a numeric User ID.")
    else:
        st.warning("Please enter a valid User ID.")
