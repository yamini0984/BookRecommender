#!/usr/bin/env python
# coding: utf-8

# In[20]:


# Import the necessary libraries at the beginning of your script
import streamlit as st
import pickle  # Import the pickle module

# Define the 'recommend' function here (you can copy the function you provided)
def recommend(book_name, model, bk, num_recommendations):
    # Find the ISBN of the input book name
    book_info = bk[bk['Book-Title'].str.contains(book_name, case=False, na=False)]
    if not book_info.empty:
        input_isbn = book_info.iloc[0]['ISBN']

        # Get all book ISBNs
        all_isbns = bk['ISBN'].unique()

        # Predict ratings for the input book for all users
        predictions = [(user, model.predict(user, input_isbn).est) for user in all_isbns]

        # Sort predictions by estimated rating in descending order
        sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

        # Get the top N recommendations
        top_n_recommendations = sorted_predictions[:num_recommendations]

        recommended_books = []
        for user, rating in top_n_recommendations:
            book_data = bk[bk['ISBN'] == user]
            if not book_data.empty:
                recommended_books.append({
                    'Book-Title': book_data['Book-Title'].values[0],
                    'Image-URL-M': book_data['Image-URL-M'].values[0]
                })

        return recommended_books

# Load the collaborative model and book dataset from the pickle files
with open('collaborative_model.pkl', 'rb') as model_file:
    collaborative_model = pickle.load(model_file)

with open('bk.pkl', 'rb') as bk_file:
    bk = pickle.load(bk_file)

# Extract unique book titles from the dataset
unique_book_titles = bk['Book-Title'].unique()

# Your Streamlit app code with layout modification
def main():
    st.title("Book Recommendation App")
    
    # Create two columns for input and output
    col1, col2 = st.columns(2)

    # Initialize recommendations as an empty list
    recommendations = []

    # Input in the left column
    with col1:
        st.subheader("Input")
        # Add a dropdown for book selection with unique book titles
        selected_book = st.selectbox("Select a book:", unique_book_titles)
        num_recommendations = st.slider("Number of Recommendations", 1, 20, 5)  # Adjust the slider range as needed
        if st.button("Get Recommendations"):
            recommendations = recommend(selected_book, collaborative_model, bk, num_recommendations)

    # Output in the right column
    with col2:
        st.subheader("Output")
        if recommendations:
            st.subheader(f"Top {num_recommendations} Recommendations:")
            for i, rec in enumerate(recommendations):
                st.write(f"{i + 1}. {rec['Book-Title']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(rec['Image-URL-M'], use_column_width=True, caption=rec['Book-Title'])
                with col2:
                    st.write("")  # Empty space for alignment

if __name__ == "__main__":
    main()


# In[ ]:




