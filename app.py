
# Import libraries
import openai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Authenticate with OpenAI API
openai.api_key = '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'

# Define function to retrieve course data
def get_courses():
    # Query OpenAI's GPT-3 for a list of courses
    prompt = "List of popular online courses"
    response = openai.Completion.create(engine='davinci', prompt=prompt, max_tokens=1024, n=1, stop=None, temperature=0.5)
    courses = response.choices[0].text.split('\n')
    # Clean up the list of courses
    courses = [course.strip() for course in courses if course.strip()]
    return courses

# Retrieve course data
courses = get_courses()

# Create a pandas DataFrame from the course data
df = pd.DataFrame(courses, columns=['course'])

# Define function to retrieve user preferences
def get_preferences():
    # Query OpenAI's GPT-3 for user preferences
    prompt = "What are your favorite topics to learn about?"
    response = openai.Completion.create(engine='davinci', prompt=prompt, max_tokens=1024, n=1, stop=None, temperature=0.5)
    preferences = response.choices[0].text.split('\n')
    # Clean up the list of preferences
    preferences = [preference.strip() for preference in preferences if preference.strip()]
    return preferences

# Define function to calculate similarity scores
def calculate_similarity_scores(preferences, df):
    # Create a TF-IDF vectorizer object
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the course data
    vectorizer.fit(df['course'])

    # Transform the user preferences into a TF-IDF vector
    pref_vector = vectorizer.transform(preferences)

    # Compute the cosine similarity between the user preferences and the courses
    try:
        similarity_scores = cosine_similarity(pref_vector, vectorizer.transform(df['course']))[0]
        if not similarity_scores.any():
            raise ValueError('No courses match your preferences.')
    except ValueError as e:
        st.error(e)
        st.stop()

    # Add the similarity scores to the DataFrame
    df['similarity_score'] = similarity_scores

    # Sort the DataFrame by similarity score in descending order
    df = df.sort_values('similarity_score', ascending=False)
    
    return df

# Define function to display top recommended courses
def display_top_courses(df):
    # Print the top 10 recommended courses
    st.subheader('Here are our top 10 recommended courses:')
    st.table(df.head(10)[['course', 'similarity_score']])

# Define function to get user input for preferences
def get_user_input():
    # Display instructions to user
    st.write('Please enter your top 1 preferences below:')

    # Retrieve user input for preferences
    preferences = []
    for i in range(1):
        preference = st.text_input(f"Preference #{i+1}:")
        if preference:
            preferences.append(preference.strip())
    
    # Check if user entered any preferences
    if not preferences:
        st.warning('Please enter at least one preference.')
        st.stop
