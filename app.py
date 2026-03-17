import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
    respose = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=66b07b358c466e9ba4fc3d94f7d16910&language=en-US'.format(movie_id))
    data = respose.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)) , reverse=True , key= lambda x:x[1])[1:7]

    recommended_movies = []
    recommended_movies_poster = []

    for i in movie_list: 
        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(movies.iloc[i[0]].title)
        
        # fetching poster of the movie via API
        recommended_movies_poster.append(fetch_poster(movie_id))
    return recommended_movies , recommended_movies_poster  

movies_dict = pickle.load(open('movies_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity_reduce.pkl','rb'))

st.title("Movie Recommendation System")

selected_movie_name = st.selectbox(
'Which Movie Would You Like To Watch Today',
movies['title'].values
)

if st.button('Recommend'):
    names , posters = recommend(selected_movie_name)

    cols = st.columns(3)
    for i in range(6):
        with cols[i % 3]:
            st.text(names[i])
            st.image(posters[i])