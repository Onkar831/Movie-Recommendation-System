import numpy as np 
import pandas as pd

# Reading data with the help of pandas
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merging both datasets
movies = movies.merge(credits , on='title')

# # Now extracting features that will be contributing or usefull in analysis 
# .Genres
# .id 
# .Keywords
# .title
# .overview
# .cast
# .crew 

movies= movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# Checking and removing missing values
movies.isnull().sum()
movies.dropna(inplace=True) 

# For genre & keywords there's str , we have to covert it into list first 
import ast
ast.literal_eval

def convert(obj): 
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L  

# Applying above function on genre & keywords

movies['genres'] = movies['genres'].apply(convert)    
movies['keywords'] = movies['keywords'].apply(convert) 

# For cast we need top 3 actors to be used as tags further , top3 actors are stored in first 3 dict of the list 

def convert3(obj): 
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+= 1
        else:
            break 

    return L   
    
movies['cast'] = movies['cast'].apply(convert3)

# For crew we will be needing only dict having job as director and fetch that name  
def fetch_director(obj): 
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L 
movies['crew'] = movies['crew'].apply(fetch_director) 

# All the features are in list except overview 
movies['overview'] = movies['overview'].apply( lambda x: x.split())

# Removing Spaces between the words 
movies['genres'] = movies['genres'].apply( lambda x : [i.replace(" " , "") for i in x])
movies['keywords'] = movies['keywords'].apply( lambda x : [i.replace(" " , "") for i in x]) 
movies['cast'] = movies['cast'].apply( lambda x : [i.replace(" " , "") for i in x]) 
movies['crew'] = movies['crew'].apply( lambda x : [i.replace(" " , "") for i in x]) 

# Combining features to make single feature for text preprocessing
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] +  movies['crew'] 

# Creating new data frame with all appropriate data needs met 
new_df = movies[['movie_id' , 'title' , 'tags']]

# Creating new data frame with all appropriate data needs met 
new_df = movies[['movie_id' , 'title' , 'tags']]

# Converting the tags dtype from list to str
new_df['tags'] = new_df['tags'].apply( lambda x:" ".join(x)) 

# Converting tags in lowercase for faster processing
new_df['tags'] = new_df['tags'].apply( lambda x:x.lower())

# Using countvectorizer to convert text to vectors
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer( max_features=5000 , stop_words='english')
vectors = cv.fit_transform( new_df['tags'] ).toarray()

# Now we will be doing stemming for faster execution 
from nltk.stem import PorterStemmer
ps = PorterStemmer()
def stem (text): 
    y = [] 

    for i in text.split():
        y.append(ps.stem(i)) 

    return " ".join(y)     
new_df['tags'] = new_df['tags'].apply(stem) 

# Creating Similarity matrix to fetch similar results 
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors) 
sorted(list(enumerate(similarity[0])) , reverse=True , key= lambda x:x[1])[1:7] 
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)) , reverse=True , key= lambda x:x[1])[1:7]

    for i in movie_list: 
        print(new_df.iloc[i[0]].title)

# Model is ready , now importing it using joblib
import pickle
pickle.dump(new_df.to_dict() , open('movies_dict.pkl' , 'wb')) 
similarity = similarity[:3550, :3550]                              # file was too large to be uploaded so got a smaller one
pickle.dump(similarity , open('similarity_reduce.pkl' , 'wb'))