import numpy as np
import pandas as pd
import ast
import sklearn
import nltk
import pickle

movies = pd.read_csv(r'C:/Users/vedan/Desktop/Projects/Movie_Recomendation_System/Dataset/tmdb_5000_movies.csv')
credits = pd.read_csv(r'C:/Users/vedan/Desktop/Projects/Movie_Recomendation_System/Dataset/tmdb_5000_credits.csv')

print(movies.head(1))
print(credits.head(1))

# print(credits.head(1)['cast'].values)

movies = movies.merge(credits, on='title')
print(movies.shape)

# Create Tags with usefull columns
# Colums kept: geners, id, keywords, title, overview, cast, crew

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
print(movies.head())

# Make new DataFrame movie_id, title, tags = merge(overview + (genres, keywords, cast[3:], crew))

print(movies.isnull().sum())
# movies = movies.dropna(inplace=True)
movies.dropna(inplace=True)
print(movies.isnull().sum())
print(movies.duplicated().sum())
print(movies.iloc[0].genres)

# Format '[{"id":28, "name": "Action"}, {"id":12, "name": "Adventure"}, {"id":14, "name": "SciFi"}]

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):  # ast.literal_eval(String): Converts string to list 
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
print(movies.head()['genres'])

movies['keywords'] = movies['keywords'].apply(convert)
print(movies.head()['keywords'])

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):  # ast.literal_eval(String): Converts string to list 
        if counter !=3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)
print(movies.head()['cast'])

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)
print(movies.head()['crew'])

print(movies.head())

print(movies['overview'][0])
movies['overview'] = movies['overview'].apply(lambda x:x.split())
print(movies.head()['overview'])
print(movies.head())

# Remove whitespaces between strings
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
print(movies.head())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
print(movies.head())

new_df = movies[['movie_id','title','tags']]
print(new_df)
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
print(new_df['tags'])
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
print(new_df['tags'])

# stemming: make past/present/future verbs same //library nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
print(vectors[0])

# print(cv.get_feature_names())
print(len(cv.get_feature_names()))

# calculate cosine distance // for high dimentional data 
# angular distance b/w vectors

from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(vectors).shape)
similarity = cosine_similarity(vectors)
print(similarity[0])
print(sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x:x[1])[1:6])

def recomend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

recomend('Batman')
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))