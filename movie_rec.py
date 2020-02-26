import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#helper functions. Use them when needed
def getTitleFromIndex(index):
    return df[df.index==index]["title"].values[0]

def getIndexFromTitle(title):
    return df[df.title==title]["index"].values[0]

#step 1 : Read CSV file
df=pd.read_csv("movie_dataset.csv")
#print(df.columns)
#Step 2: Select features
features=['keywords','cast','genres','director']
#Step 3: Create a column in DF which combines all selected features
for feature in features:
    df[feature]=df[feature].fillna("")
def combine_features(row):
    try:
        row=row["keywords"]+" "+row["cast"]+" "+row["genres"]+" "+row["director"]
        return row
    except:
        print("Error",row)
df["combined_features"]=df.apply(combine_features ,axis=1)
#print("Combined Features",df["combined_features"].head())
#Step 4: Create a count matrix from this new combined column
cv=CountVectorizer()
count_matrix=cv.fit_transform(df["combined_features"])
#Step 5: Compute the Cosine similarity based on the count_matrix
cosine_sim=cosine_similarity(count_matrix)
movieUserLikes='Avatar'
#Step 6: Get index of this movie from its title
movie_index=getIndexFromTitle(movieUserLikes)
similarMovies=list(enumerate(cosine_sim[movie_index]))
#Step 7: Get a list of similar movies in descending order of similarity score
sortedSimilarMovies=sorted(similarMovies,key=lambda x:x[1],reverse=True)
#Step 8: Print titles of first 50 movies
i=0
for element in sortedSimilarMovies:
    print(i+1,":",getTitleFromIndex(element[0]))
    i+=1
    if i>50:
        break 