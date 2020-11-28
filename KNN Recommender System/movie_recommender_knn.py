# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 13:00:10 2020

@author: DELL
"""

from math import sqrt
from random import randrange
import pandas as p
import numpy as np
from scipy import spatial
from functools import reduce
import warnings
import sklearn
from sklearn.decomposition import TruncatedSVD
import time

np.seterr('log')

dataset_list = []

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
# Calculate the Euclidean distance between two vectors
def euclidean_distance(data1, data2):
    distance = 0.0
    for i in range(len(data1)-1):
        distance += (data1[i] - data2[i])**2
    return sqrt(distance)

#Calculate the Manhattan distance 
def manhattan_distance(data1, data2):
    distance = 0.0
    for i in range(len(data1)-1):
        distance += abs(data1[i] - data2[i])
    return distance

def cosine_distance(data1, data2):
    genres_A = data1[1]
    genres_B = data2[1]
    # print(genres_A)
    # print(genres_B)
    # print(spatial.distance.cosine(genres_A, genres_B))
    genreDistance = spatial.distance.cosine(genres_A, genres_B)
    popularity_A = data1[2]
    popularity_B = data2[2]
    popularityDistance = abs(popularity_A - popularity_B)
    return genreDistance + popularityDistance

def get_neighbors(train, test_row, num_neighbors,metric):
    distances = list()
    for train_row in train:
        #print(train[train_row])
        if metric == "manhattan":
            dist = manhattan_distance(test_row, train_row) 
        elif metric == "euclidean":
            dist = euclidean_distance(test_row, train_row)
        elif metric == "cosine":
            dist = cosine_distance(test_row, train[train_row])
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i+1][0])
    return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors,metric):
    neighbors = get_neighbors(train, test_row, num_neighbors,metric)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    print(folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = 0.0
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors,metric):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors,metric)
        predictions.append(output)
    return(predictions)

def binary(genre_list):
    binaryList = []
    
    for genre in genreList:
        if genre in genre_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList

############################# Preprocess data for KNN  ############################

start = time.time()

# Step 1: Read data from Ratings and Movies CSV and store it in dataframe

ratings_cols = ['user_id', 'movieId', 'rating']
dataset_ratings = p.read_csv('ml-20m/ratings.csv', sep=',', names=ratings_cols, usecols=range(3),skiprows = 1)

movies_cols =['movieId','title','genres']
genreList = ['Action','Adventure','Animation','Children','Comedy','Crime','Documentary',
                        'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance',
                        'Sci-Fi','Thriller','War','Western','IMAX','(no genres listed)']
dataset_movies = p.read_csv('ml-20m/movies.csv',sep= ',',names=movies_cols,skiprows=1)
dataset_movies["genres"] = dataset_movies["genres"].str.split("|",n=-1, expand = False)

# Step 2 : (i) group by movie ID
#         (ii) Calculate the total number of ratings 
#         (iii) Calculate the average rating for every movie

movieProperties = dataset_ratings.groupby('movieId').agg({'rating': [np.size, np.mean]})

# Step 3: Normalize the values for dataframes

movieRatings = p.DataFrame(movieProperties['rating']['size'])
# dataset_list = movieNumRatings.values.tolist()   
# normalize_dataset(dataset_list, dataset_minmax(dataset_list))
movieNumRatings_normalized = movieRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

dataset_movies["genres"] = dataset_movies["genres"].apply(lambda x: binary(x))
dataset_movies_list = dataset_movies.values.tolist()

############### Preprocess Data for Matrix Factorization #############

# Step 1 is similar to KNN

# Step 2: Combine ratings and movies dataset
dataset_combined = p.merge(dataset_ratings,dataset_movies, on='movieId').drop(['genres'],axis=1).dropna(axis = 0, subset = ['title'])

# Step 3: Count total ratings for each movie and merge both datasets
dataset_count = (dataset_combined.
                     groupby(by = ['title'])['rating'].
                     count().reset_index().
                     rename(columns = {'rating': 'totalRatingCount'})
                     [['title','totalRatingCount']])

dataset_userRatingsInfo = dataset_combined.merge(dataset_count,left_on = 'title', right_on = 'title', how = 'left').drop_duplicates(['user_id','title'])

# Create a maitrix for factorization
dataset_userRatingsInfo.sort_values(by=['title'],ascending=False)
length = len(dataset_userRatingsInfo)
subset1 = dataset_userRatingsInfo[:2000000] 
subset2 = dataset_userRatingsInfo[2000000:4000000]
#subset3 = dataset_userRatingsInfo[10000000:15000000] 
#subset4 = dataset_userRatingsInfo[15000000:]

movieFactMatrix_1 = subset1.pivot(index = 'user_id', columns = 'title',values='rating').fillna(0)
del subset1
movieFactMatrix_2 = subset2.pivot(index = 'user_id', columns = 'title',values='rating').fillna(0)
del subset2
# movieFactMatrix_3 = subset3.pivot(index = 'user_id', columns = 'title',values='rating').fillna(0)
# del subset3
# movieFactMatrix_4 = subset4.pivot(index = 'user_id', columns = 'title',values='rating').fillna(0)
# del subset4

movie_matrix = [movieFactMatrix_1, movieFactMatrix_2]

del movieFactMatrix_1
del movieFactMatrix_2
movie_matrix_final = reduce(lambda left,right: p.merge(left,right,on='user_id'), movie_matrix)
X = movie_matrix_final.values.T
X.shape
SVD = TruncatedSVD(n_components=12, random_state=17)
matrix = SVD.fit_transform(X)
matrix.shape

movie_title = movie_matrix_final.columns
movie_title_list = list(movie_title)
warnings.filterwarnings("ignore",category =RuntimeWarning)
corr = np.corrcoef(matrix)
corr.shape

# Create dictionary to store Movie Name, Genres, Rating and Size
movieDict = {}
for line in dataset_movies_list:
    try:
       y = movieProperties.loc[line[0]].rating.get('mean')
    except KeyError:
       y = 0
    try:
        x = movieNumRatings_normalized.loc[line[0]].get('size')
    except KeyError:
        x = 0
    movieDict[line[0]] = (line[1], np.array(line[2]), x , y)
        
# Recommend the closet movies through KNN algorithm
num_neighbors = 5
metric = "cosine"
test_movie_knn = movieDict[6333]
print("Input Movie:")
print(movieDict[6333])

print("\nKNN Based Movie Recommendation:")
neighbors = get_neighbors(movieDict, test_movie_knn, num_neighbors,metric)
for row in neighbors:
    print(movieDict[row][0])

print("\nMatrix Factorization Based Movie Recommendation:")

test_movie_mf = movie_title_list.index("X2: X-Men United (2003)")
corr_mf  = corr[test_movie_mf]
recommended_movies = list(movie_title[(corr_mf >= 0.9)])
for i in range(1,6):
    print(recommended_movies[i])

end = time.time()
print("Execution time: {end - start}")
