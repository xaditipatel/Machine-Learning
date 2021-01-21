For this project, I have developed algorithm using KNN and Matrix factorisation to recommend 5 movies based on given input. 

NOTE: For matrix factorization, pivot tables creation is taking a lot of space and hence I have reduced the data.

Approach to use KNN as a recommender system: 

To use the existing KNN algorithm developed in assignment 1, I have defined a function  to compute the cosine distance between two movies on the based on the similarity of their genres and their popularity. Further, I have used the function get_neighbours() through which we can find the most suitable movies to the provided input.

Matrix Factorization: 

To implement factorization on the dataset, I have merged and sorted ratings and movies data and created a pivot table that stores the factorized values. Further used Truncated singular value decomposition to reduce dimensionality

Maximum dataset that recommender system can use
For KNN: 20m
For Matrix Factorization: 4m

Time complexity: Currently its 1 minute for both the algorithms

To scale up the recommender systems, the data pivot tables should be replaced by dictionary for matrix factorization or it should be done in batches simultaneously performing garbage collection.
