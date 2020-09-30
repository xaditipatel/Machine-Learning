# KNN algorithm evualtion with a k-fold cross validation 

The implementation also do the following:

-> Vectorized computation of the distances
-> Implemention of two distance measures : Euclidiean Distance and Manhattan Distance so that we can compare the algorithm performance using different distance measures

For this code base,the accuracy of your KNN algorithm using 10-Fold cross validation on the following datasets from the UCI-Machine Learning Repository and further compared the accuracy with that obtained with KNN from Weka.

Hayes-Roth Dataset (https://archive.ics.uci.edu/ml/datasets/Hayes-Roth)
Car Evaluation Dataset (https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)

To test the dataset, change as below:

+ Need to change the data set name in line 124 (dataSetName = 'car.data')
+ To test different distance methods, need to change on line 51(	dist = euclidean_distance(test_row, train_row) )
+ All the data set should be in the same folder as py file
