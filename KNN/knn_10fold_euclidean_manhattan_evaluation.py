# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:00:10 2020

@author: DELL

References:
https://ljvmiranda921.github.io/notebook/2017/02/09/k-nearest-neighbors/
https://machinelearningmastery.com/k-fold-cross-validation/
https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
https://machinelearningmastery.com/implement-resampling-methods-scratch-python/
https://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/
"""

from math import sqrt
import numpy as np
from random import seed
from random import randrange
from csv import reader
from sklearn import preprocessing
import pandas as p

whichDataSet = 0
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

def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = manhattan_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
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
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)

def whichDataSetFunc(dataSetName):
    global whichDataSet
    if dataSetName == 'car.data':
        whichDataSet = 1
    elif dataSetName == 'breast-cancer.data':
        whichDataSet = 2
    elif dataSetName == 'hayes-roth.data':
        whichDataSet = 3
        
# Preprocess dataset
dataSetName = 'car.data'
whichDataSetFunc(dataSetName)
if whichDataSet == 1 :
    cols =['buying','maint','doors','persons','lug_boot','safety','classes']
    dataset = p.read_csv(dataSetName,sep= ',',names=cols)
    # Encode Data
    dataset.buying.replace(('vhigh','high','med','low'),(1,2,3,4), inplace=True)
    dataset.maint.replace(('vhigh','high','med','low'),(1,2,3,4), inplace=True)
    dataset.doors.replace(('2','3','4','5more'),(1,2,3,4), inplace=True)
    dataset.persons.replace(('2','4','more'),(1,2,3), inplace=True)
    dataset.lug_boot.replace(('small','med','big'),(1,2,3), inplace=True)
    dataset.safety.replace(('low','med','high'),(1,2,3), inplace=True)
    dataset.classes.replace(('unacc','acc','good','vgood'),(1,2,3,4), inplace=True)
    dataset_list = dataset.values.tolist()   
    normalize_dataset(dataset_list, dataset_minmax(dataset_list))
elif (dataSetName == 'breast-cancer.data'):
    cols =['classes','age','menopause','tumor_size','inv_nodes','node_caps','deg_malig','breast','breast_quad','irradiat']
    dataset = p.read_csv(dataSetName,sep= ',',names=cols)
    
    # Encode Data
    dataset.classes.replace(('no-recurrence-events','recurrence-event'),(1,2), inplace=True)
    dataset.age.replace(('10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99'),(1,2,3,4,5,6,7,8,9), inplace=True)
    dataset.menopause.replace(('lt40','ge40','premeno'),(1,2,3), inplace=True)
    dataset.tumor_size.replace(('0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
    dataset.inv_nodes.replace(('0-2','4-5','6-8','9-11','12-14','15-17','18-20','21-23','24-26','27-29','30-32','33-35','36-39'),(1,2,3,4,5,6,7,8,9,10,11,12,13), inplace=True)
    dataset.node_caps.replace(('yes','no'),(1,2), inplace=True)
    dataset.deg_malig.replace(('1','2','3'),(1,2,3), inplace=True)
    dataset.breast.replace(('left','right'),(1,2), inplace=True)
    dataset.breast_quad.replace(('left-up','left-low','right-up','right-low','central'),(1,2,3,4,5), inplace=True)
    dataset.irradiat.replace(('yes','no'),(1,2), inplace=True)

    dataset_list = dataset.values.tolist()   
    normalize_dataset(dataset_list, dataset_minmax(dataset_list))
elif (dataSetName == 'hayes-roth.data'):
    cols =['name','hobby','age','educational_level','marital_status','class']
    dataset = p.read_csv(dataSetName,sep= ',',names=cols)
    dataset_list = dataset.values.tolist()   
    normalize_dataset(dataset_list, dataset_minmax(dataset_list))
# evaluate algorithm
n_folds = 10
num_neighbors = 5
scores = evaluate_algorithm(dataset_list, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
