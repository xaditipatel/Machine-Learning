# KNN algorithm evualtion with a k-fold cross validation 

The implementation also do the following:

-> Vectorized computation of the distances <br />
-> Implemention of two distance measures : Euclidiean Distance and Manhattan Distance so that we can compare the algorithm performance using different distance measures

For this code base,the accuracy of your KNN algorithm using 10-Fold cross validation on the following datasets from the UCI-Machine Learning Repository and further compared the accuracy with that obtained with KNN from Weka.

Hayes-Roth Dataset (https://archive.ics.uci.edu/ml/datasets/Hayes-Roth) <br />
Car Evaluation Dataset (https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)

To test the dataset, change as below:

+ Need to change the data set name in line 124 (dataSetName = 'car.data')
+ To test different distance methods, need to change on line 51(	dist = euclidean_distance(test_row, train_row) )
+ All the data set should be in the same folder as py file


# Evaulation Results: 


# 1.	Hayes-Roth Dataset <br /> <br />

•	Euclidean Distance <br />
Scores: [23.076923076923077, 38.46153846153847, 38.46153846153847, 46.15384615384615, 38.46153846153847, 38.46153846153847, 46.15384615384615, 30.76923076923077, 38.46153846153847, 30.76923076923077] <br />
Mean Accuracy: 36.923%  

•	Manhattan Distance <br />
Scores: [38.46153846153847, 53.84615384615385, 61.53846153846154, 23.076923076923077, 30.76923076923077, 53.84615384615385, 53.84615384615385, 30.76923076923077, 76.92307692307693, 53.84615384615385] <br />
Mean Accuracy: 47.692%

•	WEKA <br />
=== Run information === <br />

Scheme:       weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" <br />
Relation:     hayes-roth-weka.filters.unsupervised.attribute.StringToNominal-Rlast <br />
Instances:    132 <br />
Attributes:   6 <br />
              name <br />
              hobby <br />
              age <br />
              education <br />
              marital_status <br />
              classes <br />
Test mode:    10-fold cross-validation <br />

=== Classifier model (full training set) === <br />

IB1 instance-based classifier <br />
using 1 nearest neighbour(s) for classification <br />


Time taken to build model: 0 seconds <br />

=== Cross-validation === <br />
=== Summary === <br />

Correlation coefficient                  0.5852 <br />
Mean absolute error                      0.3712 <br />
Root mean squared error                  0.6798 <br />
Relative absolute error                 56.829  % <br />
Root relative squared error             88.2419 % <br />
Total Number of Instances              132     <br />


# 2.	Car Evalutaion Dataset <br />

•	Euclidean Distance <br />
Scores: [94.76744186046511, 95.93023255813954, 93.02325581395348, 93.02325581395348, 96.51162790697676, 97.67441860465115, 95.34883720930233, 94.76744186046511, 94.18604651162791, 95.34883720930233] <br />
Mean Accuracy: 95.058%<br />

•	Manhattan Distance <br />
Scores: [98.83720930232558, 97.09302325581395, 99.4186046511628, 94.18604651162791, 98.25581395348837, 95.34883720930233, 97.09302325581395, 95.93023255813954, 98.83720930232558, 96.51162790697676] <br />
Mean Accuracy: 97.151% 

•	WEKA <br />
=== Summary === <br />

Correctly Classified Instances        1210               70.0231 % <br />
Incorrectly Classified Instances       518               29.9769 % <br />
Kappa statistic                          0      <br />
Mean absolute error                      0.229  <br />
Root mean squared error                  0.3381 <br />
Relative absolute error                100      % <br />
Root relative squared error            100      % <br />
Total Number of Instances             1728     <br />
