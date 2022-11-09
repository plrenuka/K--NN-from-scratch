# Development of Distance Weighted k Nearest Neighbour Algorithm for Classification
# code by- Renuka Patil

import numpy as np
import math

def minkowski_distance(feature_data, query_instance , a):
    distance_sq_list = abs(np.subtract(feature_data, query_instance)) ** a
    minkowski_dist = (np.sum(distance_sq_list, axis = 1)) ** (1/float(a))   # adding all the values in each row and taking its inverse exponent
    return minkowski_dist

def predict_class(feature_data, class_labels, query_instance, a, k):
    distance_values = minkowski_distance(feature_data, query_instance, a)   # distance array
    sorted_indexes = np.argsort(distance_values)                            # storing the index of sorted the distance array
    sorted_indexes = sorted_indexes[0: k]                                   # selecting index of k nearest neighbours
    predicted_values = distance_values[sorted_indexes]                      # taking lowest k distance values
    classes = class_labels[sorted_indexes]                                  # get class labels of k nearest neighbours
    n = 2                                                                   # taking n = 2
    weights = (1/(predicted_values))**n                                     # calculating weights of all nearest neighbours
    unique_classes = np.unique(classes)                                     # selecting unique classes in k elements
    vote = {}
    for each in unique_classes:                                             # iterate through all unique classes
        values_at = np.where(classes == each)                               # storing indexes of the iterated class
        vote[each] = np.sum(weights[values_at])                             # calculating of votes for each uniques class
    predicted_value = max(vote, key=vote. get)                              # selecting predicted class which has highest vote value.
    return predicted_value

def calculate_accuracy(result, test_class_labels):
    compare_results = (np.array(result) == np.array(test_class_labels))
    unique, counts = np.unique(compare_results, return_counts=True)
    indx = np.where(unique == True)
    indx = indx[0][0]
    # print(indx[0][0])
    accuracy = (counts[indx] / len(test_class_labels)) * 100
    return accuracy

def main():
    a = 1
    k = 3
    result = []
    train_data = np.genfromtxt("trainingData.csv", delimiter=',')
    test_data = np.genfromtxt("testData.csv", delimiter=',')
    training_feature_inst = train_data[:, :-1]      # first 20 feature columns
    class_labels = train_data[:, -1]                # class labels i.e. last column
    test_inst = test_data[:, :-1]                   # feature columns of test data
    test_labels = test_data[:, -1]                  # class labels of test data

    for query_instance in test_inst:
        predicted_class = predict_class(training_feature_inst, class_labels, query_instance, a, k)
        result.append(predicted_class)
    print("a = ", a)
    print("k = ", k)
    print("result = ", result)
    print("test classes = ", test_labels)
    accuracy = calculate_accuracy(result, test_labels)
    print("Prediction accuracy = ", accuracy)

if __name__ == '__main__':
    main()

