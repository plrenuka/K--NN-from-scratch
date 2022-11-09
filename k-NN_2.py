import numpy as np
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler

def minkowski_distance(feature_data, query_instance , a):
    distance_sq_list = abs(np.subtract(feature_data, query_instance)) ** a
    minkowski_dist = (np.sum(distance_sq_list, axis = 1)) ** 1/float(a)
    return minkowski_dist

def predict_class(feature_data, class_lables, query_instance, a, k):
    predicted_values = minkowski_distance(feature_data, query_instance, a)
    sorted_indexes = np.argsort(predicted_values)
    sorted_indexes = sorted_indexes[0: k]
    predicted_values = predicted_values[sorted_indexes]
    classes = class_lables[sorted_indexes]
    n = 2
    weights = (1/(predicted_values))**n
    unique_classes = np.unique(classes)
    vote = {}
    for each in unique_classes:
        values_at = np.where(classes == each)       # indexes of the iterated class
        vote[each] = np.sum(weights[values_at])
    max_key = max(vote, key=vote. get)
    return max_key

def calculate_accuracy(result, test_class_lables):
    compare_results = (np.array(result) == np.array(test_class_lables))
    unique, counts = np.unique(compare_results, return_counts=True)
    indx = np.where(unique == True)
    indx = indx[0][0]
    # print(indx[0][0])
    accuracy = (counts[indx] / len(test_class_lables)) * 100
    # print((counts[indx] / len(test_class_lables)) * 100)
    return accuracy

def main():
    a = 1
    k = 7
    result = []
    train_data = np.genfromtxt("trainingData.csv", delimiter=',')
    test_data = np.genfromtxt("testData.csv", delimiter=',')

    training_feature_inst = train_data[:, :-1]      # first 20 feature columns
    class_labels = train_data[:, -1]                # class labels i.e. last column
    test_inst = test_data[:, :-1]                   # feature columns of test data
    test_labels = test_data[:, -1]                  # class labels of test data
# uncomment the below methods to impleament the respective scaling
    '''MinMaxScaler'''
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # training_feature_inst = scaler.fit_transform(training_feature_inst)
    # test_inst = scaler.transform(test_inst)
    '''Normalizer'''
    # scaler = Normalizer()
    # training_feature_inst = scaler.fit_transform(training_feature_inst)
    # test_inst = scaler.transform(test_inst)
    '''StandardScaler'''
    scaler = StandardScaler()
    training_feature_inst = scaler.fit_transform(training_feature_inst)
    test_inst = scaler.transform(test_inst)                                     #transform is used instead of fit_transform in order to keep the values of training data

    for query_instance in test_inst:
        predicted_class = predict_class(training_feature_inst, class_labels, query_instance, a, k)
        result.append(predicted_class)
    print("result = ", result)
    print("test classes = ", test_labels)
    accuracy = calculate_accuracy(result, test_labels)
    print("Prediction accuracy = ", accuracy)


if __name__ == '__main__':
    main()

