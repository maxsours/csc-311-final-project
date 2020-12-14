from sklearn.impute import KNNImputer
# from sklearn import impute
import matplotlib.pyplot as plt
from utils import *
from random import *
import torch
from torch.autograd import Variable
import knn as kn
import neural_network as nn


def sample_with_replacement(data, num_samples):
    # result = np.zeros((num_samples, data.shape[1]))
    result = []
    weights = np.ones(num_samples)
    idxs = []
    counter = 0
    while(counter < num_samples):
        r = randint(1, data.shape[0]) - 1
        if not r in idxs:
            idxs.append(r)
            # result[i] = data[r]
            result.append(data[r])
            counter += 1
        else:
            weights[counter] += 1

    # for i in range(num_samples):
        # r = randint(1, data.shape[0]) - 1
        # if not r in idxs:
        #     idxs.append(r)
        #     # result[i] = data[r]
        #     result.append(data[r])
        # else:
        #     print("here")
        #     weights[counter] += 1
        # counter += 1
    return np.array(result), np.array(weights)

def sample():
    result = {}

def bag_data(data, num_samples):
    # choosing data randomely with replacement
    # data_size = data.shape

    bag1, weights1 = sample_with_replacement(data, num_samples)
    bag2, weights2 = sample_with_replacement(data, num_samples)
    bag3, weights3 = sample_with_replacement(data, num_samples)

    # bag1 = random.sample(data, )
    # bag1 = choices(data, k=data_size)
    # bag2 = choices(data, k=data_size)
    # bag3 = choices(data, k=data_size)

    return bag1, bag2, bag3, weights1, weights2, weights3



def main():
    train_data = load_train_csv("../data")
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    bag1, bag2, bag3, weights1, weights2, weights3 = bag_data(sparse_matrix, len(sparse_matrix))
    print(val_data["user_id"])
    print(bag1.shape, bag2.shape, bag3.shape)
    k = 6
    print("Training KNN k = ", k)
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    y1 = nbrs.fit_transform(bag1.T)

    acc = sparse_matrix_item_evaluate(val_data, y1 * weights1.T)
    print("val accuracy for y1: ", acc)
    # KNN BEST TUNES:
    k = 11
    print("Training KNN k = ", k)
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    y2 = nbrs.fit_transform(bag2.T)
    acc = sparse_matrix_item_evaluate(val_data, y2 * weights2.T)
    print("val accuracy for y2: ", acc)

    k = 16
    print("Training KNN k = ", k)
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    y3 = nbrs.fit_transform(bag3.T)
    acc = sparse_matrix_item_evaluate(val_data, y3 * weights3.T)
    print(" val accuracy for y3: ", acc)
    # y2 = y2 * weights2
    # print(y2)

    avg = (y1 * weights1.T + y2 * weights2.T + y3 * weights3.T)/3
    acc = sparse_matrix_item_evaluate(val_data, avg)
    print("val accuracy for avg: ", acc)
    acc = sparse_matrix_item_evaluate(test_data, avg)
    print("test accuracy for avg: ", acc)
if __name__ == "__main__":
    main()
