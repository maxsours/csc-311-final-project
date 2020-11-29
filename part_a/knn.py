from sklearn.impute import KNNImputer
# from sklearn import impute
import matplotlib.pyplot as plt
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("user Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_item_evaluate(valid_data, mat)
    print("item Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    ks = [1, 6,  11, 16, 21, 26]
    best_k = 0
    best_acc = 0

    validation_accs = []
    for k in ks:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)

        validation_accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_k = k

    print("User collaboration's k with best accuracy is: ", best_k)

    #   calculating the accuracy of test data with the best k value
    test_acc = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print("User collaboration's final test accuracy is: ", test_acc)
    #   Plotting KNN
    plt.title("Accuracies of  Validation on ks")
    plt.plot(ks, validation_accs, label="Validation")
    plt.xlabel("k-Nearest Neighbours")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

    best_k = 0
    best_acc = 0

    validation_accs = []
    for k in ks:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)

        validation_accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_k = k

    print("item collaboration's k with best accuracy is: ", best_k)

    #   calculating the accuracy of test data with the best k value
    test_acc = knn_impute_by_item(sparse_matrix, test_data, best_k)
    print("item collaboration's final test accuracy is: ", test_acc)
    #   Plotting KNN
    plt.title("Accuracies of  Validation on ks")
    plt.plot(ks, validation_accs, label="Validation")
    plt.xlabel("k-Nearest Neighbours")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
