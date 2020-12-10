from utils import *

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id = data["user_id"]
    question_id = data["question_id"]
    is_correct = data["is_correct"]
    neg_log_p_cij = lambda i: np.logaddexp(0, beta[question_id[i]] - theta[user_id[i]]) if is_correct[i] else np.logaddexp(0, theta[user_id[i]] - beta[question_id[i]])
    log_lklihood = np.sum([neg_log_p_cij(i) for i in range(len(is_correct))])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id = data["user_id"]
    question_id = data["question_id"]
    is_correct = data["is_correct"]
    grad_theta = 0 * theta
    
    for i in range(len(is_correct)):
        grad_theta[user_id[i]] += is_correct[i] - sigmoid(theta[user_id[i]] - beta[question_id[i]])
    grad_theta /= len(theta)
    # Trying to maximize, not minimize
    theta += lr * grad_theta
    grad_beta = 0 * beta
    for i in range(len(is_correct)):
        grad_beta[question_id[i]] -= is_correct[i] - sigmoid(theta[user_id[i]] - beta[question_id[i]])
    beta += lr * grad_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.rand(542)
    beta = np.random.rand(1774)

    val_acc_lst = []
    train_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        tscore = evaluate(data=data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        train_acc_lst.append(tscore)
        #print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # Added list of training accuracies
    return theta, beta, val_acc_lst, train_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    theta, beta, val_acc_list, train_acc_list = irt(train_data, val_data, 0.15, 300)
    plt.plot(val_acc_list, label = "Validation Accuracy")
    plt.plot(train_acc_list, label = "Training Accuracy")
    plt.title("Training Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    valid_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print("Validation Accuracy:", valid_acc)
    print("Test Accuracy:", test_acc)
    
    # part (d)
    for i in range(5):
        beta_curr = beta[i]
        theta_curr = np.sort(theta)
        prob_correct = np.exp(-np.logaddexp(0, beta_curr - theta_curr))
        plt.plot(theta_curr, prob_correct, label="j = " + str(i))
    plt.title("Probability of Correct Respose vs. Theta")
    plt.ylabel("Probability")
    plt.xlabel("Theta")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
