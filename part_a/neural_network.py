import sys
sys.path.append('../')

# from knn import *
from utils import *
import torch
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import matplotlib.pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################                                                           #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = self.g(inputs)
        out = F.relu(out)
        out = self.h(out)
        out = F.sigmoid(out)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # E)
    # TODO: Add a regularizer to the cost function.  
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    print( train_data.shape)

    cost = []
    accValid = []
    accTrain = []
    lossValid = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            # FORWARD PASS
            optimizer.zero_grad()
            output = model(inputs)

            # print(model.layers[0].shape)
            # print(model.layers[1].shape)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]
            
            # BACKWARD PASS
            loss = torch.sum((output - target) ** 2.)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        reg = lamb/2 *model.get_weight_norm()
        train_loss += reg

        valid_acc, valid_loss = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {} \t "
              "Valid Cost: {}".format(epoch, train_loss, valid_acc, valid_loss))
        # train_acc = evaluate(model, zero_train_data, train_data)

        cost.append(train_loss)
        accValid.append(valid_acc)
        lossValid.append(valid_loss)
        # accTrain.append(train_acc)

    return cost, accValid, lossValid
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0
    loss = 0
    

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        prob = output[0][valid_data["question_id"][i]]
        target = valid_data["is_correct"][i]
        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
        loss += torch.sum((target - prob) ** 2.).item()
    return correct / float(total), loss


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.

    questions_num = train_matrix.shape[1]

    # print(zero_train_matrix)
    # print(train_matrix)

    i = 1
    lamb_index = 2

    k = [10, 50, 100, 200, 500]
    
    # Set optimization hyperparameters.
    lr = [0.05, 0.1, 0.05, 0.05, 0.08]
    num_epoch = [10, 15, 10, 7, 6]
    lamb = [0, 0.001, 0.01, 0.1, 1]
    # for i in range(len(k)):
    model = AutoEncoder(questions_num, k[i])
    res = train(model, lr[i], lamb[lamb_index], train_matrix, zero_train_matrix,
          valid_data, num_epoch[i])

    test_acc = evaluate(model, zero_train_matrix, test_data)
    print("Test Accuracy:", test_acc)

    epochs = np.arange(num_epoch[i])

    fig, ax = plt.subplots(1, 3)
    ax[0].plot(epochs, res[0])  
    ax[0].set_xlabel("Epoch")  
    ax[0].set_ylabel("loss")  
    ax[0].set_title("Training Loss per Epoch")

    ax[1].plot(epochs, res[1])  
    ax[1].set_xlabel("Epoch")  
    ax[1].set_ylabel("loss")  
    ax[1].set_title("Validation Accuracy per Epoch")

    ax[2].set_xlabel("Epoch")  
    ax[2].set_ylabel("Loss")  
    ax[2].set_title("Validation Loss per Epoch")
    ax[2].plot(epochs, res[2])
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
