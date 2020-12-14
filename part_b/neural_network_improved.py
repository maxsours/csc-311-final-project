import sys
from random import randint

sys.path.append('../')

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

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_nan, data_zero):
        self.data_nan = data_nan
        self.data_zero = data_zero

  def __len__(self):
        return len(self.data_nan)

  def __getitem__(self, index):
        student_data = self.data_nan[index]
        student_data_zero = self.data_zero[index]

        return student_data, student_data_zero

class AutoEncoder(nn.Module):
    def __init__(self, num_question):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_question, 64),
            nn.Sigmoid(),
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.Sigmoid(),
            nn.Linear(64, num_question),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        out = self.encoder(inputs)
        out = self.decoder(out)

        return out

# Function for adding Metadata to train data matrix
def addMetaData(zero_train_matrix, train_matrix, path="../data"):
    gender = np.zeros(train_matrix.shape[0])
    birthYear = np.zeros(train_matrix.shape[0])

    meta = load_student_meta_csv(path + "/student_meta.csv")
    for i in range(len(meta["user_id"])):
        gender[meta["user_id"][i]] = meta["gender"][i]

    gender = gender/2

    new_col = torch.from_numpy(gender).unsqueeze(1)

    new_train_matrix = torch.cat((train_matrix, new_col), 1)
    new_zero_train_matrix = torch.cat((zero_train_matrix, new_col), 1)

    new_train_matrix = new_train_matrix.type(torch.FloatTensor)
    new_zero_train_matrix = new_zero_train_matrix.type(torch.FloatTensor)

    return new_zero_train_matrix, new_train_matrix

def load_student_meta_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "gender": [],
        "data_of_birth": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["user_id"].append(int(row[0]))
                data["gender"].append(int(row[1]))
                if row[2]:
                  data["data_of_birth"].append(int(row[2][:4]))
                else:
                  data["data_of_birth"].append(-1)
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    d = data["gender"]
    for i in range(len(d)):
        if d[i] == 0:
            r = randint(1, 2)
            d[i] = r
    data["gender"] = d
    print(data["gender"])
    return data

def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, batch_size=50):
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
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    # print( train_data.shape)

    train_data_load = Dataset(train_data, zero_train_data)

    cost = []
    accValid = []
    accTrain = []
    lossValid = []

    for epoch in range(0, num_epoch):
        train_loss = 0.
        # Shuffle and split training data into batches
        train_loader = torch.utils.data.DataLoader(train_data_load, batch_size=50, num_workers=0, shuffle=True)

        for batch in train_loader:
            outputs, inputs = batch 

            inputs = Variable(inputs).unsqueeze(0)
            target = inputs.clone()

            # FORWARD PASS
            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(outputs.unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]
            
            # BACKWARD PASS
            loss = torch.sum((output - target) ** 2.)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc, valid_loss = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {} \t "
              "Valid Cost: {}".format(epoch, train_loss, valid_acc, valid_loss))

        cost.append(train_loss)
        accValid.append(valid_acc)
        lossValid.append(valid_loss)

    return cost, accValid, lossValid

def main(num_epoch, batch_size, lr):

    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    zero_train_matrix, train_matrix = addMetaData(zero_train_matrix, train_matrix)
    questions_num = train_matrix.shape[1]

    lamb_index = 0
    # Set optimization hyperparameters.
    lamb = [0, 0.001, 0.01, 0.1, 1]
    # for i in range(len(k)):
    model = AutoEncoder(questions_num)
    res = train(model, lr, lamb[lamb_index], train_matrix, zero_train_matrix,
          valid_data, num_epoch, batch_size)

    test_acc = evaluate(model, zero_train_matrix, test_data)
    print("Test Accuracy:", test_acc)

    epochs = np.arange(num_epoch)

    # fig, ax = plt.subplots(1, 3)
    # ax[0].plot(epochs, res[0])
    # ax[0].set_xlabel("Epoch")
    # ax[0].set_ylabel("loss")
    # ax[0].set_title("Training Loss per Epoch")
    plt.title("Training Loss per Epoch")
    plt.plot(epochs, res[0], label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Validation Accuracy per Epoch")
    plt.plot(epochs, res[1], label="validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Validation Loss per Epoch")
    plt.plot(epochs, res[2], label="validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend(loc='best')
    plt.show()
    # ax[1].plot(epochs, res[1])
    # ax[1].set_xlabel("Epoch")
    # ax[1].set_ylabel("loss")
    # ax[1].set_title("Validation Accuracy per Epoch")
    #
    # ax[2].set_xlabel("Epoch")
    # ax[2].set_ylabel("Loss")
    # ax[2].set_title("Validation Loss per Epoch")
    # ax[2].plot(epochs, res[2])
    # plt.show()

if __name__ == "__main__":
    epoch = 110
    batch_size = 128
    lr = 0.01
    main(epoch, batch_size, lr)