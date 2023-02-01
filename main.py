import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time

number_of_tests = 7

# stats
execution_time_on_cpu = []
execution_time_on_cuda = []
accuracy = []

# data variables
number_of_classes = 40
drop_data = 100000  # number of data to drop from each class

# model parameters
number_of_epochs = 8
learning_rate = 0.01
number_of_layers_list = [1, 3, 6, 6, 6, 6, 9]  # number_of_hidden_layers
number_of_neurons_list = [100, 200, 250, 300, 400, 500,250]  # number_of_neurons_in_each_hidden_layer

if __name__ == '__main__':
    compare_devices_test = "on"
    counter = 0
for device_test in range(number_of_tests):
    print("Test number: " + str(device_test))
    start_time = time.time()

    # model configuration #TODO
    number_of_layers = number_of_layers_list[device_test]
    number_of_neurons = number_of_layers_list[device_test]

    # which device to use
    if device_test < number_of_tests / 2:
        var_device = "cpu"
    else:
        var_device = "cuda"

    var_device = "cpu"
    # prepare doodle dataset
    all_data = np.zeros((0, 784), dtype=np.uint8)
    labels = np.zeros(1, dtype=np.uint8)
    # for i in range (0, len(os.listdir("./numpy_bitmap"))): # for all files
    for k in range(number_of_classes):
        c = np.load("./numpy_bitmap/" + os.listdir("./numpy_bitmap")[k], encoding='latin1', allow_pickle=True)
        c = np.delete(c, np.s_[0:drop_data], 0)  # delete the first drop_data number of columns
        c = c / np.max(c)  # normalize data to 0-1
        all_data = np.concatenate((all_data, c))

    print("Total number of elements: " + str(all_data.shape[0]))
    labels = np.resize(labels, all_data.shape[0])
    # prepare labels
    j = 0
    for z in range(number_of_classes):
        b = np.load("./numpy_bitmap/" + os.listdir("./numpy_bitmap")[z], encoding='latin1', allow_pickle=True)
        for j in range(j, j + b.shape[0] - drop_data):
            labels[j] = z

    # split data using scikit-learn for train and test
    partitions = [0.8, 0.2]
    train_obraz, test_obraz, train_label, test_label = train_test_split(all_data, labels, train_size=partitions[0],
                                                                        random_state=45)
    '''
    # plot some images from the dataset
    fig, axs = plt.subplots(3, 4, figsize=(10, 6))
    # loop over the plots
    for ax in axs.flatten():
        # pick a random image
        randimg2show = np.random.randint(0, len(train_obraz))
        train_obraz = np.reshape(train_obraz, (train_obraz.shape[0], train_obraz.shape[1]))

        # create the image (must be reshaped from (x,784))
        img = np.reshape(train_obraz[randimg2show], (28, 28))
        ax.imshow(img, cmap='gray')

        # title
        ax.set_title('The number %i' % train_label[randimg2show])

    plt.suptitle('How humans see the data', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, .95])
    plt.show()
    '''
    '''
    # plot
    fix, axs = plt.subplots(3,4,figsize=(10,6))
    plt.suptitle('Ttes', fontsize=20)
    for ax in axs.flatten():
        randimg2show = np.random.randint(0, len(train_obraz))
        ax.plot(train_obraz[randimg2show,:],'ko')
        ax.set_title(train_label[randimg2show])
        ax.grid()
    plt.show()
    '''

    '''
    # plot piksel intensity histogram
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(all_data.flatten(), 50)
    ax[0].set_xlabel('Pixel intensity values')
    ax[0].set_ylabel('Count')
    ax[0].set_title('Histogram of original data')
    ax[0].set_yscale('log')

    ax[1].hist(all_data.flatten(), 50)
    ax[1].set_xlabel('Pixel intensity values')
    ax[1].set_ylabel('Count')
    ax[1].set_yscale('log')
    ax[1].set_title('Histogram of normalized data')

    plt.show()
    '''

    if var_device == "cuda":
        # add data to gpu
        train_obraz = torch.tensor(train_obraz).float().cuda()
        train_label = torch.tensor(train_label).long().cuda()

        test_obraz = torch.tensor(test_obraz).float().cuda()
        test_label = torch.tensor(test_label).long().cuda()
    else:
        # add data to cpu
        train_obraz = torch.tensor(train_obraz).float()
        train_label = torch.tensor(train_label).long()

        test_obraz = torch.tensor(test_obraz).float()
        test_label = torch.tensor(test_label).long()

    # pytorch datasets
    train_data = TensorDataset(train_obraz, train_label)
    test_data = TensorDataset(test_obraz, test_label)

    # dataloader objects
    batchsize = 32
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_label))


    # create a class for the model
    def create_the_MNISTNet(number_of_layers, number_of_neurons):
        class MNISTNet(nn.Module):
            def __init__(self, number_of_layers, number_of_neurons):
                super().__init__()
                self.hidden_layers = nn.ModuleDict()
                self.number_of_layers = number_of_layers

                # input layer
                self.input = nn.Linear(784, number_of_neurons, device=var_device)

                # hidden layer
                for i in range(number_of_layers):
                    self.hidden_layers[f'hidden{i}'] = nn.Linear(number_of_neurons, number_of_neurons,
                                                                 device=var_device)
                self.fc1 = nn.Linear(number_of_neurons, number_of_neurons, device=var_device)
                # output layer
                self.output = nn.Linear(number_of_neurons, number_of_classes, device=var_device)

            # forward pass
            def forward(self, x):
                # input layer
                x = F.relu(self.input(x))

                # hidden layer
                for z in range(number_of_layers):
                    x = F.relu(self.hidden_layers[f'hidden{z}'](x))
                x = F.relu(self.fc1(x))

                # output layer
                x = torch.log_softmax(self.output(x), dim=1)

                return x

        if var_device == "cuda":
            net = MNISTNet(number_of_layers, number_of_neurons).cuda()
            lossfun = nn.NLLLoss().cuda()

        if var_device == "cpu":
            net = MNISTNet(number_of_layers, number_of_neurons)
            lossfun = nn.NLLLoss()

        # TODO SGD to adm test it
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

        return net, lossfun, optimizer


    # test model with one batch
    net, lossfun, optimizer = create_the_MNISTNet(number_of_layers, number_of_neurons)
    dataiter = iter(train_loader)
    X, y = dataiter.__next__()
    yHat = net(X)

    # loss
    loss = lossfun(yHat, y)


    # print("Loss: ", loss.item())

    #  train the model
    def function2trainTheModel():
        # number of epochs
        epochs = number_of_epochs

        # create the model
        net, lossfun, optimizer = create_the_MNISTNet(number_of_layers, number_of_neurons)

        # initialize the loss
        losses = torch.zeros(epochs)
        trainAcc = []
        testAcc = []

        # loop over epochs
        for epoch in range(epochs):
            # loop over training data batches
            batchAcc = []
            batchLoss = []
            for X, y in train_loader:
                # forward pass and loss
                yHat = net(X)
                loss = lossfun(yHat, y)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # loss from this batch
                batchLoss.append(loss.item())

                # compute accuracy
                matches = torch.argmax(yHat) == y  # booleans (false/true)
                matchesNumeric = matches.float()  # convert to numbers (0/1)
                accuracyPct = 100 * torch.mean(matchesNumeric)  # average and x100
                batchAcc.append(accuracyPct)  # add to list of accuracies
            # end of batch loop...

            # now that we've trained through the batches, get their average training accuracy
            test = torch.Tensor(batchAcc).to('cuda')
            trainAcc.append(torch.tensor(test).mean())

            # and get average losses across the batches
            losses[epoch] = np.mean(batchLoss)

            # test accuracy
            X, y = next(iter(test_loader))  # extract X,y from test dataloader
            yHat = net(X)

            # compare the following really long line of code to the training accuracy lines
            testAcc.append(100 * torch.mean((torch.argmax(yHat, axis=1) == y).float()))
            # end epochs

        return trainAcc, testAcc, losses, net


    trainAcc, testAcc, losses, net = function2trainTheModel()

    X, y = next(iter(test_loader))
    predictions = net(X).detach()

    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    if var_device == "cpu":
        execution_time_on_cpu = np.append(execution_time_on_cpu, (end_time - start_time))
    if var_device == "cuda":
        execution_time_on_cuda = np.append(execution_time_on_cuda, (end_time - start_time))

    predictions = torch.Tensor(predictions).to('cpu')
    trainAcc = torch.Tensor(trainAcc).to('cpu')
    testAcc = torch.Tensor(testAcc).to('cpu')

    '''
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].plot(losses)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_ylim([0, 3])
    ax[0].set_title('Model loss')
    ax[1].plot(trainAcc, label='Train')
    ax[1].plot(testAcc, label='Test')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].set_ylim([10, 100])
    ax[1].set_title(f'Final model test accuracy: {testAcc[-1]:.2f}%')
    ax[1].legend()
    plt.show()
    '''

    '''
    # show the predictions
    numberofSamples = 1
    plt.bar(range(number_of_classes), (torch.exp(predictions[numberofSamples])))
    plt.xticks(range(number_of_classes))
    plt.xlabel('Number')
    plt.ylabel('Probability')
    plt.title(f'True number was: {y[numberofSamples].item():.0f}')
    plt.show()

    test_obraz = torch.Tensor(test_obraz).to('cpu')
    '''

    '''
    fig, axs = plt.subplots(3, 4, figsize=(10, 6))
    for ax in axs.flatten():
        randimg2show = np.random.randint(0, len(test_obraz))
        train_obraz = np.reshape(test_obraz, (test_obraz.shape[0], test_obraz.shape[1]))

        # create the image (must be reshaped!)
        img = np.reshape(test_obraz[randimg2show], (28, 28))
        ax.imshow(img, cmap='gray')

        # title
        ax.set_title('The number %i' % test_label[randimg2show])

    plt.suptitle('Test', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, .95])
    plt.show()
    '''

    errors = 0
    for i in range(len(predictions)):
        if test_label[i] != torch.argmax(predictions[i]):
            errors += 1

    print("Errors: ", errors / len(predictions) * 100)
    accuracy = np.append(accuracy, errors / len(predictions) * 100)

# plot time of execution on cpu and cuda
plt.autoscale(enable=True, axis='both', tight=True)
x1 = list(range(1, number_of_tests + 1))
plt.ylabel("Time (s)")
plt.xlabel("Test number")
# plt.plot(x1, execution_time_on_cuda, label='CUDA')
plt.plot(x1, execution_time_on_cpu, label='CPU')
plt.legend()
plt.show()

# plot accuracy
plt.autoscale(enable=True, axis='both', tight=True)
plt.ylabel("Accuracy")
plt.xlabel("Test number")
plt.plot(x1, accuracy, label='errors %')
plt.legend()
plt.show()
