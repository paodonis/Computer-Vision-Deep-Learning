import torch
import torchvision
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import os
import cv2
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
from skimage import data
from skimage.transform import rotate, SimilarityTransform, warp
import random


class lfwDataset(Dataset):

    def __init__(self, csv_file, root_dir, transformation):

        self.root_dir = root_dir
        self.transform = transformation
        self.landmarks_frame = self.read_each_name(csv_file)

    def read_each_name(self, file_name):
        with open(file_name) as f:
            info = open(file_name).read().split()
            all_names = [[None for _ in range(3)] for _ in range(len(info)/3)]
            for x in range(0,len(info)):
                all_names[x/3][x%3] = info[x]
            return all_names
   
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img1_name = os.path.join(self.root_dir, self.landmarks_frame[idx][0]) # [idx][0]
        img2_name = os.path.join(self.root_dir, self.landmarks_frame[idx][1])
        
        image1 = Image.open(img1_name)
        image1 = image1.convert('RGB')
        image2 = Image.open(img2_name)
        image2 = image2.convert('RGB')
        
        label = self.landmarks_frame[idx][2]
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, label

class cnn(nn.Module):

    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride = (1,1), padding = 2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride = (1,1), padding = 2)
        self.conv3 = nn.Conv2d(128, 256, 5, stride = (1,1), padding = 2)
        self.conv4 = nn.Conv2d(256, 512, 5, stride = (1,1), padding = 2)

        self.linear1 = nn.Linear(131072, 1024)
        self.linear2 = nn.Linear(2048, 1)

        self.maxPool = nn.MaxPool2d(2, stride = (2,2))
        self.batchSize = 10
        
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)
        self.batch_norm5 = nn.BatchNorm2d(1024)

    def forward(self, image1, image2):
        image1 = forward_each(self, image1)
        image2 = forward_each(self, image2)
        #both_results = torch.cat((image1, image2), 1)
        #results = F.sigmoid(self.linear2(both_results))
        results = F.pairwise_distance(image1, image2)
        
        return results

def forward_each(cnn, x):
    x = cnn.conv1(x)
    x = F.relu(x)
    x = cnn.batch_norm1(x)
    x = cnn.maxPool(x)
    x = cnn.conv2(x)
    x = F.relu(x)
    x = cnn.batch_norm2(x)
    x = cnn.maxPool(x)
    x = cnn.conv3(x)
    x = F.relu(x)
    x = cnn.batch_norm3(x)
    x = cnn.maxPool(x)
    x = cnn.conv4(x)
    x = F.relu(x)
    x = cnn.batch_norm4(x)
    x = x.view((x.data.size())[0], -1)
    x = cnn.linear1(x)
    x = F.relu(x)
    x = cnn.batch_norm5(x)

    return x

'''
def accuracy(label, output): # find the average and check which one is smaller than and make it a 1
    # find average
    total_sum = 0
    average = 0
    result = 0
    counter1 = 0
    counter2 = 0
    number_of_ones = 0.0
    for i in range(0,label.size()[0]):
        if (label.data.cpu().numpy()[i] == 1.0):
            number_of_ones += 1.0
    factor = 6.0 + number_of_ones
    divisor = 12.0 + factor/3.0
    classification = np.zeros((label.size()[0],1))
    for i in range(0,label.size()[0]):
        counter1 += 1
        total_sum += output.data.cpu().numpy()[i]
    average = total_sum/divisor
    # mas 1 quiero el average mas bajo
    for i in range(0,label.size()[0]):
        if (output.data.cpu().numpy()[i] >= average):
            classification[i] = 1.0
    #print("classification", classification)
    for i in range(0,label.size()[0]):
        counter2 += 1
        if(label.data.cpu().numpy()[i] == classification[i]):
            result += 1.0
    result = result/counter2

    return result
'''

def accuracy(label, output):
    counter = 0
    result = 0
    classification = np.zeros((label.size()[0],1))
   
    for i in range(0, label.size()[0]):
        if (output.data.cpu().numpy()[i] <= 24.0):
            classification[i] = 1.0
            
    for i in range(0, label.size()[0]):
        counter += 1
        if(label.data.cpu().numpy()[i] == classification[i]):
            result += 1.0
    result = result/counter

    return result


learning_rate = 1e-4
cnn_model = cnn().cuda()

transform = transforms.Compose([transforms.Scale((128,128)), transforms.ToTensor()])

optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)


def training(cnn_model):
    margin = 1.0
    train_loss = 0
    dataset = lfwDataset(csv_file='train.txt',
                                    root_dir='lfw/', transformation = transform)

    dataloader = DataLoader(dataset, batch_size=12,
                        shuffle=True, num_workers=12)

    train_accuracy = 0
    iterations = 0
    
    for each in dataloader:
        image1 = Variable(each[0]).cuda()
        image2 = Variable(each[1]).cuda()
        label1 = np.array([float(i) for i in each[2]])
        label1 = torch.from_numpy(label1).view(label1.shape[0], -1)
        label1 = label1.type(torch.FloatTensor)
        label = Variable(label1).cuda()
        output = cnn_model(image1, image2)
        loss = torch.mean((1-label) * torch.pow(output,2) + (label) * torch.pow(torch.clamp(margin - output,min = 0.0), 2))
        train_accuracy += accuracy(label, output)

        train_loss += loss.data[0]
        #print("train loss", train_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iterations += 1.0
    train_loss = train_loss/iterations
    train_accuracy = train_accuracy/iterations
    #print("output", output, "label", label)
    #torch.save(cnn_model.state_dict(), 'my_weights')

    return train_loss, train_accuracy


def testing(cnn_model):
    #cnn_model.load_state_dict(torch.load('my_weights'))

    dataset = lfwDataset(csv_file='test.txt',
                                    root_dir='lfw/', transformation = transform)
    dataloader = DataLoader(dataset, batch_size=12,
                        shuffle=True, num_workers=12)
    test_loss = 0
    test_accuracy = 0
    iterations = 0
    margin = 1.0

    for each in dataloader:
        image1 = Variable(each[0]).cuda()
        image2 = Variable(each[1]).cuda()
        label1 = np.array([float(i) for i in each[2]])
        label1 = torch.from_numpy(label1).view(label1.shape[0], -1)
        label1 = label1.type(torch.FloatTensor)
        label = Variable(label1).cuda()
        output = cnn_model(image1, image2)
        loss = torch.mean((1-label) * torch.pow(output,2) + (label) * torch.pow(torch.clamp(margin - output,min = 0.0), 2))
        test_accuracy += accuracy(label, output)
        each_test_accuracy = accuracy(label, output)
        test_loss += loss.data[0]
        iterations += 1.0
    test_loss = test_loss/iterations
    test_accuracy = test_accuracy/iterations
        
    return test_loss, test_accuracy
    
epochs = 30
all_training_loss = list()
all_testing_loss = list()
all_training_accuracy = list()
all_testing_accuracy = list()

for epoch in range(epochs):
    print("epoch: ", epoch)
    train_loss, train_accuracy = training(cnn_model)
    train_loss = train_loss
    train_accuracy = train_accuracy
    print("train loss", train_loss)
    print("train accuracy", train_accuracy)
    all_training_loss.append(train_loss)
    all_training_accuracy.append(train_accuracy)
    test_loss, test_accuracy = testing(cnn_model)
    test_loss = test_loss
    test_accuracy = test_accuracy
    print("test loss", test_loss)
    print("test accuracy", test_accuracy)
    all_testing_loss.append(test_loss)
    all_testing_accuracy.append(test_accuracy)
    test_loss = 0

print("training loss ", all_training_loss)
print("testing loss", all_testing_loss)
print("training accuracy", all_training_accuracy)
print("testing accuracy", all_testing_accuracy)

torch.save(cnn_model.state_dict(), 'my_weights')

plt.plot(all_training_loss)
plt.plot(all_testing_loss)
plt.plot(all_testing_accuracy)
plt.plot(all_training_accuracy)

plot.show()
