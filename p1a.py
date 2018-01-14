'''
This program trains variants of deep architectures to learn when two images of someone's face are of the same person.
In order to train the algorithm and save it, call the program using "p1a.py --save WEIGHTS_FILE".
To see the results of previous training, call the program using "p1a.py --load my_weights1".
The program uses the same Convolutional Neural Network (CNN) for the two images, does a binary classification with BCE
and uses the Adam optimizer.
'''

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
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
from skimage import data
from skimage.transform import rotate, SimilarityTransform, warp
import random
import sys

epochs = 35 # select the amount of epochs

def randomTransform(lfwDataset_transformations, image): # function that performs a random rotation and random translation with probability of 0.7 each
    prob = random.randrange(0,10,1)
    check = 0
    if (prob < 7):
        angle = random.randrange(-25.0, 25.0, 1.0)
        image = rotate(image, angle, resize = True)
        check +=1
    prob = random.randrange(0,10,1)
    if (prob < 7):
        factor1 = random.randrange(7,11,1)
        sign1 = random.randrange(0,2,1)
        if (sign1 == 0):
            factor1 = factor1 * -1
        factor2 = random.randrange(7,11,1)
        sign2 = random.randrange(0,2,1)
        if (sign2 == 0):
            factor2 = factor2 * -1
        transform = SimilarityTransform(translation=(factor1, factor2))
        image = warp(image, transform)
        check += 1

    if (check > 0):           
        max_value = np.amax(image)

        (vertical,horizontal,colors) = np.shape(image)
        new_image = np.zeros((vertical,horizontal, colors), dtype = "float32")

        image = image * 255/max_value

    return image


class lfwDataset_transformations(Dataset): # class to load the images, perform random transformations and return them as a tensor with their desired size with their label

    def __init__(self, csv_file, root_dir, transformation):

        self.root_dir = root_dir
        self.transform = transformation[0]
        self.landmarks_frame = self.read_each_name(csv_file)
        self.transformation1 = transformation[1]
        self.scale1 = transformation[2]
        self.scale2 = transformation[3]


    def read_each_name(self, file_name): # read the names from the text file
        with open(file_name) as f:
            info = open(file_name).read().split()
            all_names = [[None for _ in range(3)] for _ in range(len(info)/3)]
            for x in range(0,len(info)):
                all_names[x/3][x%3] = info[x]
            return all_names
   
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img1_name = os.path.join(self.root_dir, self.landmarks_frame[idx][0]) # get the path of each image in the pair of images
        img2_name = os.path.join(self.root_dir, self.landmarks_frame[idx][1])
        
        image1 = Image.open(img1_name)

        prob = random.randrange(0,10,1)
        if (prob < 7): # perform random scaling
            prob2 = random.randrange(0,2,1)
            if (prob2 == 0):
                image1 = self.scale1(image1)
            else:
                image1 = self.scale2(image1)
        
        prob = random.randrange(0,10,1)
        if (prob < 7): # perform random mirror flipping
            image1 = self.transformation1(image1)
        
        image1 = np.asarray(image1)
        image1 = randomTransform(self, image1) #perform random rotation and translation
        image1 = np.uint8(image1)
        image1 = PIL.Image.fromarray(image1)
        image1 = image1.convert('RGB')
        
        image2 = Image.open(img2_name)

        prob = random.randrange(0,10,1)
        if (prob < 7): ##
            prob2 = random.randrange(0,2,1)
            if (prob2 == 0):
                image2 = self.scale1(image2)
            else:
                image2 = self.scale2(image2)
        
        prob = random.randrange(0,10,1)
        if (prob < 7):
            image2 = self.transformation1(image2)
            
        image2 = np.asarray(image2)
        image2 = randomTransform(self, image2)
        image2 = np.uint8(image2)
        image2 = PIL.Image.fromarray(image2)
        image2 = image2.convert('RGB')
        
        label = self.landmarks_frame[idx][2]
        if self.transform is not None: # scale the images to be of 128 x 128 and convert them into tensors
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, label


class lfwDataset_for_testing(Dataset): # load the images without applying any random transformations, just scaling them to 128x128 and converting them to tensors (used for testing)

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
        img1_name = os.path.join(self.root_dir, self.landmarks_frame[idx][0])
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
        self.conv3 = nn.Conv2d(128, 256, 3, stride = (1,1), padding = 1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride = (1,1), padding = 1)

        self.linear1 = nn.Linear(131072, 1024)
        self.linear2 = nn.Linear(2048, 1)

        self.maxPool = nn.MaxPool2d(2, stride = (2,2))
        
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)
        self.batch_norm5 = nn.BatchNorm1d(1024)

    def forward(self, image1, image2):
        image1 = forward_each(self, image1) # each image goes through the same network
        image2 = forward_each(self, image2)
        both_results = torch.cat((image1, image2), 1) # combine the results of both images
        results = F.sigmoid(self.linear2(both_results)) # convert the results to values between 0 and 1
        
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


def accuracy(label, output): # function to calculate accuracy by comparing the labels to the output of the network
    result = 0
    counter = 0
    for i in range(0, label.size()[0]):
        counter += 1.0
        value1 = label.data.cpu().numpy()[i]
        value2 = output.data.cpu().numpy()[i]
        if (value1 == value2):
            result += 1.0
    result = result/counter
    return result

learning_rate = 1e-6
criterion = nn.BCELoss()
cnn_model = cnn().cuda()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)


def training(cnn_model):
    train_loss = 0
    transform = transforms.Compose([transforms.Scale((128,128)), transforms.ToTensor()])
    aug_transform_1 = transforms.Compose([transforms.RandomHorizontalFlip()])
    scale_transform1 = transforms.Compose([transforms.Scale((128,128)), transforms.Scale((int(128*1.3), int(128*1.3))), transforms.CenterCrop((128,128))])
    scale_transform2 = transforms.Compose([transforms.Scale((128,128)), transforms.Scale((int(128*0.7), int(128*0.7))), transforms.Pad(int(128 - 0.7*128))])
    '''
    dataset = lfwDataset_for_testing(csv_file='train.txt',
                                    root_dir='lfw/', transformation = transform)
    '''
    dataset = lfwDataset_transformations(csv_file='train.txt',
                                    root_dir='lfw/', transformation = [transform, aug_transform_1, scale_transform1, scale_transform2])
    
    dataloader = DataLoader(dataset, batch_size=12,
                        shuffle=True, num_workers=12)

    train_accuracy = 0
    iterations = 0
    
    for each in dataloader: # for each pair of images loaded
        image1 = Variable(each[0]).cuda()
        image2 = Variable(each[1]).cuda()
        label1 = np.array([float(i) for i in each[2]])
        label1 = torch.from_numpy(label1).view(label1.shape[0], -1)
        label1 = label1.type(torch.FloatTensor)
        label = Variable(label1).cuda()
        output = cnn_model(image1, image2) # get the output of the network
        optimizer.zero_grad()
        loss = criterion(output, label) # calculate the loss
        loss.backward()
        optimizer.step()
        output = torch.round(output) # round to 0 and 1 in order to compare the output to the labels
        train_accuracy += accuracy(label, output) # calculate accuracy and add it up
        train_loss += loss.data[0]
        iterations += 1.0
    train_loss = train_loss/iterations
    train_accuracy = train_accuracy/iterations


    return train_loss, train_accuracy


def testing(cnn_model):
    transform = transforms.Compose([transforms.Scale((128,128)), transforms.ToTensor()])
    dataset = lfwDataset_for_testing(csv_file='test.txt',
                                    root_dir='lfw/', transformation = transform)
    dataloader = DataLoader(dataset, batch_size=12,
                        shuffle=True, num_workers=12)
    test_loss = 0
    test_accuracy = 0
    iterations = 0

    for each in dataloader:
        image1 = Variable(each[0]).cuda()
        image2 = Variable(each[1]).cuda()
        label1 = np.array([float(i) for i in each[2]])
        label1 = torch.from_numpy(label1).view(label1.shape[0], -1)
        label1 = label1.type(torch.FloatTensor)
        label = Variable(label1).cuda()
        output = cnn_model(image1, image2)
        loss = criterion(output, label)
        output = torch.round(output)
        test_accuracy += accuracy(label, output)
        test_loss += loss.data[0]
        iterations += 1.0
    test_loss = test_loss/iterations
    test_accuracy = test_accuracy/iterations
        
    return test_loss, test_accuracy


def main():

    if len(sys.argv) != 3: # if the person didn't input an argument
        print("Usage: --load/--save WEIGHS_FILE")
        return
    
    filename = sys.argv[2] # the filename to save or load the weights

    if sys.argv[1] == "--load":
        print("loading...")
        cnn_model.load_state_dict(torch.load(filename))
        test_loss, test_accuracy = testing(cnn_model)
        train_loss, train_accuracy = training(cnn_model)
        print("train loss", round(train_loss,2))
        print("train accuracy", round(train_accuracy,2))
        print("test loss", round(test_loss,2))
        print("test accuracy", round(test_accuracy,2))
        
    elif sys.argv[1] == "--save":
        print("Training... the weights will be saved at the end.")
        all_training_loss = list()
        all_testing_loss = list()
        all_training_accuracy = list()
        all_testing_accuracy = list()

        for epoch in range(epochs):
            print("epoch", epoch)
            train_loss, train_accuracy = training(cnn_model)
            print("train loss", train_loss)
            print("train accuracy", train_accuracy)
            all_training_loss.append(train_loss)
            all_training_accuracy.append(train_accuracy)
            test_loss, test_accuracy = testing(cnn_model)
            print("test loss", test_loss)
            print("test accuracy", test_accuracy)
            all_testing_loss.append(test_loss)
            all_testing_accuracy.append(test_accuracy)

        print("training loss ", all_training_loss)
        print("testing loss", all_testing_loss)
        print("training accuracy", all_training_accuracy)
        print("testing accuracy", all_testing_accuracy)

        torch.save(cnn_model.state_dict(), filename)

        plt.switch_backend('agg')
        plt.plot(all_training_loss, label = "Loss")
        plt.plot(all_testing_loss, label = "Loss")
        plt.savefig('p1a_loss', bbox_inches = 'tight')

        plt.plot(all_training_accuracy, label = "accuracy")
        plt.plot(all_testing_accuracy, label = "accuracy")
        plt.savefig('p1a_accuracy', bbox_inches = 'tight')
                
    else: # if the input arguments don't match any of the options
        print("Usage: --load/--save WEIGHS_FILE")
        return
    
    return

if __name__ == '__main__':
    main()
    
