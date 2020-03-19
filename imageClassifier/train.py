# Libraries imports
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
from os.path import isdir
from torch.autograd import Variable

# Performs train transformation on a train dataset
def train_transf(train_dir):
    # Means and standard deviations for normalize images
    mean = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]

    transf = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomRotation(30),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std_dev)])
    # Loading Data
    train_data = datasets.ImageFolder(train_dir, transform=transf)
    return train_data

# Performs test/validation transformations on a test/validadtion dataset
def test_transf(test_dir):
    # Means and standard deviations for normalize images
    mean = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]

    # Define transformation
    transf = transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std_dev)])
    # Load the Data
    test_data = datasets.ImageFolder(test_dir, transform=transf)
    return test_data

# Returns a dataloader from dataset imported
def data_loader(data, train=True):        
    if train:
        return torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else:
        return torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)

# Changing model device to cuda if it is enabled
def model_device(model):
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    else:
        model.cpu()
    return model, cuda

# Importing (first) model to be used from torchvision 
def load_model(architecture="vgg16"):
    print('Importing model: {}'.format(architecture))
    model= models.__dict__[architecture](pretrained=True)
    model.name = architecture
    
    # Freezing the features parameters
    for params in model.parameters():
        params.requires_grad=False
        
    print(model)    
    return model

# Creating a new feed forward network (untrained) 
def Classifier(model, hidden_units=4096):
    
    # Finding Input Layers
    input_features = model.classifier[0].in_features
    
    # Creating a new feed forward network (untrained) 
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                                       ('relu1', nn.ReLU()),
                                       ('dropout1', nn.Dropout(p=0.5)),
                                       ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                                       ('output', nn.LogSoftmax(dim=1))
                               ]) )
    return classifier


# Function to validate the model
def valid(model, criterion, cuda, testloader):    
    model.eval()
    accuracy = 0
    test_loss = 0
    
    for idx, (inputs, labels) in enumerate(testloader):
        if cuda:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        outputs = model.forward(inputs) #forward
        test_loss += criterion(outputs, labels).item()
        
        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1]) 
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy 
    
# Training the network
def train(model, epochs, criterion, optimizer, learning_rate, trainloader, validloader, print_every, cuda):
    print('Starting to train...')
    model.train()
    steps = 0
    
    for epoch in range(epochs):
        running_loss = 0
        print('Before enumareate.........')
        for idx, (inputs, labels) in enumerate(trainloader):
            steps+=1
            if cuda: # if cuda is available, adapt inputs and labels
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                
            # re-inicializing gradient - don't forget (it's very important :) )
            optimizer.zero_grad()            
            outputs = model.forward(inputs) # forward pass
            loss = criterion(outputs, labels) # backward pass
            loss.backward()
            
            # updating weights using gradient descending
            optimizer.step()            
            running_loss+=loss.item()
            
            # Validating 
            if steps % print_every == 0:
                with torch.no_grad():
                    valid_loss, accuracy = valid(model, criterion, cuda, validloader)
                
                print('Epoch: {}/{}'.format(epoch+1, epochs), 
                      'Training Loss: {:.4f}'.format(running_loss, print_every),
                      'Validation Loss: {:.4f}'.format(valid_loss/len(validloader)),
                      'Validation Accuracy: {:.4f}'.format(accuracy/len(validloader))
                     )
                running_loss= 0
                model.train()
            
            
    print(' Finished, model is trained')
    return model

# performs validation on the test dataset  
def validate_test(model, test_data, criterion,cuda):
    # TODO: Do validation on the test set
    accuracy, test_loss = 0, 0

    with torch.no_grad():
        test_loss, accuracy = valid(model, criterion, cuda, test_data)

    print('Accuracy: {:.2f}'.format(accuracy/len(test_data)),
         '\n Loss Rate: {:.2f}'.format(test_loss/len(test_data)))

# Saving checkpoint 
def save_checkpoint(model, save_dir, train_data):
    print('Saving checkpoint...')
    
    # TODO: Save the checkpoint 
    if isdir(save_dir):
        model.class_to_idx = train_data.class_to_idx
        checkpoint = {'structure': model.name,
                 'classifier': model.classifier,
                 'class_to_idx': model.class_to_idx,
                 'state_dict': model.state_dict()}
        torch.save(checkpoint, 'checkpoint.pth')
        print('Checkpoint was saved.')
    else:
        print('Error: Directory not found.')

        
# get arguments from the command line
def get_args():
    # parser
    parser = argparse.ArgumentParser(description='Settings of NN')
    
    # adding each argument
    parser.add_argument('--arch', type=str, help='Choose model from torchvision.models as str')    
    parser.add_argument('--save_dir', type=str,help='Define save directory for checkpoints as str.')
    parser.add_argument('--learning_rate',type=float, help='Define gradient descent learning rate as float')
    parser.add_argument('--hidden_units', type=int, help='Hidden units for DNN classifier as int')
    parser.add_argument('--epochs', type=int,help='Define the number of epochs for training as int')
    # Add GPU Option to parser
    parser.add_argument('--gpu', action="store_true", help='Use GPU + Cuda for calculations')
    #returing args
    return parser.parse_args()
        
# main function
def main():
    args = get_args() #getting args from command line
    
    # Directory for train, test and validation datasets
    raiz = 'flowers'
    train_dir, valid_dir, test_dir = raiz+'/train', raiz+'/valid', raiz+'/test' 
    
    # using transform functions to be able to create a data loader
    train_ = train_transf(train_dir)
    test_ = test_transf(test_dir)
    valid_ = test_transf(valid_dir)
    
    # Loading datasets
    trainloader = data_loader(train_, train=True)
    testloader = data_loader(test_, train=False)
    validloader = data_loader(valid_, train=False)
    
    # Loading Model (cnn part)
    arc  =  args.arch if args.arch else 'vgg16'
    model = load_model(architecture=arc)
    
    # Creating the classifier
    hidden_units = args.hidden_units if args.hidden_units else 4096
    model.classifier = Classifier(model, hidden_units)
    
    # Setting some parameters
    criterion = nn.NLLLoss()
    learning_rate = args.learning_rate if args.learning_rate else 0.001
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    epochs = args.epochs if args.epochs else 3
    print_every = 30
    steps = 0
    
    # Turn on cuda if its available
    model,cuda = model_device(model)
    print('cuda:', cuda)
    
    print('before training')
    #Training model
    model_ = train(model, epochs, criterion, optimizer, learning_rate, trainloader, validloader, print_every, cuda)
    
    # Testing the Network
    validate_test(model_, testloader, criterion, cuda)
    
    # Saving Checkpoint
    save_dir = args.save_dir if args.save_dir else './'
    save_checkpoint(model_, save_dir, train_)
    print('Done \./')
    
# Executes main
if __name__=='__main__':main()
