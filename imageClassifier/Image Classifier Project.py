#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Imports here
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import torch
import numpy as np
from torch import nn
from torch import optim
#import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
#from torch.autograd import Variable


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[41]:


# Means and standard deviations
mean = [0.485, 0.456, 0.406]
std_dev = [0.229, 0.224, 0.225]


# In[4]:


# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomRotation(30),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std_dev)]),
    'test-valid': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std_dev)])
} 

# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train':datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'test':datasets.ImageFolder(test_dir, transform=data_transforms['test-valid']),
    'valid':datasets.ImageFolder(valid_dir, transform=data_transforms['test-valid'])
} 

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'test':torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True),
    'valid':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
} 


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[5]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[6]:


# TODO: Build and train your network
model = models.vgg16(pretrained=True)


# In[7]:


# Printing model
model


# In[8]:


# Freezing the features parameters
for params in model.parameters():
    params.requires_grad=False


# In[9]:


# importing OrderedDict structure
from collections import OrderedDict


# In[10]:


# Creating a new feed forward network (untrained) 
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096, bias=True)),
                                       ('relu1', nn.ReLU()),
                                       ('dropout1', nn.Dropout(p=0.5)),
                                       ('fc2', nn.Linear(4096, 102, bias=True)),
                                       ('output', nn.LogSoftmax(dim=1))
                                      ])
                          )

model.classifier = classifier


# In[11]:


# Defining Loss and Optimizer
criterion = nn.NLLLoss()
learn_rate = 0.001
optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)


# In[12]:


# defining epochs number, print_every
epochs = 3
print_every = 30


# In[13]:


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
    


# In[14]:


def train(model, epochs, criterion, optimizer, learning_rate, trainloader, validloader, print_every, cuda):
    print('Starting to train...')
    model.train()
    steps = 0
    
    for epoch in range(epochs):
        running_loss = 0
        
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


# In[15]:


# Changing model device to cuda if it is enabled
cuda = torch.cuda.is_available()
if cuda:
    model.cuda()
else:
    model.cpu()


# In[16]:


train(model, epochs, criterion, optimizer, learn_rate, dataloaders['train'], dataloaders['valid'], print_every, cuda)


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[17]:


# TODO: Do validation on the test set
accuracy, test_loss = 0, 0

with torch.no_grad():
    test_loss, accuracy = valid(model, criterion, cuda, dataloaders['test'])

print('Accuracy: {:.2f}'.format(accuracy/len(dataloaders['test'])),
     '\n Loss Rate: {:.2f}'.format(test_loss/len(dataloaders['test'])))


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[18]:


# TODO: Save the checkpoint 
model.class_to_idx = image_datasets['train'].class_to_idx


# In[19]:


checkpoint = {'structure': 'vgg16',
             'classifier': model.classifier,
             'class_to_idx': model.class_to_idx,
             'state_dict': model.state_dict()}
torch.save(checkpoint, 'checkpoint.pth')


# In[20]:


ls


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[21]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(name):
    checkpoint = torch.load(name)
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Loading other things from checkpoint 
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict = checkpoint['state_dict']
    
    return model


# In[22]:


model = load_model('checkpoint.pth')


# In[23]:


model


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[24]:


def process_image(image, mean, std_dev):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    transf = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std_dev)
                                ])
    img = Image.open(image)
    img_transf = transf(img)
    # TODO: Process a PIL image for use in a PyTorch model
    return img_transf


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[25]:


from PIL import Image


# In[26]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[27]:


img = process_image("flowers/test/10/image_07090.jpg", mean, std_dev)
imshow(img)


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[36]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to('cpu')
    model.eval();
        
    img_torch = torch.from_numpy(np.expand_dims(process_image(image_path, mean, std_dev),axis=0)).type(torch.FloatTensor).to("cpu")
    
    with torch.no_grad():
        output = model.forward(img_torch)
    
    prob = torch.exp(output)
       
    # Find the top 5 results
    top_probs, top_labels = prob.topk(topk)    
    top_probs = np.array(top_probs.detach())[0] # This is not the correct way to do it but the correct way isnt working thanks to cpu/gpu issues so I don't care.
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    
    return top_probs, top_labels 
   


# In[ ]:





# In[37]:


probs, classes = predict("flowers/test/10/image_07090.jpg", model)

print(probs)
print(classes)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[46]:


# TODO: Display an image along with the top 5 classes
def plot_probas(path, model, cat_to_name):
    fig, (ax1, ax2) = plt.subplots(figsize=(6,7), nrows=2)
    ax1.axis('off')
    probs, classes = predict(path, model)
    class_names = [cat_to_name[i] for i in classes]
    ax2.barh(class_names, probs)
    imshow(process_image(path,mean,std_dev), ax=ax1)


# In[47]:



plot_probas("flowers/test/10/image_07090.jpg", model, cat_to_name)


# In[ ]:




