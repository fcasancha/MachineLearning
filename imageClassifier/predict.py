# Libraries Imports
import argparse
import json
from math import ceil
#import PIL
from PIL import Image
import torch
import numpy as np
from torchvision import datasets, transforms, models


# Means and standard deviations for normalize images
mean = [0.485, 0.456, 0.406]
std_dev = [0.229, 0.224, 0.225]

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(name_checkpoint, cuda=False):
    if not cuda:
        checkpoint = torch.load(name_checkpoint, map_location='cpu')
    else:
        checkpoint = torch.load(name_checkpoint)
        
    print('Loading checkpoint file..')
    checkpoint = torch.load(name_checkpoint)
    print(checkpoint['structure'])
        
    model= models.__dict__[checkpoint['structure']](pretrained=True)
    # no new gradient required
    for param in model.parameters():
        param.requires_grad = False
    
    # Loading other things from checkpoint 
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict = checkpoint['state_dict']
    
    # As we are loading model for inference, is recommended to use the following line
    model.eval()
    print('checkpoint file loaded.')
    
    return model

def process_image(image):
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

def predict(image_path, model, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to('cpu')
    model.eval();
        
    img_torch = torch.from_numpy(np.expand_dims(image_path,axis=0)).type(
        torch.FloatTensor).to("cpu")
    
    with torch.no_grad():
        output = model.forward(img_torch)
    
    prob = torch.exp(output)
       
    # Find the top 5 results
    top_probs, top_labels = prob.topk(topk)    
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers 

# Print probabity, transformed into dictionary
def print_probability(probs, flowers):
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))

# Define a parser
def arg_parser():
    parser = argparse.ArgumentParser(description="NN Settings")
    # Adding arguments 
    parser.add_argument('--image', type=str, help='Point to impage file for prediction.', required=True)
    parser.add_argument('--checkpoint',type=str, help='Point to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.')
    parser.add_argument('--category_names', type=str, help='File with Mapping from categories to real names.')
    parser.add_argument('--gpu', action="store_true", help='Use GPU + Cuda for calculations')

    # Parse args
    args = parser.parse_args()
    return args

# Main Function    
def main():
    arg = arg_parser() # receiving arguments
    # Loading checkpoint with saved model
    name_file = arg.checkpoint 
    model = load_checkpoint(name_file, cuda=True)
    #model = load_checkpoint('checkpoint.pth', cuda=True)
    
    #loading categories names from jason file
    jason_file = arg.category_names
    with open(jason_file, 'r') as file:
        cat_to_name = json.load(file)
        
    #processing the image
    img_proc = process_image(arg.image)
    
    # Verifying if cuda is enabled
    cuda = torch.cuda.is_available()
    
    #Predicting and printing results
    top_k = arg.top_k if arg.top_k else 5
    top_probs, top_labels, top_flowers = predict(img_proc, model,cat_to_name,top_k)
    
    # Print out probabilities
    print_probability(top_flowers, top_probs)

if __name__=='__main__':main()