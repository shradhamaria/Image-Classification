import argparse
import json
import PIL
import torch
import numpy as np

from math import ceil
from train import check_gpu
from torchvision import models

def arg_parser():
    # Define a parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")

    parser.add_argument('--image', default = 'flowers/test/20/image_04912.jpg',
                        type=str, 
                        help='Enter path to image.')

    parser.add_argument('--checkpoint', 
                        type=str, default = 'my_checkpoint.pth',
                        help='Point to checkpoint file as str.')
    
    parser.add_argument('--top_k', 
                        type=int, default = 3,
                        help='Enter number of top most likely classes to view.')
    
    parser.add_argument('--category_names', default = 'cat_to_name.json',
                        type=str, 
                        help='Mapping from categories to real names.')

    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')

    args = parser.parse_args()
    
    return args

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load("my_checkpoint.pth")
    
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    elif checkpoint['architecture'] == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.name = "resnet18"
    elif checkpoint['architecture'] == 'densenet201':
        model = models.densenet201(pretrained=True)
        model.name = "densenet201"    
    
    for param in model.parameters(): param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    if model.name == "resnet18":
        model.fc = checkpoint['classifier']
    else:
        model.classifier = checkpoint['classifier']
        
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

    
def resize_image(image, size):
    
    w, h = image.size

    if h > w:
        # Set width to 'size' and scale height to maintain aspect ratio
        h = int(max(h * size / w, 1))
        w = int(size)
    else:
        # Set height to 'size' and scale width to maintain aspect ratio
        w = int(max(w * size / h, 1))
        h = int(size)

    return image.resize((w, h))


def crop_image(image, size):
    
    w, h = image.size
    x0 = (w - size) / 2
    y0 = (h - size) / 2
    x1 = x0 + size
    y1 = y0 + size

    return image.crop((x0, y0, x1, y1))


def process_image(image):
    
    test_image = PIL.Image.open(image)
    # Resize image so shortest side is 256 pixels
    resized_image = resize_image(test_image, 256)

    # Crop image
    cropped_image = crop_image(resized_image, 224)

    # Convert image to float array
    np_image = np.array(cropped_image) / 255.

    # Normalize array
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # PyTorch tensors assume the color channel is the first dimension
    # but PIL assumes is the third dimension
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def predict(processed_image, loaded_model, topk, gpu_mode):
    
    # Converting to torch tensor from Numpy array
    img_tensor = torch.from_numpy(processed_image).type(torch.FloatTensor)
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0)
    
    
    loaded_model.eval()
    
    if gpu_mode == True:
        loaded_model.to('cuda')
    else:
        loaded_model.cpu()
    
    with torch.no_grad():
        # Running image through network
        output = loaded_model.forward(img_add_dim)
        

    # Calculating probabilities
    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    # Loading index and class mapping
    class_to_idx = loaded_model.class_to_idx
    # Inverting index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
        
    return probs_top_list, classes_top_list


def main():
    """
    Executing relevant functions
    """
    
    # Get Keyword Args for Prediction
    args = arg_parser()
    
    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    # Load model trained with train.py
    model = load_checkpoint(args.checkpoint)
    
    # Process Image
    image_tensor = process_image(args.image)
    print(image_tensor.shape)
    
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu); 
    
    # Carry out prediction
    probs, classes = predict(image_tensor, model, args.top_k, device)
    
    # Print out probabilities
    # Print probabilities and predicted classes
    print(probs)
    print(classes) 

    names = []
    for i in classes:
        names += [cat_to_name[i]]
    
    print(f"This flower is most likely to be a: '{names[0]}' with a probability of {round(probs[0]*100,4)}% ")

if __name__ == '__main__': main()