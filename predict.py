import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from train import check_gpu
from torchvision import models


def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, help='Point to image file for prediction.', required=True)  # Corrected 'impage' to 'image'
    parser.add_argument('--checkpoint', type=str, help='Point to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.', default=5)  # Added default value for top_k
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parser.parse_args()
    return args


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)  # Updated to use checkpoint_path argument
    
    model = models.vgg16(pretrained=True)  # Load VGG16 model
    model.name = "vgg16"

    for param in model.parameters(): 
        param.requires_grad = False  # Freeze parameters

    # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image):
    img = PIL.Image.open(image)

    original_width, original_height = img.size
    
    # Adjust size based on aspect ratio
    if original_width < original_height:
        size = [256, 256 * (original_height / original_width)]
    else: 
        size = [256 * (original_width / original_height), 256]
        
    img.thumbnail(size)  # Resize image while maintaining aspect ratio
   
    center = (original_width / 2, original_height / 2)  # Center calculation
    left, top, right, bottom = center[0] - (224 / 2), center[1] - (224 / 2), center[0] + (224 / 2), center[1] + (224 / 2)
    img = img.crop((left, top, right, bottom))  # Crop image to 224x224

    numpy_img = np.array(img) / 255.0  # Normalize pixel values

    # Normalize each color channel
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    numpy_img = (numpy_img - mean) / std
        
    # Transpose image to match PyTorch's input format
    numpy_img = numpy_img.transpose(2, 0, 1)
    
    return numpy_img


def predict(image_tensor, model, device, cat_to_name, top_k=5):
    # Ensure model is in evaluation mode
    model.eval()
    
    # Convert image tensor to a PyTorch tensor and move to the device
    image_tensor = torch.from_numpy(image_tensor).float().unsqueeze(0).to(device)
    
    with torch.no_grad():  # Disable gradient calculation
        output = model(image_tensor)  # Forward pass
        probs = torch.exp(output)  # Get probabilities

    top_probs, top_labels = probs.topk(top_k)  # Get top K probabilities and labels

    # Convert to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels.cpu().numpy()]  # Convert labels to classes
    top_flowers = [cat_to_name[lab] for lab in top_labels]  # Map labels to flower names
    
    return top_probs.cpu().numpy(), top_labels, top_flowers  # Move top_probs to CPU for further processing


def print_probability(probs, flowers):
    # Converts two lists into a dictionary to print on screen
    for i, j in enumerate(zip(flowers, probs)):
        print("Rank {}:".format(i + 1),
              "Flower: {}, likelihood: {}%".format(j[1], ceil(j[0] * 100)))  # Corrected "liklihood" to "likelihood"


def main():
    args = arg_parser()  # Parse command line arguments
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)  # Load category names from JSON

    model = load_checkpoint(args.checkpoint)  # Load the trained model
    
    image_tensor = process_image(args.image)  # Process the input image
    
    device = check_gpu(gpu_arg=args.gpu)  # Check for GPU availability
    
    top_probs, top_labels, top_flowers = predict(image_tensor, model, device, cat_to_name, args.top_k)  # Make predictions
    
    print_probability(top_flowers, top_probs)  # Print predicted probabilities


if __name__ == '__main__':
    main()  # Execute the main function
