import torch
from torchvision import transforms
from PIL import Image
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    img = Image.open(image_path)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = preprocess(img)
    return img

def predict(image_path, model, topk=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    img = process_image(image_path)
    img_tensor = img.unsqueeze(0).float()
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    probs, indices = torch.topk(torch.nn.functional.softmax(output[0], dim=0), topk)
    
    class_to_idx_inv = {model.class_to_idx[key]: key for key in model.class_to_idx}
    classes = [class_to_idx_inv[idx.item()] for idx in indices]
    
    return probs.tolist(), classes