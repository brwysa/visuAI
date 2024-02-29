import argparse
import torch
import json
from utils import load_checkpoint, process_image, predict

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("checkpoint", help="Path to the checkpoint file.")
    parser.add_argument("--top_k", type=int, default=1, help="Return top K most likely classes.")
    parser.add_argument("--category_names", default="cat_to_name.json", help="Path to a JSON file mapping categories to real names.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference.")
    
    args = parser.parse_args()
    
    model = load_checkpoint(args.checkpoint)
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    probs, classes = predict(args.image_path, model, args.top_k)
    
    for i in range(len(probs)):
        flower_name = cat_to_name[classes[i]]
        print(f"Prediction {i + 1}: {flower_name} - Probability: {probs[i]:.3f}")

if __name__ == "__main__":
    main()
