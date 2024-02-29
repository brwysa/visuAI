import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from utils import load_checkpoint

def train_model(data_dir, save_dir, arch='vgg16', hidden_units=512, learning_rate=0.01, epochs=20, gpu=False):
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError("Unsupported architecture. Please choose 'vgg16'.")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            model.eval()
            valid_loss = 0
            accuracy = 0

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    valid_loss += criterion(outputs, labels).item()

                    ps = torch.exp(outputs)
                    equality = (labels.data == ps.max(dim=1)[1])
                    accuracy += equality.type(torch.FloatTensor).mean()

            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {running_loss / len(train_loader):.3f}.. "
                  f"Validation loss: {valid_loss / len(valid_loader):.3f}.. "
                  f"Validation accuracy: {accuracy / len(valid_loader):.3f}")

    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'model': model,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(checkpoint, save_dir + '/checkpoint.pth')

def main():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint.")
    parser.add_argument("data_directory", help="Path to the directory containing training and validation data.")
    parser.add_argument("--save_dir", default=".", help="Directory to save checkpoints.")
    parser.add_argument("--arch", choices=['vgg16'], default='vgg16', help="Choose architecture.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Set learning rate.")
    parser.add_argument("--hidden_units", type=int, default=512, help="Set number of hidden units.")
    parser.add_argument("--epochs", type=int, default=20, help="Set number of epochs.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training.")
    
    args = parser.parse_args()
    
    train_model(args.data_directory, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu)

if __name__ == "__main__":
    main()
