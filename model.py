# model.py
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import io
from torchvision import models
import torch.nn as nn

# Define the model architecture (DenseNet121 as used in CheXNet)
class CheXNet(nn.Module):
    def __init__(self, num_classes=11):
        super(CheXNet, self).__init__()
        self.densenet = models.densenet121(weights='DEFAULT')
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.densenet(x)

class CheXNetClassifierChange(nn.Module):
    def __init__(self, num_classes=11):
        super(CheXNetClassifierChange, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        for param in list(self.densenet.parameters())[:-12]:
            param.requires_grad = False

        num_ftrs = self.densenet.classifier.in_features
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5 / 2),
            nn.Linear(512, num_classes)
        )

        # Replace original classifier
        self.densenet.classifier = nn.Identity()

    def forward(self, x):
        features = self.densenet(x)
        output = self.classifier(features)
        return output

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, self.num_classes)

        if self.num_classes == 1:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        if self.num_classes == 1:
            x = self.sigmoid(self.fc2(x))
        else:
            x = self.fc2(x)
        return x

class DeepCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(DeepCNN, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
        )

        # 224/2/2/2/2 = 14
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )

        if self.num_classes == 1:
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.num_classes == 1:
            return self.activation(x)
        return x

def create_resnet18_model():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    return model.to(DEVICE)

def create_resnet50_model():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 11)
    )

    return model.to(DEVICE)

def create_densenet121_model():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(weights='DEFAULT')

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )

    return model.to(DEVICE)

def create_densenet169_model():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.densenet169(weights='DEFAULT')

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 11)
    )

    return model.to(DEVICE)

def load_model_binary(model_path='static/models/densenet121_binary.pth'):
    try:
        # Check if model file exists
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, model_path)
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found at {MODEL_PATH}")
            return None

        # Set up device (CPU or GPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialize model
        model = create_densenet121_model()

        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # Set to evaluation mode

        print("CheXNet model loaded successfully")
        return model

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def load_model_multi(model_path='static/models/densenet169_model.pth'):
    try:
        # Check if model file exists
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, model_path)
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found at {MODEL_PATH}")
            return None

        # Set up device (CPU or GPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialize model
        model = create_densenet169_model()

        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # Set to evaluation mode

        print("CheXNet model loaded successfully")
        return model

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Convert grayscale to RGB if needed (CheXNet expects RGB input)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply transformations
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    return input_batch

def predict_xray_binary(model, image_data):
    if model is None:
        return [
            {"name": "No Model"}
        ]

    try:
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            image = image_data

        input_tensor = preprocess_image(image)

        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor).item()

        label = 1 if output >= 0.5 else 0
        print(output, label)
        return {
            "label": label
        }

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {
            "label": "Error"
        }

def predict_xray_multi(model, image_data):
    if model is None:
        return [
            {"name": "No Model", "probability": 45},
        ]

    try:
        class_names = ['Xẹp phổi', 'Tim to', 'Tràn dịch màng phổi', 'Thâm nhiễm phổi',
                       'Khối u phổi', 'Nốt phổi', 'Tràn khí màng phổi', 'Đông đặc phổi',
                       'Phù phổi', 'Khí phế thũng', 'Dày màng phổi']

        thresholds = np.array([0.16, 0.26, 0.19, 0.18000000000000002, 0.25, 0.2, 0.35, 0.09000000000000001, 0.13, 0.32999999999999996, 0.1])

        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            image = image_data

        input_tensor = preprocess_image(image)

        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.sigmoid(output).squeeze().cpu().numpy()

        predicted_labels = []
        for i, prob in enumerate(probabilities):
            if prob >= thresholds[i]:
                predicted_labels.append({
                    "name": class_names[i],
                    "probability": round(float(prob) * 100, 2)
                })

        top_idx = int(np.argmax(probabilities))
        top_name = class_names[top_idx]
        top_prob = round(float(probabilities[top_idx]), 4)

        return sorted(predicted_labels, key=lambda x: x["probability"], reverse=True),

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return [
            {"name": "Error", "probability": 100}
        ]


# Function to create folder for model if it doesn't exist
def ensure_model_directory():
    os.makedirs('static/models', exist_ok=True)
    if not os.path.exists('static/models/densenet121_model_binary.pth'):
        print("CheXNet model file not found. Please download and place it in the static/models directory.")
    else:
        print("CheXNet model file found.")