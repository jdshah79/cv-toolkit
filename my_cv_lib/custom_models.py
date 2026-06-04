import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class CustomClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomClassifier, self).__init__()
        # A simple CNN block for demonstration
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Linear(16 * 112 * 112, num_classes) # Assuming 224x224 input

        # Built-in image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def predict_opencv_img(self, cv2_frame):
        """Accepts a raw OpenCV frame, transforms it, and returns the class ID."""
        # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
        color_coverted = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(color_coverted)
        
        tensor = self.transform(pil_img).unsqueeze(0) # Add batch dimension
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(tensor)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.item()
