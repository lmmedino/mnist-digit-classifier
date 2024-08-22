from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

class MNISTModel(nn.Module):
    """
    Convolutional Neural Network (CNN) model for MNIST digit classification.
    The network consists of convolutional layers followed by fully connected layers.
    Dropout is applied after each fully connected layer to prevent overfitting.
    """

    def __init__(self):
        super(MNISTModel, self).__init__()
        # First convolutional layer (input: 28x28x1, output: 28x28x32)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # Second convolutional block (input: 28x28x32, output: 14x14x64 for both conv2_1 and conv2_2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions to 14x14x64

        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions to 14x14x64

        # Third convolutional block (input: 14x14x64, output: 7x7x256 each)
        self.conv3_1 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pool3_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions to 7x7x256

        self.conv3_2 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pool3_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions to 7x7x256

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(7 * 7 * 512, 1000)  # Combined output size from conv3_1 and conv3_2
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer with a dropout probability of 0.5
        self.fc2 = nn.Linear(1000, 500)
        self.dropout2 = nn.Dropout(0.5)  # Dropout layer with a dropout probability of 0.5
        self.fc3 = nn.Linear(500, 10)  # Output layer for 10 classes

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10) with class logits.
        """
        x = torch.relu(self.conv1(x))  # Conv1: Input 28x28x1 -> Output 28x28x32

        # Conv2 block: Input 28x28x32 -> Output 14x14x64 for both conv2_1 and conv2_2
        x2_1 = torch.relu(self.conv2_1(x))  # Conv2_1: Output 28x28x64
        x2_1 = self.pool2_1(x2_1)  # Pool2_1: Reduces to 14x14x64

        x2_2 = torch.relu(self.conv2_2(x))  # Conv2_2: Output 28x28x64
        x2_2 = self.pool2_2(x2_2)  # Pool2_2: Reduces to 14x14x64

        # Conv3 block: Input 14x14x64 -> Output 7x7x256 each
        x3_1 = torch.relu(self.conv3_1(x2_1))  # Conv3_1: Output 14x14x256
        x3_1 = self.pool3_1(x3_1)  # Pool3_1: Reduces to 7x7x256

        x3_2 = torch.relu(self.conv3_2(x2_2))  # Conv3_2: Output 14x14x256
        x3_2 = self.pool3_2(x3_2)  # Pool3_2: Reduces to 7x7x256

        # Concatenate the outputs of Conv3_1 and Conv3_2 along the depth dimension
        x3 = torch.cat((x3_1, x3_2), dim=1)  # Output: 7x7x512

        # Flatten the tensor for fully connected layers
        x3 = x3.view(-1, 7 * 7 * 512)  # Flatten to 1D tensor for FC layers

        # Fully connected layers with dropout
        x = torch.relu(self.fc1(x3))  # FC1: Input 7*7*512 -> Output 1000
        x = self.dropout1(x)  # Apply dropout after FC1

        x = torch.relu(self.fc2(x))  # FC2: Input 1000 -> Output 500
        x = self.dropout2(x)  # Apply dropout after FC2

        x = self.fc3(x)  # FC3: Input 500 -> Output 10 (logits for 10 classes)

        return x

# Initialize the Flask application
app = Flask(__name__)

# Load your trained model
model = MNISTModel()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Define a transform to match the training data preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict(image):
    image = transform(image).unsqueeze(0)
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item() * 100


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))

        # Convert the image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Make prediction
        prediction, confidence = predict(image)

        # Render the template with the results
        return render_template('index.html', prediction=prediction, confidence=confidence, image_data=img_str, request_made=True)

    return render_template('index.html', request_made=False)


if __name__ == "__main__":
    app.run(debug=True)
