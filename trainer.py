import torch
import logging
import os
import matplotlib.pyplot as plt

# Configure a global logger for this module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

logger = logging.getLogger(__name__)

class Trainer:
    """
        A class to handle training and validation of a PyTorch model, with functionality for early stopping,
    model saving, and metric plotting.

    Attributes:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        model_dir (str): Directory to save the model checkpoints.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        epochs (int): Number of epochs to train the model.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        higher_accuracy (float): Best accuracy achieved during training.
        early_stop (bool): Indicator for early stopping.
        counter (int): Counter for the number of epochs with no improvement.
        best_loss (float): Best validation loss achieved.
        train_losses (list): List to store training loss per epoch.
        val_losses (list): List to store validation loss per epoch.
        val_accuracies (list): List to store validation accuracy per epoch.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, model_dir='./models', patience=5, epochs=10, min_delta=0.0):
        """
        Initializes the Trainer with the given model, data loaders, loss function, and optimizer.

        Args:
            model (torch.nn.Module): The model to train.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for training the model.
            model_dir (str): Directory to save the model checkpoints. Defaults to './models'.
            patience (int): Number of epochs with no improvement after which training will be stopped. Defaults to 5.
            epochs (int): Number of epochs to train the model. Defaults to 10.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.0.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_dir = model_dir
        self.patience = patience
        self.epochs = epochs
        self.min_delta = min_delta
        self.higher_accuracy = 0.0
        self.early_stop = False
        self.counter = 0
        self.best_loss = None

        # Ensure the model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize lists to store the metrics for plotting
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self):
        """
        Performs one epoch of training.

        Returns:
            float: The average loss over the epoch.
        """
        self.model.train()
        running_loss = 0.0

        for images, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate_epoch(self):
        """
        Validates the model on the validation set.

        Returns:
            tuple: A tuple containing the validation loss and accuracy.
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return val_loss / len(self.val_loader), accuracy

    def check_early_stopping(self, val_loss):
        """
        Checks if early stopping criteria are met.

        Args:
            val_loss (float): Validation loss for the current epoch.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered.")
        else:
            self.best_loss = val_loss
            self.counter = 0

    def save_model(self, epoch, accuracy):
        """
        Saves the model if the validation accuracy has improved.

        Args:
            epoch (int): Current epoch number.
            accuracy (float): Validation accuracy for the current epoch.
        """
        if accuracy > self.higher_accuracy:
            self.higher_accuracy = accuracy
            model_path = os.path.join(self.model_dir, f'mnist_model_{epoch+1}.pth')
            torch.save(self.model.state_dict(), model_path)
            logger.info(f'Saved model with accuracy {accuracy:.2f}% at {model_path}')

    def train(self):
        """
        Trains the model for the specified number of epochs, with early stopping and model saving.

        After training, it plots and saves the training and validation loss and accuracy.
        """
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss, accuracy = self.validate_epoch()

            # Log the metrics for the current epoch
            logger.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, '
                        f'Validation Loss: {val_loss:.4f}, '
                        f'Validation Accuracy: {accuracy:.2f}%')

            # Save the model if validation accuracy improves
            self.save_model(epoch, accuracy)

            # Store the metrics for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(accuracy)

            # Check for early stopping
            self.check_early_stopping(val_loss)
            if self.early_stop:
                break
