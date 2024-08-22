import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import logging
import os

# Configure a global logger for this module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


logger = logging.getLogger(__name__)
logging.basicConfig(filename='model_performance.log', level=logging.INFO, format='%(asctime)s %(message)s')

class Tester:
    def __init__(self, model, test_loader, model_path, results_dir="results"):
        """
        Initialize the Tester with a model, test data loader, and paths for model and output files.

        Args:
            model (nn.Module): The neural network model to be tested.
            test_loader (DataLoader): DataLoader for the test data.
            model_path (str): Path to the .pth file containing the model weights.
            results_dir (str): Directory where plots and logs will be saved.
        """
        self.model = model
        self.test_loader = test_loader
        self.model_path = model_path
        self.results_dir = results_dir

        # Create directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

        # Load the model weights
        self.load_model()

    def load_model(self):
        """Load the model weights from the given file."""
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            logger.info(f"Loaded model weights from {self.model_path}")
        else:
            logger.error(f"Model file {self.model_path} not found.")
            raise FileNotFoundError(f"Model file {self.model_path} not found.")

    def compute_metrics(self, all_labels, all_preds, cm):
        """Compute and log precision, recall, F1-score, accuracy, and save metrics to file."""
        # Compute the classification report
        report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)], output_dict=True)

        # Compute overall metrics
        accuracy = np.sum(all_preds == all_labels) / len(all_labels)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        # Log and save overall metrics
        overall_metrics = {
            'Accuracy': accuracy * 100,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1-Score': f1 * 100
        }
        logger.info(f"Overall Accuracy: {overall_metrics['Accuracy']:.2f}%")
        logger.info(f"Overall Precision: {overall_metrics['Precision']:.2f}%")
        logger.info(f"Overall Recall: {overall_metrics['Recall']:.2f}%")
        logger.info(f"Overall F1-Score: {overall_metrics['F1-Score']:.2f}%")

        with open(os.path.join(self.results_dir, 'overall_metrics.txt'), 'w') as f:
            for metric, value in overall_metrics.items():
                f.write(f"{metric}: {value:.2f}%\n")

        # Compute and log class-wise metrics using the confusion matrix
        logger.info("Class-wise Precision, Recall, F1-Score, and Accuracy:")
        classwise_metrics = {}
        with open(os.path.join(self.results_dir, 'class_wise_metrics.txt'), 'w') as f:
            for class_id in range(10):
                precision = report[str(class_id)]['precision'] * 100
                recall = report[str(class_id)]['recall'] * 100
                f1 = report[str(class_id)]['f1-score'] * 100

                classwise_metrics[str(class_id)] = {
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                }
                logger.info(f"Class {class_id}: Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1-Score: {f1:.2f}%")
                f.write(f"Class {class_id}: Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1-Score: {f1:.2f}%")

        # Plot class-wise metrics
        self.plot_class_wise_metrics(classwise_metrics)
        # Plot overall metrics
        self.plot_overall_metrics(overall_metrics)

    def plot_class_wise_metrics(self, classwise_metrics):
        """Plot and save class-wise precision, recall, F1-score, and accuracy with value annotations."""
        classes = list(classwise_metrics.keys())
        metrics = ['Precision', 'Recall', 'F1-Score']
        for metric in metrics:
            values = [classwise_metrics[cls][metric] for cls in classes]
            plt.figure(figsize=(10, 6))
            bars = plt.bar(classes, values, color='skyblue')
            plt.ylim(0, 100)
            plt.yticks(np.arange(0, 101, 10))
            plt.xticks(np.arange(len(classes)), classes)
            plt.xlabel('Class')
            plt.ylabel(f'{metric} (%)')
            plt.title(f'Class-wise {metric}', pad=20)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Add value annotations on top of the bars
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

            plt.savefig(os.path.join(self.results_dir, f'class_wise_{metric.lower()}.png'))
            plt.close()
            logger.info(f"Class-wise {metric} saved to '{os.path.join(self.results_dir, f'class_wise_{metric.lower()}.png')}'")

    def plot_overall_metrics(self, overall_metrics):
        """Plot and save overall precision, recall, F1-score, and accuracy with value annotations."""
        metrics = list(overall_metrics.keys())
        values = [overall_metrics[metric] for metric in metrics]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color='salmon')
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 101, 10))
        plt.xticks(np.arange(len(metrics)), metrics)
        plt.xlabel('Metric')
        plt.ylabel('Percentage (%)')
        plt.title('Overall Metrics', pad=20)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Add value annotations on top of the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

        plt.savefig(os.path.join(self.results_dir, 'overall_metrics.png'))
        plt.close()
        logger.info(f"Overall metrics saved to '{os.path.join(self.results_dir, 'overall_metrics.png')}'")

    def test(self):
        """Test the model on the test dataset, compute and save the confusion matrix, and log the accuracy."""
        self.model.eval()  # Set the model to evaluation mode
        all_labels = []
        all_preds = []

        # Disable gradient computation for testing
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)  # Get the predicted class (index with the highest logit)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Convert lists to NumPy arrays
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # Compute the confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Save the confusion matrix using a heatmap
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(10)])
        save_cm_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(save_cm_path)
        plt.close()

        # Compute and log the metrics
        self.compute_metrics(all_labels, all_preds, cm)

    def plot_learning_curves(self, history):
        """Plot and save learning curves for loss and accuracy."""
        plt.figure()
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curves - Loss', pad=20)
        plt.legend()
        save_path = os.path.join(self.results_dir, 'learning_curve_loss.png')
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Learning curve for loss saved to '{save_path}'")

        plt.figure()
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Learning Curves - Accuracy', pad=20)
        plt.legend()
        save_path = os.path.join(self.results_dir, 'learning_curve_accuracy.png')
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Learning curve for accuracy saved to '{save_path}'")
