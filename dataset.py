import os
import csv
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import logging


#logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


class DataHandler:
    def __init__(self, dataset,root_dir='./data', val_split=0.2):
        """
        Initialize the MNISTDataHandler with the directory for data storage and the validation split ratio.

        Args:
            root_dir (str): Directory where the MNIST dataset will be stored.
            val_split (float): Fraction of the training data to use as validation data.
            dataset (torchvision.datasets): The dataset class to be used (e.g., MNIST, CIFAR10).
        """
        self.root_dir = root_dir
        self.val_split = val_split
        self.dataset = dataset

    def load_train_val_data(self):
        """
        Load and transform the MNIST dataset, splitting it into training and validation sets.

        Returns:
            tuple: A tuple containing train_dataset and val_dataset.
        """
        # Define transformations for the training dataset (with data augmentation)
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize the images to [-1, 1]
        ])

        # Download and transform the training dataset
        train_dataset = self.dataset(root=self.root_dir, train=True, download=True, transform=train_transform)

        # Split the training dataset into training and validation sets
        train_size = int((1 - self.val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        return train_dataset, val_dataset

    def load_test_data(self):
        """
        Load and transform the MNIST test dataset.

        Returns:
            Dataset: The test dataset.
        """
        # Define transformations for the test dataset (no data augmentation)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize the images to [-1, 1]
        ])

        # Download and transform the test dataset
        test_dataset = self.dataset(root=self.root_dir, train=False, download=True, transform=test_transform)
        return test_dataset

    def save_images(self, dataset, directory):
        """
        Save images and labels from a dataset to a specified directory.

        Args:
            dataset (Dataset): The dataset from which to save images and labels.
            directory (str): The directory where images and labels will be saved.
        """
        labels_file = os.path.join(directory, 'labels.csv')

        # Log that the saving process is starting
        logging.info(f"Saving images and labels to {directory}...")

        # Open a CSV file to save labels
        with open(labels_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(['image_filename', 'label'])

            for idx, (image, label) in enumerate(dataset):
                label_dir = os.path.join(directory, str(label))
                # Create a directory for each label
                os.makedirs(label_dir, exist_ok=True)
                image_filename = f'{idx}.png'
                image_path = os.path.join(label_dir, image_filename)

                # Convert the tensor to a PIL image and save
                torchvision.transforms.functional.to_pil_image(image).save(image_path)

                # Write the image filename and label to the CSV file
                writer.writerow([image_filename, label])

        # Log that the saving process is completed
        logging.info(f"Finished saving images and labels to {directory}.")
