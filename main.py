import os
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Model
from trainer import Trainer
from tester import Tester
from dataset import DataHandler
import torchvision.datasets as datasets
from optimizer import get_adam_optimizer
from utils import setup_logging, load_config

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train or test the model.")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help="Mode to run: 'train' or 'test'")
    parser.add_argument('--model_path', type=str, help="Path to the .pth model file (required for testing)", required=False)
    args = parser.parse_args()

    # Manual validation for mode (though argparse already handles this)
    if args.mode not in ['train', 'test']:
        raise ValueError(f"Invalid mode '{args.mode}'. Expected 'train' or 'test'.")

    config = load_config('configs/config.yaml')
    os.makedirs(config['results_params']['results_path'], exist_ok=True)

    # Setup logging according to the configuration
    setup_logging(config['logging_params']['log_file'])

    dataset = datasets.MNIST

    if args.mode == 'train':
        # Training mode
        data_handler = DataHandler(dataset, root_dir=config['dataset_params']['root_dir'], val_split=0.2)

        # Ensure data directories exist
        os.makedirs('data/train', exist_ok=True)
        os.makedirs('data/val', exist_ok=True)

        # Load the datasets for training and validation
        train_dataset, val_dataset = data_handler.load_train_val_data()

        # Save images to directories
        data_handler.save_images(train_dataset, 'data/train')
        data_handler.save_images(val_dataset, 'data/val')

        # Prepare data loaders
        train_loader = DataLoader(train_dataset, batch_size=config['train_params']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['train_params']['batch_size'], shuffle=False)

        # Initialize the model
        model = Model()

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = get_adam_optimizer(model, lr=config['train_params']['learning_rate'], weight_decay=config['train_params']['weight_decay'])

        # Initialize the Trainer and start training
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            model_dir=config['models_params']['save_path'],
            patience=config['train_params']['patience'],
            epochs=config['train_params']['epochs']

        )

        # Start training
        trainer.train()

    elif args.mode == 'test':
        # Testing mode requires the model path argument
        if not args.model_path:
            raise ValueError("Model path must be provided in test mode")

        data_handler = DataHandler(dataset,root_dir=config['dataset_params']['root_dir'])

        # Ensure the data directory for testing exists
        os.makedirs('data/test', exist_ok=True)

        # Load the test dataset
        test_dataset = data_handler.load_test_data()
        data_handler.save_images(test_dataset, 'data/test')

        # Prepare DataLoader for the test dataset
        test_loader = DataLoader(test_dataset, batch_size=config['test_params']['batch_size'], shuffle=False)

        # Initialize the model
        model = Model()

        # Initialize the Tester and run the test
        tester = Tester(
            model=model,
            test_loader=test_loader,
            model_path=args.model_path,
            results_dir=config['results_params']['results_path']
        )

        tester.test()


if __name__ == "__main__":
    main()
