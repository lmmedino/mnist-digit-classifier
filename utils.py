import logging
import yaml

def setup_logging(log_file="training.log"):
    """
    Sets up the logging configuration for the application.

    Args:
        log_file (str): Path to the log file where logs will be stored.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    """
    Loads the configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
