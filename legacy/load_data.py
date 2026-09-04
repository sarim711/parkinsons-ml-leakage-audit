import os
import yaml

import pandas as pd

# 1. Dynamically find the project root directory
# This script is located in src/data/
# We go up 2 levels (src/data -> src -> root) to find the project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

def load_config():
    """Helper to load config file using absolute paths"""
    config_path = os.path.join(PROJECT_ROOT, "config", "base.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_raw_data():
    """Loads the raw Parkinson's data."""
    config = load_config()
    
    # 2. Construct the absolute path to the data file
    # config['raw_data_path'] is "data/raw/parkinsons_udprs.data"
    file_path = os.path.join(PROJECT_ROOT, config["raw_data_path"])
    
    print(f"Loading data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found at {file_path}. Check if the file exists.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    df = load_raw_data()
    if df is not None:
        print(df.head())