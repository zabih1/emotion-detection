import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_params(params_path: str):
    logging.info(f"Loading parameters from {params_path}")
    try:
        with open(params_path, 'r') as file:
            config = yaml.safe_load(file)
        test_size = config['data_ingestion']['test_size']
        logging.info(f"Loaded test_size: {test_size}")
        return test_size
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def read_data(url):
    logging.info(f"Reading data from {url}")
    try:
        df = pd.read_csv(url)
        logging.info(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to read data from {url}: {e}")
        raise

def process_data(df):
    logging.info("Processing data")
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['neutral', 'sadness'])].copy()
        final_df['sentiment'].replace({'neutral': 1, 'sadness': 0}, inplace=True)
        logging.info(f"Processed data shape: {final_df.shape}")
        return final_df
    except Exception as e:
        logging.error(f"Data processing failed: {e}")
        raise

def save_data(data_path, train_data, test_data):
    logging.info(f"Saving data to {data_path}")
    try:
        os.makedirs(data_path, exist_ok=True)
        train_path = os.path.join(data_path, 'train.csv')
        test_path = os.path.join(data_path, 'test.csv')

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logging.info(f"Train data saved to {train_path}, shape: {train_data.shape}")
        logging.info(f"Test data saved to {test_path}, shape: {test_data.shape}")
    except Exception as e:
        logging.error(f"Failed to save data: {e}")
        raise

def main():
    logging.info("Starting data ingestion pipeline")
    try:
        df = read_data("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")
        final_df = process_data(df)

        test_size = load_params("params.yaml")
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        data_path = os.path.join("data", 'raw')
        save_data(data_path, train_data, test_data)

        logging.info("Data ingestion pipeline completed successfully")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
