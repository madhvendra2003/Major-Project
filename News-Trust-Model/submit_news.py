import requests
import csv
import argparse
from datetime import datetime

API_URL = "http://localhost:8080/predict"
DATA_FILE = "data/news_source_data.csv"

def get_prediction(news_text):
    """Calls the Go server to get a fake news prediction."""
    try:
        response = requests.post(API_URL, json={"text": news_text})
        response.raise_for_status()  # Raise an exception for bad status codes
        prediction = response.json().get("is_fake")
        return prediction
    except requests.exceptions.RequestException as e:
        print(f"Error calling prediction API: {e}")
        return None

def log_submission(source_id, news_text, prediction):
    """Appends the news submission details to the CSV log."""
    timestamp = datetime.now().isoformat()
    try:
        with open(DATA_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([source_id, news_text, prediction, timestamp])
        print(f"Successfully logged submission for source '{source_id}'.")
    except IOError as e:
        print(f"Error writing to data file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Submit a news article for classification and log the result.")
    parser.add_argument("--source", required=True, help="The ID or name of the news source (e.g., 'Reuters', 'NewsForYou').")
    parser.add_argument("--text", required=True, help="The full text of the news article.")
    args = parser.parse_args()

    print(f"Submitting news from source: {args.source}")
    prediction = get_prediction(args.text)

    if prediction is not None:
        print(f"Prediction received: {'FAKE' if prediction == 1 else 'REAL'} (Value: {prediction})")
        log_submission(args.source, args.text, prediction)
    else:
        print("Failed to get a prediction. The submission will not be logged.")

if __name__ == "__main__":
    main()