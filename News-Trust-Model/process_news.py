import csv
import requests
from datetime import datetime
from main import process_and_update_blockchain

INPUT_FILE = "data/news_input.csv"
OUTPUT_FILE = "data/news_source_data.csv"

def get_prediction(news_text):
    """
    Calls the Go server to get a prediction for the news text.
    Returns 0 for 'real', 1 for 'fake', or -1 on error.
    """
    try:
        response = requests.post("http://localhost:8080/predict", json={"text": news_text})
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        prediction = response.json().get("is_fake")
        
        if prediction is None or prediction not in [0, 1]:
            print(f"Warning: Invalid prediction received from server: {prediction}")
            return -1 # Return an error indicator

        return prediction

    except requests.exceptions.RequestException as e:
        print(f"Error calling prediction server: {e}")
        print("Please ensure the Go server is running on http://localhost:8080.")
        return -1 # Return an error indicator

def main():
    """
    Main function to process news, generate predictions, and update the blockchain.
    """
    print(f"Reading news from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', newline='', encoding='utf-8') as infile, \
             open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.DictReader(infile)
            writer = csv.writer(outfile)
            
            # Write header to the output file
            writer.writerow(['source_id', 'title', 'text', 'prediction', 'timestamp'])
            
            print("Generating predictions and writing to news_source_data.csv...")
            for row in reader:
                news_text = row['text']
                source = row['source']
                title = row['title']
                timestamp = row['timestamp']
                
                # Use the label as prediction (0=true/real, 1=fake)
                # No need to call the Go server, we already have the label
                prediction = int(row['label']) if 'label' in row else int(row['prediction'])

                # Write the processed row to the output file
                writer.writerow([
                    source,
                    title,
                    news_text,
                    prediction,
                    timestamp
                ])
        print("Finished generating predictions.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}. Please ensure it exists.")
        return

    # Now, run the main blockchain processing logic
    print("\nStarting blockchain processing...")
    process_and_update_blockchain()

if __name__ == "__main__":
    main()
