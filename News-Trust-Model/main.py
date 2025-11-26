from blockchain_client import BlockchainClient
from source_logic import SourceLogic
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

DATA_FILE = "data/news_source_data.csv"

def load_data(file_path):
    """Loads news submission events from the CSV file."""
    events = []
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['prediction'] = int(row['prediction'])
                row['timestamp'] = datetime.fromisoformat(row['timestamp'])
                events.append(row)
    except FileNotFoundError:
        print(f"Data file not found at {file_path}. Please create it or run submit_news.py first.")
    except (ValueError, KeyError) as e:
        print(f"Error processing data file: {e}. Make sure the CSV format is correct.")
    return events

def process_and_update_blockchain():
    """Processes news data, updates trust scores, and records them on the blockchain."""
    # 1. Initialize Blockchain and Deploy Contract
    try:
        client = BlockchainClient()  # Connects to http://127.0.0.1:7545
        contract = client.deploy_contract()
    except Exception as e:
        print(f"Error connecting to blockchain or deploying contract: {e}")
        print("Please ensure Ganache is running.")
        return

    # 2. Initialize Source Logic
    source_manager = SourceLogic(source_id="GlobalManager")

    # 3. Load and Process Events
    print("Loading news submission data...")
    events = load_data(DATA_FILE)
    if not events:
        print("No events to process.")
        return

    # Sort events by timestamp to process them in order
    events.sort(key=lambda x: x['timestamp'])

    trust_history = defaultdict(list)

    print("Processing events and updating trust scores...")
    for event in events:
        source_id = event['source_id']
        prediction = event['prediction']
        timestamp = event['timestamp']

        # Update trust score in our local model
        new_trust = source_manager.update_trust(source_id, prediction, timestamp)
        trust_history[source_id].append((timestamp, new_trust))

        # Write the latest trust score to the blockchain
        try:
            client.set_trust(source_id, new_trust)
            print(f"Source '{source_id}' at {timestamp}: prediction={'Fake' if prediction == 1 else 'Real'}, trust updated to {new_trust}.")
        except Exception as e:
            print(f"Error writing trust score to blockchain for '{source_id}': {e}")

    # 4. Display Final Trust Scores from Blockchain
    print("\n--- Final Trust Scores on Blockchain ---")
    unique_sources = sorted(list(trust_history.keys()))
    for source_id in unique_sources:
        try:
            score = client.get_trust(source_id)
            print(f"  Source '{source_id}': Trust = {score}")
        except Exception as e:
            print(f"Error retrieving trust score for '{source_id}': {e}")

    # 5. Plot Trust Score Evolution
    if not trust_history:
        print("\nNo data to plot.")
        return
        
    print("\nGenerating trust score plot...")
    plt.figure(figsize=(12, 7))
    for source_id, history in trust_history.items():
        if history:
            timestamps, scores = zip(*history)
            plt.plot(timestamps, scores, marker='o', linestyle='-', label=f'Source: {source_id}')
    
    plt.title("Source Trust Score Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Trust Score (0-100)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    # Save the plot to a file
    plot_filename = "source_trust_scores.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    
    plt.show()

def main():
    process_and_update_blockchain()

if __name__ == "__main__":
    main()