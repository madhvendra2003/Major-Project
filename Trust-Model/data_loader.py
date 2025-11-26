import csv
from datetime import datetime

def load_data(csv_file):
    events = []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse timestamp string into a datetime object
            row_ts = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
            events.append({
                'timestamp': row_ts,
                'vehicle_id': row['vehicle_id'],
                'behavior': int(row['behavior'])
            })
    # Sort events chronologically
    events.sort(key=lambda r: r['timestamp'])
    return events
