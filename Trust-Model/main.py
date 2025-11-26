from blockchain_client import BlockchainClient
from rsu_logic import RSULogic
from data_loader import load_data
from collections import defaultdict
import matplotlib.pyplot as plt

def main():
    
    client = BlockchainClient()  # default localhost Ganache

    
    contract = client.deploy_contract()

    # Map vehicle IDs (string) to Ethereum addresses (Ganache accounts)
    accounts = client.w3.eth.accounts  # Ganache provides a list of accounts
    vehicle_to_address = {}
    for vid in range(1, 11):  # Vehicle IDs 1–10
        vehicle_to_address[str(vid)] = accounts[vid - 1]

    # Initialize RSU logic
    rsu = RSULogic(rsu_id="RSU-1",decay_rate=0.1, reward_factor=0.1, penalty_factor=0.5)

    # Load dataset
    print("Loading vehicle behavior data...")
    events = load_data("data/vehicle_data.csv")

    # Track trust scores for plotting
    trust_history = defaultdict(list)

    print("Processing events and updating trust...")
    for event in events:
        vid = event['vehicle_id']
        behavior = event['behavior']
        timestamp = event['timestamp']
        # Update trust
        new_trust = rsu.update_trust(vid, behavior, timestamp)
        trust_history[vid].append((timestamp, new_trust))  # ⬅ store history
        # Write to blockchain
        vehicle_addr = vehicle_to_address.get(vid)
        if vehicle_addr:
            client.set_trust(vehicle_addr, new_trust)
            print(f"Vehicle {vid} at {timestamp}: behavior={behavior}, trust updated to {new_trust}.")
        else:
            print(f"Unknown vehicle ID {vid} in data.")

    print("\nFinal trust scores on blockchain:")
    for vid, addr in vehicle_to_address.items():
        score = client.get_trust(addr)
        print(f"  Vehicle {vid} (address {addr}): Trust = {score}")

    # Plot trust evolution for selected vehicles (on-off attackers)
    print("\nGenerating trust score plot...")
    plt.figure(figsize=(12, 6))
    for vid in ['1','2','4','5','6','8','3', '7']:  # Customize list to plot specific vehicles
        if vid in trust_history:
            timestamps, scores = zip(*trust_history[vid])
            plt.plot(timestamps, scores, label=f'Vehicle {vid}')
    plt.title("Trust Score Over Time (On-Off Attackers)")
    plt.xlabel("Timestamp")
    plt.ylabel("Trust Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("trust_scores.png")  # Save image (optional)
    plt.show()

if __name__ == "__main__":
    main()
