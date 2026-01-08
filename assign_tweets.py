import csv
import random
from datetime import datetime
import torch
import pandas as pd

# Load user IDs (from your Higgs subset)
data = torch.load("higgs_processed.pt", weights_only=False)
user_ids = list(data["user_to_idx"].keys())
num_users = len(user_ids)

# Load synthetic tweets
tweets = []
with open("tweets.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        tweets.append(row[0])

print(f"Loaded {len(tweets)} tweets")

# Ensure we have at least 10,000 interactions
tweets_extended = (tweets * 3)[:10000]  # repeat 3x and take first 10K

# Assign interaction types in order: RT, RE, MT, RT, RE, MT, ...
interaction_cycle = ["RT", "RE", "MT"]
interactions = [interaction_cycle[i % 3] for i in range(10000)]

# Simulate timestamps (e.g., during Higgs boson announcement: July 2012)
start_ts = 1341100000  # Unix timestamp
timestamps = [start_ts + i * 60 for i in range(10000)]  # 1 tweet per minute

# Generate synthetic activity
records = []
for i in range(10000):
    engager = random.choice(user_ids)
    target_user = random.choice(user_ids)
    # Ensure engager != target_user (optional)
    while target_user == engager:
        target_user = random.choice(user_ids)
    
    record = {
        "engager": engager,
        "target_user": target_user,
        "timestamp": timestamps[i],
        "interaction": interactions[i],
        "tweet": tweets_extended[i],
        "post_id": i  # will be global ID later
    }
    records.append(record)

# Save as Higgs-compatible activity file
df = pd.DataFrame(records)
df.to_csv("synthetic_higgs_activity.csv", index=False)
print("Saved 10,000 synthetic interactions.")