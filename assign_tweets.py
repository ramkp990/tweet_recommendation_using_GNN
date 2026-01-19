
'''
import csv
import random
from datetime import datetime
import pandas as pd

# ----------------------------
# 1. Load top 5,000 users from real Higgs activity
# ----------------------------
activity = pd.read_csv(
    "higgs_activity_time.txt", 
    sep=" ", 
    header=None, 
    names=["engager", "target_user", "timestamp", "interaction"]
)

user_counts = pd.concat([activity["engager"], activity["target_user"]]).value_counts()
top_users = user_counts.head(5000).index.tolist()
print(f"Loaded {len(top_users)} top users")

# ----------------------------
# 2. Read topics from file
# ----------------------------
def read_topics_from_file(filename):
    topics_prompts = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # Split by semicolon and clean up
            tweets = [tweet.strip() for tweet in line.strip().split(';') if tweet.strip()]
            if tweets:  # Only add non-empty lists
                topics_prompts.append(tweets)
    return topics_prompts

# Read topics from your file (replace 'tweets.txt' with your filename)
topics_prompts = read_topics_from_file("tweets_new.txt")
print(f"Loaded {len(topics_prompts)} topics")

# ----------------------------
# 3. Assign users to topics (no overlap)
# ----------------------------
users_per_topic = 500
topic_to_users = {}

# Shuffle users to ensure random assignment
random.shuffle(top_users)

# Check we have enough users
required_users = users_per_topic * len(topics_prompts)
if required_users > len(top_users):
    print(f"Warning: Need {required_users} users but only have {len(top_users)}")
    # Adjust users_per_topic if needed
    users_per_topic = len(top_users) // len(topics_prompts)
    print(f"Adjusting to {users_per_topic} users per topic")

# Assign non-overlapping users to each topic
for topic_id in range(len(topics_prompts)):
    start = topic_id * users_per_topic
    end = start + users_per_topic
    topic_to_users[topic_id] = top_users[start:end]
    
    # Verify no overlap with previous topics
    for prev_topic in range(topic_id):
        overlap = set(topic_to_users[topic_id]) & set(topic_to_users[prev_topic])
        if overlap:
            print(f"Error: Overlap found between topic {prev_topic} and {topic_id}: {overlap}")

print(f"Assigned {users_per_topic} unique users to each of {len(topics_prompts)} topics")

# ----------------------------
# 4. Generate tweets from prompts (no positive/negative templates)
# ----------------------------
topic_to_tweets = {}
for topic_id, prompts in enumerate(topics_prompts):
    # Use prompts directly as tweets (no template formatting)
    tweets = prompts  # Direct assignment since we removed templates
    topic_to_tweets[topic_id] = tweets
    print(f"Topic {topic_id}: {len(tweets)} tweets")

# ----------------------------
# 5. Generate synthetic interactions
# ----------------------------
records = []
start_ts = 1341100000  # Higgs announcement timestamp
interaction_types = ["RT", "RE", "MT"]

# Track which tweet index we're at for each topic
current_tweet_index = {topic_id: 0 for topic_id in range(len(topics_prompts))}

# Number of interactions to generate
num_interactions = 10000

for i in range(num_interactions):
    # Cycle through topics
    topic_id = i % len(topics_prompts)
    tweets_list = topic_to_tweets[topic_id]
    
    # Pick next tweet (cycle if needed)
    tweet = tweets_list[current_tweet_index[topic_id]]
    current_tweet_index[topic_id] = (current_tweet_index[topic_id] + 1) % len(tweets_list)
    
    # Assign interaction type
    interaction = random.choice(interaction_types)
    timestamp = start_ts + i * 60
    post_id = i

    # Pick engager and target from this topic's users
    users_pool = topic_to_users[topic_id]
    engager = random.choice(users_pool)
    target_user = random.choice(users_pool)
    
    # Make sure engager and target are different (optional but realistic)
    while target_user == engager:
        target_user = random.choice(users_pool)

    records.append({
        "engager": engager,
        "target_user": target_user,
        "timestamp": timestamp,
        "interaction": interaction,
        "tweet": tweet,
        "post_id": post_id
    })

# ----------------------------
# 6. Verify no user appears in multiple topics
# ----------------------------
user_to_topic = {}
for topic_id, users in topic_to_users.items():
    for user in users:
        if user in user_to_topic:
            print(f"Error: User {user} appears in both topic {user_to_topic[user]} and topic {topic_id}")
        else:
            user_to_topic[user] = topic_id

print(f"Verification complete. {len(user_to_topic)} users uniquely assigned to topics.")

# ----------------------------
# 7. Save to CSV
# ----------------------------
with open("synthetic_higgs_activity_new.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["engager", "target_user", "timestamp", "interaction", "tweet", "post_id"])
    writer.writeheader()
    writer.writerows(records)

print(f"✅ Saved {len(records)} synthetic interactions")
print(f"✅ Each of {len(topics_prompts)} topics has {users_per_topic} unique users")
print(f"✅ Total unique users used: {len(user_to_topic)}")


import pandas as pd
import random

# ----------------------------
# 1. Load original Higgs activity log
# ----------------------------
print("Loading original Higgs activity log...")
activity = pd.read_csv(
    "higgs_activity_time.txt",
    sep=" ",
    header=None,
    names=["engager", "target_user", "timestamp", "interaction"]
)
print(f"Loaded {len(activity)} interactions")

# ----------------------------
# 2. Get all unique users (both engagers and target_users)
# ----------------------------
all_users = sorted(set(activity["engager"]) | set(activity["target_user"]))
print(f"Found {len(all_users)} unique users")

# ----------------------------
# 3. Load topics from file
# ----------------------------
def read_topics_from_file(filename):
    topics_prompts = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tweets = [t.strip() for t in line.strip().split(';') if t.strip()]
            if tweets:
                topics_prompts.append(tweets)
    return topics_prompts

topics_prompts = read_topics_from_file("tweets_new.txt")
num_topics = len(topics_prompts)
print(f"Loaded {num_topics} topics")

# ----------------------------
# 4. Assign each USER (not just author) to ONE topic
# ----------------------------
random.seed(42)

# Shuffle users for fair assignment
shuffled_users = all_users.copy()
random.shuffle(shuffled_users)

# Assign round-robin
user_to_topic = {}
for i, user in enumerate(shuffled_users):
    topic_id = i % num_topics
    user_to_topic[user] = topic_id

print(f"Assigned each user to one of {num_topics} topics")

# ----------------------------
# 5. Filter interactions: keep only those where engager and target_user share the same topic
# ----------------------------
def same_topic(row):
    engager = row["engager"]
    target = row["target_user"]
    # Skip if either user is missing (shouldn't happen)
    if engager not in user_to_topic or target not in user_to_topic:
        return False
    return user_to_topic[engager] == user_to_topic[target]

print("Filtering interactions to enforce topic consistency...")
consistent_mask = activity.apply(same_topic, axis=1)
activity_filtered = activity[consistent_mask].copy()
print(f"Kept {len(activity_filtered)} / {len(activity)} interactions with topic-consistent users")

# ----------------------------
# 6. Add synthetic tweet based on target_user's topic
# ----------------------------
tweet_index = {i: 0 for i in range(num_topics)}

def get_tweet_for_author(author):
    topic_id = user_to_topic[author]  # author = target_user
    tweets = topics_prompts[topic_id]
    idx = tweet_index[topic_id]
    tweet = tweets[idx]
    tweet_index[topic_id] = (idx + 1) % len(tweets)
    return tweet

print("Adding synthetic tweets...")
activity_filtered["tweet"] = activity_filtered["target_user"].apply(get_tweet_for_author)
activity_filtered["post_id"] = activity_filtered.index

# ----------------------------
# 7. Save enriched, topic-consistent activity log
# ----------------------------
output_file = "synthetic_higgs_activity_with_semantics.csv"
activity_filtered.to_csv(output_file, index=False)
print(f"✅ Saved topic-consistent enriched activity log to {output_file}")
print(f"   Columns: {list(activity_filtered.columns)}")
print(f"   Sample tweet: \"{activity_filtered.iloc[0]['tweet']}\"")


import pandas as pd
import networkx as nx
import community as community_louvain  # python-louvain
import random
import json

# ----------------------------
# 1. Load data
# ----------------------------
print("Loading social network...")
social = pd.read_csv(
    "higgs_social_network.edgelist", 
    sep=" ", 
    header=None, 
    names=["follower", "followee"]
)

activity = pd.read_csv(
    "higgs_activity_time.txt",
    sep=" ",
    header=None,
    names=["engager", "target_user", "timestamp", "interaction"]
)

# Get top 5K users (as before)
user_counts = pd.concat([activity["engager"], activity["target_user"]]).value_counts()
top_users = set(user_counts.head(5000).index)
print(f"Using {len(top_users)} top users")

# Filter social graph to top users
social_sub = social[
    social["follower"].isin(top_users) & 
    social["followee"].isin(top_users)
]
print(f"Social edges among top users: {len(social_sub)}")

# ----------------------------
# 2. Build NetworkX graph
# ----------------------------
G = nx.DiGraph()  # Twitter is directed
G.add_nodes_from(top_users)
G.add_edges_from(social_sub[["follower", "followee"]].values)

# Convert to undirected for clustering (standard practice)
G_undir = G.to_undirected()

print(f"Graph nodes: {G_undir.number_of_nodes()}")
print(f"Graph edges: {G_undir.number_of_edges()}")

# ----------------------------
# 3. Run Louvain clustering
# ----------------------------
print("Running Louvain clustering...")
partition = community_louvain.best_partition(G_undir, random_state=42)

# Map: user_id → cluster_id
user_to_cluster = partition
num_clusters = len(set(partition.values()))
print(f"Found {num_clusters} communities")

# ----------------------------
# 4. Load topics
# ----------------------------
def read_topics_from_file(filename):
    topics_prompts = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tweets = [t.strip() for t in line.strip().split(';') if t.strip()]
            if tweets:
                topics_prompts.append(tweets)
    return topics_prompts

topics_prompts = read_topics_from_file("tweets_new.txt")
num_topics = len(topics_prompts)
print(f"Loaded {num_topics} topics")

# ----------------------------
# 5. Assign each CLUSTER to a TOPIC (round-robin)
# ----------------------------
cluster_ids = sorted(set(user_to_cluster.values()))
random.seed(42)
random.shuffle(cluster_ids)

cluster_to_topic = {}
for i, cid in enumerate(cluster_ids):
    cluster_to_topic[cid] = i % num_topics

print(f"Assigned {num_clusters} clusters to {num_topics} topics")

# ----------------------------
# 6. Derive user_to_topic
# ----------------------------
user_to_topic = {}
for user, cluster in user_to_cluster.items():
    user_to_topic[user] = cluster_to_topic[cluster]

# ----------------------------
# 7. Save mappings
# ----------------------------
with open("user_to_topic.json", "w") as f:
    json.dump(user_to_topic, f)

with open("cluster_to_topic.json", "w") as f:
    json.dump({str(k): v for k, v in cluster_to_topic.items()}, f)

print("✅ Saved user_to_topic.json and cluster_to_topic.json")

# ----------------------------
# 8. Optional: Print cluster stats
# ----------------------------
from collections import Counter
topic_counts = Counter(user_to_topic.values())
for topic_id in range(num_topics):
    print(f"Topic {topic_id}: {topic_counts[topic_id]} users")
    '''
# clustering_all_users.py
# generate_full_synthetic.py
import pandas as pd
import json

# Load full activity log
activity = pd.read_csv(
    "higgs_activity_time.txt",
    sep=" ",
    header=None,
    names=["engager", "target_user", "timestamp", "interaction"]
)
print(f"Loaded {len(activity)} interactions")

# Load full user_to_topic
with open("user_to_topic_full.json", "r") as f:
    user_to_topic = json.load(f)
user_to_topic = {int(k): int(v) for k, v in user_to_topic.items()}
print(f"Loaded topics for {len(user_to_topic)} users")

# Optional: Filter out any row where engager or target_user is missing (shouldn't happen)
mask = activity["engager"].isin(user_to_topic) & activity["target_user"].isin(user_to_topic)
activity_clean = activity[mask].copy()
print(f"Kept {len(activity_clean)} interactions with known users")

# Load topics
def read_topics(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip().split(';') for line in f if line.strip()]

topics_prompts = read_topics("tweets_new.txt")
tweet_index = {i: 0 for i in range(len(topics_prompts))}

def get_tweet(author):
    topic_id = user_to_topic[author]
    tweets = topics_prompts[topic_id]
    idx = tweet_index[topic_id]
    tweet = tweets[idx]
    tweet_index[topic_id] = (idx + 1) % len(tweets)
    return tweet

# Add synthetic tweets
activity_clean["tweet"] = activity_clean["target_user"].apply(get_tweet)
activity_clean["post_id"] = activity_clean.index

# Save
output_file = "synthetic_higgs_full_all_users.csv"
activity_clean.to_csv(output_file, index=False)
print(f"✅ Saved {len(activity_clean)} interactions — NO 'Default tweet.'")