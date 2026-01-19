# clustering_all_users.py
import pandas as pd
import networkx as nx
import community as community_louvain
import json

# Load full social graph
social = pd.read_csv(
    "higgs_social_network.edgelist", 
    sep=" ", 
    header=None, 
    names=["follower", "followee"]
)

# Get ALL unique users from social graph
all_users = set(social["follower"]) | set(social["followee"])
print(f"Total users in social graph: {len(all_users)}")

# Build graph with all users
G = nx.Graph()  # undirected for Louvain
G.add_nodes_from(all_users)
G.add_edges_from(social[["follower", "followee"]].values)

# Run Louvain
print("Running Louvain on full graph...")
partition = community_louvain.best_partition(G, random_state=42)

# Save user_to_cluster
user_to_cluster = partition
num_clusters = len(set(partition.values()))
print(f"Found {num_clusters} communities")

# Load topics
def read_topics(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip().split(';') for line in f if line.strip()]

topics_prompts = read_topics("tweets_new.txt")
num_topics = len(topics_prompts)

# Assign each cluster to a topic (round-robin)
cluster_ids = sorted(set(user_to_cluster.values()))
cluster_to_topic = {cid: i % num_topics for i, cid in enumerate(cluster_ids)}

# Derive user_to_topic
user_to_topic = {user: cluster_to_topic[cluster] for user, cluster in user_to_cluster.items()}

# Save
with open("user_to_topic_full.json", "w") as f:
    json.dump(user_to_topic, f)

print(f"✅ Assigned {len(user_to_topic)} users to {num_topics} topics")