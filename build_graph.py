'''
import pandas as pd
import numpy as np
import torch


# ----------------------------
# 1. Load full data
# ----------------------------
social = pd.read_csv("higgs_social_network.edgelist", sep=" ", header=None, names=["follower", "followee"])
activity = pd.read_csv("higgs_activity_time.txt", sep=" ", header=None, names=["engager", "target_user", "timestamp", "interaction"])

# Assign a unique post ID to each interaction (each row = one "post")
activity["post_id"] = activity.index

# ----------------------------
# 2. Focus on top N most active users (e.g., top 5,000)
# ----------------------------
# Count how many times each user appears (as engager OR target)
user_counts = pd.concat([activity["engager"], activity["target_user"]]).value_counts()
top_users = user_counts.head(5000).index  # top 5K most active users

# Filter activity to only involve top users
activity_sub = activity[
    activity["engager"].isin(top_users) &
    activity["target_user"].isin(top_users)
].copy()

# Keep only the first 10,000 interactions (to limit # of "posts")
#activity_sub = activity_sub.reset_index(drop=True)
#activity_sub["post_id"] = activity_sub.index + len(top_users)  # global ID
# Assign post_id as simple 0,1,2,... within the subset
activity_sub = activity_sub.reset_index(drop=True)
activity_sub["post_id"] = activity_sub.index  # 0 to 9999

# ----------------------------
# 3. Subset social graph to only these users
# ----------------------------
social_sub = social[
    social["follower"].isin(top_users) &
    social["followee"].isin(top_users)
].copy()


# Unique users and posts
users = sorted(set(activity_sub["engager"]) | set(activity_sub["target_user"]) | set(social_sub["follower"]) | set(social_sub["followee"]))
posts = sorted(activity_sub["post_id"].unique())

user_to_idx = {user: i for i, user in enumerate(users)}
#post_to_idx = {post: i + len(users) for i, post in enumerate(posts)}
U = len(users)
post_to_idx = {local_id: U + local_id for local_id in activity_sub["post_id"]}

U = len(users)
P = len(posts)
print(f"Total nodes: {U + P} (Users: {U}, Posts: {P})")

social_edges = social_sub.copy()
social_edges["follower"] = social_edges["follower"].map(user_to_idx)
social_edges["followee"] = social_edges["followee"].map(user_to_idx)
# Remove any NaN (in case some users were filtered out)
social_edges = social_edges.dropna().astype(int)

edge_index_social = torch.tensor(social_edges[["follower", "followee"]].values.T, dtype=torch.long)
print("Social edges shape:", edge_index_social.shape)

engagement = activity_sub[["engager", "post_id"]].copy()
engagement["engager"] = engagement["engager"].map(user_to_idx)
engagement["post_id"] = engagement["post_id"].map(post_to_idx)
engagement = engagement.dropna().astype(int)

edge_index_engage = torch.tensor(engagement[["engager", "post_id"]].values.T, dtype=torch.long)
print("Engagement edges shape:", edge_index_engage.shape)

authorship = activity_sub[["post_id", "target_user"]].copy()
authorship["post_id"] = authorship["post_id"].map(post_to_idx)
authorship["target_user"] = authorship["target_user"].map(user_to_idx)
authorship = authorship.dropna().astype(int)

edge_index_author = torch.tensor(authorship[["post_id", "target_user"]].values.T, dtype=torch.long)
print("Authorship edges shape:", edge_index_author.shape)


##user feature

# Initialize arrays
in_social = np.zeros(U)
out_social = np.zeros(U)
engagement_count = np.zeros(U)

# Social degrees
for _, row in social_sub.iterrows():
    f = user_to_idx.get(row["follower"])
    t = user_to_idx.get(row["followee"])
    if f is not None and t is not None:
        out_social[f] += 1
        in_social[t] += 1

# Engagement count (how many posts user initiated)
eng_counts = activity_sub["engager"].map(user_to_idx).value_counts()
for user_idx, count in eng_counts.items():
    if user_idx in user_to_idx.values():
        engagement_count[user_idx] = count

# Log-transform + stack
user_features = np.stack([
    np.log(in_social + 1),
    np.log(out_social + 1),
    np.log(engagement_count + 1)
], axis=1).astype(np.float32)

user_features = torch.tensor(user_features, dtype=torch.float)

## post feature

interaction_map = {"RT": 0, "RE": 1, "MT": 2}
post_interactions = activity_sub.set_index("post_id")["interaction"].map(interaction_map)
post_features = torch.zeros(P, 3)
for post_id, inter in post_interactions.items():
    idx = post_to_idx[post_id] - U  # local post index
    post_features[idx, inter] = 1.0

x = torch.cat([user_features, post_features], dim=0)  # shape: [U+P, feat_dim]
print("Node feature matrix shape:", x.shape)

graph = {
    "num_users": U,
    "num_posts": P,
    "x": x,  # node features
    "edge_index_dict": {
        ("user", "social", "user"): edge_index_social,
        ("user", "engages", "post"): edge_index_engage,
        ("post", "authored_by", "user"): edge_index_author,
    },
    "user_to_idx": user_to_idx,
    "post_to_idx": post_to_idx,
}
#post_features = x[4967:]  # posts start after user nodes
#print(post_features.sum(dim=1).unique())  # should be all 1.0
#print(post_features[:5])  # see first few


#torch.save(graph, "higgs_subset_graph.pt")

torch.save({
    'users': users,
    'posts': posts,
    'user_to_idx': user_to_idx,
    'post_to_idx': post_to_idx,
    'x': x,
    'edge_index_social': edge_index_social,
    'activity_sub': activity_sub  # keep for temporal split later
}, "higgs_processed.pt")
'''
# ONLY THIS CODE — SAVE AS prepare_data.py AND RUN IT ONCE
import pandas as pd
import numpy as np
import torch

social = pd.read_csv("higgs_social_network.edgelist", sep=" ", header=None, names=["follower", "followee"])
activity = pd.read_csv("higgs_activity_time.txt", sep=" ", header=None, names=["engager", "target_user", "timestamp", "interaction"])

user_counts = pd.concat([activity["engager"], activity["target_user"]]).value_counts()
top_users = user_counts.head(5000).index

activity_sub = activity[
    activity["engager"].isin(top_users) & activity["target_user"].isin(top_users)
].copy().head(10000).reset_index(drop=True)

# CRITICAL: post_id = 0,1,2,...,9999 (local)
activity_sub["post_id"] = activity_sub.index

social_sub = social[
    social["follower"].isin(top_users) & social["followee"].isin(top_users)
].copy()

users = sorted(set(activity_sub["engager"]) | set(activity_sub["target_user"]) | set(social_sub["follower"]) | set(social_sub["followee"]))
U = len(users)
P = len(activity_sub)

user_to_idx = {u: i for i, u in enumerate(users)}
# post_to_idx: local (0-9999) → global (U to U+9999)
post_to_idx = {i: U + i for i in range(P)}

# Build edges (global IDs)
social_mapped = social_sub.copy()
social_mapped["follower"] = social_mapped["follower"].map(user_to_idx)
social_mapped["followee"] = social_mapped["followee"].map(user_to_idx)
social_mapped = social_mapped.dropna().astype(int)

edge_index_social = torch.tensor(social_mapped[["follower", "followee"]].values.T, dtype=torch.long)

engagement = activity_sub[["engager", "post_id"]].copy()
engagement["engager"] = engagement["engager"].map(user_to_idx)
engagement["post_id"] = engagement["post_id"].map(post_to_idx)  # local→global
engagement = engagement.dropna().astype(int)
edge_index_engage = torch.tensor(engagement[["engager", "post_id"]].values.T, dtype=torch.long)

authorship = activity_sub[["post_id", "target_user"]].copy()
authorship["post_id"] = authorship["post_id"].map(post_to_idx)
authorship["target_user"] = authorship["target_user"].map(user_to_idx)
authorship = authorship.dropna().astype(int)
edge_index_author = torch.tensor(authorship[["post_id", "target_user"]].values.T, dtype=torch.long)

# Features
in_social = np.zeros(U); out_social = np.zeros(U); engagement_count = np.zeros(U)
for _, row in social_mapped.iterrows():
    f, t = int(row["follower"]), int(row["followee"])
    out_social[f] += 1; in_social[t] += 1

eng_counts = activity_sub["engager"].map(user_to_idx).value_counts()
for uid, cnt in eng_counts.items():
    engagement_count[int(uid)] = cnt

user_features = torch.tensor(np.stack([
    np.log(in_social + 1), np.log(out_social + 1), np.log(engagement_count + 1)
], axis=1), dtype=torch.float)

post_features = torch.zeros(P, 3)
interaction_map = {"RT": 0, "RE": 1, "MT": 2}
for i, inter in enumerate(activity_sub["interaction"]):
    post_features[i, interaction_map[inter]] = 1.0

x = torch.cat([user_features, post_features], dim=0)

torch.save({
    'user_to_idx': user_to_idx,
    'post_to_idx': post_to_idx,
    'x': x,
    'edge_index_social': edge_index_social,
    'activity_sub': activity_sub,
    'num_users': U,
    'num_posts': P,
}, "higgs_processed.pt")