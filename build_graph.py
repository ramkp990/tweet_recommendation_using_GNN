
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