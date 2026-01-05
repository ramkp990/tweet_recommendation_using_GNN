import pandas as pd

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
activity_sub = activity_sub.head(10000).reset_index(drop=True)
activity_sub["post_id"] = activity_sub.index  # new compact post IDs

# ----------------------------
# 3. Subset social graph to only these users
# ----------------------------
social_sub = social[
    social["follower"].isin(top_users) &
    social["followee"].isin(top_users)
].copy()

# ----------------------------
# 4. Final user and post node sets
# ----------------------------
users = sorted(set(top_users))
posts = sorted(activity_sub["post_id"].unique())

print(f"Users: {len(users)}")
print(f"Posts: {len(posts)}")
print(f"Social edges: {len(social_sub)}")
print(f"Interactions: {len(activity_sub)}")