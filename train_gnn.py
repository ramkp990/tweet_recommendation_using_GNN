import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv  # SAGEConv works well for heterogeneous graphs

# ----------------------------
# Load and prepare data (your code)
# ----------------------------
data = torch.load("higgs_processed.pt")
data = torch.load("higgs_processed.pt", weights_only=False)
activity_sub = data['activity_sub']

# DIAGNOSTIC: What are the actual post_id values?
print("First 5 post_id in activity_sub:", activity_sub["post_id"].head().values)
print("post_id min:", activity_sub["post_id"].min())
print("post_id max:", activity_sub["post_id"].max())
print("Expected post_id range: 0 to", len(activity_sub) - 1)

# Check post_to_idx keys
post_to_idx = data['post_to_idx']
print("First 5 keys in post_to_idx:", list(post_to_idx.keys())[:5])
print("Are post_id values in post_to_idx keys?", 
      activity_sub["post_id"].isin(post_to_idx.keys()).all())
user_to_idx = data['user_to_idx']
post_to_idx = data['post_to_idx']
x = data['x']
edge_index_social = data['edge_index_social']
activity_sub = data['activity_sub']

activity_sorted = activity_sub.sort_values("timestamp").reset_index(drop=True)
#activity_sorted["post_id"] = activity_sorted.index

n = len(activity_sorted)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_interactions = activity_sorted.iloc[:train_end]
val_interactions = activity_sorted.iloc[train_end:val_end]
test_interactions = activity_sorted.iloc[val_end:]
'''
def build_edge_index(df, user_to_idx, post_to_idx):
    df = df[["engager", "target_user", "post_id"]].copy()
    df["engager"] = df["engager"].map(user_to_idx)
    df["target_user"] = df["target_user"].map(user_to_idx)
    df["post_id"] = df["post_id"].map(post_to_idx)  # local → global (ONLY ONCE!)
    df = df.dropna().astype(int)
    engage = torch.tensor(df[["engager", "post_id"]].values.T, dtype=torch.long)
    author = torch.tensor(df[["post_id", "target_user"]].values.T, dtype=torch.long)
    return engage, author


# Build training edges
edge_index_engage_train, edge_index_author_train = build_edge_index(
    train_interactions, user_to_idx, post_to_idx
)

val_pos_edges, _ = build_edge_index(val_interactions, user_to_idx, post_to_idx)
test_pos_edges, _ = build_edge_index(test_interactions, user_to_idx, post_to_idx)  # ← NOW DEFINED!
'''

def build_edge_index_safe(df, user_to_idx, post_to_idx, num_users):
    engager = []
    post_global = []
    target_user = []

    for _, row in df.iterrows():
        u_eng = user_to_idx.get(row["engager"])
        u_tgt = user_to_idx.get(row["target_user"])
        p_local = row["post_id"]  # this is 0-9999
        p_global = post_to_idx.get(p_local)
        #print(p_global)
        if u_eng is not None and u_tgt is not None and p_global is not None:
            engager.append(u_eng)
            post_global.append(p_global)
            target_user.append(u_tgt)

    engager = torch.tensor(engager, dtype=torch.long)
    post_global = torch.tensor(post_global, dtype=torch.long)
    target_user = torch.tensor(target_user, dtype=torch.long)

    engage_edge = torch.stack([engager, post_global], dim=0)
    author_edge = torch.stack([post_global, target_user], dim=0)
    return engage_edge, author_edge

# Use it
U = len(user_to_idx)
edge_index_engage_train, edge_index_author_train = build_edge_index_safe(
    train_interactions, user_to_idx, post_to_idx, U
)
val_pos_edges, _ = build_edge_index_safe(val_interactions, user_to_idx, post_to_idx, U)
test_pos_edges, _ = build_edge_index_safe(test_interactions, user_to_idx, post_to_idx, U)


print("\n=== DEBUG TEST EDGES ===")
print("test_pos_edges shape:", test_pos_edges.shape)
print("First 5 user (src):", test_pos_edges[0, :5].numpy())
print("First 5 post (dst):", test_pos_edges[1, :5].numpy())

U = len(user_to_idx)
print(U)
print(f"User ID range: 0 to {U - 1}")
print(f"Expected post ID range: {U} to {U + 9999}")

# Check for out-of-range posts
post_ids = test_pos_edges[1].numpy()
invalid_mask = (post_ids < U) | (post_ids >= U + 10000)
if invalid_mask.any():
    print("⚠️  Invalid post IDs found:")
    print("Sample invalid post IDs:", post_ids[invalid_mask][:5])
else:
    print("✅ All post IDs are in valid range.")
# ----------------------------
# Build HeteroData graph
# ----------------------------
# ----------------------------
# Build HeteroData with LOCAL indices
# ----------------------------
graph = HeteroData()


# 1. Node features (already split logically)
num_users = len(user_to_idx)
num_posts = len(post_to_idx)

graph['user'].x = x[:num_users]      # [4967, 3]
graph['post'].x = x[num_users:]      # [10000, 3]

# 2. Convert edge indices to LOCAL IDs
# Social: user (global) → user (global) → convert to local (same as global for users)
# But ensure all user IDs are in [0, num_users)
src, dst = edge_index_social
# Filter edges to only include valid users (should be all, but safe)
mask = (src < num_users) & (dst < num_users)
graph['user', 'social', 'user'].edge_index = torch.stack([src[mask], dst[mask]], dim=0)

# Engagement: user (global) → post (global)
# Convert post global ID → local post ID: global_id - num_users
src, dst = edge_index_engage_train
post_local = dst - num_users
mask = (src < num_users) & (post_local >= 0) & (post_local < num_posts)
graph['user', 'engages', 'post'].edge_index = torch.stack([src[mask], post_local[mask]], dim=0)

# Authorship: post (global) → user (global)
src, dst = edge_index_author_train
post_local = src - num_users
mask = (post_local >= 0) & (post_local < num_posts) & (dst < num_users)
graph['post', 'authored_by', 'user'].edge_index = torch.stack([post_local[mask], dst[mask]], dim=0)

# Optional: add reverse engagement edge
# PyG can auto-add reverse, or do manually:
rev_engage = graph['user', 'engages', 'post'].edge_index.flip(0)  # [post, user]
graph['post', 'rev_engages', 'user'].edge_index = rev_engage


# ----------------------------
# Define 1-layer R-GCN (via HeteroConv)
# ----------------------------
class SimpleRGCN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv1 = HeteroConv({
            ('user', 'social', 'user'): SAGEConv((-1, -1), hidden_dim),
            ('user', 'engages', 'post'): SAGEConv((-1, -1), hidden_dim),
            ('post', 'rev_engages', 'user'): SAGEConv((-1, -1), hidden_dim),
            ('post', 'authored_by', 'user'): SAGEConv((-1, -1), hidden_dim),
        }, aggr='sum')
        self.conv2 = HeteroConv({
            ('user', 'social', 'user'): SAGEConv((-1, -1), hidden_dim),
            ('user', 'engages', 'post'): SAGEConv((-1, -1), hidden_dim),
            ('post', 'rev_engages', 'user'): SAGEConv((-1, -1), hidden_dim),
            ('post', 'authored_by', 'user'): SAGEConv((-1, -1), hidden_dim),
        }, aggr='sum')
        self.hidden_dim = hidden_dim

    def forward(self, x_dict, edge_index_dict):
        # First layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Second layer (optional but helpful)
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        return x_dict

'''
# ----------------------------
# Define 1-layer R-GCN (via HeteroConv)
# ----------------------------
class SimpleRGCN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv = HeteroConv({
            ('user', 'social', 'user'): SAGEConv((-1, -1), hidden_dim),
            ('user', 'engages', 'post'): SAGEConv((-1, -1), hidden_dim),
            ('post', 'rev_engages', 'user'): SAGEConv((-1, -1), hidden_dim),
            ('post', 'authored_by', 'user'): SAGEConv((-1, -1), hidden_dim),
        }, aggr='sum')
        self.hidden_dim = hidden_dim

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return x_dict

# ----------------------------
# Generate embeddings
# ----------------------------

model = SimpleRGCN(hidden_dim=64)
model.eval()  # no training

with torch.no_grad():
    out = model(graph.x_dict, graph.edge_index_dict)
    user_emb = out['user']      # [num_users, 64]
    post_emb = out['post']      # [num_posts, 64]

print("User embeddings shape:", user_emb.shape)
print("Post embeddings shape:", post_emb.shape)

# ----------------------------
# Example: Score a user–post pair
# ----------------------------
# Get test user and post (example: first test interaction)
U = len(user_to_idx)
# Get the first edge: (user, post)
user_idx = 0
test_user_global = test_pos_edges[0, user_idx].item()
test_post_global = test_pos_edges[1, user_idx].item()

print(f"\nGlobal user ID: {test_user_global}")
print(f"Global post ID: {test_post_global}")
print(f"Valid user range: [0, {U - 1}]")
print(f"Valid post range: [{U}, {U + 9999}]")

# Safety
assert 0 <= test_user_global < U, f"User ID {test_user_global} out of range!"
assert U <= test_post_global < U + 10000, f"Post ID {test_post_global} out of range!"

user_local = test_user_global
post_local = test_post_global - U  # now guaranteed ≥ 0

score = torch.dot(user_emb[user_local], post_emb[post_local])
print(f"Score for user {user_local}, post {post_local}: {score:.4f}")
# Save embeddings if needed
torch.save({
    'user_emb': user_emb,
    'post_emb': post_emb,
    'user_to_idx': user_to_idx,
    'post_to_idx': post_to_idx,
    'test_pos_edges': test_pos_edges,
}, "higgs_embeddings.pt")

from sklearn.metrics import ndcg_score
import numpy as np

# 1. Prepare test data structures
U = len(user_to_idx)
test_edges = test_pos_edges  # shape [2, num_test_edges]

# Group test interactions by user
from collections import defaultdict
user_test_posts = defaultdict(list)
for i in range(test_edges.shape[1]):
    u = test_edges[0, i].item()
    p_global = test_edges[1, i].item()
    p_local = p_global - U  # convert to local post index
    user_test_posts[u].append(p_local)

# Candidate set: all posts in test period (local indices)
# Since test posts have global IDs from U to U+9999, local = 0 to 9999
# But to be precise, extract from test_edges:
candidate_posts_local = set()
for i in range(test_edges.shape[1]):
    p_global = test_edges[1, i].item()
    candidate_posts_local.add(p_global - U)
candidate_posts_local = sorted(candidate_posts_local)
candidate_posts_local = torch.tensor(candidate_posts_local, dtype=torch.long)

print(f"Number of candidate posts: {len(candidate_posts_local)}")

# 2. Evaluate per user
recall_at_k = []
ndcg_at_k = []
K = 10

for user_id in user_test_posts:
    if user_id >= U:
        continue  # safety

    # Ground truth
    true_posts = user_test_posts[user_id]
    if len(true_posts) == 0:
        continue

    # Score all candidate posts for this user
    user_emb_i = user_emb[user_id].unsqueeze(0)  # [1, 64]
    candidate_embs = post_emb[candidate_posts_local]  # [num_candidates, 64]
    scores = torch.mm(user_emb_i, candidate_embs.T).squeeze(0)  # [num_candidates]

    # Get top-K predicted posts (local indices)
    topk_indices = torch.topk(scores, min(K, len(scores)))[1]
    topk_posts = candidate_posts_local[topk_indices].tolist()

    # Build binary relevance vector for NDCG
    relevance = torch.zeros(len(candidate_posts_local))
    for p in true_posts:
        if p in candidate_posts_local:
            idx = (candidate_posts_local == p).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                relevance[idx.item()] = 1.0

    # Recall@K
    hits = len(set(topk_posts) & set(true_posts))
    recall = hits / len(true_posts)
    recall_at_k.append(recall)

    # NDCG@K
    if relevance.sum() > 0:
        ndcg = ndcg_score(
            relevance.numpy().reshape(1, -1),
            scores.numpy().reshape(1, -1),
            k=K
        )
        ndcg_at_k.append(ndcg)

# 3. Final metrics
final_recall = np.mean(recall_at_k) if recall_at_k else 0.0
final_ndcg = np.mean(ndcg_at_k) if ndcg_at_k else 0.0

print(f"\n=== EVALUATION RESULTS ===")
print(f"Recall@{K}: {final_recall:.4f}")
print(f"NDCG@{K}: {final_ndcg:.4f}")
print(f"Evaluated {len(recall_at_k)} users")
'''

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import negative_sampling
import random

# ----------------------------
# Training Setup
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleRGCN(hidden_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

# Move graph to device
graph = graph.to(device)
x_dict = {'user': graph['user'].x, 'post': graph['post'].x}

# Get training engagement edges (user → post, local indices)
train_edge_index = graph['user', 'engages', 'post'].edge_index  # [2, num_train]

# Total number of users and posts
num_users = graph['user'].num_nodes
num_posts = graph['post'].num_nodes

print(f"Training on {train_edge_index.shape[1]} positive edges")
print(f"Users: {num_users}, Posts: {num_posts}")

# ----------------------------
# Training Function
# ----------------------------
def train():
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(x_dict, graph.edge_index_dict)
    user_emb = out['user']  # [num_users, 64]
    post_emb = out['post']  # [num_posts, 64]

    # Positive scores
    pos_u, pos_p = train_edge_index
    pos_scores = (user_emb[pos_u] * post_emb[pos_p]).sum(dim=1)  # [num_pos]

    # Negative sampling: for each positive, sample 1 negative post
    neg_p = torch.randint(0, num_posts, (pos_p.size(0),), device=device)
    neg_scores = (user_emb[pos_u] * post_emb[neg_p]).sum(dim=1)  # [num_pos]

    # Loss
    pos_loss = criterion(pos_scores, torch.ones_like(pos_scores))
    neg_loss = criterion(neg_scores, torch.zeros_like(neg_scores))
    loss = pos_loss + neg_loss

    loss.backward()
    optimizer.step()
    return loss.item()

# ----------------------------
# Evaluation Function (Recall@10, NDCG@10)
# ----------------------------
def evaluate(test_edges, user_emb, post_emb, K=10):
    from collections import defaultdict
    import numpy as np
    from sklearn.metrics import ndcg_score

    # Group test by user
    user_test_posts = defaultdict(list)
    candidate_posts = set()
    
    for i in range(test_edges.shape[1]):
        u = test_edges[0, i].item()
        p_global = test_edges[1, i].item()
        p_local = p_global - num_users
        user_test_posts[u].append(p_local)
        candidate_posts.add(p_local)
    
    candidate_posts = sorted(candidate_posts)
    candidate_posts = torch.tensor(candidate_posts, device=device)
    
    recall_list, ndcg_list = [], []
    
    for user_id in user_test_posts:
        if user_id >= num_users:
            continue
        true_posts = user_test_posts[user_id]
        if not true_posts:
            continue

        # Score all candidates
        scores = torch.mm(
            user_emb[user_id].unsqueeze(0),
            post_emb[candidate_posts].T
        ).squeeze(0)

        # Top-K
        topk_idx = torch.topk(scores, min(K, len(scores)))[1]
        topk_posts = candidate_posts[topk_idx].cpu().tolist()

        # Recall@K
        hits = len(set(topk_posts) & set(true_posts))
        recall = hits / len(true_posts)
        recall_list.append(recall)

        # NDCG@K
        relevance = torch.zeros(len(candidate_posts), device=device)
        for p in true_posts:
            if p in candidate_posts:
                idx = (candidate_posts == p).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    relevance[idx] = 1.0
        if relevance.sum() > 0:
            ndcg = ndcg_score(
                relevance.cpu().numpy().reshape(1, -1),
                scores.cpu().numpy().reshape(1, -1),
                k=K
            )
            ndcg_list.append(ndcg)

    return np.mean(recall_list), np.mean(ndcg_list)

# ----------------------------
# Training Loop
# ----------------------------
print("\n=== START TRAINING ===")
best_recall = 0.0
patience = 20
patience_counter = 0

for epoch in range(1, 101):  # 100 epochs
    loss = train()
    
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            out = model(x_dict, graph.edge_index_dict)
            user_emb = out['user']
            post_emb = out['post']
            
            # Move test edges to device for consistency
            test_edges_device = test_pos_edges.to(device)
            recall, ndcg = evaluate(test_edges_device, user_emb, post_emb, K=10)
            
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Recall@10: {recall:.4f} | NDCG@10: {ndcg:.4f}")
            
            # Early stopping
            if recall > best_recall:
                best_recall = recall
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), "best_rgcn_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

# ----------------------------
# Final Evaluation with Best Model
# ----------------------------
print("\n=== FINAL EVALUATION ===")
model.load_state_dict(torch.load("best_rgcn_model.pt"))
model.eval()
with torch.no_grad():
    out = model(x_dict, graph.edge_index_dict)
    user_emb = out['user']
    post_emb = out['post']
    test_edges_device = test_pos_edges.to(device)
    recall, ndcg = evaluate(test_edges_device, user_emb, post_emb, K=10)
    print(f"Best Recall@10: {recall:.4f}")
    print(f"Best NDCG@10: {ndcg:.4f}")

# Save final embeddings
torch.save({
    'user_emb': user_emb.cpu(),
    'post_emb': post_emb.cpu(),
    'user_to_idx': user_to_idx,
    'num_users': num_users,
}, "higgs_embeddings_trained.pt")