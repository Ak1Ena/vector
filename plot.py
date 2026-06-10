import matplotlib.pyplot as plt
import numpy as np
import random

def get_clusters(embeddings, threshold=0.3):
    """Simple distance-based clustering using NumPy."""
    words = sorted(embeddings.keys())
    vecs = np.array([embeddings[w] for w in words])
    
    cluster_map = {}
    current_cluster = 0
    assigned = set()
    
    for i in range(len(words)):
        if words[i] in assigned: continue
        
        cluster_map[words[i]] = current_cluster
        assigned.add(words[i])
        
        for j in range(i + 1, len(words)):
            if words[j] in assigned: continue
            dist = np.linalg.norm(vecs[i] - vecs[j])
            if dist < threshold:
                cluster_map[words[j]] = current_cluster
                assigned.add(words[j])
        current_cluster += 1
    return cluster_map, current_cluster

def plot_embeddings(embeddings):
    random.seed(42)
    plt.figure(figsize=(12, 9))
    
    # Identify groups mathematically
    cluster_map, num_clusters = get_clusters(embeddings)
    words = sorted(embeddings.keys())
    cmap = plt.cm.get_cmap('Set2')

    # Pre-calculate jittered positions so lines and points match
    pos = {}
    jitter = 0.015
    for word in words:
        vec = embeddings[word]
        pos[word] = (
            vec[0] + random.uniform(-jitter, jitter),
            vec[1] + random.uniform(-jitter, jitter)
        )

    # 1. Draw Relation Lines (Connections)
    for c_id in range(num_clusters):
        group = [w for w in words if cluster_map[w] == c_id]
        if len(group) > 1:
            color = cmap(c_id % 8)
            # Use the first word as a central "hub" for the group
            hub_x, hub_y = pos[group[0]]
            for i in range(1, len(group)):
                target_x, target_y = pos[group[i]]
                plt.plot([hub_x, target_x], [hub_y, target_y], 
                         color=color, alpha=0.3, linestyle='-', linewidth=2, zorder=1)

    # 2. Draw Points and Labels
    for i, word in enumerate(words):
        x, y = pos[word]
        cluster_id = cluster_map[word]
        color = cmap(cluster_id % 8)

        # Plot point
        plt.scatter(x, y, color=color, alpha=1.0, edgecolors='k', s=200, zorder=2)
        
        # Labels
        angle = (i * (360 / min(len(words), 8))) % 360
        rad = np.deg2rad(angle)
        dist = 14
        plt.annotate(
            word, 
            (x, y), 
            textcoords="offset points", 
            xytext=(dist * np.cos(rad), dist * np.sin(rad)), 
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec=color, lw=2),
            zorder=3
        )

    plt.xlabel("Semantic Dimension 1")
    plt.ylabel("Semantic Dimension 2")
    plt.title("Word Embeddings: Visual Relation Network")
    plt.grid(True, linestyle=':', alpha=0.4)
    
    plt.figtext(0.5, 0.01, "Lines connect words that the model considers to be in the same 'family'.", 
                ha="center", fontsize=10, style='italic', color='#555')
    
    plt.tight_layout()
    plt.show()
