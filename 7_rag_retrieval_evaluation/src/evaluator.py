import numpy as np

def calculate_mrr(relevance_scores):
    """Calculate Mean Reciprocal Rank."""
    ranks = []
    for scores in relevance_scores:
        # Find index of first relevant item (1)
        try:
            rank = scores.index(1) + 1
            ranks.append(1 / rank)
        except ValueError:
            ranks.append(0)
    return np.mean(ranks) if ranks else 0

def calculate_hit_rate(relevance_scores, k=5):
    """Calculate Hit Rate @ K."""
    hits = 0
    for scores in relevance_scores:
        if 1 in scores[:k]:
            hits += 1
    return hits / len(relevance_scores) if relevance_scores else 0

if __name__ == "__main__":
    # Example: 1 means relevant, 0 means not
    # Query 1: relevant item at pos 2
    # Query 2: relevant item at pos 1
    # Query 3: no relevant item
    test_scores = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    
    print(f"MRR: {calculate_mrr(test_scores):.4f}")
    print(f"Hit Rate @ 1: {calculate_hit_rate(test_scores, k=1):.4f}")
    print(f"Hit Rate @ 3: {calculate_hit_rate(test_scores, k=3):.4f}")
