import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

def build_graph(X_tensor):
    print("Building graph...")

    # Convert tensor to numpy
    X_np = X_tensor.numpy()

    # Compute similarity
    sim_matrix = cosine_similarity(X_np)

    threshold = 0.9
    edge_index = []

    for i in range(len(sim_matrix)):
        for j in range(len(sim_matrix)):
            if i != j and sim_matrix[i][j] > threshold:
                edge_index.append([i, j])

    # Convert to tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    print("Graph created!")
    print("Number of edges:", edge_index.shape[1])

    graph_data = Data(x=X_tensor, edge_index=edge_index)

    return graph_data