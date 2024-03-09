import numpy as np
import os
import json
def compute_distance_matrix(embeddings, metric):
    if metric == 'cosine':
        #embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        distance_matrix = np.matmul(embeddings, embeddings.T)
    elif metric == 'euclidean':
        distance_matrix = np.linalg.norm(embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :], axis=-1)
    else:
        raise NotImplementedError
    return distance_matrix
def print_distance_matrix(distance_matrix):
    # print matix in a human readable format, with column and row names
    print(' '.join(['{:>8}'.format('')]+['{:>8}'.format(str(i)) for i in range(distance_matrix.shape[0])]))
    for i in range(distance_matrix.shape[0]):
        print('{:>8}'.format(str(i)), end=' ')
        for j in range(distance_matrix.shape[1]):
            print('{:8.4f}'.format(distance_matrix[i, j]), end=' ')
        print()
    N = distance_matrix.shape[0]
    eye = np.eye(N)
    max_distance = np.max(distance_matrix-eye*1e6)
    print('Max distance: {:.4f}'.format(max_distance))
    min_distance = np.min(distance_matrix+eye*1e6)
    print('Min distance: {:.4f}'.format(min_distance))
    mean_distance = np.mean(distance_matrix*(1-eye)) * (N*N)/(N*N-N)
    print('Mean distance: {:.4f}'.format(mean_distance))
                            
    
if __name__ == '__main__':
    #embedding_path = 'data/ie_instruct_unique_id/cluster_embeddings/cluster_embeddings_InstructUIE_iota_mean_of_encoder_eval_0.npy'
    #embedding_path = 'data/ie_instruct_unique_id/cluster_embeddings/ner/cluster_embeddings_lora_65536_21_None.npy'
    embedding_path = '/home/zkhu143/iuie/data/ie_instruct_unique_id/cluster_embeddings/re/cluster_embeddings_random_4096_8.npy'
    embeddings = np.load(embedding_path)
    distance_matrix = compute_distance_matrix(embeddings, 'cosine')
    print_distance_matrix(distance_matrix)
