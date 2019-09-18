import os
import sys
import numpy as np 
import scipy
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import cg



def sm_affi(features, n):
    """
        data的相似度稀疏矩阵（n×n）
    """
    # row = []
    # col = []
    # data = []

    total_size = features.shape[0]
    
    print("nn")
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='kd_tree', n_jobs=-1).fit(features)

    # print("knn")
    # distance, indices = nbrs.kneighbors(features)

    # print("data")
    # distance_var = np.sqrt(np.mean(np.square(distance)))

    # for ind, (n_indices, n_distances) in enumerate(zip(indices, distance)):
    #     for n_ind, n_dist in zip(n_indices, n_distances):
    #         if ind != n_ind:
    #             row.append(ind)
    #             col.append(n_ind)
    #             data.append(np.exp( - n_dist / distance_var))
    # 相似度矩阵    
    # A = csr_matrix((data, (row, col)), shape=(total_size, total_size))

    A = kneighbors_graph(nbrs, n, mode='distance', include_self=False, n_jobs=-1)
    distance_var = np.sqrt(np.mean(np.square(A.data)))
    A.data = np.exp( - A.data / distance_var )

    W = A + A.transpose()
    D = W.sum(axis=1).reshape(-1)
    D = scipy.sparse.spdiags(D, 0, total_size, total_size).power(-0.5)

    # 归一化权重矩阵
    W = D * W * D

    return W


def sm_affi2(features, n):
    """
        data的相似度稀疏矩阵（n×n）
    """
    row = []
    col = []
    data = []

    total_size = features.shape[0]
    
    print("nn")
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='kd_tree', n_jobs=-1).fit(features)

    print("knn")
    distance, indices = nbrs.kneighbors(features)

    print("data")
    distance_var = np.sqrt(np.mean(np.square(distance)))

    for ind, (n_indices, n_distances) in enumerate(zip(indices, distance)):
        for n_ind, n_dist in zip(n_indices, n_distances):
            if ind != n_ind:
                row.append(ind)
                col.append(n_ind)
                data.append(np.exp( - n_dist / distance_var))

    # 相似度矩阵    
    A = csr_matrix((data, (row, col)), shape=(total_size, total_size))

    # A = kneighbors_graph(features, n, mode='distance', include_self=False, n_jobs=-1)
    # distance_var = np.sqrt(np.mean(np.square(A.data)))
    # A.data = np.exp( - A.data / distance_var )

    W = A + A.transpose()
    D = W.sum(axis=1).reshape(-1)
    D = scipy.sparse.spdiags(D, 0, total_size, total_size).power(-0.5)

    # 归一化权重矩阵
    W = D * W * D

    return W





def sm_label(label, label_indices, total_size, nb_classes):
    row = label_indices
    col = label
    data = [1 for _ in label_indices]
    # total_size = len(label)   
    return csr_matrix((data, (row, col)), shape=(total_size, nb_classes))



def conjugate_gradient_solver(A, b):

    res = []
    for i in range(b.shape[1]):
        if isinstance(b, np.ndarray):
            X, info = cg(A=A, b=b[:,i])
        else:
            X, info = cg(A=A, b=b[:,i].toarray())
        
        if int(info) != 0:
            print("Warning : info is not zero %f"%info)

        res.append(X)
    
    return np.array(res).transpose([1, 0])


def entropy_weight(Z):
    Z = np.maximum(Z, 5e-10)
    nb_classes = Z.shape[1]
    Z = Z / Z.sum(axis=1, keepdims=True)
    Z = - 1.0 *  (Z * np.log(Z)).sum(axis=1)
    W = 1.0 - Z / np.log(float(nb_classes))
    return W


def graph_laplace(feature_list, dataset, n=10, alpha=0.99):

    A = sm_affi(feature_list, n)
    b = sm_label(*dataset.label)
    I = scipy.sparse.eye(dataset.total_size)

    Y = conjugate_gradient_solver(I - A * alpha, b)
    W = entropy_weight(Y)
    Y = np.argmax(Y, axis=1)
    return Y, W




if __name__ == "__main__":

    import pickle as pkl
    import time

    test_pkl_filepath = "./feature_list_250.pkl"

    feature, emb_feature = pkl.load(open(test_pkl_filepath, 'rb'))

    start = time.time()
    W = sm_affi(feature, 10)
    end = time.time()

    print(start - end)


    start = time.time()
    W = sm_affi2(feature, 10)
    end = time.time()

    print(start - end)

