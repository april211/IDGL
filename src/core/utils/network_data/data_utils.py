# https://github.com/tkipf/gcn/blob/master/gcn/utils.py
import os
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
import torch


from ..generic_utils import *
from ..constants import VERY_SMALL_NUMBER


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize_features(mx_sp):
    """Row-normalize sparse matrix"""

    # convert to csr format for fast algorithmic operations
    mx_csr : sp.csr_matrix = mx_sp.tocsr()
    rowsum = np.array(mx_csr.sum(1))
    
    # ignore "divide by zero" warning for the isolated nodes
    with np.errstate(divide='ignore'):
        r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.

    r_mat_inv = sp.diags(r_inv, format="csr")
    mx_csr = r_mat_inv.dot(mx_csr)
    return mx_csr


def load_network_data(data_dir : str, dataset_str : str, 
                      n_neighbors=None, epsilon=None, knn_metric=None, 
                      prob_del_edge=None, prob_add_edge=None, 
                      seed=None, 
                      use_torch_sparse_adj=False):
    """
    For dataset `cora`, `citeseer`, `pubmed`, \
        all objects above must be saved using python pickle module, \
        except for `ind.dataset_str.test.index` object: 

    ind.dataset_str.x => the feature vectors of the training instances (part of the labeled instances) \
        as scipy.sparse.csr.csr_matrix object;

    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;

    ind.dataset_str.allx => all the feature vectors except for the test instances \
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;

    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;

    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;

    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;

    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;

    ind.dataset_str.test.index => the indices of test instances in graph, \
        for the inductive setting as list object.

    For dataset `citeseer`: 
        ind.citeseer.x (120) 
        ind.citeseer.tx (1000 + *15* == 1015) : 1000 normal training nodes \
                                                + 15 *isolated nodes* (discarded). 
        ind.citeseer.allx (2312)
    
    So you get: allx (2312) + tx (1015) = 3327 nodes in total, as claimed in the paper.

    And you also need to note that we only use a part of the labeled nodes. \
    For citeseer: 
        120 (train) + 500 (val.) + 1000 (test) == 1620 nodes in total, which is smaller than 3327.

    :param dataset_str: Dataset name ('cora', 'citeseer', 'pubmed', 'ogbn-arxiv')
    :return: All data input files loaded (as well the training/test data).
    """

    # If you specify any of these two parameters, 
    # you choose to use the similarity graph as input graph, instead of the original graph.
    # use `knn_size` to construct a k-nearest neighbor graph
    # use `epsilon` to construct an epsilon-graph
    assert (n_neighbors is None) or (epsilon is None)

    if dataset_str.startswith('ogbn'): # Open Graph Benchmark datasets
        from ogb.nodeproppred import NodePropPredDataset

        dataset = NodePropPredDataset(name=dataset_str)

        split_idx = dataset.get_idx_split()

        # use 32-bit int (signed) instead of 64-bit int (signed)
        idx_train, idx_val, idx_test = torch.IntTensor(split_idx["train"]), \
                     torch.IntTensor(split_idx["valid"]), torch.IntTensor(split_idx["test"])

        data = dataset[0] # This dataset has only one graph
        features = torch.FloatTensor(data[0]['node_feat'])
        labels = torch.IntTensor(data[1]).squeeze(-1)

        edge_index = data[0]['edge_index']
        adj = to_undirected(edge_index, num_nodes=data[0]['num_nodes'])

        # check the properties of the adjacency matrix:
        # 1. no self-connections        <-- skip this check
        # 2. no edge weights            <-- skip this check
        # 3. undirected graph           <-- skip this check
        # assert adj.diagonal().sum() == 0 and adj.max() <= 1 and (adj != adj.transpose()).sum() == 0

    else: # datasets: Cora, Citeseer, PubMed
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(os.path.join(data_dir, 'ind.{}.{}'.format(dataset_str, names[i])), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        # We want a ordered test feature matrix (idx = 0, 1, 2, ... ), instead of the original & unordered one.
        test_idx_reorder = parse_index_file(os.path.join(data_dir, 'ind.{}.test.index'.format(dataset_str)))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add their features as zero-vecs into the right position
            test_idx_min = test_idx_range[0]
            test_idx_max = test_idx_range[-1]
            test_idx_range_full = range(test_idx_min, test_idx_max+1)

            # `tx`, `ty` are both "unsorted", ordered by `ind.dataset_str.test.index`
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-test_idx_min, :] = tx              
            tx = tx_extended

            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-test_idx_min, :] = ty
            ty = ty_extended

        # remap the unordered test features to the ordered ones
        # Note: `test_idx_range` will always fall in the range of `tx`!
        raw_features = sp.vstack((allx, tx)).tolil()
        raw_features[test_idx_reorder, :] = raw_features[test_idx_range, :]

        features = normalize_features(raw_features)
        features = torch.FloatTensor(features.todense())

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph), dtype=np.int8)    # csr format for default

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = torch.IntTensor(np.argmax(labels, axis=1))        # labels: one-hots to scalars

        idx_train = torch.IntTensor(range(len(y)))
        idx_val = torch.IntTensor(range(len(y), len(y) + 500))
        idx_test = torch.IntTensor(test_idx_range)                 # drop the isolated nodes

    # process the graph structure
    if not n_neighbors is None:
        print('[ Using kNN-graph as input graph, knn_size == {} ... ]'.format(n_neighbors))

        adj = kneighbors_graph(features, n_neighbors,              # The matrix is of CSR format.
                               metric=knn_metric, include_self=True, 
                               n_jobs=-1).astype(np.int8, mode='safe', copy=False)

    elif not epsilon is None:
        print('[ Using epsilon-graph as input graph, epsilon == {} ... ]'.format(epsilon))

        # use cosine similarity as the distance metric.
        # TODO try different distance metrics
        feature_norm = features.div(torch.linalg.vector_norm(features, ord=2, 
                                                dim=-1, keepdim=True, dtype=torch.float32))
        
        feature_norm[torch.isnan(feature_norm)] = 0.        # check nan values
        
        attention = torch.mm(feature_norm, feature_norm.transpose(-1, -2))
        mask = (attention > epsilon).to(torch.int8, copy=False)
        adj = sp.csr_matrix(mask, dtype=np.int8, copy=False)

    else:
        print('[ Using the original input graph ... ]')

        # for experiment purpose: randomly delete or add edges
        if prob_del_edge is not None:
            adj = graph_delete_connections(prob_del_edge, seed, adj, enforce_connected=False)         

        elif prob_add_edge is not None:
            adj = graph_add_connections(prob_add_edge, seed, adj, enforce_connected=False)

    # add self-connections & sym-normalize the adjacency matrix
    adj.setdiag(values=1)
    adj_norm = sym_normalize_sparse_adj_scipy(adj)

    # choose to use the sparse format or the dense format
    if use_torch_sparse_adj:
        adj_norm = csr_sp_scipy_2_torch(adj_norm)
    else:
        adj_norm = torch.FloatTensor(adj_norm.todense())

    return adj_norm, features, labels, idx_train, idx_val, idx_test


def graph_delete_connections(prob_del, seed, sp_adj : sp.csr_matrix, enforce_connected=False):

    # TODO control ramdom source
    rnd = np.random.RandomState(seed)

    del_adj = sp_adj.toarray()
    pre_num_edges = np.sum(del_adj)
    
    # sample the whole adj. first, 
    # then take the upper triangular part and symmetrize it
    # TODO sparse implementation for `choice` and `enforce_connected`?
    sample_array = np.array([0, 1], dtype=np.int8)
    random_mask = rnd.choice(sample_array, p=[prob_del, 1. - prob_del], 
                                    size=sp_adj.shape) * np.triu(np.ones_like(sp_adj), 1)
    random_mask += random_mask.transpose()

    del_adj *= random_mask

    if enforce_connected:
        add_edges = 0
        for row_idx, row_vec in enumerate(del_adj):
            if not list(np.nonzero(row_vec)[0]):          # this node is isolated
                prev_connected = list(np.nonzero(sp_adj[row_idx, :])[0])
                other_node = rnd.choice(prev_connected)
                del_adj[row_idx, other_node] = 1          # undirected graph
                del_adj[other_node, row_idx] = 1
                add_edges += 1
        print('`graph_delete_connections` -> \
                        `enforce_connected`: # ADDED EDGES: {}.', add_edges)

    cur_num_edges = np.sum(del_adj)
    del_adj = sp.csr_matrix(del_adj, dtype=np.int8)
    print('[ Deleted {}% edges ]'.format(100 * (pre_num_edges - cur_num_edges) / pre_num_edges))
    return del_adj


def graph_add_connections(prob_add, seed, sp_adj : sp.csr_matrix):

    # TODO control ramdom source
    rnd = np.random.RandomState(seed)

    add_adj = sp_adj.toarray()
    pre_num_edges = np.sum(add_adj)

    sample_array = np.array([0, 1], dtype=np.int8)
    sampled_edges = rnd.choice(sample_array, p=[1. - prob_add, prob_add], 
                                    size=sp_adj.shape) * np.triu(np.ones_like(sp_adj), 1)
    sampled_edges += sampled_edges.transpose()

    add_adj += sampled_edges
    add_adj = (add_adj > 0).astype(np.int8, casting='safe', copy=False)

    cur_num_edges = np.sum(add_adj)
    add_adj = sp.csr_matrix(add_adj, dtype=np.int8)
    print('[ Added {}% edges ]'.format(100 * (cur_num_edges - pre_num_edges) / pre_num_edges))
    return add_adj

