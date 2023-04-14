"""
Author: Zhongxin Hu
Date: 2023-04-11
"""

import random
import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import _supervised as supervised
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.utils import check_random_state, check_symmetric
from sklearn import cluster


def same_seeds(seed):
    # Set the same seed for all random number generators
    torch.manual_seed(seed)
    if torch.cuda.is_available():#gpu usage
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLP(torch.nn.Module):
    # The architecture of Query Network and Key Network
    def __init__(self, in_size, out_size, hide_size_list):
        super(MLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size_list = hide_size_list
        self.layers = torch.nn.Sequential(#creating the MLP model
            torch.nn.Linear(self.in_size, self.hide_size_list[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hide_size_list[0], self.hide_size_list[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hide_size_list[1], self.hide_size_list[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hide_size_list[2], self.out_size),
        )
        self.reinitialize()

    def reinitialize(self):
        torch.nn.init.kaiming_uniform_(self.layers[0].weight)
        torch.nn.init.zeros_(self.layers[0].bias)
        torch.nn.init.kaiming_uniform_(self.layers[2].weight)
        torch.nn.init.zeros_(self.layers[2].bias)
        torch.nn.init.kaiming_uniform_(self.layers[4].weight)
        torch.nn.init.zeros_(self.layers[4].bias)
        # the last layer is Xavier initialization
        torch.nn.init.xavier_uniform_(self.layers[6].weight)
        torch.nn.init.zeros_(self.layers[6].bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.tanh_(x)
        return x


class SoftThreshold(torch.nn.Module):
    # soft threshold function from LISTA and ISTA-Net
    def __init__(self, threshold):
        super(SoftThreshold, self).__init__()
        self.threshold = threshold
        # b is a learnable parameter  Tb(t) := sgn(t)) max(0, |t| − b)
        self.register_parameter("b", torch.nn.Parameter(torch.from_numpy(np.zeros(shape=[self.threshold])).float()))

    def forward(self, x):
        return torch.sign(x) * torch.relu(torch.abs(x) - self.b)


class SENet(torch.nn.Module):
    # The architecture of Self-Expressive Network (SENet)
    def __init__(self,in_size, out_size, hide_size_list):
        super(SENet, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size_list = hide_size_list
        # alpha scales down the output of SENet
        self.alpha = 1.0 / 1024

        self.query_net = MLP(self.in_size, self.out_size, self.hide_size_list)#query net(MLP)
        self.key_net = MLP(self.in_size, self.out_size, self.hide_size_list)#key net(MLP)
        self.threshold = SoftThreshold(1)

    def query_output(self, x):
        query_out = self.query_net(x)
        return query_out

    def key_output(self, x):
        key_out = self.key_net(x)
        return key_out

    def coeff_output(self, query_out, key_out):
        coeff = self.threshold(torch.matmul(query_out, key_out.t()))
        return self.alpha * coeff

    def forward(self, query, key):
        query_out = self.query_output(query)
        key_out = self.key_output(key)
        coeff = self.coeff_output(query_out, key_out)
        return coeff

def regularization(x, lambd = 1.0):
    # weighted sum of l1 and l2 regularization
    return lambd * torch.abs(x).sum() + (1.0 - lambd) / 2.0 * torch.pow(x, 2).sum()


def get_sparse_rep(model, data, batch_size, top_k):
    sample_size = data.shape[0]
    top_k = min(sample_size, top_k)
    coeff_matrix = torch.empty([batch_size, sample_size], dtype=torch.float32)    # self-expressive coefficient matrix
    n_batches = sample_size // batch_size

    model.eval()
    val = []
    indices = []
    # get the top-k coefficients for each batch of samples
    with torch.no_grad():
        for i in range(n_batches):
            batch = data[i * batch_size:(i + 1) * batch_size].cuda()
            q = model.query_output(batch)
            for j in range(n_batches):
                if j != i:
                    batch_compare = data[j * batch_size: (j + 1) * batch_size].cuda()
                    k = model.key_output(batch_compare)
                    temp = model.coeff_output(q, k)
                    coeff_matrix[:, j * batch_size:(j + 1) * batch_size] = temp.cpu()
                else:
                    coeff_matrix[:, j * batch_size:(j + 1) * batch_size] = 0.0

            coeff_matrix.cpu()

            _, index = torch.topk(torch.abs(coeff_matrix), dim=1, k=top_k)

            val.append(coeff_matrix.gather(1, index).reshape([-1]).cpu().data.numpy())
            indices.append(index.reshape([-1]).cpu().data.numpy())

    val = np.concatenate(val, axis=0)
    indices = np.concatenate(indices, axis=0)
    indptr = [top_k * i for i in range(sample_size + 1)]

    sparsed_coeff = sparse.csr_matrix((val, indices, indptr), shape=[sample_size, sample_size])
    return sparsed_coeff


def evaluate(model, data, labels, num_subspaces, spectral_dim, top_k=1000, batch_size=10000):
    # evaluate the clustering performance
    sparsed_coeff = get_sparse_rep(model=model, data=data, batch_size=batch_size, top_k=top_k)
    sparsed_coeff = normalize(sparsed_coeff).astype(np.float32)

    Aff = 0.5 * (np.abs(sparsed_coeff) + np.abs(sparsed_coeff).T)    #affinity matrix
    preds = spectral_clustering(Aff, num_subspaces, spectral_dim)
    acc = clustering_accuracy(labels, preds)
    nmi = normalized_mutual_info_score(labels, preds, average_method='geometric')
    ari = adjusted_rand_score(labels, preds)
    return acc, nmi, ari


def clustering_accuracy(labels_true, labels_pred):
    # compute the clustering accuracy
    labels_true, labels_pred = supervised.check_clusterings(labels_true, labels_pred)
    value = supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)


def spectral_clustering(affinity_matrix_, n_clusters, k, seed=1, n_init=20):
    # spectral clustering prediction
    affinity_matrix_ = check_symmetric(affinity_matrix_)
    random_state = check_random_state(seed)

    laplacian = sparse.csgraph.laplacian(affinity_matrix_, normed=True)
    _, vec = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian,
                                 k=k, sigma=None, which='LA')
    embedding = normalize(vec)
    _, labels_, _ = cluster.k_means(embedding, n_clusters, random_state=seed, n_init=n_init)
    return labels_











