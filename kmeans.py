import numpy as np
import pandas as pd
from scipy.stats import mode
from PIL import Image
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import SpectralClustering


def kmeans(X: np.ndarray, k: int, centroids=None, tolerance=1e-2):  # do we need consider max_iter?
    """
    This is the standard kmeans algorithm: 
    first, initiate the centriods, when centroids=None, randomly selected k points as centroids;
    when centroids="kmeans++", initiate centroids by kmeans_plus(X,k) function.
    Then update centroids until the centroids will not change much (change<tolerance).
    """
    X_unique = np.unique(X, axis=0)  # do we need to consider X_unique
    if centroids is None:
        centroids = X_unique[np.random.choice(len(X_unique), k, replace=False)]
    if centroids == "kmeans++":
        centroids = kmeans_plus(X, k)

    C = centroids.flatten()
    C_prev = C.copy()
    diff = 9999
    iter = 0
    while diff > tolerance:
        C_prev = C.copy()
        clusters = [[] for i in range(k)]
        for i in range(X.shape[0]):
            cluster = ((centroids-X[i])**2).sum(1).argmin()
            clusters[cluster].append(i)
        for i in range(k):
            if len(clusters[i]) != 0:
                centroids[i] = np.sum(X[clusters[i]], axis=0)/len(clusters[i])
        C = centroids.flatten()
        diff = np.linalg.norm(C-C_prev)
        iter += 1

    return centroids, clusters

def kmeans_plus(X, k):
    """
    The basic idea is to randomly pick the first of k centroids. 
    Then, pick next k-1 points by selecting points that maximize 
    the minimum distance to  all existing cluster centroids. 
    So for each point, compute the minimum distance to each cluster. 
    Among those min distances to clusters for each point, 
    find the max distance. The associated point is the new centroid.
    """
    centroids_id = []
    centroids_id.append(int(np.random.choice(len(X), 1, replace=False)))
    for i in range(1, k):
        min_dist = []
        for j in range(X.shape[0]):
            min_dist.append(((X[centroids_id]-X[j])**2).sum(1).min())  
        centroids_id.append(np.argmax(min_dist))
    return X[centroids_id]

def likely_confusion_matrix(y, clusters):
    y_pred = np.zeros(len(y))
    for cluster in clusters:
        y_pred[cluster] = mode(y[cluster])[0]
    values = np.unique(y)
    n = len(values)
    confusion_matrix = {values[i]: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            confusion_matrix[values[i]].append(len(np.intersect1d(
                np.nonzero(y == values[i]), np.nonzero(y_pred == values[j]))))
    confusion_matrix_df = pd.DataFrame(confusion_matrix).T
    confusion_matrix_df.columns = [
        "pred "+str(col) for col in confusion_matrix_df.columns]
    confusion_matrix_df.index.name = "Truth"
    print(confusion_matrix_df)
    accuracy = np.mean(y == y_pred)
    print("clustering accur", accuracy)

def likely_confusion_matrix_(y, clusters):
    y_pred = np.zeros(len(y))
    for cluster in clusters:
        y_pred[cluster] = mode(y[cluster])[0]
    values = np.unique(y)
    n = len(values)
    confusion_matrix = {values[i]: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            confusion_matrix[values[i]].append(len(np.intersect1d(
                np.nonzero(y == values[i]), np.nonzero(y_pred == values[j]))))
    confusion_matrix_df = pd.DataFrame(confusion_matrix).T
    confusion_matrix_df.columns = [
        "pred "+str(col) for col in confusion_matrix_df.columns]
    confusion_matrix_df.index.name = "Truth"
    
    accuracy = np.mean(y == y_pred)
    return accuracy
    
def reassign_color(X: np.ndarray, centroids: np.ndarray): # why do I need rearrign_Grey and Color seperatly?
    """
    For each pixel, and a stable cendtriods,
    calculate the distance between the observation and each centroid and
    then return the index of the nearest centroid. 
    Covert the initial pixel to the value of the nearest centroid.
    """
    for i in range(X.shape[0]):
        distance = ((centroids-X[i])**2).sum(1)
        centroid_id = distance.argmin()
        X[i] = centroids[centroid_id]

def leaf_samples(rf, X:np.ndarray): 
    """
    Return a list of arrays where each array is the set of X sample indexes
    residing in a single leaf of some tree in rf forest. For example, if there
    are 4 leaves (in one or multiple trees), we might return:

        array([array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
               array([10, 11, 12, 13, 14, 15]), array([16, 17, 18, 19, 20]),
               array([21, 22, 23, 24, 25, 26, 27, 28, 29]))
    """
    n_trees = len(rf.estimators_)
    leaf_samples = []
    leaf_ids = rf.apply(X)  # which leaf does each X_i go to for sole tree?
    for t in range(n_trees):
        # Group by id and return sample indexes
        uniq_ids = np.unique(leaf_ids[:,t])
        sample_idxs_in_leaves = [np.where(leaf_ids[:, t] == id)[0] for id in uniq_ids]
        leaf_samples.extend(sample_idxs_in_leaves)
    return leaf_samples

def get_X_rand(X: np.ndarray):
    """Duplicate and bootstrap columns of X to get X’"""
    X_rand = X.copy()
    for i in range(X.shape[1]):
        X_rand[:,i] = np.random.choice(X[:,i], len(X), replace=True)
    return X_rand

def conjure_twoclass(X: np.ndarray):
    """Consider all X records as class 0, and X' (X_rand) as class 1.
    Create y to label X and X’ and stack [X, X'].
    """
    X_rand = get_X_rand(X)
    X_synth = np.vstack([X, X_rand])
    y_synth = np.concatenate([np.zeros(len(X)), np.ones(len(X_rand))], axis=0)
    return X_synth, y_synth


def similarity_matrix(X: np.ndarray, metric=None):
    """
    By default, return RF similarity matrix by:
        1. Consider all X records as as class 0
        2. Duplicate and bootstrap columns of X to get X’: class 1
        3. Create y to label X vs X’
        4. Train RF on stacked [X,X’] → y
        5. Walk all leaves of all trees, bumping proximity[i,j] for all 𝑥i, 𝑥j pairs in leaf; 
        divide proximities by num of leaves
    While, if metric is L2, calculate pairwise_distances: L2-norm distance, and return distance matrix.
    """
    if metric == "L2":
        return pairwise_distances(X)
    else:
        N = X.shape[0]
        X_synth, y_synth = conjure_twoclass(X)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_synth, y_synth)
        leaves = leaf_samples(rf, X)
        S = np.zeros(shape=(N, N))
        for leaf in leaves:
            for i in range(len(leaf)):
                for j in range(len(leaf)):
                    S[leaf[i],leaf[j]] += 1
        return S
