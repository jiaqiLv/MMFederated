import numpy as np
from sklearn.neighbors import NearestNeighbors
from numpy.random import uniform
from sklearn import metrics

def hopkins_statistic(data, sample_rate=0.7):
    # step1: Sample uniformly n points from data
    selected_indices = np.random.choice(data.shape[0],round(data.shape[0]*sample_rate),replace=False)
    selected_data = data[selected_indices]
    # step2: Compute the distance x_i from each real point to each nearest neighbor
    # step3: Generate a simulated data set (random_D) drawn from a random uniform 
    # distribution with n points and the same variation as the original real data set D
    n = selected_data.shape[0]
    nbrs = NearestNeighbors(n_neighbors=1).fit(selected_data)
    ujd,wjd = [],[]
    for i in range(n):
        u_dist, _ = nbrs.kneighbors(uniform(np.min(data,axis=0), np.max(data,axis=0), data.shape[1]).reshape(1,-1), 2 , return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(selected_data[i].reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
    h = sum(ujd)/(sum(ujd)+sum(wjd))
    print(h)
    return h

def Calinski_Harabasz(data,labels):
    score = metrics.calinski_harabasz_score(data,labels)
    print(score)
    return score

def Silhouette_Coefficient(data,labels):
    score = metrics.silhouette_score(data,labels)
    print(score)
    return score

def Davies_Bouldin(data,labels):
    score = metrics.davies_bouldin_score(data,labels)
    print(score)
    return score

if __name__ == '__main__':
    data = np.random.rand(512,2)
    hopkins_statistic(data)
    