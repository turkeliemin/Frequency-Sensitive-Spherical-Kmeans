import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt



def normalizer(dataset: pd.DataFrame) -> pd.DataFrame:
    array = dataset.values
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = array / norm
    new_df = pd.DataFrame(columns=dataset.columns.to_list(), data=normalized)
    return new_df

def fifs_spkmeans(dataset: pd.DataFrame, n_clusters: int, n_inits: int) -> dict:
    data_set = dataset.values
    divide_cons = len(dataset.columns.to_list()) * (len(dataset) / n_clusters)
    for rep in range(n_inits):
        clus_no = np.random.randint(0, n_clusters, len(data_set))
        next_center = np.ones((n_clusters, data_set.shape[1]))
        initial_center = np.zeros_like(next_center)
        iter = 0
        n_h = np.ones(n_clusters) * (len(dataset) / n_clusters)
        center_distance = np.zeros(len(data_set))

        # Assign initial centroids
        for cluster in range(n_clusters):
            clus_indexes = np.where(clus_no == cluster)[0]
            if clus_indexes.size > 0:
                same_clust = data_set[clus_indexes]
                clust_center = np.sum(same_clust, axis=0) / np.linalg.norm(np.sum(same_clust, axis=0))
                next_center[cluster] = clust_center

        while not np.allclose(initial_center, next_center, atol=1e-4) and iter <= 10000:
            iter += 1
            initial_center = np.copy(next_center)
            
            for index, x in enumerate(data_set):
                cents_dist = np.dot(x, next_center.T)
                distances = (1/n_h) * (cents_dist + 1 - ((n_h / divide_cons) * np.log(n_h)))
                clus_no[index] = np.argmax(distances)
                n_h[clus_no[index]] += 1 + 1/n_clusters
                n_h = n_h- 1/n_clusters
               
                center_distance[index]  = cents_dist[clus_no[index]]

                for h in range(n_clusters):
                    #temp_center = next_center + (1/n_h[:, np.newaxis]) * (x - next_center)                
                    #next_center = temp_center / np.linalg.norm(temp_center, axis=1, keepdims=True)
                    temp_center = next_center[h] + (1/n_h[h]) * (x - next_center[h])     
                    next_center[h] = temp_center / np.linalg.norm(temp_center, keepdims=True)

        if rep == 0:
            current_center = next_center
            current_clusters = clus_no
            log_likelihood = np.sum(center_distance)

        else:
            log_next = np.sum(center_distance)
            if log_next > log_likelihood:
                current_center = next_center
                current_clusters = clus_no
                log_likelihood = log_next

    return {'Cluster': current_clusters, 'Centeroids': current_center, 'Log-Likelihood': log_likelihood}


def pifs_spkmeans(dataset : pd.DataFrame, n_clusters : int, n_inits: int) -> dict:
    data_set = dataset.values
    next_center = 0
    divide_cons = len(dataset.columns.to_list()) *(len(dataset)/n_clusters)
    for rep in range(n_inits):

        clus_no = np.random.randint(0,n_clusters,len(data_set))
        next_center = np.ones((n_clusters,len(data_set.T)))
        initial_center = np.zeros_like(next_center)
        iter = 0
        n_h = np.ones(n_clusters) * (len(dataset)/n_clusters)
        center_distance = np.zeros(len(data_set))

        while (not np.allclose(initial_center,next_center,atol=1e-4)) and iter<=10000 :
            iter+=1
            initial_center = np.copy(next_center)
            #assign next centeroids
            for cluster in range(n_clusters):
                clus_indexes = np.where(np.array(clus_no) == cluster)[0]
                if np.any(clus_indexes):
                    same_clust = data_set[clus_indexes]
                    clust_center = np.sum(same_clust,axis=0) / np.linalg.norm(np.sum(same_clust,axis=0))
                    next_center[cluster] = clust_center
            
            for index,x in enumerate(data_set):
                cent_dists = np.dot(x,next_center.T)
                distances =  (1/n_h) *  (cent_dists + 1 - ((n_h/divide_cons)*np.log(n_h)))            
                clus_no[index] = np.argmax(distances)
                n_h = n_h - 1/n_clusters
                n_h[clus_no[index]] += 1+1/n_clusters 
                center_distance[index] = cent_dists[clus_no[index]]

        if rep==0:
            current_center = next_center
            current_clusters = clus_no         

            log_likelihood = np.sum(center_distance)
        else:
            #compute log likelihood
            log_next = np.sum(center_distance)

            if log_next>log_likelihood:                
                current_center = next_center
                current_clusters = clus_no
                log_likelihood = log_next

    return {'Cluster' : current_clusters, 'Centeroids' : current_center , 'Log-Likelihood' : log_likelihood}

def spkmeans(dataset : pd.DataFrame, n_clusters : int, n_inits: int) -> dict:
    data_set = dataset.values
    next_center = 0

    for rep in range(n_inits):
        rand_index = np.random.randint(0,len(data_set),n_clusters)
        next_center = data_set[rand_index]
        initial_center = np.zeros_like(next_center)
        iter = 0
        while (not np.array_equal(initial_center,next_center)):
            iter+=1

        #assign data points to clusters
            initial_center = np.copy(next_center)
            distances = np.dot(data_set,initial_center.T)            
            clus_no = np.argmax(distances,axis=1)
            center_distance = np.max(distances,axis=1)
            #assign next centeroids
            for cluster in range(n_clusters):
                clus_indexes = np.where(np.array(clus_no) == cluster)[0]
                if np.any(clus_indexes):
                    same_clust = data_set[clus_indexes] 
                    clust_center = np.sum(same_clust,axis=0) / np.linalg.norm(np.sum(same_clust,axis=0))
                    next_center[cluster] = clust_center
                
                else: 
                    nearest = np.argmax(distances.T[cluster])
                    clus_no[nearest] = cluster
                    next_center[cluster] = data_set[nearest]

        if rep==0:
            current_center = next_center
            current_clusters = clus_no         
            log_likelihood = np.sum(center_distance)

        else:
            #compute log likelihood
            log_next = np.sum(center_distance)

            if log_next>log_likelihood:                
                current_center = next_center
                current_clusters = clus_no
                log_likelihood = log_next


    return {'Cluster' : current_clusters, 'Centeroids' : current_center , 'Log-Likelihood' : log_likelihood}    

