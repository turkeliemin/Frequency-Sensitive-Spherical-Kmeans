import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


def normalizer(dataset : pd.DataFrame) -> pd.DataFrame:
    array = dataset.values

    norm = np.linalg.norm(array,axis=1,keepdims=True)
    normalized = array / norm

    new_df = pd.DataFrame(columns=dataset.columns.to_list(),data=normalized)
    return new_df


def fs_spkmeans(dataset : pd.DataFrame, n_clusters : int, n_inits: int) -> dict:
    data_set = dataset.values
    next_center = 0
    divide_cons = (len(dataset)/n_clusters) * len(dataset.columns.to_list())

    for rep in range(n_inits):

        rand_index = np.random.randint(0,len(data_set),n_clusters)
        next_center = data_set[rand_index]
        initial_center = np.zeros_like(next_center)
        iter = 0
        n_h = np.ones(n_clusters) * (len(dataset)/n_clusters)
        
        while (not np.allclose(initial_center,next_center,atol=1e-4)) and iter<=10000 :
            iter+=1
        
        #assign data points to clusters
            initial_center = np.copy(next_center)
            distances =  (1/n_h) *  (np.dot(data_set,initial_center.T) + 1 - ((n_h/divide_cons)*np.log(n_h)))            
            clus_no = np.argmax(distances,axis=1)
            center_distance = np.max(distances,axis=1)

            #assign next centeroids
            for cluster in range(n_clusters):
                clus_indexes = np.where(np.array(clus_no) == cluster)[0]
                if np.any(clus_indexes):
                    same_clust = data_set[clus_indexes]
                    clust_center = np.sum(same_clust,axis=0) / np.linalg.norm(np.sum(same_clust,axis=0))
                    next_center[cluster] = clust_center
                    n_h[cluster] = len(same_clust)
                if n_h[cluster] == 0:
                    n_h = 1
    
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



wine = pd.read_excel('dermatology.xlsx',header=None)
wine = wine.dropna()
norm_wine = normalizer(wine)
num_init = 1
lls = []

start = time.time()


for i in range(2,20):
    print(i)
    clust = wine.copy()
    cluster =fs_spkmeans(norm_wine,i,num_init)
    clust['Cluster'] = cluster['Cluster']
    clust.to_excel(f'fs_spk_excels/{i}_clusters.xlsx',index=False)
    clus_count = [len(np.where(np.asarray(clust['Cluster']) == q)[0])  for q in range(i)]
    centeroids = [str(i) for i in cluster['Centeroids']]
    cluster_centeroid = {'Cluster' : list(range(i)),'Number of Elements': clus_count, 'Centeroid' : centeroids}
    clus_data = pd.DataFrame(cluster_centeroid)
    clus_data.to_excel(f'fs_spk_excels/{i}_centers.xlsx', sheet_name= 'Cluster Data',index=False)
    lls.append(cluster['Log-Likelihood'])

plt.plot(np.arange(2,20), np.array(lls),marker = 'x',color = 'red')
plt.title('Elbow Method Analysis')
plt.xlabel('k')
plt.ylabel('Log-Likelihood values')
plt.savefig('fs-spkmeans.png')
plt.show()

end = time.time()

print('Execution time:', start -end, 'seconds')
