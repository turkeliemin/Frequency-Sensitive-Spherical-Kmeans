from algorithms import *
import time
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


dermatology = pd.read_excel('dermatology.xlsx',header=None)
dermatology = dermatology.dropna()
norm_derm = normalizer(dermatology)
num_init = 10
lls = {'spkmeans': [],'pifs-spkmeans' : [], 'fifs-spkmeans' : []}

run_times = {'spkmeans': [],'pifs-spkmeans' : [], 'fifs-spkmeans' : []}

clus_std = {'spkmeans': [],'pifs-spkmeans' : [], 'fifs-spkmeans' : []}



for i in range(2, 21):
    print(i)
    np.random.seed(61)
    
    clust = norm_derm.copy()
    start = time.time()
    cluster =spkmeans(norm_derm,i,num_init)
    end = time.time()
    clus_count = [len(np.where(np.asarray(cluster['Cluster']) == q)[0]) for q in range(i)]
    clus_std['spkmeans'].append(np.std(clus_count))
    lls['spkmeans'].append(cluster['Log-Likelihood'])
    run_times['spkmeans'].append(end-start)

    np.random.seed(61)
    clust = norm_derm.copy()
    start = time.time()

    cluster =pifs_spkmeans(norm_derm,i,num_init)
    end = time.time()

    clus_count = [len(np.where(np.asarray(cluster['Cluster']) == q)[0]) for q in range(i)]
    clus_std['pifs-spkmeans'].append(np.std(clus_count))
    lls['pifs-spkmeans'].append(cluster['Log-Likelihood'])
    run_times['pifs-spkmeans'].append(end-start)    

    np.random.seed(61)

    clust = norm_derm.copy()
    start = time.time()
    cluster =fifs_spkmeans(norm_derm,i,num_init)
    end = time.time()
    clus_count = [len(np.where(np.asarray(cluster['Cluster']) == q)[0]) for q in range(i)]
    clus_std['fifs-spkmeans'].append(np.std(clus_count))
    lls['fifs-spkmeans'].append(cluster['Log-Likelihood'])
    run_times['fifs-spkmeans'].append(end-start)


plt.figure(figsize=(12,8))
plt.plot(np.arange(2,21),clus_std['pifs-spkmeans'], marker='x', color='red',label='pifs-spkmeans')
plt.plot(np.arange(2,21),clus_std['fifs-spkmeans'], marker='.', color='black',label='fifs-spkmeans')
plt.plot(np.arange(2,21),clus_std['spkmeans'], marker='o', color='blue',label='spkmeans')
plt.title('Cluster Balancing')
plt.xticks(np.arange(2,21))
plt.ylabel('Standart Deviation of Cluster Size')
plt.xlabel('Cluster Number')
plt.legend()
plt.savefig('Cluster_Balancing.png')

plt.figure(figsize=(12,8))
plt.plot(np.arange(2,21),np.array(lls['pifs-spkmeans'])/len(norm_derm), marker='x', color='red',label='pifs-spkmeans')
plt.plot(np.arange(2,21),np.array(lls['fifs-spkmeans'])/len(norm_derm), marker='.', color='black',label='fifs-spkmeans')
plt.plot(np.arange(2,21),np.array(lls['spkmeans'])/len(norm_derm), marker='o', color='blue',label='spkmeans')
plt.title('SOF Values')
plt.xticks(np.arange(2,21))
plt.ylabel('SOF Values')
plt.xlabel('Cluster Number')
plt.legend()
plt.savefig('Sof_Values.png')



plt.figure(figsize=(12,8))
plt.plot(np.arange(2,21),run_times['pifs-spkmeans'], marker='x', color='red',label='pifs-spkmeans')
plt.plot(np.arange(2,21),run_times['fifs-spkmeans'], marker='.', color='black',label='fifs-spkmeans')
plt.plot(np.arange(2,21),run_times['spkmeans'], marker='o', color='blue',label='spkmeans')
plt.title('Run Times')
plt.xticks(np.arange(2,21))
plt.ylabel('Run Time')
plt.xlabel('Cluster Number')
plt.legend()
plt.savefig('Run Time.png')