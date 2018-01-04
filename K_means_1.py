
# coding: utf-8

# In[1]:


import numpy as np
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import math
import random
np.random.seed(100)
plt.rcParams['figure.figsize'] = (10,10)

X = np.random.multivariate_normal([0,0],[ [1,0] ,[0,1] ], 10)

plt.plot(X[:,0], X[:,1], 'bo')
plt.show()
#x.shape[1]
#print(X[0])


# In[2]:


def k_plot(clusters,centroids):
    plt.plot(clusters[0, : , 0], clusters[0, : , 1], 'ro')
    plt.plot(clusters[1, : , 0], clusters[1, : , 1], 'yo')
    plt.plot(clusters[2, : , 0], clusters[2, : , 1], 'go')
    plt.plot(centroids[0,0],centroids[0,1],'ro',marker='o',markersize=20)
    plt.plot(centroids[1,0],centroids[1,1],'yo',marker='o',markersize = 20)
    plt.plot(centroids[2,0],centroids[2,1],'go',marker='o',markersize = 20)
    
    plt.show()
    
    print(clusters[0])
    print("bosluk")
    print(centroids[0])
    
    
    
def distance(point,centroids):
    k = int(centroids.shape[0])
    dst1 = np.linalg.norm(point-centroids[0,:])
    
    min_dist = dst1
    closest_ci = 0
    for i in range(1,k):
         if(np.linalg.norm(point-centroids[i,:]) < min_dist):
            min_dist = np.linalg.norm(point-centroids[i,:])
            closest_ci = i
                 
    return closest_ci                                
  

def cluster_Arrange(centroids,shuffled_dataset,k):
    inside_cluster_c1 = 0
    inside_cluster_c2 = 0
    inside_cluster_c3 = 0
    n_row = shuffled_dataset.shape[0]
    size = int(math.ceil(float(n_row) / k))
    clusters = np.zeros(shape=(k,size*2,2))
    
    cluster_counts_dict = dict.fromkeys(range(0,3), 0)
    
    
    for i in range(0,int(n_row)):
        target_cluster = distance(shuffled_dataset[i],centroids)
        counter_tuple = []
        
     
        
        if (target_cluster == 0):

            clusters[0,inside_cluster_c1 ,:] = shuffled_dataset[i]
            inside_cluster_c1 += 1
            cluster_counts_dict[0] +=1

        elif (target_cluster == 1):

            clusters[1,inside_cluster_c2 ,:] = shuffled_dataset[i]
            inside_cluster_c2 += 1
            cluster_counts_dict[1] +=1

        elif (target_cluster == 2):

            clusters[2,inside_cluster_c3 ,:] = shuffled_dataset[i]
            inside_cluster_c3 += 1
            cluster_counts_dict[2] +=1
            
    cluster_count_vals = cluster_counts_dict.values()
    print(cluster_counts_dict)
    ret_list = []
    ret_list.append(clusters)
    ret_list.append(cluster_count_vals)
    #print(ret_list)
    return ret_list
     
       


# In[3]:


def k_means(dataset, k , max_iter = 10):
    
    shuffled_dataset = np.random.permutation(dataset)
    n_row = float(dataset.shape[0])
    size = int(math.ceil(n_row / k))
    clusters = np.zeros(shape=(k,size,2))
    
    centroids = np.zeros(shape=(k,2))
    
    cluster_c = 0
    inside_cluster_c = 0
    for row in shuffled_dataset:
        if(inside_cluster_c == size):
            inside_cluster_c = 0
            cluster_c += 1
        
        clusters[cluster_c, inside_cluster_c , :] = row
        inside_cluster_c += 1
    
    for i in range(0,k):
        centroids[i,0] = np.mean(clusters[i,:,0])
        centroids[i,1] = np.mean(clusters[i,:,1])
    
    k_plot(clusters,centroids)
    itercount = 0
    while(itercount < max_iter):
        ret_list = cluster_Arrange(centroids,shuffled_dataset,k)
        new_Clust = ret_list[0]
        counter_dict = ret_list[1]
        centroids = np.zeros(shape=(k,2))
        for i in range(0,k):
            centroids[i,0] = np.mean(new_Clust[i,0:counter_dict[i],0])
            centroids[i,1] = np.mean(new_Clust[i,0:counter_dict[i],1])
        k_plot(new_Clust,centroids)
        itercount +=1
    
k_means(X , 3 )

