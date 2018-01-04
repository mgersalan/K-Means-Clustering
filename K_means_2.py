
# coding: utf-8

# In[135]:



import numpy as np
import struct
from PIL import Image
from matplotlib.pyplot import imshow
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import math


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
    
train_x = read_idx('/Users/Gur/Desktop/ML/dataset/train-images-idx3-ubyte')
train_y = read_idx('/Users/Gur/Desktop/ML/dataset/train-labels-idx1-ubyte')
print(type(train_x))
size = 1024, 1024
img = Image.fromarray(train_x[4,:,:])
width,height = img.size
img = img.resize((256,256))
display(img)
print(train_x[4,:,:].shape)
print(train_y[4])
print(train_x[4,:,:])


# In[240]:


def dist(x,y):
    dst = 0
    
    for i in range(0,28):
        for k in range(0,28):
            dst = dst + (x[i,k] - y[i,k])**2

            
    return math.sqrt(dst)
    
def cluster_Arrange(centroids,shuffled_dataset,k):
    
    
    cluster_count = dict.fromkeys(range(0,10), 0)
    n_row = shuffled_dataset.shape[0]
    
    size = int(math.ceil(float(n_row) / k))
    clusters = np.zeros(shape=(k,500,28,28))
    
    
    for i in range(0,int(n_row)):
        
        target_cluster = distance(shuffled_dataset[i],centroids)
        clusters[target_cluster, cluster_count[target_cluster] ,:,:] = shuffled_dataset[i]
        cluster_count[target_cluster] +=1
    
    print(cluster_count.values())
    cluster_count_vals = cluster_count.values()
    ret_list = []
    ret_list.append(clusters)
    ret_list.append(cluster_count_vals)
    #print(ret_list)
    return ret_list



def distance(point,centroids):
    z = int(centroids.shape[0])
    dst1 = dist(point, centroids[0,:])
    
    min_dist = dst1
    closest_ci = 0
    for i in range(1,z):
         if(dist(point, centroids[i,:]) < min_dist):
            min_dist = dist(point, centroids[i,:])
            closest_ci = i
    
    return closest_ci                                


def display_centroids(cluster_means):
    im_to_display = []
    for prototype in cluster_means:
        
        #print(prototype.shape)
        img = Image.fromarray(prototype)
        #img = img.resize((512,512))
        img = img.convert('RGB')
        img = img.resize((256,256))
        im_to_display.append(img)
     
    f, axarr = plt.subplots(2, 5)
    axarr[0, 0].imshow(im_to_display[0])
    axarr[0, 1].imshow(im_to_display[1])
    axarr[0, 2].imshow(im_to_display[2])
    axarr[0, 3].imshow(im_to_display[3])
    axarr[0, 4].imshow(im_to_display[4])
    axarr[1, 0].imshow(im_to_display[5])
    axarr[1, 1].imshow(im_to_display[6])
    axarr[1, 2].imshow(im_to_display[7])
    axarr[1, 3].imshow(im_to_display[8])
    axarr[1, 4].imshow(im_to_display[9])
    
  
    


# In[243]:


train_x = train_x[0:500]
train_y = train_y[0:500]

def k_meansc(dataset, k , max_iter = 1):
    
    shuffled_dataset = dataset
    #shuffled_dataset = np.random.permutation(dataset)
    n_row = float(dataset.shape[0])
    size = int(math.ceil(n_row / k))
    clusters = np.zeros(shape=(k,size*2,28,28))
    
    
    centroids = np.zeros(shape=(k,28,28))

    for i in range(0,k):
        centroids[i,:,:] = shuffled_dataset[i]
    
    #display_centroids(centroids)

    itercount = 0
    flag = 0
    while(itercount < 10):
        
        
        ret_list = cluster_Arrange(centroids,shuffled_dataset,k)
        new_Clust = ret_list[0]
        counter_dict = ret_list[1]
        #print(new_Clust[5].shape)
        itercount +=1
        n_centroids = np.zeros(shape=(k,28,28))
        sample_clusters = np.zeros(shape=(k,28,28))
        for i in range(0,k):
            for j in range(0,28):
                for z in range(0,28):
                    n_centroids[i,j,z] = np.mean(new_Clust[i,0:counter_dict[i],j,z])
            
            sample_clusters[i] = new_Clust[i,0,:,:]
            
        centroids = n_centroids 
        
        
        
    
    
        
        
    return_list = []
    return_list.append(centroids)
    return_list.append(sample_clusters)
    return return_list
    
    
ret_l = k_meansc(train_x , 10 )
cluster_means = ret_l[0] 
sample_cluster_rows = ret_l[1]
print(sample_cluster_rows.shape)
print("Cluster Means")

display_centroids(cluster_means)




# In[254]:


print("Sample Clusters")
display_centroids(sample_cluster_rows)


# In[258]:
#### Same code with different initial cluster assignment, we will try to see if the accuracy changes


train_x = train_x[0:500]
train_y = train_y[0:500]

initial_instance_centroids = np.zeros(shape=(10,28,28))
visited = []
for i, label in enumerate(train_y):
    if(label not in visited):
        initial_instance_centroids[label] = train_x[i]
        visited.append(label)
    
display_centroids(initial_instance_centroids)


# In[260]:


def k_meansc(dataset, k , max_iter = 1):
    
    shuffled_dataset = dataset
    #shuffled_dataset = np.random.permutation(dataset)
    n_row = float(dataset.shape[0])
    size = int(math.ceil(n_row / k))
    clusters = np.zeros(shape=(k,size*2,28,28))
    
    
    centroids = np.zeros(shape=(k,28,28))

    for i in range(0,k):
        centroids[i] = initial_instance_centroids[i]
    
    #display_centroids(centroids)

    itercount = 0
    flag = 0
    while(itercount < 10):
        
        
        ret_list = cluster_Arrange(centroids,shuffled_dataset,k)
        new_Clust = ret_list[0]
        counter_dict = ret_list[1]
        
        itercount +=1
        n_centroids = np.zeros(shape=(k,28,28))
        sample_clusters = np.zeros(shape=(k,28,28))
        for i in range(0,k):
            for j in range(0,28):
                for z in range(0,28):
                    n_centroids[i,j,z] = np.mean(new_Clust[i,0:counter_dict[i],j,z])
            
            sample_clusters[i] = new_Clust[i,0,:,:]
            
        centroids = n_centroids 
        
        
        
    
    
        
        
    return_list = []
    return_list.append(centroids)
    return_list.append(sample_clusters)
    return return_list
    
    
ret_l = k_meansc(train_x , 10 )
cluster_means = ret_l[0] 
sample_cluster_rows = ret_l[1]
print(sample_cluster_rows.shape)
print("Cluster Means")

display_centroids(cluster_means)



