# K-Means-Clustering
Lloyd algorithm implementation of K Means Clustering with 2 case.

In the first case 100 random points generated from bivariate normal distribution and and k-means clustering applied. Visualization of each iteration (cluster centroids and data points assigned clusters) is also included in the code.

In the second case I used picked first 500, 28 x 28 images from MNIST datasets and applied k means clustering to them.
In order to highlight the importance of initial cluster selection, first, code selects the initial centroids randomly from the images and secondly selects the first instance of each class as the initial cluster mean(centroids). So that the initial cluster means all represent distinct digits. Difference between these two approaches can be seen by visualizing the final cluster centroids as prototypes of each digit and look how close they look to the digits they are supposed to represent.
