# Guessing best k for KMeans

import pandas
import numpy as np
import sklearn.cluster
import sklearn.decomposition

data = pandas.read_csv('regions.csv', header=None)

for i in range(1, data.shape[1] + 1):
    pca = sklearn.decomposition.PCA(n_components=i)
    pca.fit(data)
    print("PCA: %d components explain %.0f%% of variance in data" % (len(pca.explained_variance_ratio_), sum(pca.explained_variance_ratio_) * 100))

pca = sklearn.decomposition.PCA(n_components=0.90)
pca.fit(data)
X = pca.transform(data)

# Rough best guess is sqrt(rows/2), so to be sure we take that and
# double it to get a broad range of n_cluster parameters.
high_guess = int(round((len(X) / 2) ** 0.5 * 2))
print("Running kmeans with n_clusters from 2 to %d" % high_guess)

def euclidean_distance(p, q):
    """ Takes 2D points p and q and returns Euclidean distance """
    return (((p[0] - q[0]) ** 2) +
            ((p[1] - q[1]) ** 2)) ** 0.5

def avg_diameter(matrix, kmeans_obj, cluster_label):
    cluster_center = kmeans_obj.cluster_centers_[cluster_label]
    indexes = [i for i, e in enumerate(kmeans_obj.labels_) if e == cluster_label]
    diameters = []
    for index in indexes:
        d = euclidean_distance(cluster_center, matrix[index])
        diameters.append(d)
    return sum(diameters) / len(diameters)

std_devs = {}
for n in range(2, high_guess):
    kmeans = sklearn.cluster.KMeans(n_clusters=n)
    kmeans = kmeans.fit(X)

    # Get the average diameter of each cluster
    cluster_diameters = np.array([avg_diameter(X, kmeans, i) for i in set(kmeans.labels_)])
    # Store the standard deviation of all the average cluster diameters
    std_devs[n] = cluster_diameters.std()

print("Standard deviations for given cluster sizes: ", std_devs)
# Get the key of the smallest std dev that we found
best_size = min(std_devs, key=std_devs.get)
print("Lowest std dev found using %d clusters" % best_size)

# Run kmeans again for visualization
n = best_size
kmeans = sklearn.cluster.KMeans(n_clusters=n)
kmeans = kmeans.fit(X)


####################################
### Visualizations
####################################
# Source: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
import pylab as pl

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = X[:, 0].min() + 1, X[:, 0].max() - 1
y_min, y_max = X[:, 1].min() + 1, X[:, 1].max() - 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
pl.figure(1)
pl.clf()
pl.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
          cmap=pl.cm.Paired,
          aspect='auto', origin='lower')

pl.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
pl.scatter(centroids[:, 0], centroids[:, 1],
           marker='x', s=169, linewidths=3,
           color='w', zorder=10)
pl.xlim(x_min, x_max)
pl.ylim(y_min, y_max)
pl.xticks(())
pl.yticks(())
pl.show()
