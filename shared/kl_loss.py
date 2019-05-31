import torch
import numpy as np
from sklearn.cluster import KMeans
import util

# Ref basic operations
# https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/
# Ref KLDivLoss
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html

def assert_shape(t, shape):
	assert len(t.shape) == len(shape), (t is not None, t.shape)
	for i, v in enumerate(shape):
		assert t.shape[i] == v, t.shape

def compute_kl_div_loss_from_numpy(embeddings, kmeans):
	embeddings = torch.from_numpy(embeddings)
	embeddings.requires_grad_()
	return compute_kl_div_loss(embeddings, kmeans)

# We have no parameters from here on, so I don't think we have use for a layer
def compute_kl_div_loss(embeddings, kmeans):
	assert isinstance(embeddings, torch.Tensor)
	assert embeddings.requires_grad is True

	# Run k-means clustering if necessary
	if isinstance(kmeans, int):
		kmeans = KMeans(kmeans)
		util.debug('Running k-means on matrix of %d by %d...' % tuple(embeddings.shape))
		kmeans.fit(embeddings.data.numpy())
	centroids = kmeans.cluster_centers_

	# Get dimensions
	I = torch.tensor(kmeans.labels_.shape).prod() # Number of data points
	J = centroids.shape[0] # Number of clusters
	D = centroids.shape[1] # Number of dimensions for data pts
	assert D == embeddings.shape[1] # Dimension of centroids is same as dimensions of embedding

	# Expand embeddings
	embeddings_tensor = embeddings.view(I,1,D).expand(I,J,D)

	# Compute q numerator
	q_centroids = torch.from_numpy(centroids)
	q_centroids = q_centroids.view(1,J,D).expand(I,-1,-1)
	q_centroids.requires_grad_(False)
	q_numerator_differences = torch.add(embeddings_tensor, -1, q_centroids)
	q_numerator_sq_distances = torch.norm(q_numerator_differences, p=2, dim=2)
	assert_shape(q_numerator_sq_distances, [I,J])
	q_numerators = torch.add(q_numerator_sq_distances, 1)
	assert_shape(q_numerators, [I,J])

	# Compute q denominator
	q_denominator = torch.sum(q_numerators)
	assert 0 == q_denominator.dim(), q_denominator.shape

	# Compute q
	q = torch.div( q_denominator, q_numerators )
	assert_shape(q, [I,J])

	# Compute p numerator
	lbls, cluster_freqs = np.unique(kmeans.labels_, return_counts=True)
	cluster_freqs = cluster_freqs[np.argsort(lbls)] # Sort frequencies from cluster 0...k
	cluster_freqs = torch.from_numpy(cluster_freqs).view(1,J).expand(I,J).double()
	assert_shape(cluster_freqs, [I,J])
	p_numerators = torch.div( torch.mul(q,q), cluster_freqs )
	assert_shape(p_numerators, [I,J])

	# Computer p denominator
	p_denominator = torch.sum(p_numerators)
	assert 0 == p_denominator.dim(), p_denominator.shape

	# Compute p
	p = torch.div( p_numerators, p_denominator )
	assert_shape( p, [I,J] )

	# Compute KL Divergence
	log_p = torch.log(p)
	log_q = torch.log(q)
	log_div = torch.add( log_p, -1, log_q )
	terms = torch.mul( p, log_div )
	kl = torch.sum(terms)

	# Return
	return kl
