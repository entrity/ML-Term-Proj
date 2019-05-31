# Reference Kuhn 1955
# Kuhn, Harold W. "The Hungarian method for the assignment problem." Naval research logistics quarterly 2.1‐2 (1955): 83-97.
# 
# Reference also https://www.r-bloggers.com/matching-clustering-solutions-using-the-hungarian-method/
# which gives a simple, brief explanation of the application
# of the Hungarian algorithm for this task.

# The cost matrix is k-by-k, where k is the number of clusters.
# Rows correspond to ground-truth labelling; columns correspond
# to inferred labellings.
# The value of each cell is the sum of the sizes of the two
# indicated clusters, minus the twice size of their intersection.

import scipy.optimize
import numpy as np

class Computer(object):
	def __init__(self, k, gt_labels):
		self.k = k
		self.gt_labels = gt_labels

	def run(self, inferred_labels):
		k = self.k
		assert len(inferred_labels) == len(self.gt_labels), (len(inferred_labels), len(self.gt_labels))
		# Prepare cost matrix for Hungarian algorithm
		gt_cluser_sizes = self.gt_cluser_sizes = np.zeros((k), dtype=np.int)
		in_cluser_sizes = self.in_cluser_sizes = np.zeros((k), dtype=np.int)
		intersection = self.intersection = np.zeros((k, k), dtype=np.int)
		for i, gt_lbl in self.gt_labels:
			in_lbl = inferred_labels[i]:
			intersection[gt_lbl, in_lbl] += 1
			gt_cluser_sizes[gt_lbl] += 1
			in_cluser_sizes[in_lbl] += 1
		expand = lambda arr : np.repeat(np.expand_dims(arr, 1), k, 1)
		gt_cluser_sizes = expand(gt_cluser_sizes)
		in_cluser_sizes = expand(in_cluser_sizes)
		size_sums = gt_cluser_sizes + in_cluser_sizes.T
		self.cost = size_sums - (2*intersection)
		# Run Hungarian algorithm
		self.mapping = scipy.optimize.linear_sum_assignment( cost )
		# Compute ACC: sum of intersections over n
		n = len(inferred_labels)
		n_correct_assignments = intersection(self.mapping).sum()
		acc = n_correct_assignments / n
		return acc
