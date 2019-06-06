import sys, os
import cv2
import numpy as np
import torch
from PIL import Image
from skimage.feature import hog
import data.stl10_input as stl
from data.dataset import Dataset, SplitDataset, SavedDataset

class StarterDataset(Dataset):
	def __init__(self):
		super().__init__()
		im_arry = stl.read_all_images(stl.DATA_PATH)
		im_list = [ hog(im, multichannel=True, cells_per_block=(1,1)) for im in im_arry ]
		self.images = torch.from_numpy( np.stack(im_list) ).float()
		self.labels = torch.from_numpy( stl.read_labels(stl.LABEL_PATH) ) - 1

if __name__ == '__main__':
	starter = StarterDataset()
	traiset, testset = starter.split(0.1)
	traiset.save('data/hog-train')
	testset.save('data/hog-test')
	print('Saved HOG trainset of len %d' % len(traiset))
	print('Saved HOG testset  of len %d' % len(testset))
	SavedDataset('data/hog-train')
	SavedDataset('data/hog-test')
