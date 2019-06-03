import logging
import torch
import torch.nn as nn
import torchvision.models as models

def new():
	logging.getLogger().info('Downloading Resnet-18...')
	resnet18 = models.resnet18(pretrained=True)
	logging.getLogger().info('Resnet-18 downloaded.')
	net = Net(resnet18)
	return net

def load(fpath):
	data = torch.load(fpath)
	resnet18 = models.resnet18(pretrained=False)
	net = Net(resnet18)
	logging.info('Loading model state dict...')
	net.load_state_dict( data['model_state_dict'] )
	logging.info('Model state dict loaded.')
	return net

class Net(nn.Module):
	def __init__(self, resnet18):
		super(Net, self).__init__()
		self.features = nn.Sequential()
		for name, module in resnet18.named_children():
			if name == 'fc': break
			self.features.add_module(name, module)

	def forward(self, x):
		x = self.features(x)
		return x

if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG)
	net = new()
