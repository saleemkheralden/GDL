# %%
import time
import copy
import torch
import torch_geometric

from torch import nn, optim
from torch.nn import functional as F
from torch_geometric import nn as gnn
from torch.utils.data import DataLoader
from torch_geometric.datasets import ModelNet
from IPython.display import display, clear_output
from torch_geometric.utils import remove_self_loops
from sklearn.model_selection import train_test_split

import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(device)

# %%
class_names = [
    'bathtub', 'bed', 'chair', 'desk', 'dresser', 
    'monitor', 'night_stand', 'sofa', 'table', 'toilet'
]

# %%
class GNN(nn.Module):
	def __init__(self,
		data,
		hidden_channels,
		aggr='mean',
		lr=.1, 
		weight_decay=.01,
		criterion=None):

		super(GNN, self).__init__()

		self.in_channels = data[0].pos.shape[1]
		self.num_classes = data.num_classes
		n_heads = 1
		self.encoder = gnn.Sequential('x, edge_index', [
			(gnn.SAGEConv(
				in_channels=self.in_channels,
				out_channels=hidden_channels,
				aggr=aggr,
			), 'x, edge_index -> x'),
			nn.ReLU(inplace=True),
			(gnn.SAGEConv(
				in_channels=hidden_channels,
				out_channels=hidden_channels // 2,
				aggr=aggr
			), 'x, edge_index -> x'),
			nn.ReLU(inplace=True),
		])
		
		self.decoder = nn.Sequential(
			nn.Linear(
				n_heads * hidden_channels // 2, 
				self.num_classes
			),
			nn.Softmax(dim=-1),
		)

		self.optim = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
		self.criterion = criterion if criterion else nn.CrossEntropyLoss()
		self.reset_metrics()

	def forward(self, x):
		face = x.face
		edge_index, _ = remove_self_loops(torch.cat([face[:2], face[1:], face[[0, 2]]], dim=1))

		x = self.encoder(x.pos, edge_index)
		
		x = x.sum(dim=0) / x.shape[0]
		
		x = self.decoder(x)

		return x, x.argmax()
	

	def predict(self, x):
		return class_names[self.forward(x)[1]]
	

	def accuracy(self, y, y_hat):
		return sum(y == y_hat) / y.shape[0]

	def reset_metrics(self):
		self.metrics = {
			'loss': [],
			'train_acc': [],
			'val_acc': []
		}

	def train_model(self,
				 data,
				 n_epochs,
				 per_epoch_permute=True,
				 validate=True,
				 train_split=.8,
				 plot=False):
		
		self.train()

		if plot:
			fig = plt.figure()
			ax = fig.add_subplot(111)

		if validate:
			data = data[torch.randperm(len(data))]
			print(f'{len(data)} => ', end='')
			n_train = int(train_split * len(data))
			data, val_data = data[:n_train], data[n_train:]
			print(f'({len(data)}, {len(val_data)})')
		
		ttime = 0
		for epoch in range(n_epochs):
			ts = time.time()
			loss = 0
			self.optim.zero_grad()
			
			if per_epoch_permute:
				data = data[torch.randperm(len(data))]

			train_acc = 0

			for obj in data:				
				o, y_hat = self(obj)
				
				y = torch.zeros(self.num_classes)
				y[obj.y] = 1
				y = y.to(device)
				
				loss = loss + self.criterion(o, y)
				train_acc += 1 if y_hat == obj.y else 0
				del y, y_hat, o, obj
				torch.cuda.empty_cache()

			train_acc = train_acc / len(data)

			self.metrics['loss'].append(loss.item())
			loss.backward()
			self.optim.step()

			# train_acc = self.accuracy(y, y_hat)
			self.metrics['train_acc'].append(train_acc)


			# Validate on validation data
			if validate:
				with torch.no_grad():
					self.eval()

					val_acc = 0
					for obj in val_data:
						o, y_hat = self(obj)
						
						y = torch.zeros(self.num_classes)
						y[obj.y] = 1
						y = y.to(device)
						
						val_acc += 1 if y_hat == obj.y else 0
					
					val_acc = val_acc / len(val_data)
					self.metrics['val_acc'].append(val_acc)
			
			ttime = ttime + (time.time() - ts)

			if plot:
				ax.clear()
				clear_output()
				ax.plot(self.metrics['train_acc'], label='train accuracy')
				ax.plot(self.metrics['val_acc'], label='validation accuracy')
				# ax.set_ylim((0, 1))
				ax.legend()
				display(fig)
				plt.pause(0.2)

			if not plot:
				if (epoch == 0) or ((epoch + 1) % (n_epochs / 10) == 0):
					str_print = f'[{epoch + 1}/{n_epochs}]: loss - {loss.item():.4f}, train accuracy - {train_acc:.4f}, val accuracy - {val_acc:.4f}, avg time - {ttime / (n_epochs / 10):.3f}'
					print(str_print + ' ' * max(100 - len(str_print), 0), end='\n')
					ttime = 0

			self.train()  # Set back to training mode
			if train_acc >= .9 and val_acc >= .9:
				print(" " * 100, end='\r')
				return self.metrics

		print(" " * 100, end='\r')
		return self.metrics









# %%
dataset = ModelNet('ModelNet/', name='10')
dataset = dataset.to(device)

# %%
model = GNN(dataset, 100)
model = model.to(device)
model

# %%
idx = 0
print(f'prediction on object at idx={idx}, predicted label={model.predict(dataset[0].to(device))}, actual label={class_names[dataset[0].y]}')


# %%
model.train_model(dataset, 100)

# %%
torch.save(model.state_dict(), 'model.pkl')
