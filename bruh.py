import random
from glob import glob
from tqdm.auto import tqdm

import os
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, fps, global_max_pool, radius
from torch_geometric.nn.conv import PointConv

class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self
		

# Set experiment configs to be synced with wandb
config = AttrDict()
config.modelnet_dataset_alias = "ModelNet10" #@param ["ModelNet10", "ModelNet40"] {type:"raw"}

config.seed = 4242 #@param {type:"number"}
random.seed(config.seed)
torch.manual_seed(config.seed)

config.sample_points = 2048 #@param {type:"slider", min:256, max:4096, step:16}

config.categories = sorted([
	x.split(os.sep)[-2]
	for x in glob(os.path.join(
		config.modelnet_dataset_alias, "raw", '*', ''
	))
])

config.batch_size = 16 #@param {type:"slider", min:4, max:128, step:4}

config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(config.device)
print(f'running on {device}')

config.set_abstraction_ratio_1 = 0.748 #@param {type:"slider", min:0.1, max:1.0, step:0.01}
config.set_abstraction_radius_1 = 0.4817 #@param {type:"slider", min:0.1, max:1.0, step:0.01}
config.set_abstraction_ratio_2 = 0.3316 #@param {type:"slider", min:0.1, max:1.0, step:0.01}
config.set_abstraction_radius_2 = 0.2447 #@param {type:"slider", min:0.1, max:1.0, step:0.01}
config.dropout = 0.1 #@param {type:"slider", min:0.1, max:1.0, step:0.1}

config.learning_rate = 1e-4 #@param {type:"number"}
config.epochs = 10 #@param {type:"slider", min:1, max:100, step:1}
config.num_visualization_samples = 20 #@param {type:"slider", min:1, max:100, step:1}


pre_transform = T.NormalizeScale()
transform = T.SamplePoints(config.sample_points)


train_dataset = ModelNet(
	root=config.modelnet_dataset_alias + '/',
	name=config.modelnet_dataset_alias[-2:],
	train=True,
	transform=transform,
	pre_transform=pre_transform
)
train_loader = DataLoader(
	train_dataset,
	batch_size=config.batch_size,
	shuffle=True,
)

val_dataset = ModelNet(
	root=config.modelnet_dataset_alias + '/',
	name=config.modelnet_dataset_alias[-2:],
	train=False,
	transform=transform,
	pre_transform=pre_transform
)
val_loader = DataLoader(
	val_dataset,
	batch_size=config.batch_size,
	shuffle=False,
)

class SetAbstraction(torch.nn.Module):
	def __init__(self, ratio, r, nn):
		super().__init__()
		self.ratio = ratio
		self.r = r
		self.conv = PointConv(nn, add_self_loops=False)

	def forward(self, x, pos, batch):
		idx = fps(pos, batch, ratio=self.ratio)
		row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
						  max_num_neighbors=64)
		edge_index = torch.stack([col, row], dim=0)
		x_dst = None if x is None else x[idx]
		x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
		pos, batch = pos[idx], batch[idx]
		return x, pos, batch
	

class GlobalSetAbstraction(torch.nn.Module):
	def __init__(self, nn):
		super().__init__()
		self.nn = nn

	def forward(self, x, pos, batch):
		x = self.nn(torch.cat([x, pos], dim=1))
		x = global_max_pool(x, batch)
		pos = pos.new_zeros((x.size(0), 3))
		batch = torch.arange(x.size(0), device=batch.device)
		return x, pos, batch

class PointNet2(torch.nn.Module):
	def __init__(
		self,
		set_abstraction_ratio_1, set_abstraction_ratio_2,
		set_abstraction_radius_1, set_abstraction_radius_2, dropout
	):
		super().__init__()

		# Input channels account for both `pos` and node features.
		self.sa1_module = SetAbstraction(
			set_abstraction_ratio_1,
			set_abstraction_radius_1,
			MLP([3, 64, 64, 128])
		)
		self.sa2_module = SetAbstraction(
			set_abstraction_ratio_2,
			set_abstraction_radius_2,
			MLP([128 + 3, 128, 128, 256])
		)
		self.sa3_module = GlobalSetAbstraction(MLP([256 + 3, 256, 512, 1024]))

		self.mlp = MLP([1024, 512, 256, 10], dropout=dropout, norm=None)

	def forward(self, data):
		sa0_out = (data.x, data.pos, data.batch)
		sa1_out = self.sa1_module(*sa0_out)
		sa2_out = self.sa2_module(*sa1_out)
		sa3_out = self.sa3_module(*sa2_out)
		x, pos, batch = sa3_out

		return self.mlp(x).log_softmax(dim=-1)

# Define PointNet++ model.
model = PointNet2(
	config.set_abstraction_ratio_1,
	config.set_abstraction_ratio_2,
	config.set_abstraction_radius_1,
	config.set_abstraction_radius_2,
	config.dropout
).to(device)

# Define Optimizer
optimizer = torch.optim.Adam(
	model.parameters(), lr=config.learning_rate
)


def train_step(epoch):
	"""Training Step"""
	model.train()
	epoch_loss, correct = 0, 0
	num_train_examples = len(train_loader)
	
	progress_bar = tqdm(
		range(num_train_examples),
		desc=f"Training Epoch {epoch}/{config.epochs}"
	)
	data_iter = iter(train_loader)
	for batch_idx in progress_bar:
		data = next(data_iter).to(device)
		
		optimizer.zero_grad()
		prediction = model(data)
		loss = F.nll_loss(prediction, data.y)
		loss.backward()
		optimizer.step()
		
		epoch_loss += loss.item()
		# correct += (prediction.argmax(dim=-1) == data.y).sum().item()
		correct += prediction.max(1)[1].eq(data.y).sum().item()
	
	epoch_loss = epoch_loss / num_train_examples
	epoch_accuracy = correct / len(train_loader.dataset)
	
	print({
		"Train/Loss": epoch_loss,
		"Train/Accuracy": epoch_accuracy
	})


def val_step(epoch):
	"""Validation Step"""
	model.eval()
	epoch_loss, correct = 0, 0
	num_val_examples = len(val_loader)
	
	progress_bar = tqdm(
		range(num_val_examples),
		desc=f"Validation Epoch {epoch}/{config.epochs}"
	)
	data_iter = iter(val_loader)
	for batch_idx in progress_bar:
		data = next(data_iter).to(device)
		
		with torch.no_grad():
			prediction = model(data)
		
		loss = F.nll_loss(prediction, data.y)
		epoch_loss += loss.item()
		# correct += (prediction.argmax(dim=-1) == data.y).sum().item()
		correct += prediction.max(1)[1].eq(data.y).sum().item()

	
	epoch_loss = epoch_loss / num_val_examples
	epoch_accuracy = correct / len(val_loader.dataset)
	
	print({
		"Validation/Loss": epoch_loss,
		"Validation/Accuracy": epoch_accuracy
	})


for epoch in range(1, config.epochs + 1):
	train_step(epoch)
	val_step(epoch)


