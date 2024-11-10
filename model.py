import torch

from torch import nn
from torch.nn import functional as F
from torch_geometric import nn as gnn
from torch_geometric.utils import remove_self_loops, add_self_loops

class List(list):
	def __init__(self, e):
		self.__list = e

	def f(self, *args):
		for key in args:
			print(key)

	def __getitem__(self, keys):
		if isinstance(keys, int):
			return self.__list[keys]
		return [self.__list[e] for e in keys]

# Attempt 6
class SetAbstraction(nn.Module):
	def __init__(self, ratio, ball_query_radius, local_nn):
		super(SetAbstraction, self).__init__()
		self.ratio = ratio
		self.ball_query_radius = ball_query_radius
		self.conv = gnn.PointNetConv(local_nn, add_self_loops=False)

	def forward(self, x, pos, batch):
		idx = gnn.fps(pos, batch, ratio=self.ratio)
		row, col = gnn.radius(
			pos, pos[idx], self.ball_query_radius,
			batch, batch[idx], max_num_neighbors=64)
		edge_index = torch.stack([col, row], dim=0)
		x_dst = None if x is None else x[idx]
		x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
		pos, batch = pos[idx], batch[idx]
		return x, pos, batch

class GlobalSetAbstraction(nn.Module):
	def __init__(self, nn):
		super().__init__()
		self.nn = nn

	def forward(self, x, pos, batch):
		x = self.nn(torch.cat([x, pos], dim=1))
		x = gnn.global_max_pool(x, batch)
		pos = pos.new_zeros((x.size(0), 3))
		batch = torch.arange(x.size(0), device=batch.device)
		return x, pos, batch

class PointNet2(torch.nn.Module):
    def __init__(
        self,
        set_abstraction_ratio_1, set_abstraction_ratio_2,
        set_abstraction_radius_1, set_abstraction_radius_2, dropout
    ):
        super(PointNet2, self).__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SetAbstraction(
            set_abstraction_ratio_1,
            set_abstraction_radius_1,
            gnn.MLP([3, 64, 64, 128])
        )
        self.sa2_module = SetAbstraction(
            set_abstraction_ratio_2,
            set_abstraction_radius_2,
            gnn.MLP([128 + 3, 128, 128, 256])
        )
        self.sa3_module = GlobalSetAbstraction(gnn.MLP([256 + 3, 256, 512, 1024]))

        self.mlp = gnn.MLP([1024, 512, 256, 10], dropout=dropout, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x).log_softmax(dim=-1)


# Attempt 5
class PConv(gnn.MessagePassing):
	def __init__(self, in_channels, out_channels, aggr='max'):
		super(PConv, self).__init__(aggr=aggr)

		self.mlp = nn.Sequential(
			nn.Linear(in_channels + 3, out_channels),
			nn.ReLU(),
			nn.Linear(out_channels, out_channels)
		)

	def forward(self, h, pos, edge_index):
		return self.propagate(edge_index, h=h, pos=pos)
	
	def message(self, h_j, pos_j, pos_i):
		edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
		return self.mlp(edge_feat)

class PointNetOld(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels):
		super(PointNetOld, self).__init__()

		self.gnn_1 = PConv(in_channels, hidden_channels)
		self.gnn_2 = PConv(hidden_channels, hidden_channels)
		self.decoder = nn.Linear(hidden_channels, out_channels)
		self.relu = nn.ReLU()

	def forward(self, x):
		pos = x.pos
		face = x.face

		if face is None:
			edge_index = x.edge_index
		else:
			edge_index, _ = remove_self_loops(torch.cat([face[:2], face[1:], face[[0, 2]]], dim=1))
			
		batch = x.batch


		x = self.gnn_1(pos, pos=pos, edge_index=edge_index)
		x = self.relu(x)
		x = self.gnn_2(x, pos=pos, edge_index=edge_index)
		x = self.relu(x)

		x = gnn.global_max_pool(x, batch)
		x = self.decoder(x)
		return x, x.argmax(dim=-1)


# Attempt 4
def conv_net(in_channels, out_channels, kernel_size=1):
	return nn.Sequential(
		nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size),
		nn.BatchNorm1d(out_channels, momentum=.1),
		nn.ReLU(),
	)

def dense_net(in_channels, out_channels):
	return nn.Sequential(
		nn.Linear(in_features=in_channels, out_features=out_channels),
		nn.BatchNorm1d(out_channels, momentum=.1),
		nn.ReLU(),
	)

class Tnet(nn.Module):
	def __init__(self, in_channels, conv_channels=[32, 64, 512], dense_channels=[256, 128], out_channels=10, aggr='max'):
		super(Tnet, self).__init__()

		self.feature_dim = out_channels
		self.conv = conv_net(in_channels, conv_channels[0], 1)

		for i, e in enumerate(conv_channels[1:]):
			self.conv.extend(
				conv_net(conv_channels[i], e, 1)
			)

		self.dense = dense_net(conv_channels[-1], dense_channels[0])

		for i, e in enumerate(dense_channels[1:]):
			self.dense.extend(
				dense_net(dense_channels[i], e)
			)

		self.out_layer = nn.Linear(
			dense_channels[-1],
			self.feature_dim * self.feature_dim,
		)

	def forward(self, x):
		
		z = self.conv(x)

		# z = z.reshape(*z.shape[:-1])
		z = torch.max(z, dim=-1).values
		# z = z.reshape(1, -1)

		z = self.dense(z)
		z = self.out_layer(z)
		z_mat = z.reshape((z.shape[0], self.feature_dim, self.feature_dim))
		z = z_mat @ x

		id_mat = torch.eye(self.feature_dim).to(z_mat.device)
		reg_loss = torch.mean((z_mat - id_mat).pow(2))

		# z = (t).reshape(*t.shape[:-1])
		return z, reg_loss
	
class PointNet(nn.Module):
	def __init__(self, in_channels, hidden_channels=None, out_channels=10, aggr='mean'):
		super(PointNet, self).__init__()
		default_hidden = [3, 32, 32, 32, 32, 64, 512, 256, 128]
		if hidden_channels is None:
			hidden_channels = default_hidden
		assert len(hidden_channels) == len(default_hidden), f'hidden channels should be of length {len(default_hidden)}. {default_hidden}'

		self.tnet1 = Tnet(in_channels=in_channels, out_channels=hidden_channels[0])

		self.conv1 = nn.Sequential(
			conv_net(hidden_channels[0], hidden_channels[1]),
			conv_net(hidden_channels[1], hidden_channels[2]),
		)

		self.tnet2 = Tnet(in_channels=hidden_channels[2], out_channels=hidden_channels[3])

		self.conv2 = nn.Sequential(
			conv_net(hidden_channels[3], hidden_channels[4]),
			conv_net(hidden_channels[4], hidden_channels[5]),
			conv_net(hidden_channels[5], hidden_channels[6]),
		)


		self.dense = nn.Sequential(
			dense_net(hidden_channels[6], hidden_channels[7]),
			nn.Dropout(.5),
			dense_net(hidden_channels[7], hidden_channels[8]),
			nn.Dropout(.5),
			nn.Linear(hidden_channels[8], out_channels)
		)

	def forward(self, x):
		x, reg_loss1 = self.tnet1(x)
		x = self.conv1(x)
		x, reg_loss2 = self.tnet2(x)
		x = self.conv2(x)
		
		x = torch.max(x, dim=-1).values
		
		x = self.dense(x)
		return x, reg_loss1 + reg_loss2
	
	# def predict(self, x):
	# 	return F.softmax(self(x), dim=-1)
		

# Attempt 3
class PGConvKernel(gnn.MessagePassing):
	def __init__(self, in_dim, feature_dim, out_dim, aggr='max'):
		super(PGConvKernel, self).__init__(aggr=aggr)

		self.fc1 = nn.Sequential(
			nn.Linear(feature_dim, out_dim),
			nn.ReLU(),
		)

		self.fc2 = nn.Sequential(
			nn.Linear(2 * out_dim, feature_dim),
			nn.ReLU()
		)

		self.mlp1 = nn.Sequential(
			nn.Linear(in_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 128),
			nn.ReLU(),
			nn.Linear(128, feature_dim)
		)

		self.mlp2 = nn.Sequential(
			nn.Linear(feature_dim, 32),
			nn.Tanh(),
			nn.Linear(32, 64),
			nn.Tanh(),
			nn.Linear(64, out_dim)
		)

		self.mlp3 = nn.Sequential(
			nn.Linear(feature_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, feature_dim)
		)

	def forward(self, x, edge_index):
		f_in = x.normal
		p_in = x.pos

		f_pos = self.mlp1(p_in)
		
		f_1 = f_in + f_pos
		f_i = self.fc1(f_1)

		return self.propagate(edge_index, f=f_i, f_1=f_1)
	
	def message(self, f_i, f_j):
		delta_f_ij = f_i - f_j
		return delta_f_ij
	
	def aggregate(self, inputs, index):
		return self.mlp2(torch.max(inputs, dim=0)[0])
	
	def update(self, aggr_out, f, f_1):
		x = torch.cat([f, aggr_out], dim=-1)
		x = self.fc2(x)
		f_2 = x + f_1
		f_out = self.mlp3(f_2) + f_2
		return f_out

class PointViG(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(PointViG, self).__init__()
		
		self.embedding = nn.Sequential(
			nn.Linear(in_dim, 64),
		)

		self.encoder = gnn.Sequential(
			'x, edge_index', [
				(PGConvKernel(64, 8, 64), 'x, edge_index -> x'),
				# nn.Tanh(),
				nn.Dropout(p=.3),
				(PGConvKernel(64, 16, 128), 'x, edge_index -> x'),
				(PGConvKernel(128, 32, 256), 'x, edge_index -> x'),
			]
		)

		self.mlp = nn.Sequential(
			nn.Linear(256, 128),
			nn.Tanh(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, out_dim),
		)

	def forward(self, x, edge_index):
		x = self.embedding(x)
		x = self.encoder(x, edge_index)
		x = self.mlp(x)
		return x



# Attempt 2
class PNET(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, class_names, aggr='mean', k_neighbors=5):
		super(PNET, self).__init__()

		self.in_channels = in_channels
		# self.edge_conv = nn.Sequential(
		# 	nn.Linear(2 * in_channels, out_channels)
		# )

		# self.gnn_1 = gnn.EdgeConv(
		# 	self.edge_conv,
		# 	aggr=aggr,
		# )

		self.gnn_1 = gnn.SAGEConv(
			in_channels=self.in_channels,
			out_channels=hidden_channels[0],
			aggr=aggr,
		)

		self.conv = nn.Sequential()
		for i, e in enumerate(hidden_channels[1:]):
			self.conv.extend(
				nn.Sequential(
					nn.Conv1d(hidden_channels[i], e, kernel_size=1, stride=1, padding=0),
					nn.Dropout(round((torch.rand(1) * .6).item(), 3)),
					# nn.BatchNorm1d(e),
					nn.ReLU(),
				)
			)

		self.decoder_1 = nn.Linear(256, 128)
		self.decoder_2 = nn.Linear(256 + 128, 64)
		self.decoder_3 = nn.Linear(128 + 64, 64)
		self.decoder_4 = nn.Linear(64 + 64, 64)
		self.out_layer = nn.Linear(64, out_channels)

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

	def forward(self, x):
		face = x.face

		if face is None:
			edge_index = x.edge_index
		else:
			edge_index, _ = remove_self_loops(torch.cat([face[:2], face[1:], face[[0, 2]]], dim=1))

		batch = x.batch

		x = self.gnn_1(x.pos, edge_index)
		x = self.relu(x)

		x = gnn.global_mean_pool(x, batch)
		x = x.reshape(*x.shape, 1)
		x = self.conv(x)
		# x = self.mlp(x.T)
		x = x.reshape(*x.shape[:-1])
		z_1 = self.decoder_1(x)
		x = torch.concat([z_1, x], dim=1)

		z_2 = self.decoder_2(x)
		x = torch.concat([z_2, z_1], dim=1)

		z_3 = self.decoder_3(x)
		x = torch.concat([z_3, z_2], dim=1)

		x = self.decoder_4(x)

		x = self.out_layer(x)
		
		return x, x.argmax(dim=-1)


# Attempt 1
class GNN(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, class_names, aggr='mean', k_neighbors=5):
		super(GNN, self).__init__()
		self.k_neighbors = k_neighbors

		self.in_channels = in_channels
		self.gnn_1 = gnn.SAGEConv(
			in_channels=self.in_channels,
			out_channels=hidden_channels[0],
			aggr=aggr,
		)

		self.gnn_2 = gnn.SAGEConv(
			in_channels=hidden_channels[0],
			out_channels=hidden_channels[1],
			aggr=aggr,
		)

		self.gnn_3 = gnn.GATConv(
			in_channels=hidden_channels[0] + hidden_channels[1],
			out_channels=hidden_channels[1],
			aggr=aggr,
			heads=2,
			concat=False,
			dropout=.3,
		)

		self.bn = nn.BatchNorm1d(hidden_channels[1])

		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()

		self.decoder = nn.Sequential(
			nn.Linear(
				in_features=hidden_channels[1],
				out_features=hidden_channels[2],
			),
			nn.BatchNorm1d(hidden_channels[2]),
			nn.ReLU(),
			nn.Dropout(.3),
			nn.Linear(
				in_features=hidden_channels[2],
				out_features=hidden_channels[2] // 2,
			),
			nn.BatchNorm1d(hidden_channels[2] // 2),
			nn.ReLU(),
			nn.Linear(
				in_features=hidden_channels[2] // 2,
				out_features=out_channels,
			),
			nn.BatchNorm1d(out_channels),
		)

		self.class_names = class_names

	def forward(self, x):
		face = x.face

		if face is None:
			edge_index = x.edge_index
		else:
			edge_index, _ = remove_self_loops(torch.cat([face[:2], face[1:], face[[0, 2]]], dim=1))

		batch = x.batch

		x_1 = self.gnn_1(x.pos, edge_index)
		x_1 = self.relu(x_1)
		
		x_2 = self.gnn_2(x_1, edge_index)
		x_2 = self.relu(x_2)
		
		x_3 = torch.concat([x_1, x_2], dim=-1)
		x_3 = self.gnn_3(x_3, edge_index)

		# x_1 = gnn.global_mean_pool(x_1, batch)
		# x_2 = gnn.global_mean_pool(x_2, batch)
		x_3 = gnn.global_mean_pool(x_3, batch)

		z = x_3

		# z = self.bn(z)

		z = self.decoder(z)
		return z, z.argmax(dim=-1)


	def predict(self, x):
		return [self.class_names[e] for e in self(x)[1]]
	







