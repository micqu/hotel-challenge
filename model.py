import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	"""
	Fully connected network
	"""

	def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.0):
		super(Net, self).__init__()

		self.input_dim = input_dim
		self.hidden_dims = hidden_dims
		self.output_dim = output_dim
		self.dropout_p = dropout_p

		self.dims = [self.input_dim]
		self.dims.extend(hidden_dims)
		self.dims.append(self.output_dim)

		self.layers = nn.ModuleList([])

		for i in range(len(self.dims) - 1):
			in_dim = self.dims[i]
			out_dim = self.dims[i + 1]
			self.layers.append(nn.Linear(in_dim, out_dim, bias=True))
		
		# Same initialization across all models
		self.__init_net_weights__()

	def __init_net_weights__(self):
		for m in self.layers:
			m.weight.data.normal_(0.0, 0.1)
			m.bias.data.fill_(0.1)

	def forward(self, x):
		x = x.view(-1, self.input_dim)

		for i, layer in enumerate(self.layers):
			x = layer(x)

			# Do not apply ReLU on the final layer
			if i < (len(self.layers) - 1):
				x = F.relu(x)

            # No dropout on output layer
			if i < (len(self.layers) - 1):
				x = F.dropout(x, p=self.dropout_p, training=self.training)

		# x = F.sigmoid(x)
		x = F.softmax(x, dim=1)
		# return F.log_softmax(x,dim=-1)
		return x