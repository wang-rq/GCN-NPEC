import torch
import torch.nn as nn
# from torchsummary import summary

from layers import MultiHeadAttention
from data import generate_data
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GCNLayer(nn.Module):
    def __init__(self,hidden_dim):
        super(GCNLayer,self).__init__()
        # node GCN Layers
        self.W_node = nn.Linear(hidden_dim, hidden_dim)
        self.V_node_in = nn.Linear(hidden_dim, hidden_dim)
        self.V_node = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim = hidden_dim, num_heads = 1)
        self.Relu=nn.ReLU()
        self.Ln1_node = nn.LayerNorm(hidden_dim)
        self.Ln2_node =nn.LayerNorm(hidden_dim)

        # edge GCN Layers
        self.W_edge = nn.Linear(hidden_dim,hidden_dim)
        self.V_edge_in = nn.Linear(hidden_dim, hidden_dim)
        self.V_edge = nn.Linear(2 * hidden_dim,hidden_dim)
        self.W1_edge = nn.Linear(hidden_dim, hidden_dim)
        self.W2_edge = nn.Linear(hidden_dim, hidden_dim)
        self.W3_edge =nn.Linear(hidden_dim,hidden_dim)
        self.Relu =nn.ReLU()
        self.Ln1_edge = nn.LayerNorm(hidden_dim)
        self.Ln2_edge = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim
 
    def forward(self, x, e,neighbor_index):
        # node embedding
        batch_size=x.size(0)
        node_num =x.size(1)
        node_hidden_dim =x.size(-1)
        t=x.unsqueeze(1).repeat(1,node_num,1,1)
#         print(neighbor_index.shape)
#         print(neighbor_index[1])
        neighbor_index = neighbor_index.unsqueeze(3).repeat(1,1,1,node_hidden_dim)
#         print(neighbor_index.shape)
#         print(neighbor_index[1])
        neighbor = t.gather(2, neighbor_index)
#         print(neighbor.shape)
#         print(neighbor[1])
        neighbor = neighbor.view(batch_size, node_num,-1,node_hidden_dim)
        
#         print(x.shape, neighbor.shape)
        att, _ = self.attn(x, neighbor, neighbor)
        out = self.W_node(att)
        h_nb_node = self.Ln1_node(x + self.Relu(out))
        h_node=self.Ln2_node(h_nb_node+self.Relu(self.V_node(torch.cat([self.V_node_in(x),h_nb_node], dim=-1))))
  
        # edge embedding
        x_from =x.unsqueeze(2).repeat(1,1,node_num,1)
        x_to=x.unsqueeze(1).repeat(1, node_num, 1,1)
        h_nb_edge =self.Ln1_edge(e+self.Relu(self.W_edge(self.W1_edge(e)+self.W2_edge(x_from)+self.W3_edge(x_to))))
        h_edge =self.Ln2_edge(h_nb_edge + self.Relu(self.V_edge(torch.cat([self.V_edge_in(e), h_nb_edge],dim=-1))))
        
        return h_node, h_edge
    
    
class GCN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, node_hidden_dim, edge_hidden_dim, gcn_num_layers, k=10):
        super(GCN, self).__init__()
        self.k = k
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.gcn_num_layers = gcn_num_layers

        self.node_W1 = nn.Linear(2, self.node_hidden_dim)
        self.node_W2 = nn.Linear(2, self.node_hidden_dim // 2)
        self.node_W3 = nn.Linear(1, self.node_hidden_dim // 2)
        self.edge_W4 = nn.Linear(1, self.edge_hidden_dim // 2)
        self.edge_W5 = nn.Linear(1, self.edge_hidden_dim // 2)
        self.nodes_embedding = nn.Linear(
            self.node_hidden_dim, self.node_hidden_dim, bias=False)
        self.edges_embedding = nn.Linear(
            self.edge_hidden_dim, self.edge_hidden_dim, bias=False)
        self.gcn_layers = nn.ModuleList(
            [GCNLayer(self.node_hidden_dim) for i in range(self.gcn_num_layers)])
        self.Relu = nn.ReLU()

    def forward(self,pack):
        de, cus, demand, dis = pack
#         print(de.shape)
#         print(cus.shape)
        node = torch.cat((de.unsqueeze(-2),cus),axis = 1)
        batch_size =node.size(0)
#         print(node.shape)
#         print(batch_size)
        node_num = node.size(1)
#         print(node_num)
        # node=torch.cat([node,timewin],dim=2)
        # edge =torch.cat([dis.unsqueeze(3),timedis.unsqueeze(3)], dim=3)
        '''
        edge = dis.unsqueeze(3).cuda()
        '''
        edge = dis.unsqueeze(3).to(device)
#         device = torch.device('cuda:0')
        '''
        self_edge=torch.arange(0,node_num).unsqueeze(0).repeat(batch_size,1).unsqueeze(2).cuda()
        '''
        self_edge = torch.arange(0, node_num).unsqueeze(0).repeat(batch_size, 1).unsqueeze(2).to(device)
#         print(self_edge.shape)
        order=dis.sort(2)[1]
#         print('order', order.shape)
        '''
        neighbor_index=order[:,:,1:self.k+1].cuda()
        '''
        neighbor_index = order[:, :, 1:self.k + 1].to(device)
#         print('-------------------', neighbor_index.shape)
#         print(neighbor_index)
        with torch.no_grad():
            '''
            a=torch.zeros_like(dis).cuda()
            '''
            a = torch.zeros_like(dis).to(device)
            a=torch.scatter(a,2,neighbor_index,1)
            a=torch.scatter(a,2,self_edge,-1)
#         print(a)
#         print(node.shape)
        '''
        depot=node[:,0,:].cuda()
        '''
        depot = node[:, 0, :].to(device)
        dedmand = torch.zeros((1))
        '''
        demand=demand.unsqueeze(2).cuda()
        customer =node[:,1:,].cuda()
        '''
        demand=demand.unsqueeze(2).to(device)
        customer =node[:,1:,].to(device)
#         print(depot.shape, demand.shape, customer.shape)
        # Node and edge embedding
#         print(depot)
        depot_embedding=self.Relu(self.node_W1(depot))
#         print(self.node_W2(customer).shape, self.node_W3(demand).shape)
        customer_embedding=self.Relu(torch.cat([self.node_W2(customer),self.node_W3(demand)],dim=2))
        x=torch.cat([depot_embedding.unsqueeze(1),customer_embedding],dim=1)
#         print(edge.shape, a.shape)
        e=self.Relu(torch.cat([self.edge_W4(edge),self.edge_W5(a.unsqueeze(3))],dim=3))
        x=self.nodes_embedding(x)
        e=self.edges_embedding(e)
        for layer in range(self.gcn_num_layers):
            x,e =self.gcn_layers[layer](x,e,neighbor_index) #BxVxH,BxVxVxH
        return x, e
    

class Normalization(nn.Module):

	def __init__(self, embed_dim, normalization = 'batch'):
		super().__init__()

		normalizer_class = {
			'batch': nn.BatchNorm1d,
			'instance': nn.InstanceNorm1d}.get(normalization, None)
		self.normalizer = normalizer_class(embed_dim, affine=True)
		# Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
	# 	self.init_parameters()

	# def init_parameters(self):
	# 	for name, param in self.named_parameters():
	# 		stdv = 1. / math.sqrt(param.size(-1))
	# 		param.data.uniform_(-stdv, stdv)

	def forward(self, x):

		if isinstance(self.normalizer, nn.BatchNorm1d):
			# (batch, num_features)
			# https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989
			return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
		
		elif isinstance(self.normalizer, nn.InstanceNorm1d):
			return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
		else:
			assert self.normalizer is None, "Unknown normalizer type"
			return x


class ResidualBlock_BN(nn.Module):
	def __init__(self, MHA, BN, **kwargs):
		super().__init__(**kwargs)
		self.MHA = MHA
		self.BN = BN

	def forward(self, x, mask = None):
		if mask is None:
			return self.BN(x + self.MHA(x))
		return self.BN(x + self.MHA(x, mask))

class SelfAttention(nn.Module):
	def __init__(self, MHA, **kwargs):
		super().__init__(**kwargs)
		self.MHA = MHA

	def forward(self, x, mask = None):
		return self.MHA([x, x, x], mask = mask)

class EncoderLayer(nn.Module):
	# nn.Sequential):
	def __init__(self, n_heads = 8, FF_hidden = 512, embed_dim = 128, **kwargs):
		super().__init__(**kwargs)
		self.n_heads = n_heads
		self.FF_hidden = FF_hidden
		self.BN1 = Normalization(embed_dim, normalization = 'batch')
		self.BN2 = Normalization(embed_dim, normalization = 'batch')

		self.MHA_sublayer = ResidualBlock_BN(
				SelfAttention(
					MultiHeadAttention(n_heads = self.n_heads, embed_dim = embed_dim, need_W = True)
				),
			self.BN1
			)

		self.FF_sublayer = ResidualBlock_BN(
			nn.Sequential(
					nn.Linear(embed_dim, FF_hidden, bias = True),
					nn.ReLU(),
					nn.Linear(FF_hidden, embed_dim, bias = True)
			),
			self.BN2
		)
		
	def forward(self, x, mask=None):
		"""	arg x: (batch, n_nodes, embed_dim)
			return: (batch, n_nodes, embed_dim)
		"""
		return self.FF_sublayer(self.MHA_sublayer(x, mask = mask))
		
class GraphAttentionEncoder(nn.Module):
	def __init__(self, embed_dim = 128, n_heads = 8, n_layers = 3, FF_hidden = 512):
		super().__init__()
		self.init_W_depot = torch.nn.Linear(2, embed_dim, bias = True)
		self.init_W = torch.nn.Linear(3, embed_dim, bias = True)
		self.encoder_layers = nn.ModuleList([EncoderLayer(n_heads, FF_hidden, embed_dim) for _ in range(n_layers)])
	
	def forward(self, x, mask = None):
		""" x[0] -- depot_xy: (batch, 2) --> embed_depot_xy: (batch, embed_dim)
			x[1] -- customer_xy: (batch, n_nodes-1, 2)
			x[2] -- demand: (batch, n_nodes-1)
			--> concated_customer_feature: (batch, n_nodes-1, 3) --> embed_customer_feature: (batch, n_nodes-1, embed_dim)
			embed_x(batch, n_nodes, embed_dim)

			return: (node embeddings(= embedding for all nodes), graph embedding(= mean of node embeddings for graph))
				=((batch, n_nodes, embed_dim), (batch, embed_dim))
		"""
		x = torch.cat([self.init_W_depot(x[0])[:, None, :],
				self.init_W(torch.cat([x[1], x[2][:, :, None]], dim = -1))], dim = 1)
	
		for layer in self.encoder_layers:
			x = layer(x, mask)

		return (x, torch.mean(x, dim = 1))




if __name__ == '__main__':
	batch = 5
	n_nodes = 21
	encoder = GraphAttentionEncoder(n_layers = 1)
	data = generate_data(n_samples = batch, n_customer = n_nodes-1)
	# mask = torch.zeros((batch, n_nodes, 1), dtype = bool)
	output = encoder(data, mask = None)
	print('output[0].shape:', output[0].size())
	print('output[1].shape', output[1].size())
	
	# summary(encoder, [(2), (20,2), (20)])
	cnt = 0
	for i, k in encoder.state_dict().items():
		print(i, k.size(), torch.numel(k))
		cnt += torch.numel(k)
	print(cnt)

	# output[0].mean().backward()
	# print(encoder.init_W_depot.weight.grad)

