import torch
import torch.nn as nn

from data import generate_data
from encoder import GCN
from decoder import DecoderCell, ClassificationDecoder

# class AttentionModel(nn.Module):
class Model(nn.Module):
	
	def __init__(self, embed_dim = 128, n_encode_layers = 3, n_heads = 8, tanh_clipping = 10., FF_hidden = 512):
		super().__init__()
		
# 		self.Encoder = GraphAttentionEncoder(embed_dim, n_heads, n_encode_layers, FF_hidden)
		self.Encoder = GCN(embed_dim, embed_dim, embed_dim, embed_dim, n_encode_layers, 5)
		self.se_Decoder = DecoderCell(embed_dim, n_heads, tanh_clipping)
		self.cl_Decoder = ClassificationDecoder(embed_dim)

	def to_ground(self, pi):
		with torch.no_grad():
			n = len(pi)
    # 		print(pi)
    # 		print(pi.size(0))
			dist = torch.zeros((n, 21, 21))
			for num in range(pi.size(0)):      
				for i in range(21 - 1):
    # 				print(pi[num][i], pi[num][i+1])
					dist[num][int(pi[num][i])][int(pi[num][i+1])] = 1
			return dist
		
	def forward(self, x, return_pi = True, decode_type = 'greedy'):
		node, e = self.Encoder(x)
		se_decoder_output = self.se_Decoder(x, node, return_pi = True, decode_type = decode_type)
		cl_decoder_output = self.cl_Decoder(e)
		cost, ll, pi = se_decoder_output
		ground = self.to_ground(pi)
		return cost, ll, pi, ground, cl_decoder_output
		
if __name__ == '__main__':
	model = Model()
	# model = AttentionModel()
	model.train()
	data = generate_data(n_samples = 5, n_customer = 20, seed = 123)
	return_pi = True
	output = model(data, decode_type = 'sampling', return_pi = return_pi)
	if return_pi:
		cost, ll, pi = output
		print('\ncost: ', cost.size(), cost)
		print('\nll: ', ll.size(), ll)
		print('\npi: ', pi.size(), pi)
	else:
		print(output[0])# cost: (batch)
		print(output[1])# ll: (batch)

	cnt = 0
	for i, k in model.state_dict().items():
		print(i, k.size(), torch.numel(k))
		cnt += torch.numel(k)
	print('total parameters:', cnt)

	# output[1].mean().backward()
	# print(model.Decoder.Wout.weight.grad)
	# print(model.Encoder.init_W_depot.weight.grad)