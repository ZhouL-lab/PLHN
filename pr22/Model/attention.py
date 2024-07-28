# The implementation of Transformer;
import torch
from torch import nn
from torch.nn import functional as F

"""This script defines 3D different multi-head attention layers.
"""

class multihead_attention_3d(nn.Module):
	def __init__(self, value, channels,total_key_filters, total_value_filters,output_filters, num_heads,col):
		super(multihead_attention_3d, self).__init__()
		self.training=value
		self.total_key_filters=total_key_filters
		self.total_value_filters=total_value_filters
		self.num_heads=num_heads
		self.col=col
		self.q = nn.Conv3d(channels, total_key_filters, kernel_size=1, padding=0, stride=1)
		self.v=nn.Conv3d(channels, total_key_filters, kernel_size=1, padding=0, stride=1)
		self.k=nn.Conv3d(channels, total_value_filters, kernel_size=1, padding=0, stride=1)
		self.transf=nn.Conv3d(total_value_filters, output_filters, kernel_size=3, padding=1, stride=1)

	def forward(self, x):
		input=x
		q = self.q(x)
		v = self.v(x)
		k=self.k(x)
		###
		total_key_filters=self.total_key_filters
		total_value_filters=self.total_value_filters
		num_heads=self.num_heads

		q = split_heads_3d(q, num_heads)  ##

		v = split_heads_3d(v, num_heads)
		k = split_heads_3d(k, num_heads)

		key_filters_per_head = total_key_filters // num_heads
		q *= key_filters_per_head ** -0.5
		# attention
		x = global_attention_3d(q, k, v, self.training,self.col)
		x = x.permute([0, 2, 3, 4, 1, 5])
		#print('x size {}'.format(x.size()))
		x = combine_last_two_dimensions(x)
		x = x.permute([0,4,1,2, 3])
		#print('x size {}'.format(x.size()))
		# x = Conv3D(x, output_filters, 1, 1, use_bias=True)
		y1 = self.transf(x)
		#print('x size {}'.format(x.size()))
		out=input+y1

		return out


def split_heads_3d(x, num_heads):
	"""Split channels (last dimension) into multiple heads (becomes dimension 1).
	
	Args:
		x: a Tensor with shape [batch, d, h, w, channels]
		num_heads: an integer
	
	Returns:
		a Tensor with shape [batch, num_heads, d, h, w, channels / num_heads]
	"""
	#tf.transpose(split_last_dimension(x, num_heads), [0, 4, 1, 2, 3, 5])
	out=split_last_dimension(x, num_heads)
	out=out.permute([0, 4, 1, 2, 3, 5])

	return out


def split_last_dimension(x, n):
	"""Reshape x so that the last dimension becomes two dimensions.
	The first of these two dimensions is n.

	Args:
		x: a Tensor with shape [..., m]
		n: an integer.

	Returns:
		a Tensor with shape [..., n, m/n]
	"""
	x=x.permute([0,2,3,4,1])
	old_shape =list(x.size()) #x.get_shape().dims
	last = old_shape[-1]
	new_shape = old_shape[:-1] + [n] + [last // n if last else None]

	#ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
	#ret.set_shape(new_shape)  ##new_reshape
	#ret = torch.reshape(x, torch.cat(torch.tensor([old_shape[:-1], [n, -1]]), 0))
	ret=torch.reshape(x,old_shape[:-1] + [n,-1])
	ret=torch.reshape(x,new_shape)

	return ret


def global_attention_3d(q, k, v, training, col,name=None):
	"""global self-attention.
	Args:
		q: a Tensor with shape [batch, heads, _d, _h, _w, channels_k]
		k: a Tensor with shape [batch, heads, d, h, w, channels_k]
		v: a Tensor with shape [batch, heads, d, h, w, channels_v]
		name: an optional string
	Returns:
		a Tensor of shape [batch, heads, _d, _h, _w, channels_v]
	"""
	#new_shape = tf.concat([tf.shape(q)[0:-1], [v.shape[-1].value]], 0) ##
	#new_shape = torch.cat([list(torch.size(q))[0:-1], [list(torch.size(v))[-1].value]], 0)  ##
	qsize=list(q.size())
	new_shape=qsize[0:-1]+[list(v.size())[-1]]
	# flatten q,k,v
	q_new = flatten_3d(q)
	k_new = flatten_3d(k)
	v_new = flatten_3d(v)

	# attention
	output = dot_product_attention(q_new, k_new, v_new, bias=None,
				training=training,col=col, dropout_rate=0.5)

	# putting the representations back in the right place
	output = scatter_3d(output, new_shape)

	return output

def flatten_3d(x):
	"""flatten x."""

	#x_shape = tf.shape(x)
	x_shape = list(x.size())
	# [batch, heads, length, channels], length = d*h*w
	x = reshape_range(x, 2, 5, x_shape[2]*x_shape[3]*x_shape[4])

	return x

def reshape_range(tensor, i, j, shape):
	"""Reshapes a tensor between dimensions i and j."""

	#target_shape = tf.concat(
	#		[tf.shape(tensor)[:i], shape, tf.shape(tensor)[j:]],
	#		axis=0)
	tshape=list(tensor.size())
	target_shape =tshape[:i]+[shape]+tshape[j:]
	#torch.cat([list(torch.size(tensor))[:i], shape, list(torch.size(tensor))[j:]],axis=0)

	return torch.reshape(tensor, target_shape)

def scatter_3d(x, shape):
	"""scatter x."""

	#x = tf.reshape(x, shape)
	x=torch.reshape(x,shape)

	return x


def dot_product_attention(q, k, v, bias, training,col, dropout_rate=0.0, name=None):
	"""Dot-product attention.
	Args:
		q: a Tensor with shape [batch, heads, length_q, channels_k]
		k: a Tensor with shape [batch, heads, length_kv, channels_k]
		v: a Tensor with shape [batch, heads, length_kv, channels_v]
		bias: bias Tensor
		dropout_rate: a floating point number
		name: an optional string

	Returns:
		A Tensor with shape [batch, heads, length_q, channels_v]
	"""
##本质上是3个矩阵相乘；
	# [batch, num_heads, length_q, length_kv]
	#logits = tf.matmul(q, k, transpose_b=True)
	#logits = torch.mul(q, k)
	if col: ##如果是channel-wise attention
		logits=torch.matmul(k.permute(0,1,3,2),q)
	else:  ##如果是localization-wise attention
		logits = torch.matmul(q,k.permute(0, 1, 3, 2))
	#weights = tf.nn.softmax(logits, name="attention_weights") ##
	weights= F.softmax(logits,dim =-1)
	#print('weigts={}'.format(weights.size()))
	# dropping out the attention links for each of the heads
	#weights = tf.layers.dropout(weights, dropout_rate, training) ##
	#if training:
	#	weights=torch.nn.Dropout(0.5)(weights)
	#out=torch.mul(weights, v) #tf.matmul(weights, v)
	if col:
		out = torch.matmul(v,weights)  # tf.matmul(weights, v)
	else:
		out = torch.matmul(weights,v)

	return out

def combine_last_two_dimensions(x):
	"""Reshape x so that the last two dimension become one.

	Args:
		x: a Tensor with shape [..., a, b]

	Returns:
		a Tensor with shape [..., a*b]
	"""

	#old_shape = x.get_shape().dims
	old_shape = list(x.size())

	a, b = old_shape[-2:]
	new_shape = old_shape[:-2] + [a * b if a and b else None]
	#print(new_shape)
	#ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
	ret = torch.reshape(x, new_shape)
	#ret.set_shape(new_shape) ##进行了合并的操作

	return ret
