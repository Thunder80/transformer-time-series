import torch
import torch.nn as nn

class Time2Vec(nn.Module):
   def __init__(self, seq_len):
      super().__init__()
      self.output_dim = output_dim
      self.input_dim = input_dim
      self.w0 = nn.Parameter(torch.Tensor(1, seq_len))
      self.phi0 = nn.Parameter(torch.Tensor(1, seq_len))
      self.W = nn.Parameter(torch.Tensor(seq_len, output_dim-1))
      self.Phi = nn.Parameter(torch.Tensor(seq_len, output_dim-1))
      self.reset_parameters()

   def reset_parameters(self):
      nn.init.uniform_(self.w0, 0, 1)
      nn.init.uniform_(self.phi0, 0, 1)
      nn.init.uniform_(self.W, 0, 1)
      nn.init.uniform_(self.Phi, 0, 1)

   def forward(self, x):
      x = tf.math.reduce_mean(x[:,:,:4], axis=-1) 
      time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
      time_linear = torch.unsqueeze(time_linear, 1)

      time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.  bias_periodic)
      time_periodic = tf.expand_dims(time_periodic, axis=-1)
      return tf.concat([time_linear, time_periodic], axis=-1) 