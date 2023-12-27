import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys
from collections import OrderedDict

class TransformerTFA(nn.Module):
    # def __init__(self, hidden_dim = 16, in_channels = 128, max_seq_len = 7, num_classes = 5, nhead=1): ###, n_layers = 1
    def __init__(self, hidden_dim = 128, in_channels = 1024, max_seq_len = 7, num_classes = 5, nhead=1): ###, n_layers = 1
        super().__init__()
        # Patch embeddings
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1)   ### stride default 1 check
        # self.patch_emb = patch_emb
        
        # Positional embeddings for each location
        # self.positional_encodings = nn.Parameter(torch.zeros(max_seq_len, 1, hidden_dim), requires_grad=True)
        self.positional_encodings = nn.Parameter(torch.randn(max_seq_len, 1, hidden_dim), requires_grad=True)
        
        # Classification head
        self.classification = nn.Linear(hidden_dim, num_classes)
        
        ### only for n_layers = 1
        dim_feedforward = hidden_dim*1   #*4
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation="relu", batch_first=False)
        # # Make copies of the transformer layer
        # self.transformer_layers = clone_module_list(transformer_layer, n_layers)

        # `[CLS]` token embedding
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, hidden_dim), requires_grad=True)
        
    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, seq_len]`
        """
        # Get patch embeddings. # Rearrange to shape `[seq_len, batch_size, hidden_dim]`
        x = self.conv(x).squeeze(-1)
        x = x.permute(2, 0, 1)
        
        # Add positional embeddings
        x = x + self.positional_encodings  ##[:x.shape[0]]
        
        # Concatenate the `[CLS]` token embeddings before feeding the transformer
        cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x])

        ### transformer encoder with no attention masking
        x = self.encoder_layer(x)
        
        # Get the transformer output of the `[CLS]` token (which is the first in the sequence).
        x = x[0]

        # Classification head, to get logits
        x = self.classification(x)

        ### pred logits
        return x

class TransformerTFAwLMR(nn.Module):
    # def __init__(self, hidden_dim = 16, in_channels = 128, max_seq_len = 7, num_classes = 5, nhead=1): ###, n_layers = 1
    def __init__(self, hidden_dim = 128, in_channels = 1024, max_seq_len = 7, num_classes = 5, nhead=1, class_counts=None, l=0.25, d=1.0, omega=20, epsilon=0.1):
        super().__init__()
        ### LMR inits:
        self.class_counts = class_counts
        self._num_classes = len(class_counts)
        self.l = l
        self.d = d
        self.omega = omega
        self.epsilon = epsilon
        # self.class_counts = torch.tensor(self.class_counts)
        # pre-compute c(y)
        self.set_class_weights()
        # set mask which identifies classes which are few-shot
        self.set_fs_classes()
        
        # Patch embeddings
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1)   ### stride default 1 check
        # self.patch_emb = patch_emb
        
        # Positional embeddings for each location
        # self.positional_encodings = nn.Parameter(torch.zeros(max_seq_len, 1, hidden_dim), requires_grad=True)
        self.positional_encodings = nn.Parameter(torch.randn(max_seq_len, 1, hidden_dim), requires_grad=True)
        
        # Classification head
        self.classification = nn.Linear(hidden_dim, num_classes)
        
        ### only for n_layers = 1
        dim_feedforward = hidden_dim*1   #*4
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation="relu", batch_first=False)
        # # Make copies of the transformer layer
        # self.transformer_layers = clone_module_list(transformer_layer, n_layers)

        # `[CLS]` token embedding
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, hidden_dim), requires_grad=True)
        
    ### LMR specific functions
    # pre-compute c(y)
    def set_class_weights(self):
        # Eq. 1 in paper.
        tilde_C = 1.0 / torch.log((self.class_counts * self.d) + self.epsilon)
        # Eq. 2 in paper.
        numerator = tilde_C - torch.min(tilde_C)
        denominator = torch.max(tilde_C) - torch.min(tilde_C)
        self.c = numerator / denominator * self.l

    # identify classes which are few-shot. 1 for few-shot, 0 otherwise    
    def set_fs_classes(self):
        fs_classes = torch.where(self.class_counts <= self.omega, 1, 0)
        self.fs_classes = fs_classes #nn.Parameter(fs_classes, requires_grad=False)
        
    # compute reconstructions for each sample and combine with original sample
    # based on the class size contribution
    def reconstruct(self, x, y):
        n_batch, dim = x.shape

        # calculate similarities
        x_norm = F.layer_norm(x, [dim]) / torch.sqrt(torch.tensor(dim))
        sim = torch.matmul(x_norm, x_norm.t())

        # mask to remove similarity to self
        self_mask = torch.eye(n_batch).to(x.device)

        # mask to remove similarity to few-shot classes
        fs_mask = torch.index_select(input=self.fs_classes.to(x.device), dim=0, index=y)

        # Eq. 3 in paper.
        # combine masks to create exclusion mask E. 
        # entries are either 1 (ignore) or 0 (don't ignore)
        E = self_mask + fs_mask
        E = torch.where(E >= 1.0, 1.0, 0.0)

        # Eq. 4 in paper.
        # apply mask and softmax to calculate W. 
        sim = sim - 1e5 * E
        sim = F.softmax(sim, dim=-1)

        # get contribution of reconstruction for each sample, based on class count
        contrib = torch.index_select(input=self.c.to(x.device), dim=0, index=y)

        # Eq. 5 in paper.
        # combine reconstructions with original samples, to get R. 
        reconstructions = torch.matmul(sim, x)
        R = (reconstructions.t() * contrib).t() + (x.t() * (1 - contrib)).t()
        return R


    # perform pairwise label mixing
    def pairwise_mix(self, x, y):
        n_batch = x.shape[0]

        # generate one-hot labels ready for mixing
        y_oh = F.one_hot(y, self.class_counts.shape[0])

        # beta selects other samples to mix with
        beta = torch.randint(low=0, high=n_batch, size=[n_batch])
        beta = F.one_hot(beta, n_batch)

        # alpha are the pairwise mixing weights
        # set half the elements of alpha to 1, otherwise random
        alpha = torch.where(torch.rand(n_batch) > 0.5, torch.rand(n_batch), torch.ones(n_batch))

        # Eq. 6 in paper.
        # Mixing mask M. 
        M = (torch.eye(n_batch) * alpha + beta.t() * (1 - alpha)).t()
        M = M.to(x.device)
        
        # Eq. 7 in main paper.
        x = torch.matmul(M, x)
        y = torch.matmul(M, y_oh.float())
        return x, y

    def forward(self, x, y):
        """
        * `x` is the input image of shape `[batch_size, channels, seq_len]`
        """
        # Get patch embeddings. # Rearrange to shape `[seq_len, batch_size, hidden_dim]`
        x = self.conv(x).squeeze(-1)
        x = x.permute(2, 0, 1)
        
        # Add positional embeddings
        x = x + self.positional_encodings  ##[:x.shape[0]]
        
        # Concatenate the `[CLS]` token embeddings before feeding the transformer
        cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x])

        ### transformer encoder with no attention masking
        x = self.encoder_layer(x)
        
        # Get the transformer output of the `[CLS]` token (which is the first in the sequence).
        x = x[0]

        if self.training:
            x = self.reconstruct(x, y)
            x, y = self.pairwise_mix(x, y)
        else:
            ### one hot to match the format in train (below line not required as anyways normal labels can be used during validation)
            y = F.one_hot(y, self._num_classes).float()
            
        # Classification head, to get logits
        x = self.classification(x)

        ### pred logits
        return x, y

class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x



class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        return logits
    
    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)
    
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)


class InceptionI3dTavg(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3dTavg, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        # x = self.logits(self.dropout(self.avg_pool(x)))
        x = self.logits(self.dropout(torch.mean(self.avg_pool(x), dim=-3).unsqueeze(2)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3).squeeze(2)
        # logits is batch X time X classes, which is what we want to work with
        return logits
    
    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)

class InceptionI3dTavgwLMR_noyield(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    ### LMR inputs
    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,
                 class_counts=None, l=0.25, d=1.0, omega=20, epsilon=0.1, nested_lmr=False):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3dTavgwLMR_noyield, self).__init__()
        ### LMR inits:
        self.nested_lmr = nested_lmr
        self.class_counts = class_counts
        self.l = l
        self.d = d
        self.omega = omega
        self.epsilon = epsilon
        # self.class_counts = torch.tensor(self.class_counts)
        # pre-compute c(y)
        self.set_class_weights()
        # set mask which identifies classes which are few-shot
        self.set_fs_classes()
        
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
    
    ### LMR specific functions
    # pre-compute c(y)
    def set_class_weights(self):
        # Eq. 1 in paper.
        tilde_C = 1.0 / torch.log((self.class_counts * self.d) + self.epsilon)
        # Eq. 2 in paper.
        numerator = tilde_C - torch.min(tilde_C)
        denominator = torch.max(tilde_C) - torch.min(tilde_C)
        self.c = numerator / denominator * self.l
        ### assigning 0 weightage for LMR for the yielding class
        self.c[0] = 0.0

    # identify classes which are few-shot. 1 for few-shot, 0 otherwise    
    def set_fs_classes(self):
        fs_classes = torch.where(self.class_counts <= self.omega, 1, 0)
        self.fs_classes = fs_classes #nn.Parameter(fs_classes, requires_grad=False)
        
    # compute reconstructions for each sample and combine with original sample
    # based on the class size contribution
    def reconstruct(self, x, y):
        n_batch, dim = x.shape

        # calculate similarities
        x_norm = F.layer_norm(x, [dim]) / torch.sqrt(torch.tensor(dim))
        sim = torch.matmul(x_norm, x_norm.t())

        # mask to remove similarity to self
        self_mask = torch.eye(n_batch).to(x.device)

        # mask to remove similarity to few-shot classes
        fs_mask = torch.index_select(input=self.fs_classes.to(x.device), dim=0, index=y)

        # Eq. 3 in paper.
        # combine masks to create exclusion mask E. 
        # entries are either 1 (ignore) or 0 (don't ignore)
        E = self_mask + fs_mask
        E = torch.where(E >= 1.0, 1.0, 0.0)

        # Eq. 4 in paper.
        # apply mask and softmax to calculate W. 
        sim = sim - 1e5 * E
        sim = F.softmax(sim, dim=-1)

        # get contribution of reconstruction for each sample, based on class count
        contrib = torch.index_select(input=self.c.to(x.device), dim=0, index=y)

        # Eq. 5 in paper.
        # combine reconstructions with original samples, to get R. 
        reconstructions = torch.matmul(sim, x)
        R = (reconstructions.t() * contrib).t() + (x.t() * (1 - contrib)).t()
        return R


    # perform pairwise label mixing
    def pairwise_mix(self, x, y):
        n_batch = x.shape[0]

        # generate one-hot labels ready for mixing
        y_oh = F.one_hot(y, self.class_counts.shape[0])

        # beta selects other samples to mix with
        beta = torch.randint(low=0, high=n_batch, size=[n_batch])
        beta = F.one_hot(beta, n_batch)

        # alpha are the pairwise mixing weights
        # set half the elements of alpha to 1, otherwise random
        alpha = torch.where(torch.rand(n_batch) > 0.5, torch.rand(n_batch), torch.ones(n_batch))

        # Eq. 6 in paper.
        # Mixing mask M. 
        M = (torch.eye(n_batch) * alpha + beta.t() * (1 - alpha)).t()
        M = M.to(x.device)
        
        # Eq. 7 in main paper.
        x = torch.matmul(M, x)
        y = torch.matmul(M, y_oh.float())
        return x, y

    ### nested LMR
    def nested_reconstruct(self, x, y_oh):
        n_batch, dim = x.shape

        # calculate similarities
        x_norm = F.layer_norm(x, [dim]) / torch.sqrt(torch.tensor(dim))
        sim = torch.matmul(x_norm, x_norm.t())

        # mask to remove similarity to self
        self_mask = torch.eye(n_batch).to(x.device)

        # # mask to remove similarity to few-shot classes
        # fs_mask = torch.index_select(input=self.fs_classes.to(x.device), dim=0, index=y)
        ### this can be estimated using weighted label-count values (but not req. since neither HDD nor METEOR contains few shot classes)

        # Eq. 3 in paper.
        # combine masks to create exclusion mask E. 
        # entries are either 1 (ignore) or 0 (don't ignore)
        E = self_mask #+ fs_mask
        E = torch.where(E >= 1.0, 1.0, 0.0)

        # Eq. 4 in paper.
        # apply mask and softmax to calculate W. 
        sim = sim - 1e5 * E
        sim = F.softmax(sim, dim=-1)

        # get contribution of reconstruction for each sample, based on class count
        # contrib = torch.index_select(input=self.c.to(x.device), dim=0, index=y)
        contrib = torch.sum(y_oh * self.c.to(x.device), 1)

        # Eq. 5 in paper.
        # combine reconstructions with original samples, to get R. 
        reconstructions = torch.matmul(sim, x)
        R = (reconstructions.t() * contrib).t() + (x.t() * (1 - contrib)).t()
        return R


    # perform pairwise label mixing
    def nested_pairwise_mix(self, x, y_oh):
        n_batch = x.shape[0]

        # # generate one-hot labels ready for mixing
        # y_oh = F.one_hot(y, self.class_counts.shape[0])

        # beta selects other samples to mix with
        beta = torch.randint(low=0, high=n_batch, size=[n_batch])
        beta = F.one_hot(beta, n_batch)

        # alpha are the pairwise mixing weights
        # set half the elements of alpha to 1, otherwise random
        alpha = torch.where(torch.rand(n_batch) > 0.5, torch.rand(n_batch), torch.ones(n_batch))

        # Eq. 6 in paper.
        # Mixing mask M. 
        M = (torch.eye(n_batch) * alpha + beta.t() * (1 - alpha)).t()
        M = M.to(x.device)
        
        # Eq. 7 in main paper.
        x = torch.matmul(M, x)
        y = torch.matmul(M, y_oh.float())
        return x, y
    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x, y):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        ### LMR (with avg pooling along the temporal dimension)
        # x = self.logits(self.dropout(self.avg_pool(x)))
        x = torch.mean(self.avg_pool(x).squeeze(-1).squeeze(-1), dim=-1)
        if self.training:
            x = self.reconstruct(x, y)
            x, y = self.pairwise_mix(x, y)
            
            ### Apply Nested-LMR with 0.5 prob.
            if(torch.rand(1)[0] > 0.5 and self.nested_lmr==True):
                x = self.nested_reconstruct(x, y)
                x, y = self.nested_pairwise_mix(x, y)
        else:
            ### one hot to match the format in train (below line not required as anyways normal labels can be used during validation)
            y = F.one_hot(y, self._num_classes).float()
        x = self.logits(self.dropout(x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3).squeeze(2)
        ## logits is batch X classes, which is what we want to work with
        return logits, y
    
    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


class InceptionI3dTavgwLMR(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    ### LMR inputs
    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,
                 class_counts=None, l=0.25, d=1.0, omega=20, epsilon=0.1, nested_lmr=False):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3dTavgwLMR, self).__init__()
        ### LMR inits:
        self.nested_lmr = nested_lmr
        self.class_counts = class_counts
        self.l = l
        self.d = d
        self.omega = omega
        self.epsilon = epsilon
        # self.class_counts = torch.tensor(self.class_counts)
        # pre-compute c(y)
        self.set_class_weights()
        # set mask which identifies classes which are few-shot
        self.set_fs_classes()
        
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
    
    ### LMR specific functions
    # pre-compute c(y)
    def set_class_weights(self):
        # Eq. 1 in paper.
        tilde_C = 1.0 / torch.log((self.class_counts * self.d) + self.epsilon)
        # Eq. 2 in paper.
        numerator = tilde_C - torch.min(tilde_C)
        denominator = torch.max(tilde_C) - torch.min(tilde_C)
        self.c = numerator / denominator * self.l
        
    # identify classes which are few-shot. 1 for few-shot, 0 otherwise    
    def set_fs_classes(self):
        fs_classes = torch.where(self.class_counts <= self.omega, 1, 0)
        self.fs_classes = fs_classes #nn.Parameter(fs_classes, requires_grad=False)
        
    # compute reconstructions for each sample and combine with original sample
    # based on the class size contribution
    def reconstruct(self, x, y):
        n_batch, dim = x.shape

        # calculate similarities
        x_norm = F.layer_norm(x, [dim]) / torch.sqrt(torch.tensor(dim))
        sim = torch.matmul(x_norm, x_norm.t())

        # mask to remove similarity to self
        self_mask = torch.eye(n_batch).to(x.device)

        # mask to remove similarity to few-shot classes
        fs_mask = torch.index_select(input=self.fs_classes.to(x.device), dim=0, index=y)

        # Eq. 3 in paper.
        # combine masks to create exclusion mask E. 
        # entries are either 1 (ignore) or 0 (don't ignore)
        E = self_mask + fs_mask
        E = torch.where(E >= 1.0, 1.0, 0.0)

        # Eq. 4 in paper.
        # apply mask and softmax to calculate W. 
        sim = sim - 1e5 * E
        sim = F.softmax(sim, dim=-1)

        # get contribution of reconstruction for each sample, based on class count
        contrib = torch.index_select(input=self.c.to(x.device), dim=0, index=y)

        # Eq. 5 in paper.
        # combine reconstructions with original samples, to get R. 
        reconstructions = torch.matmul(sim, x)
        R = (reconstructions.t() * contrib).t() + (x.t() * (1 - contrib)).t()
        return R


    # perform pairwise label mixing
    def pairwise_mix(self, x, y):
        n_batch = x.shape[0]

        # generate one-hot labels ready for mixing
        y_oh = F.one_hot(y, self.class_counts.shape[0])

        # beta selects other samples to mix with
        beta = torch.randint(low=0, high=n_batch, size=[n_batch])
        beta = F.one_hot(beta, n_batch)

        # alpha are the pairwise mixing weights
        # set half the elements of alpha to 1, otherwise random
        alpha = torch.where(torch.rand(n_batch) > 0.5, torch.rand(n_batch), torch.ones(n_batch))

        # Eq. 6 in paper.
        # Mixing mask M. 
        M = (torch.eye(n_batch) * alpha + beta.t() * (1 - alpha)).t()
        M = M.to(x.device)
        
        # Eq. 7 in main paper.
        x = torch.matmul(M, x)
        y = torch.matmul(M, y_oh.float())
        return x, y

    ### nested LMR
    def nested_reconstruct(self, x, y_oh):
        n_batch, dim = x.shape

        # calculate similarities
        x_norm = F.layer_norm(x, [dim]) / torch.sqrt(torch.tensor(dim))
        sim = torch.matmul(x_norm, x_norm.t())

        # mask to remove similarity to self
        self_mask = torch.eye(n_batch).to(x.device)

        # # mask to remove similarity to few-shot classes
        # fs_mask = torch.index_select(input=self.fs_classes.to(x.device), dim=0, index=y)
        ### this can be estimated using weighted label-count values (but not req. since neither HDD nor METEOR contains few shot classes)

        # Eq. 3 in paper.
        # combine masks to create exclusion mask E. 
        # entries are either 1 (ignore) or 0 (don't ignore)
        E = self_mask #+ fs_mask
        E = torch.where(E >= 1.0, 1.0, 0.0)

        # Eq. 4 in paper.
        # apply mask and softmax to calculate W. 
        sim = sim - 1e5 * E
        sim = F.softmax(sim, dim=-1)

        # get contribution of reconstruction for each sample, based on class count
        # contrib = torch.index_select(input=self.c.to(x.device), dim=0, index=y)
        contrib = torch.sum(y_oh * self.c.to(x.device), 1)

        # Eq. 5 in paper.
        # combine reconstructions with original samples, to get R. 
        reconstructions = torch.matmul(sim, x)
        R = (reconstructions.t() * contrib).t() + (x.t() * (1 - contrib)).t()
        return R


    # perform pairwise label mixing
    def nested_pairwise_mix(self, x, y_oh):
        n_batch = x.shape[0]

        # # generate one-hot labels ready for mixing
        # y_oh = F.one_hot(y, self.class_counts.shape[0])

        # beta selects other samples to mix with
        beta = torch.randint(low=0, high=n_batch, size=[n_batch])
        beta = F.one_hot(beta, n_batch)

        # alpha are the pairwise mixing weights
        # set half the elements of alpha to 1, otherwise random
        alpha = torch.where(torch.rand(n_batch) > 0.5, torch.rand(n_batch), torch.ones(n_batch))

        # Eq. 6 in paper.
        # Mixing mask M. 
        M = (torch.eye(n_batch) * alpha + beta.t() * (1 - alpha)).t()
        M = M.to(x.device)
        
        # Eq. 7 in main paper.
        x = torch.matmul(M, x)
        y = torch.matmul(M, y_oh.float())
        return x, y
    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x, y):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        ### LMR (with avg pooling along the temporal dimension)
        # x = self.logits(self.dropout(self.avg_pool(x)))
        x = torch.mean(self.avg_pool(x).squeeze(-1).squeeze(-1), dim=-1)
        if self.training:
            x = self.reconstruct(x, y)
            x, y = self.pairwise_mix(x, y)
            
            ### Apply Nested-LMR with 0.5 prob.
            if(torch.rand(1)[0] > 0.5 and self.nested_lmr==True):
                x = self.nested_reconstruct(x, y)
                x, y = self.nested_pairwise_mix(x, y)
        else:
            ### one hot to match the format in train (below line not required as anyways normal labels can be used during validation)
            y = F.one_hot(y, self._num_classes).float()
        x = self.logits(self.dropout(x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3).squeeze(2)
        ## logits is batch X classes, which is what we want to work with
        return logits, y
    
    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


class InceptionI3dNoTavgwLMR(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    ### LMR inputs
    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,
                 class_counts=None, l=0.25, d=1.0, omega=20, epsilon=0.1, nested_lmr=False):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3dNoTavgwLMR, self).__init__()
        ### LMR inits:
        self.nested_lmr = nested_lmr
        self.class_counts = class_counts
        self.l = l
        self.d = d
        self.omega = omega
        self.epsilon = epsilon
        # self.class_counts = torch.tensor(self.class_counts)
        # pre-compute c(y)
        self.set_class_weights()
        # set mask which identifies classes which are few-shot
        self.set_fs_classes()
        
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
    
    ### LMR specific functions
    # pre-compute c(y)
    def set_class_weights(self):
        # Eq. 1 in paper.
        tilde_C = 1.0 / torch.log((self.class_counts * self.d) + self.epsilon)
        # Eq. 2 in paper.
        numerator = tilde_C - torch.min(tilde_C)
        denominator = torch.max(tilde_C) - torch.min(tilde_C)
        self.c = numerator / denominator * self.l

    # identify classes which are few-shot. 1 for few-shot, 0 otherwise    
    def set_fs_classes(self):
        fs_classes = torch.where(self.class_counts <= self.omega, 1, 0)
        self.fs_classes = fs_classes #nn.Parameter(fs_classes, requires_grad=False)
        
    # compute reconstructions for each sample and combine with original sample
    # based on the class size contribution
    def reconstruct(self, x, y):
        n_batch, dim = x.shape

        # calculate similarities
        x_norm = F.layer_norm(x, [dim]) / torch.sqrt(torch.tensor(dim))
        sim = torch.matmul(x_norm, x_norm.t())

        # mask to remove similarity to self
        self_mask = torch.eye(n_batch).to(x.device)

        # mask to remove similarity to few-shot classes
        fs_mask = torch.index_select(input=self.fs_classes.to(x.device), dim=0, index=y)

        # Eq. 3 in paper.
        # combine masks to create exclusion mask E. 
        # entries are either 1 (ignore) or 0 (don't ignore)
        E = self_mask + fs_mask
        E = torch.where(E >= 1.0, 1.0, 0.0)

        # Eq. 4 in paper.
        # apply mask and softmax to calculate W. 
        sim = sim - 1e5 * E
        sim = F.softmax(sim, dim=-1)

        # get contribution of reconstruction for each sample, based on class count
        contrib = torch.index_select(input=self.c.to(x.device), dim=0, index=y)

        # Eq. 5 in paper.
        # combine reconstructions with original samples, to get R. 
        reconstructions = torch.matmul(sim, x)
        R = (reconstructions.t() * contrib).t() + (x.t() * (1 - contrib)).t()
        return R


    # perform pairwise label mixing
    def pairwise_mix(self, x, y):
        n_batch = x.shape[0]

        # generate one-hot labels ready for mixing
        y_oh = F.one_hot(y, self.class_counts.shape[0])

        # beta selects other samples to mix with
        beta = torch.randint(low=0, high=n_batch, size=[n_batch])
        beta = F.one_hot(beta, n_batch)

        # alpha are the pairwise mixing weights
        # set half the elements of alpha to 1, otherwise random
        alpha = torch.where(torch.rand(n_batch) > 0.5, torch.rand(n_batch), torch.ones(n_batch))

        # Eq. 6 in paper.
        # Mixing mask M. 
        M = (torch.eye(n_batch) * alpha + beta.t() * (1 - alpha)).t()
        M = M.to(x.device)
        
        # Eq. 7 in main paper.
        # x = torch.matmul(M, x)
        y = torch.matmul(M, y_oh.float())
        return M, y

    ### nested LMR
    def nested_reconstruct(self, x, y_oh):
        n_batch, dim = x.shape

        # calculate similarities
        x_norm = F.layer_norm(x, [dim]) / torch.sqrt(torch.tensor(dim))
        sim = torch.matmul(x_norm, x_norm.t())

        # mask to remove similarity to self
        self_mask = torch.eye(n_batch).to(x.device)

        # # mask to remove similarity to few-shot classes
        # fs_mask = torch.index_select(input=self.fs_classes.to(x.device), dim=0, index=y)
        ### this can be estimated using weighted label-count values (but not req. since neither HDD nor METEOR contains few shot classes)

        # Eq. 3 in paper.
        # combine masks to create exclusion mask E. 
        # entries are either 1 (ignore) or 0 (don't ignore)
        E = self_mask #+ fs_mask
        E = torch.where(E >= 1.0, 1.0, 0.0)

        # Eq. 4 in paper.
        # apply mask and softmax to calculate W. 
        sim = sim - 1e5 * E
        sim = F.softmax(sim, dim=-1)

        # get contribution of reconstruction for each sample, based on class count
        # contrib = torch.index_select(input=self.c.to(x.device), dim=0, index=y)
        contrib = torch.sum(y_oh * self.c.to(x.device), 1)

        # Eq. 5 in paper.
        # combine reconstructions with original samples, to get R. 
        reconstructions = torch.matmul(sim, x)
        R = (reconstructions.t() * contrib).t() + (x.t() * (1 - contrib)).t()
        return R

    # perform pairwise label mixing
    def nested_pairwise_mix(self, x, y_oh):
        n_batch = x.shape[0]

        # # generate one-hot labels ready for mixing
        # y_oh = F.one_hot(y, self.class_counts.shape[0])

        # beta selects other samples to mix with
        beta = torch.randint(low=0, high=n_batch, size=[n_batch])
        beta = F.one_hot(beta, n_batch)

        # alpha are the pairwise mixing weights
        # set half the elements of alpha to 1, otherwise random
        alpha = torch.where(torch.rand(n_batch) > 0.5, torch.rand(n_batch), torch.ones(n_batch))

        # Eq. 6 in paper.
        # Mixing mask M. 
        M = (torch.eye(n_batch) * alpha + beta.t() * (1 - alpha)).t()
        M = M.to(x.device)
        
        # Eq. 7 in main paper.
        # x = torch.matmul(M, x)
        y = torch.matmul(M, y_oh.float())
        return M, y
    

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x, y):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        ### LMR (with avg pooling along the temporal dimension)
        # x = self.logits(self.dropout(self.avg_pool(x)))
        # x = torch.mean(self.avg_pool(x).squeeze(-1).squeeze(-1), dim=-1)
        
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        # print(x.size(), y.size())
        x_reconstructed = []
        y_tlabels = []
        nested_lmr_prob = torch.rand(1)[0]
        for xti in range(x.size(-1)):
            xtf = x[...,xti]
            if self.training:
                xtf = self.reconstruct(xtf, y)
                if(xti==0):
                    Mix_mat, yl = self.pairwise_mix(xtf, y)
                xtf = torch.matmul(Mix_mat, xtf)
                
                ### Apply Nested-LMR with 0.5 prob.
                if(nested_lmr_prob > 0.5 and self.nested_lmr==True):
                    xtf = self.nested_reconstruct(xtf, yl)
                    if(xti==0):
                        Mix_mat_nested, yl = self.nested_pairwise_mix(xtf, yl)
                    xtf = torch.matmul(Mix_mat_nested, xtf)
                    
            else:
                ### one hot to match the format in train (below line not required as anyways normal labels can be used during validation)
                yl = F.one_hot(y, self._num_classes).float()
            x_reconstructed.append(xtf)
            y_tlabels.append(yl)
        x_reconstructed = torch.stack(x_reconstructed,-1)
        y_tlabels = torch.stack(y_tlabels,-1)
        
        x_reconstructed = x_reconstructed.unsqueeze(-1).unsqueeze(-1).permute(0, 1, 3, 4, 2)
        # print(x_reconstructed.size(), y_tlabels.size())
        x = self.logits(self.dropout(x_reconstructed))
        # print(x.size())
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(2)  #.squeeze(3)
        ## logits is batch X classes, which is what we want to work with
        # return logits, y
        # print(logits.size(), x.squeeze(3).squeeze(2).size())
        return logits, y_tlabels
    
    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)

class InceptionI3dBeforeLMR(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3dBeforeLMR, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()


    ### LMR specific functions
    def replace_logits_wfclayer(self, num_classes):
        self._num_classes = num_classes
        self.logits = nn.Linear(1024, num_classes)
    
    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x, y):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        ### Pre-train before applying LMR to these features (with avg pooling along the temporal dimension)
        x = torch.mean(self.avg_pool(x).squeeze(-1).squeeze(-1), dim=-1)
        y = F.one_hot(y, self._num_classes).float()
        
        logits = self.logits(self.dropout(x))
        return logits, y
    
    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


class InceptionI3dwLMR(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    ### LMR inputs
    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,
                 class_counts=None, l=0.25, d=1.0, omega=20, epsilon=0.1):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
        
        super(InceptionI3dwLMR, self).__init__()
        ### LMR inits:
        self.class_counts = class_counts
        self.l = l
        self.d = d
        self.omega = omega
        self.epsilon = epsilon
        # self.class_counts = torch.tensor(self.class_counts)
        # pre-compute c(y)
        self.set_class_weights()
        # set mask which identifies classes which are few-shot
        self.set_fs_classes()

        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
    
    ### LMR specific functions
    def replace_logits_wfclayer(self):
        self.logits = nn.Linear(1024, self.class_counts.shape[0])
    
    # pre-compute c(y)
    def set_class_weights(self):
        # Eq. 1 in paper.
        tilde_C = 1.0 / torch.log((self.class_counts * self.d) + self.epsilon)
        # Eq. 2 in paper.
        numerator = tilde_C - torch.min(tilde_C)
        denominator = torch.max(tilde_C) - torch.min(tilde_C)
        self.c = numerator / denominator * self.l

    # identify classes which are few-shot. 1 for few-shot, 0 otherwise    
    def set_fs_classes(self):
        fs_classes = torch.where(self.class_counts <= self.omega, 1, 0)
        self.fs_classes = fs_classes #nn.Parameter(fs_classes, requires_grad=False)
        
    # compute reconstructions for each sample and combine with original sample
    # based on the class size contribution
    def reconstruct(self, x, y):
        n_batch, dim = x.shape

        # calculate similarities
        x_norm = F.layer_norm(x, [dim]) / torch.sqrt(torch.tensor(dim))
        sim = torch.matmul(x_norm, x_norm.t())

        # mask to remove similarity to self
        self_mask = torch.eye(n_batch).to(x.device)

        # mask to remove similarity to few-shot classes
        fs_mask = torch.index_select(input=self.fs_classes.to(x.device), dim=0, index=y)

        # Eq. 3 in paper.
        # combine masks to create exclusion mask E. 
        # entries are either 1 (ignore) or 0 (don't ignore)
        E = self_mask + fs_mask
        E = torch.where(E >= 1.0, 1.0, 0.0)

        # Eq. 4 in paper.
        # apply mask and softmax to calculate W. 
        sim = sim - 1e5 * E
        sim = F.softmax(sim, dim=-1)

        # get contribution of reconstruction for each sample, based on class count
        contrib = torch.index_select(input=self.c.to(x.device), dim=0, index=y)

        # Eq. 5 in paper.
        # combine reconstructions with original samples, to get R. 
        reconstructions = torch.matmul(sim, x)
        R = (reconstructions.t() * contrib).t() + (x.t() * (1 - contrib)).t()
        return R


    # perform pairwise label mixing
    def pairwise_mix(self, x, y):
        n_batch = x.shape[0]

        # generate one-hot labels ready for mixing
        y_oh = F.one_hot(y, self.class_counts.shape[0])

        # beta selects other samples to mix with
        beta = torch.randint(low=0, high=n_batch, size=[n_batch])
        beta = F.one_hot(beta, n_batch)

        # alpha are the pairwise mixing weights
        # set half the elements of alpha to 1, otherwise random
        alpha = torch.where(torch.rand(n_batch) > 0.5, torch.rand(n_batch), torch.ones(n_batch))

        # Eq. 6 in paper.
        # Mixing mask M. 
        M = (torch.eye(n_batch) * alpha + beta.t() * (1 - alpha)).t()
        M = M.to(x.device)
        
        # Eq. 7 in main paper.
        x = torch.matmul(M, x)
        y = torch.matmul(M, y_oh.float())
        return x, y


    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x, y):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        ### LMR (with max pooling along the temporal dimension)
        # x = torch.max(self.avg_pool(x).squeeze(-1).squeeze(-1), dim=-1)[0]
        x = torch.mean(self.avg_pool(x).squeeze(-1).squeeze(-1), dim=-1)
        if self.training:
            x = self.reconstruct(x, y)
            x, y = self.pairwise_mix(x, y)
        else:
            ### one hot to match the format in train (below line not required as anyways normal labels can be used during validation)
            y = F.one_hot(y, self._num_classes).float()
        logits = self.logits(self.dropout(x))
        return logits, y
    
        ### No LMR
        # x = self.logits(self.dropout(self.avg_pool(x)))
        # if self._spatial_squeeze:
        #     logits = x.squeeze(3).squeeze(3)
        # # logits is batch X time X classes, which is what we want to work with
        # return logits
        

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


class InceptionI3dRNN(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    ### LMR inputs
    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
        
        super(InceptionI3dRNN, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
        self.build()
        
        # ### Temporal feature aggregation
        # self.num_layers = 1
        # self.hidden_size = 32 #8 #16
        # self.sampfeatdim = 128
        # self.fc = None
        # ### is Relu required in between?
        # self.gru = None
        
        
    def add_RNNagg(self, num_classes):
        self._num_classes = num_classes
        ### Temporal feature aggregation
        self.num_layers = 1
        self.hidden_size = 32 #32 #8 #16
        self.sampfeatdim = 256 #128
        self.fc = nn.Linear(1024, self.sampfeatdim)
        ### is Relu required in between?
        self.gru = nn.GRU(self.sampfeatdim, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.dropout1 =  nn.Dropout(0.1)
        self.logits = nn.Linear(self.hidden_size*2, self._num_classes)
        
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        ### LMR (with max pooling along the temporal dimension)
        # x = torch.max(self.avg_pool(x).squeeze(-1).squeeze(-1), dim=-1)[0]
        # x = torch.mean(self.avg_pool(x).squeeze(-1).squeeze(-1), dim=-1)
        xt = self.dropout(self.avg_pool(x).squeeze(-1).squeeze(-1))
        # print(xt.shape)
        
        xcat = None 
        for xit in range(xt.shape[-1]):
            if(xcat is None):
                xcat = self.fc(xt[...,xit]).unsqueeze(-1)
            else:
                xcat = torch.cat([xcat, self.fc(xt[...,xit]).unsqueeze(-1)], -1)
        xcat = xcat.transpose(2,1)
        # print(xcat.shape)
        h0 = torch.zeros(self.num_layers * 2, xcat.size(0), self.hidden_size).to(x.device)
        out, hidden_st = self.gru(xcat, h0)
        x = out[:, -1, :]
        # print(x.shape, out.shape, hidden_st.shape)
        
        logits = self.logits(self.dropout1(x))
        return logits

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)

class InceptionI3dRNNwLMR(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    ### LMR inputs
    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,
                 class_counts=None, l=0.25, d=1.0, omega=20, epsilon=0.1):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
        
        super(InceptionI3dRNNwLMR, self).__init__()
        ### LMR inits:
        self.class_counts = class_counts
        self.l = l
        self.d = d
        self.omega = omega
        self.epsilon = epsilon
        # self.class_counts = torch.tensor(self.class_counts)
        # pre-compute c(y)
        self.set_class_weights()
        # set mask which identifies classes which are few-shot
        self.set_fs_classes()

        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        # self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
        #                      kernel_shape=[1, 1, 1],
        #                      padding=0,
        #                      activation_fn=None,
        #                      use_batch_norm=False,
        #                      use_bias=True,
        #                      name='logits')
        
        self.build()
        
        ### Temporal feature aggregation
        self.num_layers = 1
        self.hidden_size = 32 #32 #8 #16
        self.sampfeatdim = 256 #128
        self.fc = nn.Linear(1024, self.sampfeatdim)
        ### is Relu required in between?
        self.gru = nn.GRU(self.sampfeatdim, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.dropout1 =  nn.Dropout(0.1)
        self.logits = nn.Linear(self.hidden_size*2, self._num_classes)
        

    ### LMR specific functions
    # pre-compute c(y)
    def set_class_weights(self):
        # Eq. 1 in paper.
        tilde_C = 1.0 / torch.log((self.class_counts * self.d) + self.epsilon)
        # Eq. 2 in paper.
        numerator = tilde_C - torch.min(tilde_C)
        denominator = torch.max(tilde_C) - torch.min(tilde_C)
        self.c = numerator / denominator * self.l

    # identify classes which are few-shot. 1 for few-shot, 0 otherwise    
    def set_fs_classes(self):
        fs_classes = torch.where(self.class_counts <= self.omega, 1, 0)
        self.fs_classes = fs_classes #nn.Parameter(fs_classes, requires_grad=False)
        
    # compute reconstructions for each sample and combine with original sample
    # based on the class size contribution
    def reconstruct(self, x, y):
        n_batch, dim = x.shape

        # calculate similarities
        x_norm = F.layer_norm(x, [dim]) / torch.sqrt(torch.tensor(dim))
        sim = torch.matmul(x_norm, x_norm.t())

        # mask to remove similarity to self
        self_mask = torch.eye(n_batch).to(x.device)

        # mask to remove similarity to few-shot classes
        fs_mask = torch.index_select(input=self.fs_classes.to(x.device), dim=0, index=y)

        # Eq. 3 in paper.
        # combine masks to create exclusion mask E. 
        # entries are either 1 (ignore) or 0 (don't ignore)
        E = self_mask + fs_mask
        E = torch.where(E >= 1.0, 1.0, 0.0)

        # Eq. 4 in paper.
        # apply mask and softmax to calculate W. 
        sim = sim - 1e5 * E
        sim = F.softmax(sim, dim=-1)

        # get contribution of reconstruction for each sample, based on class count
        contrib = torch.index_select(input=self.c.to(x.device), dim=0, index=y)

        # Eq. 5 in paper.
        # combine reconstructions with original samples, to get R. 
        reconstructions = torch.matmul(sim, x)
        R = (reconstructions.t() * contrib).t() + (x.t() * (1 - contrib)).t()
        return R


    # perform pairwise label mixing
    def pairwise_mix(self, x, y):
        n_batch = x.shape[0]

        # generate one-hot labels ready for mixing
        y_oh = F.one_hot(y, self.class_counts.shape[0])

        # beta selects other samples to mix with
        beta = torch.randint(low=0, high=n_batch, size=[n_batch])
        beta = F.one_hot(beta, n_batch)

        # alpha are the pairwise mixing weights
        # set half the elements of alpha to 1, otherwise random
        alpha = torch.where(torch.rand(n_batch) > 0.5, torch.rand(n_batch), torch.ones(n_batch))

        # Eq. 6 in paper.
        # Mixing mask M. 
        M = (torch.eye(n_batch) * alpha + beta.t() * (1 - alpha)).t()
        M = M.to(x.device)
        
        # Eq. 7 in main paper.
        x = torch.matmul(M, x)
        y = torch.matmul(M, y_oh.float())
        return x, y


    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x, y):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        ### LMR (with max pooling along the temporal dimension)
        # x = torch.max(self.avg_pool(x).squeeze(-1).squeeze(-1), dim=-1)[0]
        # x = torch.mean(self.avg_pool(x).squeeze(-1).squeeze(-1), dim=-1)
        xt = self.dropout(self.avg_pool(x).squeeze(-1).squeeze(-1))
        # print(xt.shape)
        
        xcat = None 
        for xit in range(xt.shape[-1]):
            if(xcat is None):
                xcat = self.fc(xt[...,xit]).unsqueeze(-1)
            else:
                xcat = torch.cat([xcat, self.fc(xt[...,xit]).unsqueeze(-1)], -1)
        xcat = xcat.transpose(2,1)
        # print(xcat.shape)
        h0 = torch.zeros(self.num_layers * 2, xcat.size(0), self.hidden_size).to(x.device)
        out, hidden_st = self.gru(xcat, h0)
        x = out[:, -1, :]
        # print(x.shape, out.shape, hidden_st.shape)
        
        if self.training:
            x = self.reconstruct(x, y)
            x, y = self.pairwise_mix(x, y)
        else:
            ### one hot to match the format in train (below line not required as anyways normal labels can be used during validation)
            y = F.one_hot(y, self._num_classes).float()
        logits = self.logits(self.dropout1(x))
        return logits, y
    
        ### No LMR
        # x = self.logits(self.dropout(self.avg_pool(x)))
        # if self._spatial_squeeze:
        #     logits = x.squeeze(3).squeeze(3)
        # # logits is batch X time X classes, which is what we want to work with
        # return logits
        

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)



class InceptionI3dRNNdefective(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    ### LMR inputs
    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):   ##, dropout_keep_prob=0.2, 0.5
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
        
        super(InceptionI3dRNNdefective, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
        self.build()
        
        # ### Temporal feature aggregation
        # self.num_layers = 1
        # self.hidden_size = 32 #8 #16
        # self.sampfeatdim = 128
        # self.fc = None
        # ### is Relu required in between?
        # self.gru = None
        
        
    def add_RNNagg(self, num_classes):
        self._num_classes = num_classes
        ### Temporal feature aggregation
        self.num_layers = 1
        self.hidden_size = 32 #32 #8 #16
        self.sampfeatdim = 128
        self.fc = nn.Linear(1024, self.sampfeatdim)
        ### is Relu required in between?
        self.gru = nn.GRU(self.sampfeatdim, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        # self.dropout1 =  nn.Dropout(0.1)
        self.logits = nn.Linear(self.hidden_size*2, self._num_classes)
        
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        ### LMR (with max pooling along the temporal dimension)
        # x = torch.max(self.avg_pool(x).squeeze(-1).squeeze(-1), dim=-1)[0]
        # x = torch.mean(self.avg_pool(x).squeeze(-1).squeeze(-1), dim=-1)
        xt = self.avg_pool(x).squeeze(-1).squeeze(-1)
        # print(xt.shape)
        
        xcat = None 
        for xit in range(xt.shape[-1]):
            if(xcat is None):
                xcat = self.fc(xt[...,xit]).unsqueeze(-1)
            else:
                xcat = torch.cat([xcat, self.fc(xt[...,xit]).unsqueeze(-1)], -1)
        xcat = xcat.transpose(2,1)
        # print(xcat.shape)
        h0 = torch.zeros(self.num_layers * 2, xcat.size(0), self.hidden_size).to(x.device)
        out, hidden_st = self.gru(xcat, h0)
        x = out[:, -1, :]
        # print(x.shape, out.shape, hidden_st.shape)
        
        logits = self.logits(self.dropout(x))
        return logits

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


class InceptionI3dT3dconv(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3dT3dconv, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()


    def add_reinit_headlayers(self, num_classes):
        self.temporal_pool = nn.Linear(7, 1)
        self.reluf = F.relu
        self._num_classes = num_classes
        self.logits = nn.Linear(1024, num_classes)
        
        # self.temporal_pool = Unit3D(in_channels=384+384+128+128, output_channels=384+384+128+128,
        #                      kernel_shape=[7, 1, 1],
        #                      stride=(7, 1, 1),
        #                      padding=0,
        #                      activation_fn=None,
        #                      use_batch_norm=False,
        #                      use_bias=True,
        #                      name='temporal_pool')
        # self._num_classes = num_classes
        # self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
        #                      kernel_shape=[1, 1, 1],
        #                      padding=0,
        #                      activation_fn=None,
        #                      use_batch_norm=False,
        #                      use_bias=True,
        #                      name='logits')
        
    # def replace_logits(self, num_classes):
    #     self._num_classes = num_classes
    #     self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
    #                          kernel_shape=[1, 1, 1],
    #                          padding=0,
    #                          activation_fn=None,
    #                          use_batch_norm=False,
    #                          use_bias=True,
    #                          name='logits')
        
    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        x = self.avg_pool(x).squeeze(3).squeeze(3)
        logits = self.logits(self.dropout(self.reluf(self.temporal_pool(x).squeeze(2))))
        # # x = self.logits(self.dropout(self.avg_pool(x)))
        # x = self.logits(self.dropout(self.temporal_pool(self.avg_pool(x))))
        # # x = self.logits(self.dropout(torch.mean(self.avg_pool(x), dim=-3).unsqueeze(2)))
        # if self._spatial_squeeze:
        #     logits = x.squeeze(3).squeeze(3).squeeze(2)
        # # logits is batch X time X classes, which is what we want to work with
        return logits
    
    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


class InceptionI3dT3dconvwLMR(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,
                 class_counts=None, l=0.25, d=1.0, omega=20, epsilon=0.1):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3dT3dconvwLMR, self).__init__()
        ### LMR inits:
        self.class_counts = class_counts
        self.l = l
        self.d = d
        self.omega = omega
        self.epsilon = epsilon
        # self.class_counts = torch.tensor(self.class_counts)
        # pre-compute c(y)
        self.set_class_weights()
        # set mask which identifies classes which are few-shot
        self.set_fs_classes()

        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        # self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
        #                      kernel_shape=[1, 1, 1],
        #                      padding=0,
        #                      activation_fn=None,
        #                      use_batch_norm=False,
        #                      use_bias=True,
        #                      name='logits')
        self.temporal_pool = nn.Linear(7, 1)
        self.reluf = F.relu
        self.logits = nn.Linear(1024, num_classes)

        self.build()


#     def add_reinit_headlayers(self, num_classes):
#         self.temporal_pool = nn.Linear(7, 1)
#         self.reluf = F.relu
#         self._num_classes = num_classes
#         self.logits = nn.Linear(1024, num_classes)
        
    ### LMR specific functions
    # pre-compute c(y)
    def set_class_weights(self):
        # Eq. 1 in paper.
        tilde_C = 1.0 / torch.log((self.class_counts * self.d) + self.epsilon)
        # Eq. 2 in paper.
        numerator = tilde_C - torch.min(tilde_C)
        denominator = torch.max(tilde_C) - torch.min(tilde_C)
        self.c = numerator / denominator * self.l

    # identify classes which are few-shot. 1 for few-shot, 0 otherwise    
    def set_fs_classes(self):
        fs_classes = torch.where(self.class_counts <= self.omega, 1, 0)
        self.fs_classes = fs_classes #nn.Parameter(fs_classes, requires_grad=False)
        
    # compute reconstructions for each sample and combine with original sample
    # based on the class size contribution
    def reconstruct(self, x, y):
        n_batch, dim = x.shape

        # calculate similarities
        x_norm = F.layer_norm(x, [dim]) / torch.sqrt(torch.tensor(dim))
        sim = torch.matmul(x_norm, x_norm.t())

        # mask to remove similarity to self
        self_mask = torch.eye(n_batch).to(x.device)

        # mask to remove similarity to few-shot classes
        fs_mask = torch.index_select(input=self.fs_classes.to(x.device), dim=0, index=y)

        # Eq. 3 in paper.
        # combine masks to create exclusion mask E. 
        # entries are either 1 (ignore) or 0 (don't ignore)
        E = self_mask + fs_mask
        E = torch.where(E >= 1.0, 1.0, 0.0)

        # Eq. 4 in paper.
        # apply mask and softmax to calculate W. 
        sim = sim - 1e5 * E
        sim = F.softmax(sim, dim=-1)

        # get contribution of reconstruction for each sample, based on class count
        contrib = torch.index_select(input=self.c.to(x.device), dim=0, index=y)

        # Eq. 5 in paper.
        # combine reconstructions with original samples, to get R. 
        reconstructions = torch.matmul(sim, x)
        R = (reconstructions.t() * contrib).t() + (x.t() * (1 - contrib)).t()
        return R


    # perform pairwise label mixing
    def pairwise_mix(self, x, y):
        n_batch = x.shape[0]

        # generate one-hot labels ready for mixing
        y_oh = F.one_hot(y, self.class_counts.shape[0])

        # beta selects other samples to mix with
        beta = torch.randint(low=0, high=n_batch, size=[n_batch])
        beta = F.one_hot(beta, n_batch)

        # alpha are the pairwise mixing weights
        # set half the elements of alpha to 1, otherwise random
        alpha = torch.where(torch.rand(n_batch) > 0.5, torch.rand(n_batch), torch.ones(n_batch))

        # Eq. 6 in paper.
        # Mixing mask M. 
        M = (torch.eye(n_batch) * alpha + beta.t() * (1 - alpha)).t()
        M = M.to(x.device)
        
        # Eq. 7 in main paper.
        x = torch.matmul(M, x)
        y = torch.matmul(M, y_oh.float())
        return x, y

    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x, y):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        x = self.avg_pool(x).squeeze(3).squeeze(3)
        x = self.reluf(self.temporal_pool(x).squeeze(2))
        if self.training:
            x = self.reconstruct(x, y)
            x, y = self.pairwise_mix(x, y)
        else:
            ### one hot to match the format in train (below line not required as anyways normal labels can be used during validation)
            y = F.one_hot(y, self._num_classes).float()
        logits = self.logits(self.dropout(x))
        return logits, y
    
    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)

class InceptionI3dTTFA(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    ### LMR inputs
    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
        
        super(InceptionI3dTTFA, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
        self.build()
        
        
    def add_TTFA(self, num_classes):
        self._num_classes = num_classes
        ### Temporal feature aggregation
        # self.logits = TransformerTFA(hidden_dim = 128, in_channels = 1024, max_seq_len = 7, num_classes = num_classes, nhead=1)
        # self.logits = TransformerTFA(hidden_dim = 32, in_channels = 1024, max_seq_len = 7, num_classes = num_classes, nhead=1)
        self.logits = TransformerTFA(hidden_dim = 64, in_channels = 1024, max_seq_len = 7, num_classes = num_classes, nhead=1)
        
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        ### rectified dropout blunder below (but do so for other models above)
        x=self.avg_pool(x).squeeze(-1)#.squeeze(-1)
        # print(x.shape)
        logits = self.logits(self.dropout(x))
        return logits

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


class InceptionI3dTTFAwLMR(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    ### LMR inputs
    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,
                 class_counts=None, l=0.25, d=1.0, omega=20, epsilon=0.1):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
        
        super(InceptionI3dTTFAwLMR, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        # self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
        #                      kernel_shape=[1, 1, 1],
        #                      padding=0,
        #                      activation_fn=None,
        #                      use_batch_norm=False,
        #                      use_bias=True,
        #                      name='logits')
        
        ### Temporal feature aggregation
        # self.logits = TransformerTFAwLMR(hidden_dim = 128, in_channels = 1024, max_seq_len = 7, num_classes = num_classes, nhead=1, class_counts=class_counts, l=l, d=d, omega=omega, epsilon=epsilon)
        self.logits = TransformerTFAwLMR(hidden_dim = 32, in_channels = 1024, max_seq_len = 7, num_classes = num_classes, nhead=1, class_counts=class_counts, l=l, d=d, omega=omega, epsilon=epsilon)
        
        self.build()
        
        
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x, y):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        ### rectified dropout blunder below (but do so for other models above)
        x=self.dropout(self.avg_pool(x).squeeze(-1))#.squeeze(-1)
        ### the temporal dimension won't work for LMR
        # print(x.shape)
        
        logits, y = self.logits(x, y)
        return logits, y

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)

