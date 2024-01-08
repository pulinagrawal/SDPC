import torch
import torch.nn.functional as f
import torch.nn as nn
import numpy as np
from SPC_2L.DataTools import norm
from torch.autograd import Function

# The `LayerPC` class is a PyTorch module that represents a layer in a neural network, 
# with methods for forward and backward convolution operations, as well as transformations between layers.
class LayerPC(nn.Module):

    def __init__(self, dico_shape=None, stride=1, pad=0, out_pad=0, a=1, b=1, v=1, v_size=None, bias=False,
                 transform=None, drop=None, normalize_dico=True, dico_load=None, seed=None, groups=1):

        super(LayerPC,self).__init__()

        self.params={'dico_shape': dico_shape,
                     'stride': stride,
                     'out_pad': out_pad,
                     'pad':pad,
                     'a': a,
                     'b': b,
                     'v': v,
                     'normalize_dico': normalize_dico,
                     'groups': groups}

        if seed is not None:
            torch.manual_seed(seed)
        self.dico_shape = dico_shape
        self.stride = stride
        self.out_pad = out_pad
        self.pad = pad
        self.a = a
        self.b = b

        if bias:
            self.bias = nn.Parameter(torch.zeros(1,dico_shape[0],1,1))
        else:
            self.bias = 0

        # v is a scaling factor per filter (i.e., per channel) not per unit/neuron 
            # evidenced by 1,v_size,1,1
        self.v_size = v_size
        if self.v_size is None:
            self.v = v
        else:
            self.v = v * torch.ones(1,v_size,1,1)
            self.v = nn.Parameter(self.v)

        self.groups = groups

        self.transform = transform
        self.drop = drop
        self.normalize_dico = normalize_dico
        self.dico = self.init_dico(dico_load)

    def init_dico(self, dico_load):
        """
        The function `init_dico` initializes a dictionary either by generating random values or by loading an existing
        dictionary.
        
        :param dico_load: The parameter `dico_load` is a dictionary that can be loaded as an existing dictionary
        :return: The function `init_dico` returns either a randomly initialized dictionary `dico` or a loaded dictionary
        `dico_load` as a `nn.Parameter` object.
        """
        if dico_load is None:
            if self.dico_shape is None:
                raise('you need either to define a Dictionary size (N x C x H x W) or to load an existing Dictionary')
            else:
                dico = torch.randn(self.dico_shape)
                if self.normalize_dico:
                    dico /= norm(dico) # l2 norm makes it all positive
                else:
                    dico *= np.sqrt(2/(self.dico_shape[-1]*self.dico_shape[-2]*self.dico_shape[-3]))
                return nn.Parameter(dico)
        else:
            return nn.Parameter(dico_load)

    def forward(self, x):

        return f.conv2d(x, self.dico, stride=self.stride, groups=self.groups, padding=self.pad) + self.bias


    def backward(self, x):
        """
        The function performs a backward convolution operation on the input tensor.
        
        :param x: The parameter `x` represents the input tensor to the backward function. It is the tensor that will be passed
        through the backward operation
        :return: the variable `x`.
        """
        if self.drop is not None:
            x = self.drop(self,x)
        from torch.nn import ConvTranspose2d  
        x = f.conv_transpose2d(x, self.dico, stride=self.stride, padding=self.pad, output_padding=self.out_pad, groups=self.groups)
        return x

    def to_next(self, x):
        if self.transform is None:
            return self.v * x
        else:
            return self.v * self.transform.to_next(self, x)

    def to_previous(self, x):
        if self.transform is None:
            return x
        else:
            return self.transform.to_previous(self, x)


class Network(object):

    def __init__(self, layers, input_size, verbose=True):

        self.nb_layers = len(layers)
        self.layers = layers
        self.sparse_map_size = [None] * self.nb_layers
        self.input_size = input_size

        self.param_transform = []
        # this code block is creating a list called `param_transform` that stores tuples representing the kernel size 
        # and stride of each layer's transformation function.
        for i in range(self.nb_layers):
            if self.layers[i].transform is not None:
                to_append = (self.layers[i].transform.kernel_size, self.layers[i].transform.stride)
            else :
                to_append = (1,1)
            self.param_transform.append(to_append)


        # This code block initializes the network structure by creating an input tensor of random values and 
        # passing it through each layer of the network. It also prints out the size of the output tensor at 
        # each layer, as well as the size of the transformed tensor if a transformation function is applied.
        # This is useful for understanding the structure of the network and verifying that the sizes of the 
        # tensors are as expected.
        if input_size is not None:

            input = torch.rand(input_size)
            network_structure = 'NETWORK STRUCTURE : \n Input : {0}'.format(input_size)

            for i in range(self.nb_layers):
                if i == 0:
                    sparse_map = self.layers[i].forward(input)
                else:
                    sparse_map = self.layers[i].forward(self.layers[i-1].to_next(sparse_map))
                self.sparse_map_size[i] = sparse_map.size()
                network_structure += '\n Layer {0} : {1}'.format(i + 1, list(sparse_map.size()))
                if self.layers[i].transform is not None:
                    network_structure += '\n Layer {0} transformed : {1}'.format(i + 1, list((self.layers[i].to_next(sparse_map).size())))

            if verbose:
                print(network_structure)

            self.to_device(torch.discover_device())
        
    def to_device(self, device):
        self.layers = [self.layers[i].to(device) for i in range(self.nb_layers)]

    def project_dico(self, j, cpu=False):
        """
        The function `project_dico` projects a dictionary `dico` through the layers of a neural network in reverse order.
        
        :param j: The parameter "j" represents the index of the layer in the neural network. It is used to specify which layer's
        dictionary (dico) should be processed
        :param cpu: The `cpu` parameter is a boolean flag that determines whether the data should be detached and moved to the
        CPU before returning it. If `cpu` is `True`, the data is detached and moved to the CPU. If `cpu` is `False` (default),
        the data is detached but, defaults to False (optional)
        :return: the variable "dico".
        """

        if cpu:
            dico = self.layers[j].dico.data.detach().cpu()
        else:
            dico = self.layers[j].dico.data.detach()

        for i in range(j-1,-1,-1):
            dico = self.layers[i].to_previous(dico)
            dico = self.layers[i].backward(dico)

        return dico

    def turn_off(self, i=None):
        """
        The `turn_off` function is used to turn off the gradient computation for the parameters of a neural network model.
        
        :param i: The parameter `i` is an optional argument that specifies the index of the layer for which the `requires_grad`
        attribute of its parameters should be set to `False`. If `i` is not provided, then the code will iterate over all layers
        in the model and set the `requires_grad`
        """

        if i is None:
            for i in range(self.nb_layers):
                for param in self.layers[i].parameters():
                    param.requires_grad = False
        else:
            for param in self.layers[i].parameters():
                param.requires_grad = False

    def turn_on(self, i=None):
        """
        The `turn_on` function is used to enable gradient computation for the specified layer or all layers in a neural network
        model.
        
        :param i: The parameter `i` is an optional argument that specifies the index of a specific layer in the neural network.
        If `i` is not provided (i.e., `None`), then the code will iterate over all layers in the network and set the
        `requires_grad` attribute of all parameters in
        """

        if i is None:
            for i in range(self.nb_layers):
                for param in self.layers[i].parameters():
                    param.requires_grad = True
        else:
            for param in self.layers[i].parameters():
                param.requires_grad = True

'''
class MaxPool2d(nn.Module):

    def __init__(self, kernel_size, stride=1):
        super(MaxPool2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
        self.unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride)
        if torch.cuda.is_available():
            self.pool = self.pool.to(torch.discover_device())
            self.unpool = self.unpool.to(torch.discover_device())

        #self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
        #self.unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride)

    def to_next(self, layer, x):

        x, _ = self.pool(x)
        return x

    def to_previous(self, layer, x):

        w = (x.size()[-2] - 1) * self.stride + self.kernel_size
        h = (x.size()[-1] - 1) * self.stride + self.kernel_size
        c_size = x.size()[0]
        d_size = layer.dico.data.size()[0]

        _, idx = self.pool(torch.randn(c_size, d_size, w, h).to(torch.discover_device()))

        #if torch.cuda.is_available:
        #    _, idx = self.pool(torch.randn(c_size, d_size, w, h).to(torch.discover_device()))
        #else:
        #    _, idx = self.pool(torch.randn(c_size, d_size, w, h))

        return self.unpool(x, idx)


## with padding
class AvgPool2d(nn.Module):

    def __init__(self, kernel_size, stride=1):
        super(AvgPool2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = torch.nn.AvgPool2d(kernel_size, stride=stride)
        self.unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride)
        if torch.cuda.is_available():
            self.pool = self.pool.to(torch.discover_device())
            self.unpool = self.unpool.to(torch.discover_device())
        self.init = True

        # self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
        # self.unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride)

    def to_next(self, layer, x):

        if self.init == True:
            self.input_size = x.size()
            self.init = False
        x = self.pool(x)

        return x

    def to_previous(self, layer, x):
        x = f.interpolate(x, size=self.input_size[-1], mode='bilinear', align_corners=True)
        return x


class RescaleFeedback(Function):

    @staticmethod
    def forward(ctx, input, v):
        ctx.save_for_backward(input, v)
        return  v*input

    @staticmethod
    def backward(ctx, grad_output):
        input, v = ctx.saved_tensors
        grad_input = grad_v =  None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.div(v**2)
        if ctx.needs_input_grad[1]:
            grad_v = None#(grad_output*input).sum()


        return grad_input, grad_v


class MaxPool2d_b(nn.Module):

    def __init__(self, kernel_size, stride=1):
        super(MaxPool2d_b, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
        self.unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride)
        if torch.cuda.is_available():
            self.pool = self.pool.to(torch.discover_device())
            self.unpool = self.unpool.to(torch.discover_device())
        self.init = True

        # self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
        # self.unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride)

    def to_next(self, layer, x):

        if self.init == True:
            self.input_size = x.size()
            self.init = False
        x, _ = self.pool(x)

        return x

    def to_previous(self, layer, x):
        x = f.interpolate(x, size=self.input_size[-1], mode='bilinear', align_corners=True)
        return x

'''