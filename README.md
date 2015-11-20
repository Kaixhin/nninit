# nninit

Weight initialisation schemes for Torch7 neural network modules. Works with `nn`, and therefore `nngraph`. Supported modules:

- nn.Linear
- nn.TemporalConvolution
- nn.SpatialConvolution / cudnn.SpatialConvolution
- nn.VolumetricConvolution / cudnn.VolumetricConvolution

## Installation

```sh
luarocks install https://raw.githubusercontent.com/Kaixhin/nninit/master/rocks/nninit-scm-1.rockspec
```

## Example

```lua
local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
require 'nninit'

local X = torch.ones(1, 3, 3):cuda()

local model = nn.Sequential()
model:add(cudnn.SpatialConvolution(1, 1, 2, 2):init('orthogonal'))
model:add(nn.View(4))
model:add(nn.Linear(4, 4):init('kaiming', 'uniform', 'lrelu', 1/3))
model:add(nn.RReLU(1/3, 1/3))
model:add(nn.Linear(4, 5):init('normal', 'w', 1, 0.4))
model:add(nn.Linear(5, 3):init('xavier', 'normal', 1.1))
model:add(nn.Linear(3, 2):init('sparse', 0.2):init('constant', 'b', 0))
model:add(nn.LogSoftMax())
model:cuda()

print(model:forward(X))
```

## Usage

**nninit** uses method chaining on modules - call `init` with the method name and any parameters needed by the method. For example, `nn.Linear(5, 3):init('constant', 'w', 1)`.

### Gains

Optional gains can be calculated depending on the succeeding nonlinearity. If `gain` is a number it is used directly; if `gain` is a string the following mapping is used. By default the `gain` parameter is `linear`.

| Gain      | Parameters | Mapping                     |
|-----------|------------|-----------------------------|
| 'linear'  |            | 1                           |
| 'sigmoid' |            | 1                           |
| 'relu'    |            | sqrt(2)                     |
| 'lrelu'   | leakiness  | sqrt(2 / (1 + leakiness^2)) |

### Initialiser Methods

#### constant(wb, val)
Fills weights if `wb` is `'w'`, or biases if `wb` is `w`, with a constant value.

#### normal(wb, mean, stdv)
Fills weights if `wb` is `'w'`, or biases if `wb` is `w`, ~ N(mean, stdv).

#### uniform(wb, a, b)
Fills weights if `wb` is `'w'`, or biases if `wb` is `w`, ~ U(a, b).

#### eye()
Fills weights with an `m x n` identity matrix (ones on the diagonals, zeros elsewhere).

#### xavier(dist, [gain])
Fills weights with `stdv = gain * sqrt(2 / (fanIn + fanOut))`. Uses the uniform distribution by default.  
Also known as Glorot initialisation.

> Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. In *International Conference on Artificial Intelligence and Statistics*.

#### kaiming(dist, [gain])
Fills weights with `stdv = gain * sqrt(1 / fanIn)`. Uses the normal distribution by default.  
Also known as He initialisation.

> He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *arXiv preprint arXiv:1502.01852*.

#### orthogonal([gain])
Fills weights with a (normally distributed) random orthogonal matrix.

> Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. *arXiv preprint arXiv:1312.6120*.

#### sparse(sparsity)
Sets `(1 - sparsity)` percent of the weights to 0, where `sparsity` is between 0 and 1. For example, a `sparsity` of 0.2 drops out 80% of the weights.

> Martens, J. (2010). Deep learning via Hessian-free optimization. In *Proceedings of the 27th International Conference on Machine Learning (ICML-10)*.

## Acknowledgements

- [Lasagne](https://github.com/Lasagne/Lasagne)
- [Purdue e-Lab Torch Toolbox](https://github.com/e-lab/torch-toolbox)
