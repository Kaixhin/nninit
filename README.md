# nninit

Weight initialisation schemes for Torch7 neural network modules. Works with `nn`, and therefore `nngraph`. Supported modules:

- nn.Linear
- nn.TemporalConvolution
- nn.SpatialConvolution / cudnn.SpatialConvolution
- nn.VolumetricConvolution / cudnn.VolumetricConvolution

Readme contents:

- [Installation](#installation)
- [Example](#example)
- [Usage](#usage)
- [Development](#development)
- [Acknowledgements](#acknowledgements)

## Installation

```sh
luarocks install nninit
```

## Example

```lua
local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
require 'nninit'

local X = torch.ones(1, 3, 3):cuda()

local model = nn.Sequential()
model:add(cudnn.SpatialConvolution(1, 1, 2, 2):wInit('orthogonal'))
model:add(nn.View(4))
model:add(nn.Linear(4, 4):wInit('kaiming', 'uniform', 'lrelu', 1/3))
model:add(nn.RReLU(1/3, 1/3))
model:add(nn.Linear(4, 5):wInit('normal', 1, 0.4))
model:add(nn.Linear(5, 3):wInit('xavier', 'normal', 1.1))
model:add(nn.Linear(3, 2):wInit('sparse', 0.2):bInit('constant', 0))
model:add(nn.LogSoftMax())
model:cuda()

print(model:forward(X))
```

## Usage

**nninit** adds 2 methods to `nn.Module`: `wInit` for weight initialisation and `bInit` for bias initialisation. It uses method chaining, where both methods return the module, allowing calls to be composed (see above for an example). Call `wInit` or `bInit` with the function name and any parameters needed by the function.

### wInit Functions

#### constant(val)
Fills weights with the constant `val`.

#### addConstant(val)
Adds to current weights with the constant `val`.

#### mulConstant(val)
Multiplies current weights with the constant `val`.

#### normal(mean, stdv)
Fills weights ~ N(`mean`, `stdv`).

#### addNormal(mean, stdv)
Adds to current weights with ~ N(`mean`, `stdv`).

#### uniform(a, b)
Fills weights ~ U(`a`, `b`).

#### addUniform(a, b)
Adds to current weights with ~ U(`a`, `b`).

#### eye()
Fills weights with an `m x n` identity matrix (ones on the diagonals, zeros elsewhere).

#### xavier([dist, [gain]])
Fills weights with `stdv = gain * sqrt(2 / (fanIn + fanOut))`. Uses the uniform distribution by default.  
Also known as Glorot initialisation.

> Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. In *International Conference on Artificial Intelligence and Statistics*.

#### kaiming([dist, [gain]])
Fills weights with `stdv = gain * sqrt(1 / fanIn)`. Uses the normal distribution by default.  
Also known as He initialisation.

> He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *arXiv preprint arXiv:1502.01852*.

#### orthogonal([gain])
Fills weights with a (normally distributed) random orthogonal matrix.

> Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. *arXiv preprint arXiv:1312.6120*.

#### sparse(sparsity)
Sets `(1 - sparsity)` percent of the weights to 0, where `sparsity` is between 0 and 1. For example, a `sparsity` of 0.2 drops out 80% of the weights.

> Martens, J. (2010). Deep learning via Hessian-free optimization. In *Proceedings of the 27th International Conference on Machine Learning (ICML-10)*.

### bInit Functions

#### constant(val)
Fills biases with the constant `val`.

#### addConstant(val)
Adds to current biases with the constant `val`.

#### mulConstant(val)
Multiplies current biases with the constant `val`.

#### normal(mean, stdv)
Fills biases ~ N(`mean`, `stdv`).

#### addNormal(mean, stdv)
Adds to current biases with ~ N(`mean`, `stdv`).

#### uniform(a, b)
Fills biases ~ U(`a`, `b`).

#### addUniform(a, b)
Adds to current biases with ~ U(`a`, `b`).

### Dists

The 2 types of distribution supported are `'normal'` and `'uniform'`.

### Gains

Optional gains can be calculated depending on the succeeding nonlinearity. If `gain` is a number it is used directly; if `gain` is a string the following mapping is used. By default the `gain` parameter is `'linear'`.

| Gain      | Parameters | Mapping                     |
|-----------|------------|-----------------------------|
| 'linear'  |            | 1                           |
| 'sigmoid' |            | 1                           |
| 'relu'    |            | sqrt(2)                     |
| 'lrelu'   | leakiness  | sqrt(2 / (1 + leakiness^2)) |

## Development

To develop **nninit**/use it to test new initialisation schemes, `git clone`/download this repo and use `luarocks make rocks/nninit-scm-1.rockspec` to install **nninit** locally.

## Acknowledgements

- [Lasagne](https://github.com/Lasagne/Lasagne)
- [Purdue e-Lab Torch Toolbox](https://github.com/e-lab/torch-toolbox)
