# nninit

Weight initialisation schemes for Torch7 neural network modules. Works with `nn`, and therefore `nngraph`. Allows basic indexing to modify subsets of weights. Supported modules:

- nn.Linear / nn.LinearNoBias
- nn.TemporalConvolution
- nn.SpatialConvolution / cudnn.SpatialConvolution
- nn.VolumetricConvolution / cudnn.VolumetricConvolution

Readme contents:

- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Development](#development)
- [Acknowledgements](#acknowledgements)

## Installation

```sh
luarocks install nninit
```

## Usage

**nninit** adds 2 methods to `nn.Module`: `wInit` for weight initialisation and `bInit` for bias initialisation. It uses method chaining, where both methods return the module, allowing calls to be composed (see above for an example). Call `wInit` or `bInit` with the function name and any parameters needed by the function. All functions are supported by `wInit`; functions supported by `bInit` are noted explicitly.

### Functions

#### constant(val, [indices])
**Supported by `bInit`**.  
Fills weights/biases with the constant `val`.

#### addConstant(val, [indices])
**Supported by `bInit`**.  
Adds to current weights/biases with the constant `val`.

#### mulConstant(val, [indices])
**Supported by `bInit`**.  
Multiplies current weights/biases with the constant `val`.

#### normal(mean, stdv, [indices])
**Supported by `bInit`**.  
Fills weights/biases ~ N(`mean`, `stdv`).

#### addNormal(mean, stdv, [indices])
**Supported by `bInit`**.  
Adds to current weights/biases with ~ N(`mean`, `stdv`).

#### uniform(a, b, [indices])
**Supported by `bInit`**.  
Fills weights/biases ~ U(`a`, `b`).

#### addUniform(a, b, [indices])
**Supported by `bInit`**.  
Adds to current weights/biases with ~ U(`a`, `b`).

#### eye()
Fills weights with the identity matrix (for linear layers).  
Fills filters with the Dirac delta function (for convolutional layers). Normalises by the number of input layers.

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

### Indices

`indices` accepts anything called by the [indexing operator](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor--dim1dim2--or--dim1sdim1e-dim2sdim2e-), e.g. `{{1, 3}, {}}`.

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

## Example

```lua
local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
require 'rnn'
require 'nninit'

local batchSize = 5
local imgSize = 16
local nChannels = 3
local nFilters = 8
local rho = 6
local hiddenSize = 2

local cnn = nn.Sequential()
cnn:add(cudnn.SpatialConvolution(nChannels, nFilters, 2, 2):wInit('eye'):wInit('mulConstant', 1/2):wInit('addNormal', 0, 0.01):bInit('constant', 0))
cnn:add(nn.View(nFilters*15*15))
cnn:add(nn.Linear(nFilters*15*15, nFilters):wInit('kaiming', 'uniform', 'lrelu', 1/3))
cnn:add(nn.RReLU(1/3, 1/3))
cnn:add(nn.Linear(nFilters, 6):wInit('orthogonal', 'relu'))
cnn:add(cudnn.ReLU())
cnn:add(nn.Linear(6, 4):wInit('xavier', 'normal', 1.1))
cnn:add(nn.Linear(4, hiddenSize):wInit('sparse', 0.2):bInit('constant', 0))

local model = nn.Sequential()
model:add(nn.Sequencer(cnn))
local lstm = nn.FastLSTM(hiddenSize, hiddenSize, rho)
-- Note that chaining will pass through the module initialised, never parents
lstm.i2g:bInit('constant', 1, {{3*hiddenSize-1, 3*hiddenSize}}) -- High forget gate bias
model:add(nn.Sequencer(lstm))
model:cuda()

local inputs = {}
for i = 1, rho do
  table.insert(inputs, torch.ones(batchSize, nChannels, imgSize, imgSize):cuda())
end
print(model:forward(inputs))
```

## Development

To develop **nninit**/use it to test new initialisation schemes, `git clone`/download this repo and use `luarocks make rocks/nninit-scm-1.rockspec` to install **nninit** locally.

## Acknowledgements

- [Lasagne](https://github.com/Lasagne/Lasagne)
- [Purdue e-Lab Torch Toolbox](https://github.com/e-lab/torch-toolbox)
