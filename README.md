# nninit

Parameter initialisation schemes for Torch7 neural network modules. Works with `nn`, and therefore `nngraph`. Allows arbitrary indexing of weights/biases/parameters. Supported modules:

- nn.Linear / nn.LinearNoBias
- nn.LookupTable
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

**`nninit`** adds an `init` method to `nn.Module`, with the following API:

```lua
module:init(accessor, initialiser, ...)
```

The [`accessor`](#accessors) argument is used to extract the tensor to be initialised from the module. The [`initialiser`](#initialisers) argument is a function that takes the module, tensor, and further options; it adjusts the tensor and returns the module, allowing `init` calls to be chained. `nninit` comes with several initialiser functions. `...` represents additional arguments for the initialiser function.

### Accessors

The `accessor` argument is used to extract the tensor to be initialised from the module. It can either be a string, table, or function. 

#### string

The tensor is accessed as a property of the module. For example:

```lua
module:init('weight', nninit.constant, 1)
```

#### table

The tensor is first accessed as a property of the module from the first element, and a subtensor is then extracted using Torch's [indexing operator](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor--dim1dim2--or--dim1sdim1e-dim2sdim2e-) applied to the second element. For example:

```lua
module:init({'weight', {{1, 5}, {}}}, nninit.uniform, -1, 1)
```

#### function

The tensor must be returned as the result of the function applied to the module. For example:

```lua
module:init(function(m) return m.weight:narrow(1, 1, 10) end, nninit.normal, 0, 0.01)
```

### Initialisers

#### nninit.copy(module, tensor, init)
Copies the `init` tensor to the tensor to be initialised.

#### nninit.constant(module, tensor, val)
Fills tensor with the constant `val`.

#### nninit.addConstant(module, tensor, val)
Adds to current tensor with the constant `val`.

#### nninit.mulConstant(module, tensor, val)
Multiplies current tensor by the constant `val`.

#### nninit.normal(module, tensor, mean, stdv)
Fills tensor ~ N(`mean`, `stdv`).

#### nninit.addNormal(module, tensor, mean, stdv)
Adds to current tensor with ~ N(`mean`, `stdv`).

#### nninit.uniform(module, tensor, a, b)
Fills tensor ~ U(`a`, `b`).

#### nninit.addUniform(module, tensor, a, b)
Adds to current tensor with ~ U(`a`, `b`).

#### nninit.eye(module, tensor)
**Only supports the module weights as the tensor. Relies on the module type to determine appropriate identity.**  
Fills weights with the identity matrix (for linear layers/lookup tables).  
Fills filters with the Dirac delta function (for convolutional layers). Normalises by the number of input layers.

#### nninit.xavier(module, tensor, [{[dist], [gain]}])
Fills tensor with `stdv = gain * sqrt(2 / (fanIn + fanOut))`. Uses the uniform distribution by default.  
Optional named parameters [`dist`](#dists) and [`gain`](#gains) can be passed in via a table.  
Also known as Glorot initialisation.

> Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. In *International Conference on Artificial Intelligence and Statistics*.

#### nninit.kaiming(module, tensor, [{[dist], [gain]}])
Fills tensor with `stdv = gain * sqrt(1 / fanIn)`. Uses the normal distribution by default.  
Optional named parameters [`dist`](#dists) and [`gain`](#gains) can be passed in via a table.  
Also known as He initialisation.

> He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *arXiv preprint arXiv:1502.01852*.

#### nninit.orthogonal(module, tensor, [{[gain]}])
**Only supports tensors with at least 2 dimensions.**  
Fills tensor with a (normally distributed) random orthogonal matrix.  
Optional named parameter [`gain`](#gains) can be passed in via a table.

> Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. *arXiv preprint arXiv:1312.6120*.

#### nninit.sparse(module, tensor, sparsity)
Sets `(1 - sparsity)` percent of the tensor to 0, where `sparsity` is between 0 and 1. For example, a `sparsity` of 0.2 drops out 80% of the tensor.

> Martens, J. (2010). Deep learning via Hessian-free optimization. In *Proceedings of the 27th International Conference on Machine Learning (ICML-10)*.

### Dists

The 2 types of distribution supported are `'normal'` and `'uniform'`.

### Gains

Gains can be calculated depending on the succeeding nonlinearity. If `gain` is a number it is used directly; if `gain` is a string the following mapping is used. By default gains (where applicable) are set to 1.

| Gain      | Parameters | Mapping                     |
|-----------|------------|-----------------------------|
| 'linear'  |            | 1                           |
| 'sigmoid' |            | 1                           |
| 'relu'    |            | sqrt(2)                     |
| 'lrelu'   | leakiness  | sqrt(2 / (1 + leakiness^2)) |

If the `gain` must be calculated from additional parameters, `gain` must be passed as table with the string as the first element as well as named parameters. For example:

```lua
module:init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.3}})
```

## Example

```lua
local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
require 'rnn'
local nninit = require 'nninit'

local getBias = function(module)
  return module.bias
end

local batchSize = 5
local imgSize = 16
local nChannels = 3
local nFilters = 8
local rho = 6
local hiddenSize = 2

local cnn = nn.Sequential()
cnn:add(cudnn.SpatialConvolution(nChannels, nFilters, 2, 2):init('weight', nninit.eye)
                                                           :init('weight', nninit.mulConstant, 1/2)
                                                           :init('weight', nninit.addNormal, 0, 0.01)
                                                           :init(getBias, nninit.constant, 0))
cnn:add(nn.View(nFilters*15*15))
cnn:add(nn.Linear(nFilters*15*15, nFilters):init('weight', nninit.kaiming, {
  dist = 'uniform',
  gain = {'lrelu', leakiness = 0.3}
}))
cnn:add(nn.RReLU(1/3, 1/3))
cnn:add(nn.Linear(nFilters, 6):init('weight', nninit.orthogonal, {gain = 'relu'}))
cnn:add(cudnn.ReLU())
cnn:add(nn.Linear(6, 4):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
cnn:add(nn.Linear(4, hiddenSize):init('weight', nninit.sparse, 0.2)
                                :init(getBias, nninit.constant, 0))

local model = nn.Sequential()
model:add(nn.Sequencer(cnn))
local lstm = nn.FastLSTM(hiddenSize, hiddenSize, rho)
-- Note that chaining will pass through the module initialised, never parents
lstm.i2g:init({'bias', {{3*hiddenSize+1, 4*hiddenSize}}}, nninit.constant, 1) -- High forget gate bias
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
