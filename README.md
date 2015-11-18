# nninit

Weight initialisation schemes for Torch7 neural network modules.

## Installation

```sh
luarocks install https://raw.githubusercontent.com/Kaixhin/nninit/master/rocks/nninit-scm-1.rockspec
```

## Example

```lua
require 'nn'
local nninit = require 'nninit'

local model = nn.Sequential()
model:add(nninit.xavier(nn.SpatialConvolution(1, 1, 2, 2)))
model:add(nn.View(4))
model:add(nninit.kaiming(nn.Linear(4, 3), 'uniform', 'lrelu', 1/3))
model:add(nn.RReLU(1/3, 1/3))
model:add(nninit.constant(nn.Linear(3, 2), 1))
model:add(nn.LogSoftMax())

print(model:forward(X))
```

## Usage

**nninit** wraps modules - call the desired method with the module and parameters.

### Gains

Optional gains can be calculated depending on the succeeding nonlinearity. By default the `gainType` parameter is `linear`. The mappings are as follows:

| Gain Type | Parameters | Mapping                     |
|-----------|------------|-----------------------------|
| linear    |            | 1                           |
| sigmoid   |            | 1                           |
| relu      |            | sqrt(2)                     |
| lrelu     | leakiness  | sqrt(2 / (1 + leakiness^2)) |

### Initialisers

#### nninit.constant(module, val)
Fills weights with a constant value.

#### nninit.normal(module, mean, stdv)
Fills weights ~ N(mean, stdv).

#### nninit.uniform(module, a, b)
Fills weights ~ U(a, b).

#### nninit.xavier(module, dist, gainType)
Fills weights with stdv = gain * sqrt(2 / (fanIn + fanOut)). Zeroes biases. Uses the uniform distribution by default.
Also known as Glorot initialisation.

Ref: Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. In *International Conference on Artificial Intelligence and Statistics*.

#### nninit.kaiming(module, dist, gainType)
Fills weights with stdv = gain * sqrt(1 / fanIn). Zeroes biases. Uses the normal distribution by default.
Also known as He initialisation.

Ref: He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *arXiv preprint arXiv:1502.01852*.

## Acknowledgements

- [Lasagne](https://github.com/Lasagne/Lasagne)
- [Purdue e-Lab Torch Toolbox](https://github.com/e-lab/torch-toolbox)
