# nninit

Weight initialisation schemes for Torch7 neural network modules.

## Installation

```sh
luarocks install https://raw.githubusercontent.com/Kaixhin/nninit/master/rocks/nninit-scm-1.rockspec
```

## Usage

```lua
require 'nn'
local nninit = require 'nninit'

local model = nn.Sequential()
model:add(nninit.xavier(nn.SpatialConvolution(1, 1, 2, 2)))
model:add(nn.View(4))
model:add(nninit.kaiming(nn.Linear(4, 3), 'relu', 'uniform'))
model:add(nn.ReLU())
model:add(nninit.constant(nn.Linear(3, 2), 1))
model:add(nn.LogSoftMax())

print(model:forward(X))
```

## Acknowledgements

- [Lasagne](https://github.com/Lasagne/Lasagne)
- [Purdue e-Lab Torch Toolbox](https://github.com/e-lab/torch-toolbox)
