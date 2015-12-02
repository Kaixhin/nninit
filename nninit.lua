local nn = require 'nn'

-- Helper functions

-- Calculates fan in and fan out of module
local function calcFan(module)
  local typename = torch.type(module)

  if typename == 'nn.Linear' or typename == 'nn.LinearNoBias' then
    return module.weight:size(2), module.weight:size(1)
  elseif typename:find('TemporalConvolution') then
    return module.weight:size(2), module.weight:size(1)
  elseif typename:find('SpatialConvolution') then
    return module.nInputPlane * module.kW * module.kH, module.nOutputPlane * module.kW * module.kH
  elseif typename:find('VolumetricConvolution') then
    return module.nInputPlane * module.kT * module.kW * module.kH, module.nOutputPlane * module.kT * module.kW * module.kH
  else
    error("Unsupported module")
  end
end

-- Returns the gain or calculates if given a gain type (with optional args)
local function calcGain(gain)
  -- Return gain if a number already
  if type(gain) == 'number' then
    return gain
  end

  -- Extract gain string if table
  if type(gain) == 'table' then
    local args = gain
    gain = gain[1]
  end

  -- Process gain strings with optional args
  if gain == 'linear' or gain == 'sigmoid' then
    return 1
  elseif gain == 'relu' then
    return math.sqrt(2)
  elseif gain == 'lrelu' then
    return math.sqrt(2 / (1 + math.pow(args.leakiness, 2)))
  end

  -- Return 1 by default
  return 1
end

-- init method

-- Add init to nn.Module
nn.Module.init = function(self, accessor, initialiser, ...)
  -- Extract tensor to initialise
  local tensor
  if type(accessor) == 'string' then
    tensor = self[accessor]
  elseif type(accessor) == 'table' then
    tensor = self[accessor[1]][accessor[2]]
  elseif type(accessor) == 'function' then
    tensor = accessor(self)
  else
    error("Unsupported accessor")
  end

  -- Initialise tensor (given module and options)
  initialiser(self, tensor, ...)

  -- Return module for chaining
  return self
end

-- nninit

local nninit = {}

-- Copies another tensor to the tensor to be initialised
nninit.copy = function(module, tensor, init)
  tensor:copy(init)

  return module
end

-- Fills tensor with a constant value
nninit.constant = function(module, tensor, val)
  tensor:fill(val)

  return module
end

-- Adds to current tensor with a constant value
nninit.addConstant = function(module, tensor, val)
  tensor:add(val)

  return module
end

-- Multiplies current tensor by a constant value
nninit.mulConstant = function(module, tensor, val)
  tensor:mul(val)

  return module
end

-- Fills tensor ~ N(mean, stdv)
nninit.normal = function(module, tensor, mean, stdv)
  tensor:normal(mean, stdv)

  return module
end

-- Adds to current tensor with ~ N(mean, stdv)
nninit.addNormal = function(module, tensor, mean, stdv)
  tensor:add(torch.Tensor(tensor:size()):normal(mean, stdv))

  return module
end

-- Fills tensor ~ U(a, b)
nninit.uniform = function(module, tensor, a, b)
  tensor:uniform(a, b)

  return module
end

-- Adds to current tensor with ~ U(a, b)
nninit.addUniform = function(module, tensor, a, b)
  tensor:add(torch.Tensor(tensor:size()):uniform(a, b))

  return module
end

-- Fills weights with the identity matrix (for linear layers)
-- Fills filters with the Dirac delta function (for convolutional layers)
-- TODO: Fix for new API
nninit.eye = function(self)
  local typename = torch.type(self)

  if typename == 'nn.Linear' or typename == 'nn.LinearNoBias' then
    local I = torch.eye(self.weight:size(2), self.weight:size(1))
    self.weight:copy(I)
  elseif typename:find('TemporalConvolution') then
    self.weight:zero()
    for i = 1, self.inputFrameSize do
      self.weight[{{}, {(i-1)*self.kW + math.ceil(self.kW/2)}}]:fill(1/self.inputFrameSize)
    end
  elseif typename:find('SpatialConvolution') then
    self.weight:zero():view(self.nInputPlane, self.nOutputPlane, self.kW, self.kH)[{{}, {}, math.ceil(self.kW/2), math.ceil(self.kH/2)}]:fill(1/self.nInputPlane)
  elseif typename:find('VolumetricConvolution') then
    self.weight:zero():view(self.nInputPlane, self.nOutputPlane, self.kT, self.kW, self.kH)[{{}, {}, math.ceil(self.kT/2), math.ceil(self.kW/2), math.ceil(self.kH/2)}]:fill(1/self.nInputPlane)
  else
    error("Unsupported module")
  end

  return self
end

--[[
--  Glorot, X., & Bengio, Y. (2010)
--  Understanding the difficulty of training deep feedforward neural networks
--  In International Conference on Artificial Intelligence and Statistics
--
--  Also known as Glorot initialisation
--]]
nninit.xavier = function(module, tensor, options)
  local fanIn, fanOut = calcFan(module)
  gain = calcGain(options.gain)
  dist = options.dist or 'uniform' -- Uniform by default

  local stdv = gain * math.sqrt(2 / (fanIn + fanOut))
  if dist == 'uniform' then
    local b = stdv * math.sqrt(3)
    tensor:uniform(-b, b)
  elseif dist == 'normal' then
    tensor:normal(0, stdv)
  end

  return module
end

--[[
--  He, K., Zhang, X., Ren, S., & Sun, J. (2015)
--  Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification
--  arXiv preprint arXiv:1502.01852
--
--  Also known as He initialisation
--]]
nninit.kaiming = function(module, tensor, options)
  local fanIn = calcFan(module)
  gain = calcGain(options.gain)
  dist = options.dist or 'normal' -- Normal by default

  local stdv = gain * math.sqrt(1 / fanIn)
  if dist == 'uniform' then
    local b = stdv * math.sqrt(3)
    tensor:uniform(-b, b)
  elseif dist == 'normal' then
    tensor:normal(0, stdv)
  end

  return module
end

--[[
--  Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013)
--  Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
--  arXiv preprint arXiv:1312.6120
--]]
-- TODO: Fix for new API
nninit.orthogonal = function(self, gain, ...)
  local fanIn, fanOut = calcFan(self)
  gain = gain or 'linear' -- Linear by default
  gain = calcGain(gain, ...)

  -- Construct random matrix
  local randMat = torch.Tensor(fanOut, fanIn):normal(0, 1)
  local U, __, V = torch.svd(randMat, 'S')

  -- Pick out orthogonal matrix
  local W
  if fanOut > fanIn then
    W = U
  else
    W = V:narrow(1, 1, fanOut)
  end
  -- Resize
  W:resize(self.weight:size())
  -- Multiply by gain
  W:mul(gain)

  self.weight:copy(W)

  return self
end

--[[
-- Martens, J. (2010)
-- Deep learning via Hessian-free optimization
-- In Proceedings of the 27th International Conference on Machine Learning (ICML-10)
--]]
nninit.sparse = function(module, tensor, sparsity)
  local nElements = tensor:nElement()
  local nSparseElements = math.floor(sparsity * nElements)
  local randIndices = torch.randperm(nElements):long()
  local sparseIndices = randIndices:narrow(1, 1, nSparseElements)

  -- Zero out selected indices
  tensor:view(nElements):indexFill(1, sparseIndices, 0)

  return module
end

return nninit
