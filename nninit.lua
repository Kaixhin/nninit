local nn = require 'nn'

-- Calculates fan in and fan out of module
local function calcFan(module)
  local typename = torch.type(module)

  if typename == 'nn.Linear' then
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

-- Returns the gain or calculates if given a gain type (with optional parameters)
local function calcGain(gain, ...)
  local args = {...}

  -- Return gain if a number already
  if type(gain) == 'number' then
    return gain
  end

  if gain == 'linear' or gain == 'sigmoid' then
    return 1
  elseif gain == 'relu' then
    return math.sqrt(2)
  elseif gain == 'lrelu' then
    local leakiness = args[1]
    return math.sqrt(2 / (1 + math.pow(leakiness, 2)))
  end
end

-- Fills weights/biases with a constant value
local constant = function(self, wb, val)
  if wb == 'w' then
    self.weight:fill(val)
  elseif wb == 'b' then
    self.bias:fill(val)
  end

  return self
end

-- Adds to current weights/biases with a constant value
local addConstant = function(self, wb, val)
  if wb == 'w' then
    self.weight:add(val)
  elseif wb == 'b' then
    self.bias:add(val)
  end

  return self
end

-- Multiplies current weights/biases with a constant value
local mulConstant = function(self, wb, val)
  if wb == 'w' then
    self.weight:mul(val)
  elseif wb == 'b' then
    self.bias:mul(val)
  end

  return self
end

-- Fills weights/biases ~ N(mean, stdv)
local normal = function(self, wb, mean, stdv)
  if wb == 'w' then
    self.weight:normal(mean, stdv)
  elseif wb == 'b' then
    self.bias:normal(mean, stdv)
  end

  return self
end

-- Adds to current weights/biases with ~ N(mean, stdv)
local addNormal = function(self, wb, mean, stdv)
  local noise

  if wb == 'w' then
    noise = torch.Tensor(self.weight:size()):normal(mean, stdv)
    self.weight:add(noise)
  elseif wb == 'b' then
    noise = torch.Tensor(self.bias:size()):normal(mean, stdv)
    self.bias:add(noise)
  end

  return self
end

-- Fills weights/biases ~ U(a, b)
local uniform = function(self, a, b)
  if wb == 'w' then
    self.weight:uniform(a, b)
  elseif wb == 'b' then
    self.bias:uniform(a, b)
  end

  return self
end

-- Adds to current weights/biases with ~ U(a, b)
local addUniform = function(self, wb, a, b)
  local noise

  if wb == 'w' then
    noise = torch.Tensor(self.weight:size()):uniform(a, b)
    self.weight:add(noise)
  elseif wb == 'b' then
    noise = torch.Tensor(self.bias:size()):uniform(a, b)
    self.bias:add(noise)
  end

  return self
end

-- Fills weights with the identity matrix
local eye = function(self)
  local fanIn, fanOut = calcFan(self)
  local I = torch.eye(fanOut, fanIn)
  -- Resize
  I:resize(self.weight:size())
  
  self.weight:copy(I)

  return self
end

--[[
--  Glorot, X., & Bengio, Y. (2010)
--  Understanding the difficulty of training deep feedforward neural networks
--  In International Conference on Artificial Intelligence and Statistics
--
--  Also known as Glorot initialisation
--]]
local xavier = function(self, dist, gain, ...)
  local fanIn, fanOut = calcFan(self)
  gain = gain or 'linear' -- Linear by default
  gain = calcGain(gain, ...)
  dist = dist or 'uniform' -- Uniform by default

  local stdv = gain * math.sqrt(2 / (fanIn + fanOut))
  if dist == 'uniform' then
    local b = stdv * math.sqrt(3)
    self.weight:uniform(-b, b)
  elseif dist == 'normal' then
    self.weight:normal(0, stdv)
  end

  return self
end

--[[
--  He, K., Zhang, X., Ren, S., & Sun, J. (2015)
--  Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification
--  arXiv preprint arXiv:1502.01852
--
--  Also known as He initialisation
--]]
local kaiming = function(self, dist, gain, ...)
  local fanIn = calcFan(self)
  gain = gain or 'linear' -- Linear by default
  gain = calcGain(gain, ...)
  dist = dist or 'normal' -- Normal by default

  local stdv = gain * math.sqrt(1 / fanIn)
  if dist == 'uniform' then
    local b = stdv * math.sqrt(3)
    self.weight:uniform(-b, b)
  elseif dist == 'normal' then
    self.weight:normal(0, stdv)
  end

  return self
end

--[[
--  Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013)
--  Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
--  arXiv preprint arXiv:1312.6120
--]]
local orthogonal = function(self, gain, ...)
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
local sparse = function(self, sparsity)
  local nElements = self.weight:nElement()
  local nSparseElements = math.floor(sparsity * nElements)
  local randIndices = torch.randperm(nElements):long()
  local sparseIndices = randIndices:narrow(1, 1, nSparseElements)

  -- Zero out selected indices
  self.weight:view(nElements):indexFill(1, sparseIndices, 0)

  return self
end

-- Add wInit to nn.Module
nn.Module.wInit = function(self, fn, ...)
  if fn == 'constant' then
    return constant(self, 'w', ...)
  elseif fn == 'addConstant' then
    return addConstant(self, 'w', ...)
  elseif fn == 'mulConstant' then
    return mulConstant(self, 'w', ...)
  elseif fn == 'normal' then
    return normal(self, 'w', ...)
  elseif fn == 'addNormal' then
    return addNormal(self, 'w', ...)
  elseif fn == 'uniform' then
    return uniform(self, 'w', ...)
  elseif fn == 'addUniform' then
    return addUniform(self, 'w', ...)
  elseif fn == 'eye' then
    return eye(self, ...)
  elseif fn == 'xavier' then
    return xavier(self, ...)
  elseif fn == 'kaiming' then
    return kaiming(self, ...)
  elseif fn == 'orthogonal' then
    return orthogonal(self, ...)
  elseif fn == 'sparse' then
    return sparse(self, ...)
  end
end

-- Add bInit to nn.Module
nn.Module.bInit = function(self, fn, ...)
  if fn == 'constant' then
    return constant(self, 'b', ...)
  elseif fn == 'addConstant' then
    return addConstant(self, 'b', ...)
  elseif fn == 'mulConstant' then
    return mulConstant(self, 'b', ...)
  elseif fn == 'normal' then
    return normal(self, 'b', ...)
  elseif fn == 'addNormal' then
    return addNormal(self, 'b', ...)
  elseif fn == 'uniform' then
    return uniform(self, 'b', ...)
  elseif fn == 'addUniform' then
    return addUniform(self, 'b', ...)
  end
end
