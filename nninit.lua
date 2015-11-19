local nn = require 'nn'

-- Calculates fan in and fan out of module
local function calcFan(module)
  local typename = module.__typename

  if typename == 'nn.Linear' then
    return module.weight:size(2), module.weight:size(1)
  elseif typename == 'nn.TemporalConvolution' then
    return module.weight:size(2), module.weight:size(1)
  elseif typename == 'nn.SpatialConvolution' or typename == 'cudnn.SpatialConvolution' then
    return module.nInputPlane * module.kW * module.kH, module.nOutputPlane * module.kW * module.kH
  elseif typename == 'nn.VolumetricConvolution' or typename == 'cudnn.VolumetricConvolution' then
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

-- Fills weights/biases ~ N(mean, stdv)
local normal = function(self, wb, mean, stdv)
  if wb == 'w' then
    self.weight:normal(mean, stdv)
  elseif wb == 'b' then
    self.bias:normal(mean, stdv)
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

-- Add to nn.Module
nn.Module.init = function(self, fn, ...)
  if fn == 'constant' then
    return constant(self, ...)
  elseif fn == 'normal' then
    return normal(self, ...)
  elseif fn == 'uniform' then
    return uniform(self, ...)
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
