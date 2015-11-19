local nninit = {}

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

-- Fills weights with a constant value
nninit.constant = function(module, val)
  module.weight:fill(val)

  return module
end

-- Fills biases with a constant value
nninit.biasConstant = function(module, val)
  module.bias:fill(val)

  return module
end

-- Fills weights ~ N(mean, stdv)
nninit.normal = function(module, mean, stdv)
  module.weight:normal(mean, stdv)

  return module
end

-- Fills biases ~ N(mean, stdv)
nninit.biasNormal = function(module, mean, stdv)
  module.bias:normal(mean, stdv)

  return module
end

-- Fills weights ~ U(a, b)
nninit.uniform = function(module, a, b)
  module.weight:uniform(a, b)

  return module
end

-- Fills biases ~ U(a, b)
nninit.biasUniform = function(module, a, b)
  module.bias:uniform(a, b)

  return module
end

--[[
--  Glorot, X., & Bengio, Y. (2010)
--  Understanding the difficulty of training deep feedforward neural networks
--  In International Conference on Artificial Intelligence and Statistics
--
--  Also known as Glorot initialisation
--]]
nninit.xavier = function(module, dist, gain, ...)
  local fanIn, fanOut = calcFan(module)
  gain = gain or 'linear' -- Linear by default
  gain = calcGain(gain, ...)
  dist = dist or 'uniform' -- Uniform by default

  local stdv = gain * math.sqrt(2 / (fanIn + fanOut))
  if dist == 'uniform' then
    local b = stdv * math.sqrt(3)
    module.weight:uniform(-b, b)
  elseif dist == 'normal' then
    module.weight:normal(0, stdv)
  end
  module.bias:zero()

  return module
end

--[[
--  He, K., Zhang, X., Ren, S., & Sun, J. (2015)
--  Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification
--  arXiv preprint arXiv:1502.01852
--
--  Also known as He initialisation
--]]
nninit.kaiming = function(module, dist, gain, ...)
  local fanIn = calcFan(module)
  gain = gain or 'linear' -- Linear by default
  gain = calcGain(gain, ...)
  dist = dist or 'normal' -- Normal by default

  local stdv = gain * math.sqrt(1 / fanIn)
  if dist == 'uniform' then
    local b = stdv * math.sqrt(3)
    module.weight:uniform(-b, b)
  elseif dist == 'normal' then
    module.weight:normal(0, stdv)
  end
  module.bias:zero()

  return module
end

--[[
--  Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013)
--  Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
--  arXiv preprint arXiv:1312.6120
--]]
nninit.orthogonal = function(module, gain, ...)
  local fanIn, fanOut = calcFan(module)
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
  W:resize(module.weight:size())
  -- Multiply by gain
  W:mul(gain)

  module.weight:copy(W)
  module.bias:zero()

  return module
end

--[[
-- Martens, J. (2010)
-- Deep learning via Hessian-free optimization
-- In Proceedings of the 27th International Conference on Machine Learning (ICML-10)
--]]
nninit.sparse = function(module, sparsity)
  local nElements = module.weight:nElement()
  local nSparseElements = math.floor(sparsity * nElements)
  local randIndices = torch.randperm(nElements):long()
  local sparseIndices = randIndices:narrow(1, 1, nSparseElements)

  -- Zero out selected indices
  module.weight:view(nElements):indexFill(1, sparseIndices, 0)

  return module
end

return nninit
