local nninit = {}

-- Calculates fan in and fan out of module
local function calcFan(module)
  local typename = module.__typename

  if typename == 'nn.Linear' then
    return module.weight:size(2), module.weight:size(1)
  elseif typename == 'nn.TemporalConvolution' then
    return module.weight:size(2), module.weight:size(1)
  elseif typename == 'nn.SpatialConvolution' then
    return module.nInputPlane * module.kW * module.kH, module.nOutputPlane * module.kW * module.kH
  elseif typename == 'nn.VolumetricConvolution' then
    return module.nInputPlane * module.kT * module.kW * module.kH, module.nOutputPlane * module.kT * module.kW * module.kH
  end
end

-- Calculates gain given a type and optional parameters
local function calcGain(gainType, ...)
  local args = {...}

  if gainType == 'linear' or gainType == 'sigmoid' then
    return 1
  elseif gainType == 'relu' then
    return math.sqrt(2)
  elseif gainType == 'lrelu' then
    local leakiness = args[1]
    return math.sqrt(2 / (1 + math.pow(leakiness, 2)))
  end
end

-- Fills weights with a constant value
nninit.constant = function(module, val)
  module.weight:fill(val)

  return module
end

-- Fills weights ~ N(mean, stdv)
nninit.normal = function(module, mean, stdv)
  module.weight:normal(mean, stdv)

  return module
end

-- Fills weights ~ U(a, b)
nninit.uniform = function(module, a, b)
  module.weight:uniform(a, b)

  return module
end

--[[
--  Glorot, X., & Bengio, Y. (2010)
--  Understanding the difficulty of training deep feedforward neural networks
--  In International Conference on Artificial Intelligence and Statistics
--
--  Also known as Glorot initialisation
--]]
nninit.xavier = function(module, dist, gainType, ...)
  local fanIn, fanOut = calcFan(module)
  gainType = gainType or 'linear' -- Linear by default
  local gain = calcGain(gainType, ...)
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
nninit.kaiming = function(module, dist, gainType, ...)
  local fanIn = calcFan(module)
  gainType = gainType or 'linear' -- Linear by default
  local gain = calcGain(gainType, ...)
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
nninit.orthogonal = function(module, gainType, ...)
  local fanIn, fanOut = calcFan(module)
  gainType = gainType or 'linear' -- Linear by default
  local gain = calcGain(gainType, ...)

  -- Construct random matrix
  local randMat = torch.Tensor(fanOut, fanIn):normal(0, 1)
  local U, __, V = torch.svd(randMat, 'A')

  -- Pick out orthogonal matrix
  local W
  if fanOut > fanIn then
    W = U:narrow(2, 1, fanIn)
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
