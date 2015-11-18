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

return nninit
