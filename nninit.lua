local nninit = {}

-- Calculates fan in and fan out of module
local function calcFan(module)
  local typename = module.__typename

  if typename == 'nn.Linear' then
    return module.weight:size(2), module.weight:size(1)
  elseif typename == 'nn.SpatialConvolution' then
    return module.nInputPlane * module.kH * module.kW, module.nOutputPlane * module.kH * module.kW
  elseif typename == 'nn.TemporalConvolution' then
    return module.weight:size(2), module.weight:size(1)
  elseif typename == 'nn.HorizontalConvolution' then
    return module.kH * module.kW, module.kH * module.kW
  elseif typename == 'nn.VerticalConvolution' then
    return module.kH * module.kW, module.kH * module.kW
  elseif typename == 'nn.LateralConvolution' then
    return module.nInputPlane, module.nOutputPlane
  end
end

nninit.constant = function(module, val)
  module.weight:fill(val)

  return module
end

nninit.normal = function(module, mean, stdv)
  module.weight:normal(mean, stdv)

  return module
end

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
nninit.xavier = function(module, dist)
  local fanIn, fanOut = calcFan(module)
  dist = dist or 'uniform' -- Uniform by default

  local stdv = math.sqrt(2 / (fanIn + fanOut))
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
nninit.kaiming = function(module, dist)
  local fanIn = calcFan(module)
  dist = dist or 'normal' -- Normal by default

  local stdv = math.sqrt(1 / fanIn)
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
