local nn = require 'nn'
local hasSignal, signal = pcall(require, 'signal')

-- Helper functions

-- Calculates fan in and fan out of module
local function calcFan(module)
  local typename = torch.type(module)
  if typename == 'nn.Linear' or
     typename == 'nn.LinearNoBias' or
     typename == 'nn.LookupTable' then
    return module.weight:size(2), module.weight:size(1)
  elseif typename:find('TemporalConvolution') then
    return module.weight:size(2), module.weight:size(1)
  elseif typename:find('SpatialConvolution') or typename:find('SpatialFullConvolution') then
    return module.nInputPlane * module.kW * module.kH, module.nOutputPlane * module.kW * module.kH
  elseif typename:find('VolumetricConvolution') or typename:find('VolumetricFullConvolution') then
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
  local args
  if type(gain) == 'table' then
    args = gain
    gain = gain[1]
  end

  -- Process gain strings with optional args
  if gain == 'linear' or gain == 'sigmoid' then
    return 1
  elseif gain == 'tanh' then
    return 5 / 3
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
-- TODO: Generalise for arbitrary tensors?
nninit.eye = function(module, tensor)
  if module.weight ~= tensor then
    error("nninit.eye only supports 'weight' tensor")
  end

  local typename = torch.type(module)

  if typename == 'nn.Linear' or
     typename == 'nn.LinearNoBias' or
     typename == 'nn.LookupTable' then
    local I = torch.eye(tensor:size(1), tensor:size(2))
    tensor:copy(I)
  elseif typename:find('TemporalConvolution') then
    tensor:zero()
    for i = 1, module.inputFrameSize do
      tensor[{{}, {(i-1)*module.kW + math.ceil(module.kW/2)}}]:fill(1/module.inputFrameSize)
    end
  elseif typename:find('SpatialConvolution') or typename:find('SpatialFullConvolution') then
    tensor:zero():view(module.nInputPlane, module.nOutputPlane, module.kW, module.kH)[{{}, {}, math.ceil(module.kW/2), math.ceil(module.kH/2)}]:fill(1/module.nInputPlane)
  elseif typename:find('VolumetricConvolution') or typename:find('VolumetricFullConvolution') then
    tensor:zero():view(module.nInputPlane, module.nOutputPlane, module.kT, module.kW, module.kH)[{{}, {}, math.ceil(module.kT/2), math.ceil(module.kW/2), math.ceil(module.kH/2)}]:fill(1/module.nInputPlane)
  else
    error("Unsupported module for 'eye'")
  end

  return module
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
  options = options or {}
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
  options = options or {}
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
nninit.orthogonal = function(module, tensor, options)
  local sizes = tensor:size()
  if #sizes < 2 then
    error("nninit.orthogonal only supports tensors with 2 or more dimensions")
  end

  -- Calculate "fan in" and "fan out" for arbitrary tensors based on module conventions
  local fanIn = sizes[2]
  local fanOut = sizes[1]
  for d = 3, #sizes do
    fanIn = fanIn * sizes[d]
  end

  options = options or {}
  gain = calcGain(options.gain)

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
  W:resize(tensor:size())
  -- Multiply by gain
  W:mul(gain)

  tensor:copy(W)

  return module
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

--[[
--  Aghajanyan, A. (2017)
--  Convolution Aware Initialization
--  arXiv preprint arXiv:1702.06295
--]]
nninit.convolutionAware = function(module, tensor, options)
  -- The author of the paper provided a reference implementation for Keras: https://github.com/farizrahman4u/keras-contrib/pull/60

  -- Make sure that the signal library is available, which provides the Fourier transform
  if hasSignal == false then
    error("nninit.convolutionAware requires the signal library, please make sure to install it: https://github.com/soumith/torch-signal")
  end

  -- Check the size of the convolution tensor, right now, only 2d convolution tensors are supported
  local sizes = tensor:size()
  if #sizes ~= 4 then
    error("nninit.convolutionAware only supports 2d convolutions, feel free to issue a pull request to extend this implementation")
  end

  -- Store the sizes of the convolution tensor to make the implementation easier to read
  local filterCount = sizes[1]
  local filterStacks = sizes[2]
  local filterRows = sizes[3]
  local filterCols = sizes[4]

  -- Due to the irfft2 interface of the signal library, we currently have to restrict the filter size
  if filterRows ~= filterCols then
    error("nninit.convolutionAware requires the filters to have the same number of rows and columns, feel free to issue a pull request to extend this implementation")
  end

  -- Calculate "fanIn" and "fanOut" for 2d convolution tensors based on module conventions
  local fanIn = filterStacks * filterRows * filterCols
  local fanOut = filterCount * filterRows * filterCols

  -- Setup options where "std" specifies the noise to break symmetry in the inverse Fourier transform
  options = options or {}
  gain = calcGain(options.gain)
  std = options.std or 0.05

  -- Specify the variables for the frequency domain tensor
  local fourierTensor = signal.rfft2(torch.zeros(filterRows, filterCols))
  local fourierRows = fourierTensor:size(1)
  local fourierCols = fourierTensor:size(2)
  local fourierSize = fourierRows * fourierCols

  -- Specify the variables for the orthogonal tensor buffer
  local orthogonalIndex = fourierSize
  local orthogonalTensor = nil

  -- For each filter, create a suitable basis tensor and perform an inverse Fourier transform to obtain the filter coefficients
  for filterIndex = 1, filterCount do
    basisTensor = torch.zeros(filterStacks, fourierSize)

    -- Create a suitable basis tensor using the orthogonal tensor buffer, making sure to refill the buffer should it be empty
    for basisIndex = 1, filterStacks do
      if orthogonalIndex == fourierSize then
        local randomTensor = torch.zeros(fourierSize, fourierSize):normal(0.0, 1.0)
        local symmetricTensor = randomTensor + randomTensor:t() - torch.diag(randomTensor:diag())

        orthogonalIndex = 0
        orthogonalTensor, _, _ = torch.svd(symmetricTensor)
      end

      -- Copy a column from the orthogonal tensor buffer into the basis tensor
      orthogonalIndex = orthogonalIndex + 1
      basisTensor[{ { basisIndex }, {} }] = orthogonalTensor[{ {}, { orthogonalIndex } }]
    end

    basisTensor = basisTensor:view(filterStacks, fourierRows, fourierCols)

    -- Perform the inverse Fourier transform from the basis tensor to obtain the filter coefficients, making sure to break the symmetry
    for basisIndex = 1, filterStacks do
      fourierTensor[{ {}, {}, { 1 } }] = basisTensor[{ { basisIndex }, {} }]
      fourierTensor[{ {}, {}, { 2 } }]:zero()

      -- Unlike the Numpy implementation, the inverse Fourier transform in the signal library does sadly only support a single size argument
      tensor[{ { filterIndex }, { basisIndex }, {}, {} }] = signal.irfft2(fourierTensor, filterRows) + torch.zeros(filterRows, filterCols):normal(0.0, std)
    end

    -- Clear the orthogonal tensor buffer, we do not want to reuse it for the next filter
    orthogonalIndex = fourierSize
    orthogonalTensor = nil
  end

  -- Scale the filter variance to match the variance scheme defined by He-normal initialization
  tensor:mul(gain * torch.sqrt((1.0 / fanIn) * (1.0 / tensor:var())))

  return module
end

return nninit
