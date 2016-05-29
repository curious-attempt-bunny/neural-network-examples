require 'nn'

local model = nn.Sequential();  -- make a multi-layer perceptron
local inputs = 2*8; local outputs = 8+1; local HUs = 18; -- parameters
model:add(nn.Linear(inputs, HUs))
model:add(nn.Sigmoid())
model:add(nn.Linear(HUs, outputs))
model:add(nn.Sigmoid())

local criterion = nn.MSECriterion()

local batchSize = 256*256
local batchInputs = torch.Tensor(batchSize, inputs)
local batchLabels = torch.Tensor(batchSize, outputs)

local lhs = 0
local rhs = 0
for i=1,batchSize do
  local input = torch.Tensor(inputs)
  for j=1,8 do
    input[j] = bit.band(bit.rshift(lhs, j-1), 1)
  end
  for j=1,8 do
    input[j+8] = bit.band(bit.rshift(rhs, j-1), 1)
  end
  local sum = lhs+rhs
  local output = torch.Tensor(outputs)
  for j=1,9 do
    output[j] = bit.band(bit.rshift(sum, j-1), 1)
  end
  batchInputs[i]:copy(input)
  batchLabels[i]:copy(output)

  lhs = lhs + 1
  if lhs == 256 then
    lhs = 0
    rhs = rhs + 1
  end
end

local params, gradParams = model:getParameters()
for i=1,params:size(1) do
    params[i] = math.random() - 0.5
end

local optimState = {learningRate=0.3}

require 'optim'

for epoch=1,5000 do
    local chunks = 256*32
    optimState = {learningRate= 0.03}
    if epoch < 500 then
        optimState = {learningRate= 0.1}
    end
    if epoch < 250 then
        optimState = {learningRate= 0.3}
    end
    if epoch < 50 then
        optimState = {learningRate= 0.9}
    end
    local shuffle_indexes = torch.randperm(batchInputs:size(1))
    local chunkSize = batchSize / chunks
    for b=0,chunks-1 do
        local batch = torch.Tensor(chunkSize, inputs)
        local labels = torch.Tensor(chunkSize, outputs)
        for i=1,chunkSize do
            local j = shuffle_indexes[b*chunkSize + i]

            batch[i] = batchInputs[j]
            labels[i] = batchLabels[j]
        end

        local function feval(params)
            gradParams:zero()

            local outputs = model:forward(batch)
            local loss = criterion:forward(outputs, labels)
            local dloss_doutput = criterion:backward(outputs, labels)
            model:backward(batch, dloss_doutput)

            return loss,gradParams
        end

        optim.sgd(feval, params, optimState)
    end
      -- print(loss)
    if epoch % 10 == 0 then
        local accuracy = 0.0
        for i=1,batchSize do
            local input = batchInputs[i]
            local labels = batchLabels[i]
            local output = model:forward(input)
            local accurate = true
            for j=1,9 do
                if output[j] >= 0.5 then
                    output[j] = 1
                else
                    output[j] = 0
                end
                if not(output[j] == labels[j]) then
                    accurate = false
                end
            end
            if accurate then
                accuracy = accuracy + 1
            end
        end
      --  print(loss)
        print(accuracy/batchSize)
    end
end

local lhs = 10
local rhs = 23
local input = torch.Tensor(inputs)
for j=1,8 do
input[j] = bit.band(bit.rshift(lhs, j-1), 1)
end
for j=1,8 do
input[j+8] = bit.band(bit.rshift(rhs, j-1), 1)
end
local output = model:forward(input)
for j=1,9 do
    if output[j] >= 0.5 then
        output[j] = 1
    else
        output[j] = 0
    end
end
print(input)
print(output)
