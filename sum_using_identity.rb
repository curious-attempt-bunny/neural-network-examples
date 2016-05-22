def scale(i)
  i.to_f / 100.0
end

def reverse_scale(i)
  i * 100.0
end

# identity function
def activation(a)
  a
end

# identity function has a simple gradient error * input
def activation_derived(x, weights, input)
  error(x, weights)*input
end

def evaluate(x, weights)
  activation(1.0*weights[0] + x[0]*weights[1] + x[1]*weights[2])
end

def error(x, weights)
  x[2] - evaluate(x, weights)
end

def cost(x, weights)
  (error(x,weights)**2) / 2.0
end

def derived_cost(x, weights)
  [
    activation_derived(x, weights, 1.0),
    activation_derived(x, weights, x[0]),
    activation_derived(x, weights, x[1])
  ]
end

def mean_squared_error(data_set, weights)
  data_set.map { |x| error(x, weights)**2 }.inject(&:+) / data_set.size
end

entire_data_set = []
0.upto(50).each do |lhs|
  0.upto(50).each do |rhs|
    entire_data_set << [scale(lhs), scale(rhs), scale(lhs + rhs)]
  end
end

entire_data_set.shuffle!
training_data = entire_data_set[0...(entire_data_set.size*0.8)]
test_data = entire_data_set[(entire_data_set.size*0.8)...(entire_data_set.size*0.9)]
validation_data = entire_data_set[(entire_data_set.size*0.9)..-1]

weights = [rand()-0.5, rand()-0.5, rand()-0.5]

puts "entire_data_set.size = #{entire_data_set.size}"
puts "training_data.size   = #{training_data.size}"
puts "test_data.size       = #{test_data.size}"
puts "validation_data.size = #{validation_data.size}"

learning_rate = 0.01
epoch = 1
previous_test_error = mean_squared_error(validation_data, weights)
puts "Test error prior to training: #{previous_test_error}"
while true
  training_data.shuffle!

  training_data.each do |x|
    # puts "1.0 * #{weights[0]} + #{x[0]} * #{weights[1]} + #{x[1]} * #{weights[2]} = #{evaluate(x,weights)} (target: #{x[2]}, error: #{error(x,weights)})"
    derived_cost = derived_cost(x, weights)
    # puts "  #{derived_cost.inspect}"
    derived_cost.each_with_index do |component_derived_cost, i|
      # puts "component_derived_cost: #{component_derived_cost} for #{i}: #{weights[i]}"
      weights[i] += learning_rate * component_derived_cost
    end
    # puts "  resulting weights: #{weights.inspect}"
    # puts "  resulting error: #{error(x,weights)}"
  end

  puts "Epoch #{epoch}"
  puts "  Weights          #{weights.inspect}"
  puts "  Training   error #{mean_squared_error(training_data, weights)}"
  puts "  Test       error #{mean_squared_error(test_data, weights)}"
  puts "  Validation error #{mean_squared_error(validation_data, weights)}"

  break if (previous_test_error - mean_squared_error(test_data, weights)).abs < 0.00001

  previous_test_error = mean_squared_error(test_data, weights)

  epoch += 1
end

puts

puts "Inspection of validation data:"
puts

validation_data.each do |x|
  puts "inputs #{reverse_scale(x[0])} (#{reverse_scale(x[0]).round}) & #{reverse_scale(x[1])} (#{reverse_scale(x[1]).round}) outputs #{reverse_scale(evaluate(x,weights))} (#{reverse_scale(evaluate(x,weights)).round}) with target of #{reverse_scale(x[2])} error of #{error(x,weights).abs}"
end

