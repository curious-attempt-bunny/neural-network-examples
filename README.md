# Summing two numbers

See `sum_using_identity.rb`.

This is the simple problem of summing two numbers. Points of note are:
* There's no hidden layer. Just the two input units connected directly to the output unit.
* Inputs and outputs are scaled to the range 0 through 1.
* The activation function is the identity function.
* Training is halted once the change in mean squared error for the test set between epochs becomes tiny.
* Learning rates as high as 0.1 did well. The code here is using 0.01 so that the improvement over time is easier to see between epochs.
* Example trained weights are [0.007, 0.986, 0.989] which is close to the perfect solution of [0.0, 1.0, 1.0].

Example output:

```
...
Epoch 10
  Weights          [0.009599184925396284, 0.9787101464236859, 0.983655022429862]
  Training   error 1.5676162831454145e-05
  Test       error 1.5585174217532254e-05
  Validation error 1.5459221579425437e-05
Epoch 11
  Weights          [0.006514308014261394, 0.985870965394665, 0.9889739240579075]
  Training   error 7.017675304883861e-06
  Test       error 6.992045697560452e-06
  Validation error 6.9695414785398326e-06

Inspection of validation data:

inputs 6.0 (6) & 14.000000000000002 (14) outputs 20.412291530604836 (20) with target of 20.0 error of 0.004122915306048358
inputs 24.0 (24) & 32.0 (32) outputs 55.959499540751146 (56) with target of 56.00000000000001 error of 0.0004050045924886003
inputs 36.0 (36) & 6.0 (6) outputs 42.076629099981524 (42) with target of 42.0 error of 0.0007662909998152334
inputs 21.0 (21) & 18.0 (18) outputs 39.15625170775644 (39) with target of 39.0 error of 0.0015625170775643848
...
```