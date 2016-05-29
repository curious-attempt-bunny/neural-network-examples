# Summing two numbers (2 scalar inputs, 1 scalar output)

See [sum_using_identity.rb](https://github.com/curious-attempt-bunny/neural-network-examples/blob/master/sum_using_identity.rb).

## Overview

This is the simple problem of summing two numbers. Points of note are:
* There's no hidden layer. Just the two input units connected directly to the output unit.
* Inputs and outputs are [scaled](https://github.com/curious-attempt-bunny/neural-network-examples/blob/master/sum_using_identity.rb#L2) to the range 0 through 1.
* The activation function is the [identity function](https://github.com/curious-attempt-bunny/neural-network-examples/blob/master/sum_using_identity.rb#L9-L12).
* Training is halted once the change in mean squared error for the test set between epochs [becomes tiny](https://github.com/curious-attempt-bunny/neural-network-examples/blob/master/sum_using_identity.rb#L87).
* [Learning rates](https://github.com/curious-attempt-bunny/neural-network-examples/blob/master/sum_using_identity.rb#L62) as high as 0.1 did well. The code here is using 0.01 so that the improvement over time is easier to see between epochs.
* Example trained weights are [0.007, 0.986, 0.989] which is close to the perfect solution of [0.0, 1.0, 1.0].

## Gotchas

* Get the [derivation with respect to each of the weights](https://github.com/curious-attempt-bunny/neural-network-examples/blob/master/sum_using_identity.rb#L31-L37) correct. Refer to [https://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error](https://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error) for details. Derivation of oj with respect to netj is 1 for the identity function.
* It's `target minus output` for the [error](https://github.com/curious-attempt-bunny/neural-network-examples/blob/master/sum_using_identity.rb#L23-L25). If you invert that then you'll need to flip a sign elsewhere for it to work.
* The mean squared errors across the data sets and weights are good for [debugging](https://github.com/curious-attempt-bunny/neural-network-examples/blob/master/sum_using_identity.rb#L81-L85).
* The [termination criteria](https://github.com/curious-attempt-bunny/neural-network-examples/blob/master/sum_using_identity.rb#L87) need to be very sensitive because the outputs are sensitive to small errors (10 scales to 0.10, and 11 scales to 0.11, so an error of 0.01 or squared error of 0.0001 is huge).
* [Shuffling the entire data set](https://github.com/curious-attempt-bunny/neural-network-examples/blob/master/sum_using_identity.rb#L50) is very important.
* [Rounding the output](https://github.com/curious-attempt-bunny/neural-network-examples/blob/master/sum_using_identity.rb#L100) rather than flooring the output is key when inspecting the validation data.

## Example output

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

# Summing two numbers (2*8 bitwise inputs, 9 bitwise outputs)

See [sum_using_bit_represenation.lua](https://github.com/curious-attempt-bunny/neural-network-examples/blob/master/sum_using_bit_represenation.lua). You'll need [Torch](http://torch.ch/) installed in order to run this.

## Overview

* The network uses sigmoid activation functions.
* The full range of all possible inputs was provided in the training set. So this was a recall exercise (i.e. no test or validation set).
* Training was very slow going. It looks 500 epochs and a number of techniques speed things up!
* Minibatches of 4 gave the best results (selected from a shuffled full dataset)
* An early learning rate of 0.9 yielded strong early epoch results.
* A schedule for the learning rate dropped it to 0.3, 0.1, and 0.01 for epochs 51..250, 251..500, 501+ respectively.

## Gotchas

* You need patience for this! It took 500 epochs to reach 100% recall on the dataset.
* Bisecting on the number of hidden units led to 18 hidden units as the point where it converged on a solution quickly.
* Checking the accuracy of the network was very slow so it's done once every 10 epochs.


## Example output

Note that this is outputting the accuracy on the full dataset every 10 epochs (i.e. not the training error on the output nodes):
```
0.21949768066406
0.50169372558594
0.62513732910156
0.85713195800781
0.9383544921875
0.95703125
0.96766662597656
0.97686767578125
0.98966979980469
0.98698425292969
0.9952392578125
0.9962158203125
0.99430847167969
0.99769592285156
0.9962158203125
0.99459838867188
0.99739074707031
0.99713134765625
0.99668884277344
0.99742126464844
0.99897766113281
0.99504089355469
0.99601745605469
0.99737548828125
0.99940490722656
0.99958801269531
0.99957275390625
0.99984741210938
0.99708557128906
0.99980163574219
0.99885559082031
0.99714660644531
0.99977111816406
0.99990844726562
0.99977111816406
0.99989318847656
0.99940490722656
0.99937438964844
0.99931335449219
0.99977111816406
0.99989318847656
0.999267578125
0.99989318847656
0.99995422363281
0.99995422363281
0.9962158203125
0.99995422363281
0.99919128417969
0.99966430664062
0.99996948242188
1
```