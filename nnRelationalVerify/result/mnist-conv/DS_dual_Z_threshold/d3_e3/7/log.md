## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.5986378539754209


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.0269389, -5.4159231, -7.0269389, -5.4159231, -1.0962763, 1.0962763)
1: (2.2580481, 3.4206123, 2.2580481, 3.4206123, -0.8407953, 0.8407953)
2: (-4.7107487, -3.5419888, -4.7107487, -3.5419888, -0.6490864, 0.6490864)
3: (-10.8725948, -9.2918243, -10.8725948, -9.2918243, -1.0560126, 1.0560126)
4: (-5.3605938, -4.1134462, -5.3605938, -4.1134462, -0.8966975, 0.8966975)
5: (-8.8907557, -7.6184897, -8.8907557, -7.6184897, -1.0834274, 1.0834274)
6: (-6.2716517, -4.6520371, -6.2716517, -4.6520371, -1.0614564, 1.0614562)
7: (-8.6629391, -7.6370931, -8.6629391, -7.6370931, -0.6802588, 0.6802588)
8: (1.1841860, 2.2911892, 1.1841860, 2.2911892, -0.7355913, 0.7355913)
9: (-9.1918335, -7.7131872, -9.1918335, -7.7131872, -1.0761302, 1.0761302)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.60 + 34.50 = 58.10 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.4733920, upper bound: 0.4733934
