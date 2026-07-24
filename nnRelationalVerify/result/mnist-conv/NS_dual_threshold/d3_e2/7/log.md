## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.4590685438731504


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.4139881, -11.2255135, -12.4139881, -11.2255135, -0.6577733, 0.6577733)
1: (-6.9266186, -6.2103581, -6.9266186, -6.2103581, -0.4071734, 0.4071734)
2: (10.3175907, 11.1374578, 10.3175907, 11.1374578, -0.4527867, 0.4527868)
3: (-2.9911165, -2.1709232, -2.9911165, -2.1709232, -0.4422367, 0.4422365)
4: (-10.0884972, -8.9663677, -10.0884972, -8.9663677, -0.5930830, 0.5930830)
5: (-15.2414322, -14.1796751, -15.2414322, -14.1796751, -0.5463638, 0.5463638)
6: (-14.8709164, -13.7281513, -14.8709164, -13.7281513, -0.5891317, 0.5891320)
7: (-4.1133437, -3.2332451, -4.1133437, -3.2332451, -0.4156903, 0.4156903)
8: (-1.1690698, -0.3937376, -1.1690698, -0.3937376, -0.4013501, 0.4013501)
9: (-9.5523605, -8.4703913, -9.5523605, -8.4703913, -0.5556037, 0.5556038)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.93 + 32.61 = 56.54 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.4014909, upper bound: 0.4014919
