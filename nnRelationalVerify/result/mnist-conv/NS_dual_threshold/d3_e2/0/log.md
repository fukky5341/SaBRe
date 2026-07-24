## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.4956641551718251


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.5620794, -4.6123977, -5.5620794, -4.6123977, -0.6276813, 0.6276810)
1: (-7.8563643, -6.7703190, -7.8563643, -6.7703190, -0.5915331, 0.5915332)
2: (-4.9149446, -3.8728876, -4.9149446, -3.8728876, -0.6851301, 0.6851301)
3: (6.6139617, 7.7194357, 6.6139617, 7.7194357, -0.5454841, 0.5454841)
4: (-13.6602554, -12.1493416, -13.6602554, -12.1493416, -0.7471505, 0.7471504)
5: (-1.0872331, -0.0750502, -1.0872331, -0.0750502, -0.6599443, 0.6599443)
6: (-10.4108829, -9.1494446, -10.4108829, -9.1494446, -0.6827829, 0.6827829)
7: (-8.5220213, -7.3562346, -8.5220213, -7.3562346, -0.6137354, 0.6137354)
8: (-3.7094417, -2.6971564, -3.7094417, -2.6971564, -0.5617008, 0.5617007)
9: (-5.8063717, -4.7306633, -5.8063717, -4.7306633, -0.6551542, 0.6551542)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.19 + 35.49 = 57.68 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -0.4037484, upper bound: 0.4037484
