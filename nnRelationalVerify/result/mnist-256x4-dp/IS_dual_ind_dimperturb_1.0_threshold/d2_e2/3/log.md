## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.0794310745157396


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.3936425, 0.5104463, -0.3936425, 0.5104463, -0.9040889, 0.9040889)
1: (-0.3611194, 0.3743429, -0.3611194, 0.3743429, -0.7354623, 0.7354623)
2: (-0.3189284, 0.4941514, -0.3189284, 0.4941514, -0.8130797, 0.8130797)
3: (-0.2711910, 0.4512931, -0.2711910, 0.4512931, -0.7224841, 0.7224841)
4: (-0.3643050, 0.3685117, -0.3643050, 0.3685117, -0.7328166, 0.7328166)
5: (-0.3684157, 0.4413534, -0.3684157, 0.4413534, -0.8097690, 0.8097690)
6: (-0.3243454, 0.4679304, -0.3243454, 0.4679304, -0.7922758, 0.7922758)
7: (-0.3948485, 0.4353864, -0.3948485, 0.4353864, -0.8302349, 0.8302349)
8: (0.1751941, 1.0736474, 0.1751941, 1.0736474, -0.8984532, 0.8984532)
9: (-0.4112116, 0.5059593, -0.4112116, 0.5059593, -0.9171709, 0.9171709)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.27 + 2.72 = 3.99 seconds
status: Status.VERIFIED
relational distance
Output dim: 8, lower bound: -0.8374108, upper bound: 0.8374108
