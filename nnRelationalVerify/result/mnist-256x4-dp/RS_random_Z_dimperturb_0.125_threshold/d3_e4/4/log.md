## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00085992


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0070511, 0.0083576, 0.0070511, 0.0083576, -0.0009603, 0.0009603)
1: (0.0023410, 0.0025297, 0.0023410, 0.0025297, -0.0001387, 0.0001387)
2: (0.0097392, 0.0104615, 0.0097392, 0.0104615, -0.0005309, 0.0005309)
3: (-0.0046078, -0.0038607, -0.0046078, -0.0038607, -0.0005491, 0.0005491)
4: (0.0001425, 0.0009512, 0.0001425, 0.0009512, -0.0005944, 0.0005944)
5: (0.0032132, 0.0039785, 0.0032132, 0.0039785, -0.0005625, 0.0005625)
6: (-0.0095514, -0.0065146, -0.0095514, -0.0065146, -0.0022320, 0.0022320)
7: (0.0063157, 0.0104515, 0.0063157, 0.0104515, -0.0030398, 0.0030398)
8: (0.9936627, 0.9965761, 0.9936627, 0.9965761, -0.0021413, 0.0021413)
9: (-0.0127793, -0.0101348, -0.0127793, -0.0101348, -0.0019437, 0.0019437)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.80 + 1.30 = 3.10 seconds
status: Status.VERIFIED
relational distance
Output dim: 8, lower bound: -0.0008330, upper bound: 0.0008331
