## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00527912


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0013408, 0.0013704, -0.0013408, 0.0013704, -0.0026131, 0.0026131)
1: (-0.0036403, -0.0025478, -0.0036403, -0.0025478, -0.0010924, 0.0010924)
2: (0.0320190, 0.0337702, 0.0320190, 0.0337702, -0.0017513, 0.0017513)
3: (-0.0031223, -0.0010598, -0.0031223, -0.0010598, -0.0017995, 0.0017995)
4: (-0.0022593, -0.0007658, -0.0022593, -0.0007658, -0.0014936, 0.0014936)
5: (0.0112880, 0.0137995, 0.0112880, 0.0137995, -0.0025115, 0.0025115)
6: (-0.0039703, -0.0023184, -0.0039703, -0.0023184, -0.0016518, 0.0016518)
7: (0.9757870, 0.9768119, 0.9757870, 0.9768119, -0.0010250, 0.0010250)
8: (-0.0142849, -0.0072790, -0.0142849, -0.0072790, -0.0070059, 0.0070059)
9: (0.0002105, 0.0042658, 0.0002105, 0.0042658, -0.0040553, 0.0040553)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.16 + 1.37 = 2.53 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0006178, upper bound: 0.0006178
