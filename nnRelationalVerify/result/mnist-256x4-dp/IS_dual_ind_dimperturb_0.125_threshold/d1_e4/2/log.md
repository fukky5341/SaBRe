## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 2.776e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0001893, 0.0006363, 0.0001893, 0.0006363, -0.0001625, 0.0001625)
1: (-0.0034543, -0.0033686, -0.0034543, -0.0033686, -0.0000264, 0.0000264)
2: (0.0151680, 0.0157357, 0.0151680, 0.0157357, -0.0001992, 0.0001992)
3: (1.0067874, 1.0069059, 1.0067874, 1.0069059, -0.0000606, 0.0000606)
4: (-0.0042097, -0.0041218, -0.0042097, -0.0041218, -0.0000297, 0.0000297)
5: (0.0041238, 0.0044655, 0.0041238, 0.0044655, -0.0001236, 0.0001236)
6: (-0.0025926, -0.0025742, -0.0025926, -0.0025742, -0.0000106, 0.0000106)
7: (-0.0125965, -0.0118059, -0.0125965, -0.0118059, -0.0003357, 0.0003357)
8: (-0.0132850, -0.0123641, -0.0132850, -0.0123641, -0.0003034, 0.0003034)
9: (0.0019976, 0.0024353, 0.0019976, 0.0024353, -0.0001406, 0.0001406)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 1.23 = 2.69 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -0.0000275, upper bound: 0.0000275
