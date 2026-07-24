## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00147475


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0003332, 0.0028727, 0.0003332, 0.0028727, -0.0015084, 0.0015084)
1: (0.0013704, 0.0017373, 0.0013704, 0.0017373, -0.0002179, 0.0002179)
2: (0.0127716, 0.0141757, 0.0127716, 0.0141757, -0.0008339, 0.0008339)
3: (-0.0014714, -0.0000193, -0.0014714, -0.0000193, -0.0008625, 0.0008625)
4: (-0.0040161, -0.0024440, -0.0040161, -0.0024440, -0.0009337, 0.0009337)
5: (0.0064262, 0.0079139, 0.0064262, 0.0079139, -0.0008836, 0.0008836)
6: (0.0031971, 0.0090997, 0.0031971, 0.0090997, -0.0035059, 0.0035059)
7: (-0.0149497, -0.0069109, -0.0149497, -0.0069109, -0.0047747, 0.0047747)
8: (0.9786830, 0.9843456, 0.9786830, 0.9843456, -0.0033634, 0.0033634)
9: (-0.0016773, 0.0034629, -0.0016773, 0.0034629, -0.0030531, 0.0030531)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 1.27 = 2.77 seconds
status: Status.ADV_EXAMPLE
