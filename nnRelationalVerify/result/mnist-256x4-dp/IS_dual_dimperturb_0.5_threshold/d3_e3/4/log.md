## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0041507


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9914924, 0.9984019, 0.9914924, 0.9984019, -0.0069095, 0.0069095)
1: (-0.0033838, -0.0016622, -0.0033838, -0.0016622, -0.0017217, 0.0017217)
2: (-0.0012454, 0.0078785, -0.0012454, 0.0078785, -0.0091239, 0.0091239)
3: (-0.0048591, -0.0007063, -0.0048591, -0.0007063, -0.0041528, 0.0041528)
4: (0.0002868, 0.0020528, 0.0002868, 0.0020528, -0.0017659, 0.0017659)
5: (-0.0026069, 0.0088685, -0.0026069, 0.0088685, -0.0114754, 0.0114754)
6: (-0.0007101, 0.0022025, -0.0007101, 0.0022025, -0.0029126, 0.0029126)
7: (-0.0049749, 0.0025609, -0.0049749, 0.0025609, -0.0075358, 0.0075358)
8: (-0.0021804, 0.0017826, -0.0021804, 0.0017826, -0.0039630, 0.0039630)
9: (-0.0039309, 0.0006644, -0.0039309, 0.0006644, -0.0045953, 0.0045953)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.60 + 2.11 = 3.71 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0038805, upper bound: 0.0038805
