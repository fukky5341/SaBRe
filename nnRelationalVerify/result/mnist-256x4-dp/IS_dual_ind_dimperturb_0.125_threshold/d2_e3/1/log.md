## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 9.720972e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0069794, 0.0071147, 0.0069794, 0.0071147, -0.0000663, 0.0000663)
1: (0.0016005, 0.0018625, 0.0016005, 0.0018625, -0.0001284, 0.0001284)
2: (0.0019049, 0.0040185, 0.0019049, 0.0040185, -0.0010357, 0.0010357)
3: (-0.0028715, -0.0026827, -0.0028715, -0.0026827, -0.0000925, 0.0000925)
4: (0.0074042, 0.0083202, 0.0074042, 0.0083202, -0.0004488, 0.0004488)
5: (-0.0017618, -0.0016250, -0.0017618, -0.0016250, -0.0000670, 0.0000670)
6: (0.9932335, 0.9934843, 0.9932335, 0.9934843, -0.0001229, 0.0001229)
7: (0.0000201, 0.0016781, 0.0000201, 0.0016781, -0.0008125, 0.0008125)
8: (0.0009946, 0.0015141, 0.0009946, 0.0015141, -0.0002545, 0.0002545)
9: (-0.0103510, -0.0093143, -0.0103510, -0.0093143, -0.0005080, 0.0005080)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.60 + 1.29 = 2.88 seconds
status: Status.VERIFIED
relational distance
Output dim: 6, lower bound: -0.0000890, upper bound: 0.0000891
