## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.40684923


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1575796, 0.1929853, -0.1575796, 0.1929853, -0.3505649, 0.3505649)
1: (0.6776892, 1.0640551, 0.6776892, 1.0640551, -0.3863659, 0.3863659)
2: (-0.1251361, 0.1759242, -0.1251361, 0.1759242, -0.3010603, 0.3010603)
3: (-0.0812532, 0.1257458, -0.0812532, 0.1257458, -0.2069990, 0.2069990)
4: (-0.1330110, 0.1204199, -0.1330110, 0.1204199, -0.2534309, 0.2534309)
5: (-0.1271938, 0.1388515, -0.1271938, 0.1388515, -0.2660453, 0.2660453)
6: (-0.1737246, 0.1496684, -0.1737246, 0.1496684, -0.3233930, 0.3233930)
7: (-0.1273452, 0.1651041, -0.1273452, 0.1651041, -0.2924494, 0.2924494)
8: (-0.0828770, 0.2278950, -0.0828770, 0.2278950, -0.3107721, 0.3107721)
9: (-0.1581204, 0.1631413, -0.1581204, 0.1631413, -0.3212618, 0.3212618)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 2.06 = 3.56 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.3387955, upper bound: 0.3387955
