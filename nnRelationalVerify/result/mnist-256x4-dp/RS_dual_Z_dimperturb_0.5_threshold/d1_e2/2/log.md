## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00146637


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040755, -0.0037957, -0.0040755, -0.0037957, -0.0002798, 0.0002798)
1: (-0.0052829, -0.0043827, -0.0052829, -0.0043827, -0.0009002, 0.0009002)
2: (0.9698058, 0.9710016, 0.9698058, 0.9710016, -0.0011958, 0.0011958)
3: (0.0259440, 0.0324190, 0.0259440, 0.0324190, -0.0049797, 0.0049797)
4: (-0.0031587, -0.0026224, -0.0031587, -0.0026224, -0.0005363, 0.0005363)
5: (0.0138684, 0.0145757, 0.0138684, 0.0145757, -0.0007073, 0.0007073)
6: (0.0045969, 0.0050593, 0.0045969, 0.0050593, -0.0004624, 0.0004624)
7: (-0.0161799, -0.0145019, -0.0161799, -0.0145019, -0.0016781, 0.0016781)
8: (0.0038928, 0.0052241, 0.0038928, 0.0052241, -0.0013313, 0.0013313)
9: (0.0045413, 0.0071206, 0.0045413, 0.0071206, -0.0025793, 0.0025793)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.13 + 1.46 = 2.58 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0010761, upper bound: 0.0010761
