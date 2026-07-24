## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00021588


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0035682, 0.0041159, 0.0035682, 0.0041159, -0.0002685, 0.0002685)
1: (0.0018378, 0.0019169, 0.0018378, 0.0019169, -0.0000388, 0.0000388)
2: (0.0120843, 0.0123871, 0.0120843, 0.0123871, -0.0001484, 0.0001484)
3: (-0.0021823, -0.0018691, -0.0021823, -0.0018691, -0.0001535, 0.0001535)
4: (-0.0020135, -0.0016745, -0.0020135, -0.0016745, -0.0001662, 0.0001662)
5: (0.0056980, 0.0060188, 0.0056980, 0.0060188, -0.0001573, 0.0001573)
6: (0.0003076, 0.0015805, 0.0003076, 0.0015805, -0.0006240, 0.0006240)
7: (-0.0047092, -0.0029756, -0.0047092, -0.0029756, -0.0008498, 0.0008498)
8: (0.9858966, 0.9871178, 0.9858966, 0.9871178, -0.0005986, 0.0005986)
9: (-0.0041937, -0.0030852, -0.0041937, -0.0030852, -0.0005434, 0.0005434)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 1.22 = 2.68 seconds
status: Status.ADV_EXAMPLE
