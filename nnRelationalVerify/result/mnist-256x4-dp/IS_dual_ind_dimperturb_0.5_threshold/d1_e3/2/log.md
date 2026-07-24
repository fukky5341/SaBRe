## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00109205


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0043871, -0.0040989, -0.0043871, -0.0040989, -0.0002093, 0.0002093)
1: (0.0024438, 0.0040393, 0.0024438, 0.0040393, -0.0011592, 0.0011592)
2: (0.0059419, 0.0095064, 0.0059419, 0.0095064, -0.0025897, 0.0025897)
3: (0.0033283, 0.0048304, 0.0033283, 0.0048304, -0.0010913, 0.0010913)
4: (1.0096629, 1.0154904, 1.0096629, 1.0154904, -0.0042338, 0.0042338)
5: (0.0041212, 0.0052549, 0.0041212, 0.0052549, -0.0008236, 0.0008236)
6: (-0.0125814, -0.0111061, -0.0125814, -0.0111061, -0.0010719, 0.0010719)
7: (-0.0104082, -0.0102200, -0.0104082, -0.0102200, -0.0001367, 0.0001367)
8: (-0.0031807, -0.0021613, -0.0031807, -0.0021613, -0.0007406, 0.0007406)
9: (-0.0073507, -0.0022477, -0.0073507, -0.0022477, -0.0037075, 0.0037075)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 1.42 = 2.75 seconds
status: Status.ADV_EXAMPLE
