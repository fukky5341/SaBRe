## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00473928


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0094880, -0.0031697, -0.0094880, -0.0031697, -0.0038371, 0.0038371)
1: (-0.0050448, -0.0043393, -0.0050448, -0.0043393, -0.0007054, 0.0007054)
2: (0.0328518, 0.0492802, 0.0328518, 0.0492802, -0.0111326, 0.0111326)
3: (0.0007693, 0.0110240, 0.0007693, 0.0110240, -0.0049693, 0.0049693)
4: (-0.0041314, -0.0022687, -0.0041314, -0.0022687, -0.0018627, 0.0018627)
5: (0.0100700, 0.0118353, 0.0100700, 0.0118353, -0.0017653, 0.0017653)
6: (-0.0170294, -0.0022342, -0.0170294, -0.0022342, -0.0072822, 0.0072822)
7: (0.9546633, 0.9748665, 0.9546633, 0.9748665, -0.0202032, 0.0202032)
8: (-0.0054905, 0.0006441, -0.0054905, 0.0006441, -0.0061346, 0.0061346)
9: (-0.0040282, -0.0009918, -0.0040282, -0.0009918, -0.0030363, 0.0030363)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 1.23 = 2.68 seconds
status: Status.ADV_EXAMPLE
