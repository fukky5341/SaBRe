## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0002144


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9913708, 0.9922087, 0.9913708, 0.9922087, -0.0004610, 0.0004610)
1: (-0.0034141, -0.0032053, -0.0034141, -0.0032053, -0.0001149, 0.0001149)
2: (0.0069326, 0.0080391, 0.0069326, 0.0080391, -0.0006088, 0.0006088)
3: (-0.0049322, -0.0044286, -0.0049322, -0.0044286, -0.0002771, 0.0002771)
4: (0.0018697, 0.0020838, 0.0018697, 0.0020838, -0.0001178, 0.0001178)
5: (0.0076789, 0.0090705, 0.0076789, 0.0090705, -0.0007657, 0.0007657)
6: (-0.0007614, -0.0004081, -0.0007614, -0.0004081, -0.0001943, 0.0001943)
7: (-0.0051075, -0.0041936, -0.0051075, -0.0041936, -0.0005028, 0.0005028)
8: (-0.0022501, -0.0017695, -0.0022501, -0.0017695, -0.0002644, 0.0002644)
9: (0.0001880, 0.0007453, 0.0001880, 0.0007453, -0.0003066, 0.0003066)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 1.23 = 2.65 seconds
status: Status.ADV_EXAMPLE
