## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00100386


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0020963, -0.0014430, -0.0020963, -0.0014430, -0.0002276, 0.0002276)
1: (-0.0025601, -0.0008349, -0.0025601, -0.0008349, -0.0005478, 0.0005478)
2: (0.0044763, 0.0059228, 0.0044763, 0.0059228, -0.0004767, 0.0004767)
3: (-0.0041638, -0.0039946, -0.0041638, -0.0039946, -0.0000744, 0.0000744)
4: (0.0047826, 0.0059318, 0.0047826, 0.0059318, -0.0003949, 0.0003949)
5: (-0.0011309, 0.0000377, -0.0011309, 0.0000377, -0.0003795, 0.0003795)
6: (-0.0056376, -0.0049875, -0.0056376, -0.0049875, -0.0002318, 0.0002318)
7: (0.0010766, 0.0021783, 0.0010766, 0.0021783, -0.0004165, 0.0004165)
8: (-0.0004251, -0.0002618, -0.0004251, -0.0002618, -0.0000644, 0.0000644)
9: (1.0054893, 1.0084306, 1.0054893, 1.0084306, -0.0009589, 0.0009589)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.41 + 1.22 = 2.63 seconds
status: Status.VERIFIED
relational distance
Output dim: 9, lower bound: -0.0009259, upper bound: 0.0009260
