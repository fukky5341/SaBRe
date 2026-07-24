## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0148662


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0010110, 0.0237579, 0.0010110, 0.0237579, -0.0227469, 0.0227469)
1: (-0.0058332, 0.0041669, -0.0058332, 0.0041669, -0.0100001, 0.0100001)
2: (-0.0022324, 0.0138009, -0.0022324, 0.0138009, -0.0160333, 0.0160333)
3: (-0.0041212, 0.0062953, -0.0041212, 0.0062953, -0.0104166, 0.0104166)
4: (-0.0049182, -0.0003567, -0.0049182, -0.0003567, -0.0045615, 0.0045615)
5: (-0.0023527, 0.0075168, -0.0023527, 0.0075168, -0.0098695, 0.0098695)
6: (-0.0103676, 0.0075240, -0.0103676, 0.0075240, -0.0178916, 0.0178916)
7: (-0.0200446, 0.0050339, -0.0200446, 0.0050339, -0.0250785, 0.0250785)
8: (0.9780714, 0.9960835, 0.9780714, 0.9960835, -0.0180121, 0.0180121)
9: (-0.0115908, 0.0055211, -0.0115908, 0.0055211, -0.0171119, 0.0171119)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.48 + 2.36 = 3.84 seconds
status: Status.VERIFIED
relational distance
Output dim: 8, lower bound: -0.0139744, upper bound: 0.0139744
