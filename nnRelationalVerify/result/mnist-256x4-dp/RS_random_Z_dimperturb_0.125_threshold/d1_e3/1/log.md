## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00010188


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040958, -0.0040894, -0.0040958, -0.0040894, -0.0000028, 0.0000028)
1: (-0.0060437, -0.0058036, -0.0060437, -0.0058036, -0.0001057, 0.0001057)
2: (0.9692107, 0.9694989, 0.9692107, 0.9694989, -0.0001269, 0.0001269)
3: (0.0192096, 0.0213350, 0.0192096, 0.0213350, -0.0009356, 0.0009356)
4: (-0.0023157, -0.0021540, -0.0023157, -0.0021540, -0.0000712, 0.0000712)
5: (0.0149299, 0.0150933, 0.0149299, 0.0150933, -0.0000719, 0.0000719)
6: (0.0045654, 0.0046449, 0.0045654, 0.0046449, -0.0000350, 0.0000350)
7: (-0.0133074, -0.0127566, -0.0133074, -0.0127566, -0.0002425, 0.0002425)
8: (0.0061717, 0.0066087, 0.0061717, 0.0066087, -0.0001924, 0.0001924)
9: (0.0088250, 0.0096110, 0.0088250, 0.0096110, -0.0003460, 0.0003460)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.25 = 2.58 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0000908, upper bound: 0.0000909
