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
Threshold: 0.00072416


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040707, -0.0040616, -0.0040707, -0.0040616, -0.0000043, 0.0000043)
1: (-0.0051028, -0.0047633, -0.0051028, -0.0047633, -0.0001606, 0.0001606)
2: (0.9703399, 0.9707474, 0.9703399, 0.9707474, -0.0001927, 0.0001927)
3: (0.0275379, 0.0305433, 0.0275379, 0.0305433, -0.0014217, 0.0014217)
4: (-0.0030160, -0.0027874, -0.0030160, -0.0027874, -0.0001081, 0.0001081)
5: (0.0142221, 0.0144531, 0.0142221, 0.0144531, -0.0001093, 0.0001093)
6: (0.0048768, 0.0049891, 0.0048768, 0.0049891, -0.0000532, 0.0000532)
7: (-0.0156938, -0.0149149, -0.0156938, -0.0149149, -0.0003684, 0.0003684)
8: (0.0042784, 0.0048963, 0.0042784, 0.0048963, -0.0002923, 0.0002923)
9: (0.0054198, 0.0065312, 0.0054198, 0.0065312, -0.0005257, 0.0005257)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.16 + 1.30 = 2.46 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0001885, upper bound: 0.0001885
