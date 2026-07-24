## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 9.720972e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0070018, 0.0071132, 0.0070018, 0.0071132, -0.0000619, 0.0000619)
1: (0.0016439, 0.0018597, 0.0016439, 0.0018597, -0.0001200, 0.0001200)
2: (0.0019275, 0.0036681, 0.0019275, 0.0036681, -0.0009677, 0.0009677)
3: (-0.0028402, -0.0026847, -0.0028402, -0.0026847, -0.0000864, 0.0000864)
4: (0.0075561, 0.0083104, 0.0075561, 0.0083104, -0.0004193, 0.0004193)
5: (-0.0017603, -0.0016477, -0.0017603, -0.0016477, -0.0000626, 0.0000626)
6: (0.9932752, 0.9934816, 0.9932752, 0.9934816, -0.0001148, 0.0001148)
7: (0.0002950, 0.0016604, 0.0002950, 0.0016604, -0.0007591, 0.0007591)
8: (0.0010808, 0.0015085, 0.0010808, 0.0015085, -0.0002378, 0.0002378)
9: (-0.0103400, -0.0094862, -0.0103400, -0.0094862, -0.0004746, 0.0004746)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 1.29 = 2.70 seconds
status: Status.VERIFIED
relational distance
Output dim: 6, lower bound: -0.0000762, upper bound: 0.0000762
