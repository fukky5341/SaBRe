## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01412451


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0070859, 0.0021950, -0.0070859, 0.0021950, -0.0092809, 0.0092809)
1: (-0.0059161, -0.0006881, -0.0059161, -0.0006881, -0.0052280, 0.0052280)
2: (0.0286883, 0.0445115, 0.0286883, 0.0445115, -0.0158232, 0.0158232)
3: (-0.0035656, 0.0049713, -0.0035656, 0.0049713, -0.0085368, 0.0085368)
4: (-0.0082029, 0.0042583, -0.0082029, 0.0042583, -0.0124612, 0.0124612)
5: (0.0065115, 0.0175010, 0.0065115, 0.0175010, -0.0109895, 0.0109895)
6: (-0.0174815, 0.0049197, -0.0174815, 0.0049197, -0.0224012, 0.0224012)
7: (0.9636511, 0.9787615, 0.9636511, 0.9787615, -0.0151104, 0.0151104)
8: (-0.0224543, 0.0060451, -0.0224543, 0.0060451, -0.0284994, 0.0284994)
9: (-0.0075022, 0.0099707, -0.0075022, 0.0099707, -0.0174728, 0.0174728)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.58 + 1.60 = 3.18 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0111076, upper bound: 0.0111076
