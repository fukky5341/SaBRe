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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040705, -0.0040618, -0.0040705, -0.0040618, -0.0000042, 0.0000042)
1: (-0.0050978, -0.0047690, -0.0050978, -0.0047690, -0.0001566, 0.0001566)
2: (0.9703459, 0.9707404, 0.9703459, 0.9707404, -0.0001879, 0.0001879)
3: (0.0275824, 0.0304926, 0.0275824, 0.0304926, -0.0013859, 0.0013859)
4: (-0.0030122, -0.0027908, -0.0030122, -0.0027908, -0.0001054, 0.0001054)
5: (0.0142260, 0.0144497, 0.0142260, 0.0144497, -0.0001065, 0.0001065)
6: (0.0048784, 0.0049873, 0.0048784, 0.0049873, -0.0000518, 0.0000518)
7: (-0.0156807, -0.0149265, -0.0156807, -0.0149265, -0.0003592, 0.0003592)
8: (0.0042888, 0.0048872, 0.0042888, 0.0048872, -0.0002850, 0.0002850)
9: (0.0054385, 0.0065147, 0.0054385, 0.0065147, -0.0005125, 0.0005125)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.23 + 1.29 = 2.52 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0001825, upper bound: 0.0001825
