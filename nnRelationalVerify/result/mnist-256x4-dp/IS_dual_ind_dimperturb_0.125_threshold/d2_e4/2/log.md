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
Threshold: 0.0014131413


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0000843, 0.0005999, -0.0000843, 0.0005999, -0.0006843, 0.0006843)
1: (-0.0037019, -0.0027882, -0.0037019, -0.0027882, -0.0007817, 0.0007817)
2: (0.0327333, 0.0333002, 0.0327333, 0.0333002, -0.0005669, 0.0005669)
3: (-0.0026175, -0.0015589, -0.0026175, -0.0015589, -0.0010586, 0.0010586)
4: (-0.0026585, -0.0014534, -0.0026585, -0.0014534, -0.0007907, 0.0007907)
5: (0.0124442, 0.0130833, 0.0124442, 0.0130833, -0.0006390, 0.0006390)
6: (-0.0032098, -0.0015903, -0.0032098, -0.0015903, -0.0010339, 0.0010339)
7: (0.9760064, 0.9769463, 0.9760064, 0.9769463, -0.0007845, 0.0007845)
8: (-0.0122892, -0.0105043, -0.0122892, -0.0105043, -0.0017849, 0.0017849)
9: (0.0020774, 0.0031181, 0.0020774, 0.0031181, -0.0010407, 0.0010407)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 1.37 = 2.95 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0005283, upper bound: 0.0005283
