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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040700, -0.0040614, -0.0040700, -0.0040614, -0.0000041, 0.0000041)
1: (-0.0050788, -0.0047557, -0.0050788, -0.0047557, -0.0001546, 0.0001546)
2: (0.9703687, 0.9707565, 0.9703687, 0.9707565, -0.0001855, 0.0001855)
3: (0.0277507, 0.0306105, 0.0277507, 0.0306105, -0.0013682, 0.0013682)
4: (-0.0030211, -0.0028036, -0.0030211, -0.0028036, -0.0001041, 0.0001041)
5: (0.0142169, 0.0144368, 0.0142169, 0.0144368, -0.0001052, 0.0001052)
6: (0.0048847, 0.0049917, 0.0048847, 0.0049917, -0.0000512, 0.0000512)
7: (-0.0157112, -0.0149701, -0.0157112, -0.0149701, -0.0003546, 0.0003546)
8: (0.0042646, 0.0048526, 0.0042646, 0.0048526, -0.0002813, 0.0002813)
9: (0.0053949, 0.0064525, 0.0053949, 0.0064525, -0.0005059, 0.0005059)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 1.32 = 2.50 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0001777, upper bound: 0.0001778
