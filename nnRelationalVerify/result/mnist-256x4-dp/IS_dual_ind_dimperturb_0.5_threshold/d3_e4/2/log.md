## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0181036


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042417, -0.0015653, -0.0042417, -0.0015653, -0.0026764, 0.0026764)
1: (-0.0027538, 0.0032342, -0.0027538, 0.0032342, -0.0056532, 0.0056532)
2: (0.0077405, 0.0211185, 0.0077405, 0.0211185, -0.0133780, 0.0133780)
3: (-0.0023959, 0.0040725, -0.0023959, 0.0040725, -0.0064684, 0.0064684)
4: (0.9894158, 1.0125500, 0.9894158, 1.0125500, -0.0231342, 0.0231342)
5: (-0.0033097, 0.0047027, -0.0033097, 0.0047027, -0.0080124, 0.0080124)
6: (-0.0118370, -0.0062999, -0.0118370, -0.0062999, -0.0055371, 0.0055371)
7: (-0.0103133, -0.0033694, -0.0103133, -0.0033694, -0.0069438, 0.0069438)
8: (-0.0065013, -0.0026757, -0.0065013, -0.0026757, -0.0038257, 0.0038257)
9: (-0.0047759, 0.0179509, -0.0047759, 0.0179509, -0.0192025, 0.0192025)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.80 + 2.58 = 4.38 seconds
status: Status.VERIFIED
relational distance
Output dim: 4, lower bound: -0.0147968, upper bound: 0.0147967
