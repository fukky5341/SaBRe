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
0: (-0.0040709, -0.0040614, -0.0040709, -0.0040614, -0.0000044, 0.0000044)
1: (-0.0051119, -0.0047539, -0.0051119, -0.0047539, -0.0001629, 0.0001629)
2: (0.9703289, 0.9707586, 0.9703289, 0.9707586, -0.0001955, 0.0001955)
3: (0.0274572, 0.0306261, 0.0274572, 0.0306261, -0.0014420, 0.0014420)
4: (-0.0030223, -0.0027813, -0.0030223, -0.0027813, -0.0001097, 0.0001097)
5: (0.0142158, 0.0144593, 0.0142158, 0.0144593, -0.0001108, 0.0001108)
6: (0.0048738, 0.0049922, 0.0048738, 0.0049922, -0.0000539, 0.0000539)
7: (-0.0157153, -0.0148940, -0.0157153, -0.0148940, -0.0003737, 0.0003737)
8: (0.0042614, 0.0049129, 0.0042614, 0.0049129, -0.0002965, 0.0002965)
9: (0.0053891, 0.0065610, 0.0053891, 0.0065610, -0.0005332, 0.0005332)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.21 + 1.30 = 2.51 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0001903, upper bound: 0.0001903
