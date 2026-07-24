## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.044602215


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0133614, 0.0039744, -0.0133614, 0.0039744, -0.0173358, 0.0173358)
1: (-0.0088904, 0.0085930, -0.0088904, 0.0085930, -0.0174834, 0.0174834)
2: (0.9385499, 0.9828106, 0.9385499, 0.9828106, -0.0442606, 0.0442606)
3: (0.0056092, 0.0551789, 0.0056092, 0.0551789, -0.0495696, 0.0495696)
4: (-0.0280572, 0.0330979, -0.0280572, 0.0330979, -0.0611551, 0.0611551)
5: (0.0050801, 0.0360547, 0.0050801, 0.0360547, -0.0309747, 0.0309747)
6: (-0.0180389, 0.0184402, -0.0180389, 0.0184402, -0.0364790, 0.0364790)
7: (-0.0371548, 0.0025968, -0.0371548, 0.0025968, -0.0397517, 0.0397517)
8: (-0.0186400, 0.0306223, -0.0186400, 0.0306223, -0.0492623, 0.0492623)
9: (-0.0211006, 0.0146403, -0.0211006, 0.0146403, -0.0357409, 0.0357409)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.97 + 1.94 = 3.91 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0328590, upper bound: 0.0328590
