## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.03610424


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0039820, 0.0478403, 0.0039820, 0.0478403, -0.0270925, 0.0270925)
1: (0.0033230, 0.0043076, 0.0033230, 0.0043076, -0.0008630, 0.0008630)
2: (0.0192061, 0.0330986, 0.0192061, 0.0330986, -0.0086659, 0.0086659)
3: (0.0321295, 0.0598560, 0.0321295, 0.0598560, -0.0170158, 0.0170158)
4: (-0.0120757, -0.0050303, -0.0120757, -0.0050303, -0.0050179, 0.0050179)
5: (0.0286036, 0.0440200, 0.0286036, 0.0440200, -0.0093966, 0.0093966)
6: (-0.0034756, 0.0386132, -0.0034756, 0.0386132, -0.0258311, 0.0258311)
7: (-0.0066416, -0.0062888, -0.0066416, -0.0062888, -0.0003528, 0.0003528)
8: (0.7334405, 0.8587427, 0.7334405, 0.8587427, -0.0764898, 0.0764898)
9: (0.0767484, 0.0889098, 0.0767484, 0.0889098, -0.0075307, 0.0075307)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 1.16 = 2.52 seconds
status: Status.ADV_EXAMPLE
