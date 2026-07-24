## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00011788


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041714, -0.0041579, -0.0041714, -0.0041579, -0.0000091, 0.0000091)
1: (-0.0088733, -0.0083676, -0.0088733, -0.0083676, -0.0003397, 0.0003397)
2: (0.9658151, 0.9664220, 0.9658151, 0.9664220, -0.0004077, 0.0004077)
3: (-0.0058363, -0.0013595, -0.0058363, -0.0013595, -0.0030070, 0.0030070)
4: (-0.0005896, -0.0002491, -0.0005896, -0.0002491, -0.0002287, 0.0002287)
5: (0.0166744, 0.0170185, 0.0166744, 0.0170185, -0.0002311, 0.0002311)
6: (0.0036290, 0.0037964, 0.0036290, 0.0037964, -0.0001124, 0.0001124)
7: (-0.0074259, -0.0062657, -0.0074259, -0.0062657, -0.0007793, 0.0007793)
8: (0.0108378, 0.0117582, 0.0108378, 0.0117582, -0.0006182, 0.0006182)
9: (0.0172174, 0.0188729, 0.0172174, 0.0188729, -0.0011120, 0.0011120)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 1.23 = 2.48 seconds
status: Status.ADV_EXAMPLE
