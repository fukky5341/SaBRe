## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.045187955


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0147140, 0.0016869, -0.0147140, 0.0016869, -0.0164009, 0.0164009)
1: (-0.0159254, 0.0003092, -0.0159254, 0.0003092, -0.0162346, 0.0162346)
2: (0.0258554, 0.0719360, 0.0258554, 0.0719360, -0.0460806, 0.0460806)
3: (-0.0041989, 0.0296865, -0.0041989, 0.0296865, -0.0234659, 0.0234659)
4: (-0.0088418, 0.0027675, -0.0088418, 0.0027675, -0.0116093, 0.0116093)
5: (0.0079119, 0.0166966, 0.0079119, 0.0166966, -0.0087847, 0.0087847)
6: (-0.0328940, 0.0070042, -0.0328940, 0.0070042, -0.0387826, 0.0387826)
7: (0.9191754, 0.9830317, 0.9191754, 0.9830317, -0.0638564, 0.0638564)
8: (-0.0145441, 0.0206706, -0.0145441, 0.0206706, -0.0304386, 0.0304386)
9: (-0.0199322, 0.0109980, -0.0199322, 0.0109980, -0.0309302, 0.0309302)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 1.60 = 3.00 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0348608, upper bound: 0.0348608
