## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000228159


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0000701, 0.0005921, 0.0000701, 0.0005921, -0.0005221, 0.0005221)
1: (-0.0030723, -0.0028080, -0.0030723, -0.0028080, -0.0002643, 0.0002643)
2: (0.0329255, 0.0332879, 0.0329255, 0.0332879, -0.0003624, 0.0003624)
3: (-0.0025945, -0.0021483, -0.0025945, -0.0021483, -0.0004462, 0.0004462)
4: (-0.0018891, -0.0015389, -0.0018891, -0.0015389, -0.0002628, 0.0002628)
5: (0.0125881, 0.0130756, 0.0125881, 0.0130756, -0.0004875, 0.0004875)
6: (-0.0031152, -0.0027024, -0.0031152, -0.0027024, -0.0002865, 0.0002865)
7: (0.9760268, 0.9762814, 0.9760268, 0.9762814, -0.0002546, 0.0002546)
8: (-0.0122673, -0.0109055, -0.0122673, -0.0109055, -0.0013618, 0.0013618)
9: (0.0023096, 0.0031037, 0.0023096, 0.0031037, -0.0007940, 0.0007940)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 1.20 = 2.45 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0001527, upper bound: 0.0001527
