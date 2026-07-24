## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00021588


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0035729, 0.0040966, 0.0035729, 0.0040966, -0.0002537, 0.0002537)
1: (0.0018385, 0.0019141, 0.0018385, 0.0019141, -0.0000367, 0.0000367)
2: (0.0120949, 0.0123845, 0.0120949, 0.0123845, -0.0001403, 0.0001403)
3: (-0.0021713, -0.0018718, -0.0021713, -0.0018718, -0.0001451, 0.0001451)
4: (-0.0020106, -0.0016864, -0.0020106, -0.0016864, -0.0001571, 0.0001571)
5: (0.0057092, 0.0060160, 0.0057092, 0.0060160, -0.0001486, 0.0001486)
6: (0.0003523, 0.0015696, 0.0003523, 0.0015696, -0.0005897, 0.0005897)
7: (-0.0046943, -0.0030365, -0.0046943, -0.0030365, -0.0008032, 0.0008032)
8: (0.9859071, 0.9870750, 0.9859071, 0.9870750, -0.0005658, 0.0005658)
9: (-0.0041548, -0.0030947, -0.0041548, -0.0030947, -0.0005136, 0.0005136)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.56 + 1.19 = 2.74 seconds
status: Status.ADV_EXAMPLE
