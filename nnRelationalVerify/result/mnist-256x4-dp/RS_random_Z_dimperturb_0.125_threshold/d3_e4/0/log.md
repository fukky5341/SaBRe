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
Threshold: 0.00020622


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040924, -0.0040683, -0.0040924, -0.0040683, -0.0000142, 0.0000142)
1: (-0.0059149, -0.0050128, -0.0059149, -0.0050128, -0.0005323, 0.0005323)
2: (0.9693654, 0.9704480, 0.9693654, 0.9704480, -0.0006388, 0.0006388)
3: (0.0203499, 0.0283349, 0.0203499, 0.0283349, -0.0047113, 0.0047113)
4: (-0.0028481, -0.0022408, -0.0028481, -0.0022408, -0.0003583, 0.0003583)
5: (0.0143919, 0.0150057, 0.0143919, 0.0150057, -0.0003622, 0.0003622)
6: (0.0046080, 0.0049066, 0.0046080, 0.0049066, -0.0001761, 0.0001761)
7: (-0.0151215, -0.0130521, -0.0151215, -0.0130521, -0.0012210, 0.0012210)
8: (0.0047325, 0.0063742, 0.0047325, 0.0063742, -0.0009687, 0.0009687)
9: (0.0062364, 0.0091893, 0.0062364, 0.0091893, -0.0017422, 0.0017422)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.85 + 1.29 = 3.13 seconds
status: Status.ADV_EXAMPLE
