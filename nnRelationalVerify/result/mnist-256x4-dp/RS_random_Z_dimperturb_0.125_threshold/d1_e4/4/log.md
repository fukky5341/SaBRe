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
Threshold: 0.00167296


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0070717, -0.0040512, -0.0070717, -0.0040512, -0.0010648, 0.0010648)
1: (-0.0047750, -0.0044378, -0.0047750, -0.0044378, -0.0001189, 0.0001189)
2: (0.0350326, 0.0425046, 0.0350326, 0.0425046, -0.0026342, 0.0026342)
3: (0.0021705, 0.0069713, 0.0021705, 0.0069713, -0.0016925, 0.0016925)
4: (-0.0034190, -0.0025285, -0.0034190, -0.0025285, -0.0003139, 0.0003139)
5: (0.0102514, 0.0108729, 0.0102514, 0.0108729, -0.0002191, 0.0002191)
6: (-0.0113712, -0.0042985, -0.0113712, -0.0042985, -0.0024934, 0.0024934)
7: (0.9636220, 0.9723262, 0.9636220, 0.9723262, -0.0030686, 0.0030686)
8: (-0.0047325, -0.0021352, -0.0047325, -0.0021352, -0.0009156, 0.0009156)
9: (-0.0013669, -0.0010766, -0.0013669, -0.0010766, -0.0001024, 0.0001024)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.48 + 1.16 = 2.65 seconds
status: Status.ADV_EXAMPLE
