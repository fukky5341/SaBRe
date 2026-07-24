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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0117690, 0.0029149, -0.0117690, 0.0029149, -0.0146839, 0.0146839)
1: (-0.0081479, 0.0067054, -0.0081479, 0.0067054, -0.0148534, 0.0148534)
2: (0.9426727, 0.9803103, 0.9426727, 0.9803103, -0.0376376, 0.0376376)
3: (0.0060435, 0.0534235, 0.0060435, 0.0534235, -0.0473800, 0.0473800)
4: (-0.0230246, 0.0294046, -0.0230246, 0.0294046, -0.0524292, 0.0524292)
5: (0.0061799, 0.0315062, 0.0061799, 0.0315062, -0.0253263, 0.0253263)
6: (-0.0151180, 0.0158905, -0.0151180, 0.0158905, -0.0310085, 0.0310085)
7: (-0.0348590, -0.0003190, -0.0348590, -0.0003190, -0.0345399, 0.0345399)
8: (-0.0146095, 0.0274415, -0.0146095, 0.0274415, -0.0420510, 0.0420510)
9: (-0.0184411, 0.0144797, -0.0184411, 0.0144797, -0.0329208, 0.0329208)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.69 + 1.82 = 3.51 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0283569, upper bound: 0.0283569
