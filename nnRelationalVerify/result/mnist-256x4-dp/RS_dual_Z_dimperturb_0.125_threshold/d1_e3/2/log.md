## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00035784


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0043151, -0.0042566, -0.0043151, -0.0042566, -0.0000270, 0.0000270)
1: (0.0033169, 0.0036408, 0.0033169, 0.0036408, -0.0001494, 0.0001494)
2: (0.0068322, 0.0075557, 0.0068322, 0.0075557, -0.0003338, 0.0003338)
3: (0.0041503, 0.0044553, 0.0041503, 0.0044553, -0.0001407, 0.0001407)
4: (1.0128521, 1.0140350, 1.0128521, 1.0140350, -0.0005458, 0.0005458)
5: (0.0047416, 0.0049718, 0.0047416, 0.0049718, -0.0001062, 0.0001062)
6: (-0.0122130, -0.0119135, -0.0122130, -0.0119135, -0.0001382, 0.0001382)
7: (-0.0103612, -0.0103230, -0.0103612, -0.0103230, -0.0000176, 0.0000176)
8: (-0.0026228, -0.0024159, -0.0026228, -0.0024159, -0.0000955, 0.0000955)
9: (-0.0060763, -0.0050404, -0.0060763, -0.0050404, -0.0004779, 0.0004779)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 1.22 = 2.60 seconds
status: Status.VERIFIED
relational distance
Output dim: 4, lower bound: -0.0003389, upper bound: 0.0003389
