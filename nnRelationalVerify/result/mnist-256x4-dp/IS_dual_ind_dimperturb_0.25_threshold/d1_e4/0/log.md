## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 8.477e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0037564, -0.0023438, -0.0037564, -0.0023438, -0.0007302, 0.0007302)
1: (-0.0045763, -0.0043047, -0.0045763, -0.0043047, -0.0000964, 0.0000964)
2: (0.0097870, 0.0115929, 0.0097870, 0.0115929, -0.0008918, 0.0008918)
3: (1.0085841, 1.0090213, 1.0085841, 1.0090213, -0.0002503, 0.0002503)
4: (-0.0035050, -0.0032240, -0.0035050, -0.0032240, -0.0001312, 0.0001312)
5: (0.0010761, 0.0021570, 0.0010761, 0.0021570, -0.0005552, 0.0005552)
6: (-0.0025335, -0.0024837, -0.0025335, -0.0024837, -0.0000456, 0.0000456)
7: (-0.0096095, -0.0071904, -0.0096095, -0.0071904, -0.0015012, 0.0015012)
8: (-0.0054757, -0.0025282, -0.0054757, -0.0025282, -0.0013249, 0.0013249)
9: (-0.0029288, -0.0015261, -0.0029288, -0.0015261, -0.0006004, 0.0006004)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 1.36 = 2.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0001268, upper bound: 0.0001269

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.ADV_EXAMPLE
time: 0.60 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.ADV_EXAMPLE
time: 0.45 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 2.72 + 1.19 = 3.90 seconds
