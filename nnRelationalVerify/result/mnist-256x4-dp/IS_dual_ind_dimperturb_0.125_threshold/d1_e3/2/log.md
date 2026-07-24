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
0: (-0.0043204, -0.0042536, -0.0043204, -0.0042536, -0.0000292, 0.0000292)
1: (0.0032999, 0.0036697, 0.0032999, 0.0036697, -0.0001617, 0.0001617)
2: (0.0067675, 0.0075939, 0.0067675, 0.0075939, -0.0003613, 0.0003613)
3: (0.0041343, 0.0044825, 0.0041343, 0.0044825, -0.0001523, 0.0001523)
4: (1.0127896, 1.0141406, 1.0127896, 1.0141406, -0.0005907, 0.0005907)
5: (0.0047295, 0.0049923, 0.0047295, 0.0049923, -0.0001149, 0.0001149)
6: (-0.0122397, -0.0118977, -0.0122397, -0.0118977, -0.0001495, 0.0001495)
7: (-0.0103647, -0.0103210, -0.0103647, -0.0103210, -0.0000191, 0.0000191)
8: (-0.0026338, -0.0023974, -0.0026338, -0.0023974, -0.0001033, 0.0001033)
9: (-0.0061688, -0.0049857, -0.0061688, -0.0049857, -0.0005172, 0.0005172)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 1.22 = 2.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0003894, upper bound: 0.0003894

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 85
type: A, layer: 3, pos: 131
type: A, layer: 3, pos: 111

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 85

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.ADV_EXAMPLE
time: 0.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0003645, upper bound: 0.0003645
time: 0.39 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 2.65 + 1.00 = 3.65 seconds
