## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00017731


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0017202, -0.0013912, -0.0017202, -0.0013912, -0.0001675, 0.0001675)
1: (-0.0086759, -0.0078409, -0.0086759, -0.0078409, -0.0004251, 0.0004251)
2: (0.0296475, 0.0301655, 0.0296475, 0.0301655, -0.0002637, 0.0002637)
3: (0.0032359, 0.0042032, 0.0032359, 0.0042032, -0.0004924, 0.0004924)
4: (-0.0077179, -0.0068685, -0.0077179, -0.0068685, -0.0004323, 0.0004323)
5: (0.0108149, 0.0111366, 0.0108149, 0.0111366, -0.0001638, 0.0001638)
6: (0.0044949, 0.0057225, 0.0044949, 0.0057225, -0.0006249, 0.0006249)
7: (0.9812046, 0.9820637, 0.9812046, 0.9820637, -0.0004373, 0.0004373)
8: (-0.0067159, -0.0057949, -0.0067159, -0.0057949, -0.0004688, 0.0004688)
9: (-0.0011718, -0.0005634, -0.0011718, -0.0005634, -0.0003097, 0.0003097)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 1.36 = 2.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0002559, upper bound: 0.0002559

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002400, upper bound: 0.0002302
time: 0.54 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002438, upper bound: 0.0002437
time: 0.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.26 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 7, lower bound: -0.0002400, upper bound: 0.0002302
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 7, lower bound: -0.0002438, upper bound: 0.0002437

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0016929, -0.0013833, -0.0017101, -0.0013914, -0.0001332, 0.0001521
1: -0.0086065, -0.0078210, -0.0086501, -0.0078415, -0.0003380, 0.0003859
2: 0.0296905, 0.0301779, 0.0296634, 0.0301651, -0.0002097, 0.0002394
3: 0.0032128, 0.0041228, 0.0032366, 0.0041733, -0.0004470, 0.0003915
4: -0.0076473, -0.0068483, -0.0076917, -0.0068691, -0.0003438, 0.0003925
5: 0.0108416, 0.0111442, 0.0108248, 0.0111363, -0.0001302, 0.0001487
6: 0.0044656, 0.0056205, 0.0044957, 0.0056846, -0.0005673, 0.0004969
7: 0.9811841, 0.9819922, 0.9812051, 0.9820371, -0.0003970, 0.0003477
8: -0.0067379, -0.0058714, -0.0067152, -0.0058233, -0.0004256, 0.0003728
9: -0.0011212, -0.0005489, -0.0011530, -0.0005638, -0.0002462, 0.0002811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002274, upper bound: 0.0002138
time: 0.52 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.ADV_EXAMPLE
time: 0.44 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 2.86 + 3.69 = 6.55 seconds
