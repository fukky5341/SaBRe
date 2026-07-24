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
Threshold: 4.776e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0034727, -0.0028317, -0.0034727, -0.0028317, -0.0002348, 0.0002348)
1: (-0.0045226, -0.0043974, -0.0045226, -0.0043974, -0.0000429, 0.0000429)
2: (0.0101485, 0.0109680, 0.0101485, 0.0109680, -0.0002965, 0.0002965)
3: (1.0087203, 1.0089228, 1.0087203, 1.0089228, -0.0000799, 0.0000799)
4: (-0.0034078, -0.0032801, -0.0034078, -0.0032801, -0.0000455, 0.0000455)
5: (0.0012929, 0.0017835, 0.0012929, 0.0017835, -0.0001794, 0.0001794)
6: (-0.0025222, -0.0024990, -0.0025222, -0.0024990, -0.0000105, 0.0000105)
7: (-0.0087853, -0.0076862, -0.0087853, -0.0076862, -0.0004267, 0.0004267)
8: (-0.0044588, -0.0031157, -0.0044588, -0.0031157, -0.0004739, 0.0004739)
9: (-0.0026502, -0.0020092, -0.0026502, -0.0020092, -0.0002237, 0.0002237)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 1.28 = 2.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0000604, upper bound: 0.0000605

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000585
time: 0.45 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000585, upper bound: 0.0000585
time: 0.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.05 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.05
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000585
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.05
Output dim: 3, lower bound: -0.0000585, upper bound: 0.0000585

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0034726, -0.0028533, -0.0034727, -0.0028388, -0.0002269, 0.0002152
1: -0.0045225, -0.0044025, -0.0045225, -0.0043991, -0.0000407, 0.0000377
2: 0.0101486, 0.0109399, 0.0101485, 0.0109589, -0.0002856, 0.0002703
3: 1.0087274, 1.0089226, 1.0087227, 1.0089228, -0.0000721, 0.0000769
4: -0.0034033, -0.0032801, -0.0034063, -0.0032801, -0.0000413, 0.0000437
5: 0.0012930, 0.0017669, 0.0012930, 0.0017780, -0.0001733, 0.0001642
6: -0.0025222, -0.0024991, -0.0025222, -0.0024990, -0.0000104, 0.0000104
7: -0.0087539, -0.0076864, -0.0087750, -0.0076863, -0.0004005, 0.0004167
8: -0.0044096, -0.0031160, -0.0044427, -0.0031158, -0.0004275, 0.0004542
9: -0.0026501, -0.0020335, -0.0026502, -0.0020172, -0.0002139, 0.0002007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000508, upper bound: 0.0000563
time: 0.44 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.ADV_EXAMPLE
time: 0.45 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 2.78 + 3.47 = 6.25 seconds
