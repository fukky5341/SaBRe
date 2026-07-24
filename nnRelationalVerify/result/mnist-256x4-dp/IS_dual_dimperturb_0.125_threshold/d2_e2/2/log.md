## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0004916


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0001825, 0.0007238, 0.0001825, 0.0007238, -0.0003998, 0.0003998)
1: (0.9943464, 0.9954928, 0.9943464, 0.9954928, -0.0008467, 0.0008467)
2: (-0.0079146, -0.0076438, -0.0079146, -0.0076438, -0.0002000, 0.0002000)
3: (0.0029164, 0.0035936, 0.0029164, 0.0035936, -0.0005002, 0.0005002)
4: (0.0027992, 0.0036820, 0.0027992, 0.0036820, -0.0006520, 0.0006520)
5: (0.0037947, 0.0050776, 0.0037947, 0.0050776, -0.0009475, 0.0009475)
6: (-0.0008682, 0.0003181, -0.0008682, 0.0003181, -0.0008761, 0.0008761)
7: (-0.0074740, -0.0069253, -0.0074740, -0.0069253, -0.0004053, 0.0004053)
8: (0.0080940, 0.0081663, 0.0080940, 0.0081663, -0.0000534, 0.0000534)
9: (-0.0031081, -0.0023246, -0.0031081, -0.0023246, -0.0005787, 0.0005787)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.26 = 2.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0006086, upper bound: 0.0006086

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005664, upper bound: 0.0005840
time: 0.42 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005839, upper bound: 0.0005840
time: 0.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.02 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.02
Output dim: 1, lower bound: -0.0005664, upper bound: 0.0005840
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.02
Output dim: 1, lower bound: -0.0005839, upper bound: 0.0005840

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0001825, 0.0007070, 0.0001825, 0.0007238, -0.0003966, 0.0003801
1: 0.9943820, 0.9954926, 0.9943464, 0.9954928, -0.0008050, 0.0008399
2: -0.0079146, -0.0076523, -0.0079146, -0.0076438, -0.0001984, 0.0001901
3: 0.0029165, 0.0035726, 0.0029164, 0.0035936, -0.0004962, 0.0004756
4: 0.0027993, 0.0036545, 0.0027992, 0.0036820, -0.0006467, 0.0006199
5: 0.0037949, 0.0050376, 0.0037947, 0.0050776, -0.0009398, 0.0009008
6: -0.0008312, 0.0003179, -0.0008682, 0.0003181, -0.0008330, 0.0008690
7: -0.0074569, -0.0069254, -0.0074740, -0.0069253, -0.0003853, 0.0004020
8: 0.0080963, 0.0081662, 0.0080940, 0.0081663, -0.0000507, 0.0000529
9: -0.0030838, -0.0023247, -0.0031081, -0.0023246, -0.0005502, 0.0005741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005664, upper bound: 0.0005664
time: 0.43 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005664, upper bound: 0.0005840
time: 0.43 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0001625, 0.0007051, 0.0001825, 0.0007216, -0.0004181, 0.0003943
1: 0.9943861, 0.9955351, 0.9943512, 0.9954927, -0.0008349, 0.0008854
2: -0.0079246, -0.0076532, -0.0079146, -0.0076450, -0.0002091, 0.0001972
3: 0.0028914, 0.0035702, 0.0029164, 0.0035908, -0.0005231, 0.0004933
4: 0.0027666, 0.0036514, 0.0027993, 0.0036783, -0.0006818, 0.0006429
5: 0.0037474, 0.0050331, 0.0037948, 0.0050722, -0.0009908, 0.0009343
6: -0.0008271, 0.0003619, -0.0008632, 0.0003180, -0.0008639, 0.0009162
7: -0.0074550, -0.0069050, -0.0074717, -0.0069253, -0.0003996, 0.0004238
8: 0.0080965, 0.0081689, 0.0080943, 0.0081663, -0.0000526, 0.0000558
9: -0.0030810, -0.0022956, -0.0031049, -0.0023246, -0.0005707, 0.0006052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005840, upper bound: 0.0005664
time: 0.46 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005840, upper bound: 0.0005840
time: 0.46 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.29 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 1, lower bound: -0.0005664, upper bound: 0.0005664
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 1, lower bound: -0.0005664, upper bound: 0.0005840
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 1, lower bound: -0.0005840, upper bound: 0.0005664
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 1, lower bound: -0.0005840, upper bound: 0.0005840

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0001825, 0.0007070, 0.0001825, 0.0007070, -0.0003769, 0.0003769
1: 0.9943820, 0.9954926, 0.9943820, 0.9954926, -0.0007982, 0.0007982
2: -0.0079146, -0.0076523, -0.0079146, -0.0076523, -0.0001885, 0.0001885
3: 0.0029165, 0.0035726, 0.0029165, 0.0035726, -0.0004715, 0.0004715
4: 0.0027993, 0.0036545, 0.0027993, 0.0036545, -0.0006146, 0.0006146
5: 0.0037949, 0.0050376, 0.0037949, 0.0050376, -0.0008932, 0.0008932
6: -0.0008312, 0.0003179, -0.0008312, 0.0003179, -0.0008259, 0.0008259
7: -0.0074569, -0.0069254, -0.0074569, -0.0069254, -0.0003821, 0.0003821
8: 0.0080963, 0.0081662, 0.0080963, 0.0081662, -0.0000503, 0.0000503
9: -0.0030838, -0.0023247, -0.0030838, -0.0023247, -0.0005456, 0.0005456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005540, upper bound: 0.0005279
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005537, upper bound: 0.0005562
time: 0.43 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0001825, 0.0007070, 0.0001625, 0.0007051, -0.0003799, 0.0004006
1: 0.9943820, 0.9954926, 0.9943861, 0.9955351, -0.0008484, 0.0008046
2: -0.0079146, -0.0076523, -0.0079246, -0.0076532, -0.0001900, 0.0002004
3: 0.0029165, 0.0035726, 0.0028914, 0.0035702, -0.0004753, 0.0005012
4: 0.0027993, 0.0036545, 0.0027666, 0.0036514, -0.0006195, 0.0006533
5: 0.0037949, 0.0050376, 0.0037474, 0.0050331, -0.0009003, 0.0009494
6: -0.0008312, 0.0003179, -0.0008271, 0.0003619, -0.0008779, 0.0008325
7: -0.0074569, -0.0069254, -0.0074550, -0.0069050, -0.0004061, 0.0003851
8: 0.0080963, 0.0081662, 0.0080965, 0.0081689, -0.0000535, 0.0000507
9: -0.0030838, -0.0023247, -0.0030810, -0.0022956, -0.0005799, 0.0005499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005540, upper bound: 0.0005497
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.ADV_EXAMPLE
time: 0.43 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 2.58 + 9.89 = 12.47 seconds
