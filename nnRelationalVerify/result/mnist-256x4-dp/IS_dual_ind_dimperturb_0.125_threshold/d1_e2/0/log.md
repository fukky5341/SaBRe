## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 4.3265e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0010676, 0.0010988, 0.0010676, 0.0010988, -0.0000289, 0.0000289)
1: (0.9936378, 0.9937697, 0.9936378, 0.9937697, -0.0001040, 0.0001040)
2: (-0.0063696, -0.0055556, -0.0063696, -0.0055556, -0.0006107, 0.0006107)
3: (0.0039279, 0.0040014, 0.0039279, 0.0040014, -0.0000545, 0.0000545)
4: (0.0028078, 0.0034512, 0.0028078, 0.0034512, -0.0004948, 0.0004948)
5: (0.0062196, 0.0063955, 0.0062196, 0.0063955, -0.0001759, 0.0001759)
6: (-0.0013320, -0.0010495, -0.0013320, -0.0010495, -0.0002113, 0.0002113)
7: (-0.0082210, -0.0080671, -0.0082210, -0.0080671, -0.0001539, 0.0001539)
8: (0.0055509, 0.0066205, 0.0055509, 0.0066205, -0.0007660, 0.0007660)
9: (-0.0036822, -0.0035981, -0.0036822, -0.0035981, -0.0000841, 0.0000841)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.31 + 1.27 = 2.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0000937, upper bound: 0.0000937

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000925, upper bound: 0.0000916
time: 0.45 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000926, upper bound: 0.0000926
time: 0.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.03 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.03
Output dim: 1, lower bound: -0.0000925, upper bound: 0.0000916
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.03
Output dim: 1, lower bound: -0.0000926, upper bound: 0.0000926

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0010668, 0.0010987, 0.0010677, 0.0010988, -0.0000297, 0.0000288
1: 0.9936388, 0.9937717, 0.9936380, 0.9937695, -0.0001026, 0.0001059
2: -0.0063593, -0.0055522, -0.0063678, -0.0055559, -0.0005991, 0.0006083
3: 0.0039271, 0.0040007, 0.0039280, 0.0040013, -0.0000551, 0.0000536
4: 0.0028052, 0.0034430, 0.0028081, 0.0034498, -0.0004928, 0.0004858
5: 0.0062218, 0.0063962, 0.0062200, 0.0063954, -0.0001736, 0.0001763
6: -0.0013284, -0.0010483, -0.0013314, -0.0010496, -0.0002072, 0.0002105
7: -0.0082217, -0.0080691, -0.0082210, -0.0080674, -0.0001542, 0.0001519
8: 0.0055465, 0.0066069, 0.0055514, 0.0066182, -0.0007630, 0.0007495
9: -0.0036822, -0.0035956, -0.0036822, -0.0035983, -0.0000839, 0.0000866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000916, upper bound: 0.0000916
time: 0.47 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000916, upper bound: 0.0000916
time: 0.45 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0010679, 0.0010988, 0.0010676, 0.0010988, -0.0000286, 0.0000289
1: 0.9936387, 0.9937689, 0.9936378, 0.9937697, -0.0001030, 0.0001032
2: -0.0063609, -0.0055570, -0.0063696, -0.0055556, -0.0005967, 0.0006100
3: 0.0039282, 0.0040008, 0.0039279, 0.0040014, -0.0000542, 0.0000538
4: 0.0028089, 0.0034443, 0.0028078, 0.0034512, -0.0004942, 0.0004835
5: 0.0062215, 0.0063952, 0.0062196, 0.0063955, -0.0001740, 0.0001756
6: -0.0013290, -0.0010500, -0.0013320, -0.0010495, -0.0002064, 0.0002110
7: -0.0082208, -0.0080688, -0.0082210, -0.0080671, -0.0001537, 0.0001523
8: 0.0055528, 0.0066090, 0.0055509, 0.0066205, -0.0007651, 0.0007479
9: -0.0036822, -0.0035991, -0.0036822, -0.0035981, -0.0000841, 0.0000831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000916, upper bound: 0.0000925
time: 0.49 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000916, upper bound: 0.0000926
time: 0.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.28 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 1, lower bound: -0.0000916, upper bound: 0.0000916
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 1, lower bound: -0.0000916, upper bound: 0.0000916
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 1, lower bound: -0.0000916, upper bound: 0.0000925
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 1, lower bound: -0.0000916, upper bound: 0.0000926

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010668, 0.0010987, 0.0010668, 0.0010987, -0.0000297, 0.0000297
1: 0.9936388, 0.9937717, 0.9936388, 0.9937717, -0.0001049, 0.0001049
2: -0.0063593, -0.0055522, -0.0063593, -0.0055522, -0.0005989, 0.0005989
3: 0.0039271, 0.0040007, 0.0039271, 0.0040007, -0.0000545, 0.0000545
4: 0.0028052, 0.0034430, 0.0028052, 0.0034430, -0.0004857, 0.0004857
5: 0.0062218, 0.0063962, 0.0062218, 0.0063962, -0.0001744, 0.0001744
6: -0.0013284, -0.0010483, -0.0013284, -0.0010483, -0.0002071, 0.0002071
7: -0.0082217, -0.0080691, -0.0082217, -0.0080691, -0.0001526, 0.0001526
8: 0.0055465, 0.0066069, 0.0055465, 0.0066069, -0.0007493, 0.0007493
9: -0.0036822, -0.0035956, -0.0036822, -0.0035956, -0.0000866, 0.0000866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000836, upper bound: 0.0000791
time: 0.47 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000837, upper bound: 0.0000836
time: 0.47 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010668, 0.0010987, 0.0010679, 0.0010988, -0.0000297, 0.0000285
1: 0.9936388, 0.9937717, 0.9936387, 0.9937689, -0.0001020, 0.0001053
2: -0.0063593, -0.0055522, -0.0063609, -0.0055570, -0.0005985, 0.0006027
3: 0.0039271, 0.0040007, 0.0039282, 0.0040008, -0.0000548, 0.0000534
4: 0.0028052, 0.0034430, 0.0028089, 0.0034443, -0.0004887, 0.0004854
5: 0.0062218, 0.0063962, 0.0062215, 0.0063952, -0.0001734, 0.0001748
6: -0.0013284, -0.0010483, -0.0013290, -0.0010500, -0.0002070, 0.0002085
7: -0.0082217, -0.0080691, -0.0082208, -0.0080688, -0.0001529, 0.0001517
8: 0.0055465, 0.0066069, 0.0055528, 0.0066090, -0.0007556, 0.0007488
9: -0.0036822, -0.0035956, -0.0036822, -0.0035991, -0.0000831, 0.0000866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000836, upper bound: 0.0000791
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000837, upper bound: 0.0000836
time: 0.45 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010679, 0.0010988, 0.0010668, 0.0010987, -0.0000285, 0.0000297
1: 0.9936387, 0.9937689, 0.9936388, 0.9937717, -0.0001053, 0.0001020
2: -0.0063609, -0.0055570, -0.0063593, -0.0055522, -0.0006027, 0.0005985
3: 0.0039282, 0.0040008, 0.0039271, 0.0040007, -0.0000534, 0.0000548
4: 0.0028089, 0.0034443, 0.0028052, 0.0034430, -0.0004854, 0.0004887
5: 0.0062215, 0.0063952, 0.0062218, 0.0063962, -0.0001748, 0.0001734
6: -0.0013290, -0.0010500, -0.0013284, -0.0010483, -0.0002085, 0.0002070
7: -0.0082208, -0.0080688, -0.0082217, -0.0080691, -0.0001517, 0.0001529
8: 0.0055528, 0.0066090, 0.0055465, 0.0066069, -0.0007488, 0.0007556
9: -0.0036822, -0.0035991, -0.0036822, -0.0035956, -0.0000866, 0.0000831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000836, upper bound: 0.0000803
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.ADV_EXAMPLE
time: 0.48 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 2.58 + 12.49 = 15.06 seconds
