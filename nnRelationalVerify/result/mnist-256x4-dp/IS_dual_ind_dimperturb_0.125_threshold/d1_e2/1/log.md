## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 5.355e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0037534, -0.0032044, -0.0037534, -0.0032044, -0.0005489, 0.0005489)
1: (0.0059952, 0.0063946, 0.0059952, 0.0063946, -0.0003994, 0.0003994)
2: (0.0110625, 0.0120882, 0.0110625, 0.0120882, -0.0008484, 0.0008484)
3: (-0.0035024, -0.0030271, -0.0035024, -0.0030271, -0.0004656, 0.0004656)
4: (0.0049632, 0.0051212, 0.0049632, 0.0051212, -0.0000919, 0.0000919)
5: (-0.0014222, -0.0010917, -0.0014222, -0.0010917, -0.0003305, 0.0003305)
6: (-0.0055791, -0.0054107, -0.0055791, -0.0054107, -0.0001684, 0.0001684)
7: (-0.0030461, -0.0027083, -0.0030461, -0.0027083, -0.0003378, 0.0003378)
8: (-0.0024707, -0.0017940, -0.0024707, -0.0017940, -0.0006767, 0.0006767)
9: (1.0004603, 1.0005240, 1.0004603, 1.0005240, -0.0000638, 0.0000638)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 1.22 = 2.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0000585, upper bound: 0.0000585

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000583, upper bound: 0.0000583
time: 0.49 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000583, upper bound: 0.0000583
time: 0.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 9, lower bound: -0.0000583, upper bound: 0.0000583
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 9, lower bound: -0.0000583, upper bound: 0.0000583

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037534, -0.0032386, -0.0037534, -0.0032047, -0.0005486, 0.0005148
1: 0.0060213, 0.0063946, 0.0059955, 0.0063946, -0.0003733, 0.0003991
2: 0.0110625, 0.0120270, 0.0110625, 0.0120876, -0.0008477, 0.0007847
3: -0.0034716, -0.0030271, -0.0035022, -0.0030271, -0.0004330, 0.0004652
4: 0.0049734, 0.0051212, 0.0049633, 0.0051212, -0.0000721, 0.0000917
5: -0.0014050, -0.0010917, -0.0014220, -0.0010917, -0.0003133, 0.0003303
6: -0.0055682, -0.0054107, -0.0055790, -0.0054107, -0.0001575, 0.0001683
7: -0.0030461, -0.0027317, -0.0030461, -0.0027085, -0.0003375, 0.0003144
8: -0.0024310, -0.0017940, -0.0024703, -0.0017940, -0.0006371, 0.0006763
9: 1.0004603, 1.0005231, 1.0004603, 1.0005242, -0.0000639, 0.0000628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000583, upper bound: 0.0000583
time: 0.46 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000583, upper bound: 0.0000583
time: 0.48 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0037663, -0.0032464, -0.0037534, -0.0032101, -0.0005563, 0.0005069
1: 0.0060306, 0.0064027, 0.0059998, 0.0063946, -0.0003640, 0.0004029
2: 0.0110392, 0.0120132, 0.0110625, 0.0120780, -0.0008750, 0.0007732
3: -0.0034615, -0.0030165, -0.0034971, -0.0030271, -0.0004261, 0.0004751
4: 0.0049766, 0.0051248, 0.0049650, 0.0051212, -0.0000709, 0.0001196
5: -0.0014012, -0.0010850, -0.0014193, -0.0010917, -0.0003095, 0.0003342
6: -0.0055655, -0.0054065, -0.0055773, -0.0054107, -0.0001548, 0.0001708
7: -0.0030511, -0.0027406, -0.0030461, -0.0027126, -0.0003385, 0.0003055
8: -0.0024221, -0.0017788, -0.0024640, -0.0017940, -0.0006281, 0.0006852
9: 1.0004594, 1.0005373, 1.0004603, 1.0005239, -0.0000645, 0.0000770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000583, upper bound: 0.0000583
time: 0.45 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000583, upper bound: 0.0000583
time: 0.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.23 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 9, lower bound: -0.0000583, upper bound: 0.0000583
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 9, lower bound: -0.0000583, upper bound: 0.0000583
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 9, lower bound: -0.0000583, upper bound: 0.0000583
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 9, lower bound: -0.0000583, upper bound: 0.0000583

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037534, -0.0032386, -0.0037534, -0.0032386, -0.0005148, 0.0005148
1: 0.0060213, 0.0063946, 0.0060213, 0.0063946, -0.0003733, 0.0003733
2: 0.0110625, 0.0120270, 0.0110625, 0.0120270, -0.0007847, 0.0007847
3: -0.0034716, -0.0030271, -0.0034716, -0.0030271, -0.0004330, 0.0004330
4: 0.0049734, 0.0051212, 0.0049734, 0.0051212, -0.0000721, 0.0000721
5: -0.0014050, -0.0010917, -0.0014050, -0.0010917, -0.0003133, 0.0003133
6: -0.0055682, -0.0054107, -0.0055682, -0.0054107, -0.0001575, 0.0001575
7: -0.0030461, -0.0027317, -0.0030461, -0.0027317, -0.0003144, 0.0003144
8: -0.0024310, -0.0017940, -0.0024310, -0.0017940, -0.0006371, 0.0006371
9: 1.0004603, 1.0005231, 1.0004603, 1.0005231, -0.0000628, 0.0000628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000510, upper bound: 0.0000444
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000445, upper bound: 0.0000426
time: 0.47 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037534, -0.0032386, -0.0037663, -0.0032464, -0.0005069, 0.0005277
1: 0.0060213, 0.0063946, 0.0060306, 0.0064027, -0.0003815, 0.0003640
2: 0.0110625, 0.0120270, 0.0110392, 0.0120132, -0.0007835, 0.0008215
3: -0.0034716, -0.0030271, -0.0034615, -0.0030165, -0.0004479, 0.0004278
4: 0.0049734, 0.0051212, 0.0049766, 0.0051248, -0.0000998, 0.0000915
5: -0.0014050, -0.0010917, -0.0014012, -0.0010850, -0.0003200, 0.0003095
6: -0.0055682, -0.0054107, -0.0055655, -0.0054065, -0.0001617, 0.0001548
7: -0.0030461, -0.0027317, -0.0030511, -0.0027406, -0.0003055, 0.0003194
8: -0.0024310, -0.0017940, -0.0024221, -0.0017788, -0.0006523, 0.0006281
9: 1.0004603, 1.0005231, 1.0004594, 1.0005373, -0.0000770, 0.0000637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000510, upper bound: 0.0000444
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000445, upper bound: 0.0000426
time: 0.40 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0037663, -0.0032464, -0.0037534, -0.0032386, -0.0005277, 0.0005069
1: 0.0060306, 0.0064027, 0.0060213, 0.0063946, -0.0003640, 0.0003815
2: 0.0110392, 0.0120132, 0.0110625, 0.0120270, -0.0008215, 0.0007835
3: -0.0034615, -0.0030165, -0.0034716, -0.0030271, -0.0004278, 0.0004479
4: 0.0049766, 0.0051248, 0.0049734, 0.0051212, -0.0000915, 0.0000998
5: -0.0014012, -0.0010850, -0.0014050, -0.0010917, -0.0003095, 0.0003200
6: -0.0055655, -0.0054065, -0.0055682, -0.0054107, -0.0001548, 0.0001617
7: -0.0030511, -0.0027406, -0.0030461, -0.0027317, -0.0003194, 0.0003055
8: -0.0024221, -0.0017788, -0.0024310, -0.0017940, -0.0006281, 0.0006523
9: 1.0004594, 1.0005373, 1.0004603, 1.0005231, -0.0000637, 0.0000770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000510, upper bound: 0.0000444
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000425, upper bound: 0.0000425
time: 0.42 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0037663, -0.0032464, -0.0037663, -0.0032464, -0.0005199, 0.0005199
1: 0.0060306, 0.0064027, 0.0060306, 0.0064027, -0.0003721, 0.0003721
2: 0.0110392, 0.0120132, 0.0110392, 0.0120132, -0.0007895, 0.0007895
3: -0.0034615, -0.0030165, -0.0034615, -0.0030165, -0.0004370, 0.0004370
4: 0.0049766, 0.0051248, 0.0049766, 0.0051248, -0.0000709, 0.0000709
5: -0.0014012, -0.0010850, -0.0014012, -0.0010850, -0.0003161, 0.0003161
6: -0.0055655, -0.0054065, -0.0055655, -0.0054065, -0.0001589, 0.0001589
7: -0.0030511, -0.0027406, -0.0030511, -0.0027406, -0.0003105, 0.0003105
8: -0.0024221, -0.0017788, -0.0024221, -0.0017788, -0.0006433, 0.0006433
9: 1.0004594, 1.0005373, 1.0004594, 1.0005373, -0.0000778, 0.0000778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000510, upper bound: 0.0000444
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000425, upper bound: 0.0000425
time: 0.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.15 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 9, lower bound: -0.0000510, upper bound: 0.0000444
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 9, lower bound: -0.0000445, upper bound: 0.0000426
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 9, lower bound: -0.0000510, upper bound: 0.0000444
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 9, lower bound: -0.0000445, upper bound: 0.0000426
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 9, lower bound: -0.0000510, upper bound: 0.0000444
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 9, lower bound: -0.0000425, upper bound: 0.0000425
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 9, lower bound: -0.0000510, upper bound: 0.0000444
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 9, lower bound: -0.0000425, upper bound: 0.0000425

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.48 + 14.04 = 16.52 seconds
