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
Threshold: 0.0029930225


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0002245, 0.0017627, -0.0002245, 0.0017627, -0.0019872, 0.0019872)
1: (0.9920171, 0.9970278, 0.9920171, 0.9970278, -0.0050107, 0.0050107)
2: (-0.0086045, -0.0025493, -0.0086045, -0.0025493, -0.0057238, 0.0057238)
3: (0.0025766, 0.0046549, 0.0025766, 0.0046549, -0.0020783, 0.0020783)
4: (0.0013672, 0.0052175, 0.0013672, 0.0052175, -0.0038503, 0.0038503)
5: (0.0031408, 0.0080109, 0.0031408, 0.0080109, -0.0048701, 0.0048701)
6: (-0.0021078, 0.0000522, -0.0021078, 0.0000522, -0.0021600, 0.0021600)
7: (-0.0093804, -0.0058737, -0.0093804, -0.0058737, -0.0035067, 0.0035067)
8: (-0.0014898, 0.0095568, -0.0014898, 0.0095568, -0.0109517, 0.0109517)
9: (-0.0058226, 0.0005370, -0.0058226, 0.0005370, -0.0063596, 0.0063596)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 3.02 = 4.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0032357, upper bound: 0.0032357

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031940, upper bound: 0.0031231
time: 2.49 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031940, upper bound: 0.0031939
time: 2.20 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 4.82 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 4.82
Output dim: 1, lower bound: -0.0031940, upper bound: 0.0031231
IS_B2, status: Status.UNKNOWN, split count: 1, time: 4.82
Output dim: 1, lower bound: -0.0031940, upper bound: 0.0031939

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -0.0002223, 0.0017475, -0.0001687, 0.0016401, -0.0018624, 0.0019162
1: 0.9920551, 0.9970222, 0.9923263, 0.9968869, -0.0048318, 0.0046960
2: -0.0085379, -0.0025545, -0.0080667, -0.0026816, -0.0055178, 0.0051778
3: 0.0025789, 0.0046390, 0.0026350, 0.0045267, -0.0019478, 0.0020041
4: 0.0013698, 0.0051649, 0.0014321, 0.0047924, -0.0034227, 0.0037328
5: 0.0031462, 0.0079738, 0.0032778, 0.0077104, -0.0045642, 0.0046960
6: -0.0020847, 0.0000503, -0.0019211, 0.0000038, -0.0020885, 0.0019714
7: -0.0093537, -0.0058776, -0.0091640, -0.0059723, -0.0033814, 0.0032865
8: -0.0014778, 0.0094694, -0.0011850, 0.0088502, -0.0102316, 0.0105568
9: -0.0057741, 0.0005300, -0.0054302, 0.0003582, -0.0061323, 0.0059602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031231
time: 3.07 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031231
time: 2.88 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -0.0002245, 0.0017627, -0.0002174, 0.0017253, -0.0019499, 0.0019801
1: 0.9920171, 0.9970278, 0.9921111, 0.9970097, -0.0049926, 0.0049167
2: -0.0086045, -0.0025493, -0.0084405, -0.0025662, -0.0057072, 0.0055360
3: 0.0025766, 0.0046549, 0.0025840, 0.0046158, -0.0020393, 0.0020709
4: 0.0013672, 0.0052175, 0.0013755, 0.0050880, -0.0037208, 0.0038420
5: 0.0031408, 0.0080109, 0.0031583, 0.0079194, -0.0047785, 0.0048526
6: -0.0021078, 0.0000522, -0.0020509, 0.0000460, -0.0021538, 0.0021031
7: -0.0093804, -0.0058737, -0.0093145, -0.0058863, -0.0034941, 0.0034408
8: -0.0014898, 0.0095568, -0.0014508, 0.0093415, -0.0107295, 0.0109129
9: -0.0058226, 0.0005370, -0.0057030, 0.0005141, -0.0063368, 0.0062400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031939
time: 2.13 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031940
time: 1.84 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.11 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031231
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031231
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031939
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031940

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001687, 0.0016401, -0.0001687, 0.0016401, -0.0018087, 0.0018087
1: 0.9923263, 0.9968869, 0.9923263, 0.9968869, -0.0045606, 0.0045606
2: -0.0080667, -0.0026816, -0.0080667, -0.0026816, -0.0050445, 0.0050445
3: 0.0026350, 0.0045267, 0.0026350, 0.0045267, -0.0018917, 0.0018917
4: 0.0014321, 0.0047924, 0.0014321, 0.0047924, -0.0033604, 0.0033604
5: 0.0032778, 0.0077104, 0.0032778, 0.0077104, -0.0044327, 0.0044327
6: -0.0019211, 0.0000038, -0.0019211, 0.0000038, -0.0019249, 0.0019249
7: -0.0091640, -0.0059723, -0.0091640, -0.0059723, -0.0031918, 0.0031918
8: -0.0011850, 0.0088502, -0.0011850, 0.0088502, -0.0099369, 0.0099369
9: -0.0054302, 0.0003582, -0.0054302, 0.0003582, -0.0057884, 0.0057884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029082, upper bound: 0.0030387
time: 2.02 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029023, upper bound: 0.0029017
time: 1.99 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002174, 0.0017253, -0.0001687, 0.0016401, -0.0018575, 0.0018940
1: 0.9921111, 0.9970097, 0.9923263, 0.9968869, -0.0047758, 0.0046834
2: -0.0084405, -0.0025662, -0.0080667, -0.0026816, -0.0054260, 0.0051664
3: 0.0025840, 0.0046158, 0.0026350, 0.0045267, -0.0019426, 0.0019808
4: 0.0013755, 0.0050880, 0.0014321, 0.0047924, -0.0034170, 0.0036559
5: 0.0031583, 0.0079194, 0.0032778, 0.0077104, -0.0045521, 0.0046416
6: -0.0020509, 0.0000460, -0.0019211, 0.0000038, -0.0020547, 0.0019671
7: -0.0093145, -0.0058863, -0.0091640, -0.0059723, -0.0033422, 0.0032777
8: -0.0014508, 0.0093415, -0.0011850, 0.0088502, -0.0102047, 0.0104298
9: -0.0057030, 0.0005141, -0.0054302, 0.0003582, -0.0060612, 0.0059444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030978, upper bound: 0.0030804
time: 2.44 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031041, upper bound: 0.0030818
time: 2.62 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001687, 0.0016401, -0.0002174, 0.0017253, -0.0018940, 0.0018575
1: 0.9923263, 0.9968869, 0.9921111, 0.9970097, -0.0046834, 0.0047758
2: -0.0080667, -0.0026816, -0.0084405, -0.0025662, -0.0051664, 0.0054260
3: 0.0026350, 0.0045267, 0.0025840, 0.0046158, -0.0019808, 0.0019426
4: 0.0014321, 0.0047924, 0.0013755, 0.0050880, -0.0036559, 0.0034170
5: 0.0032778, 0.0077104, 0.0031583, 0.0079194, -0.0046416, 0.0045521
6: -0.0019211, 0.0000038, -0.0020509, 0.0000460, -0.0019671, 0.0020547
7: -0.0091640, -0.0059723, -0.0093145, -0.0058863, -0.0032777, 0.0033422
8: -0.0011850, 0.0088502, -0.0014508, 0.0093415, -0.0104298, 0.0102047
9: -0.0054302, 0.0003582, -0.0057030, 0.0005141, -0.0059444, 0.0060612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030804, upper bound: 0.0031454
time: 2.15 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030818, upper bound: 0.0031528
time: 2.28 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002174, 0.0017253, -0.0002174, 0.0017253, -0.0019427, 0.0019427
1: 0.9921111, 0.9970097, 0.9921111, 0.9970097, -0.0048986, 0.0048986
2: -0.0084405, -0.0025662, -0.0084405, -0.0025662, -0.0055194, 0.0055194
3: 0.0025840, 0.0046158, 0.0025840, 0.0046158, -0.0020318, 0.0020318
4: 0.0013755, 0.0050880, 0.0013755, 0.0050880, -0.0037125, 0.0037125
5: 0.0031583, 0.0079194, 0.0031583, 0.0079194, -0.0047610, 0.0047610
6: -0.0020509, 0.0000460, -0.0020509, 0.0000460, -0.0020969, 0.0020969
7: -0.0093145, -0.0058863, -0.0093145, -0.0058863, -0.0034282, 0.0034282
8: -0.0014508, 0.0093415, -0.0014508, 0.0093415, -0.0106906, 0.0106906
9: -0.0057030, 0.0005141, -0.0057030, 0.0005141, -0.0062172, 0.0062172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030388, upper bound: 0.0029080
time: 2.18 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029019, upper bound: 0.0029017
time: 1.95 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.31 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 1, lower bound: -0.0029082, upper bound: 0.0030387
IS_B1_A1_A2, status: Status.VERIFIED, split count: 3, time: 5.31
Output dim: 1, lower bound: -0.0029023, upper bound: 0.0029017
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 1, lower bound: -0.0030978, upper bound: 0.0030804
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 1, lower bound: -0.0031041, upper bound: 0.0030818
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 1, lower bound: -0.0030804, upper bound: 0.0031454
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 1, lower bound: -0.0030818, upper bound: 0.0031528
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 1, lower bound: -0.0030388, upper bound: 0.0029080
IS_B2_A2_B2, status: Status.VERIFIED, split count: 3, time: 5.31
Output dim: 1, lower bound: -0.0029019, upper bound: 0.0029017

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0001476, 0.0016380, -0.0001687, 0.0016401, -0.0017877, 0.0018066
1: 0.9923315, 0.9968337, 0.9923263, 0.9968869, -0.0045554, 0.0045074
2: -0.0080573, -0.0027314, -0.0080667, -0.0026816, -0.0050351, 0.0049939
3: 0.0026570, 0.0045244, 0.0026350, 0.0045267, -0.0018697, 0.0018895
4: 0.0014565, 0.0047851, 0.0014321, 0.0047924, -0.0033359, 0.0033530
5: 0.0033294, 0.0077052, 0.0032778, 0.0077104, -0.0043811, 0.0044274
6: -0.0019178, -0.0000144, -0.0019211, 0.0000038, -0.0019217, 0.0019067
7: -0.0091603, -0.0060094, -0.0091640, -0.0059723, -0.0031880, 0.0031546
8: -0.0010701, 0.0088379, -0.0011850, 0.0088502, -0.0098215, 0.0099246
9: -0.0054234, 0.0002908, -0.0054302, 0.0003582, -0.0057816, 0.0057210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028536, upper bound: 0.0030095
time: 2.14 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028644, upper bound: 0.0030132
time: 2.46 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002125, 0.0017249, -0.0001215, 0.0016348, -0.0018473, 0.0018463
1: 0.9921125, 0.9969972, 0.9923394, 0.9967679, -0.0046554, 0.0046578
2: -0.0084384, -0.0025778, -0.0080434, -0.0027933, -0.0053123, 0.0051405
3: 0.0025891, 0.0046153, 0.0026843, 0.0045211, -0.0019320, 0.0019310
4: 0.0013812, 0.0050863, 0.0014869, 0.0047741, -0.0033929, 0.0035994
5: 0.0031704, 0.0079181, 0.0033934, 0.0076975, -0.0045271, 0.0045248
6: -0.0020501, 0.0000418, -0.0019130, -0.0000370, -0.0020131, 0.0019548
7: -0.0093136, -0.0058949, -0.0091547, -0.0060555, -0.0032581, 0.0032598
8: -0.0014240, 0.0093387, -0.0009276, 0.0088197, -0.0101489, 0.0101694
9: -0.0057015, 0.0004984, -0.0054133, 0.0002072, -0.0059087, 0.0059117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028813, upper bound: 0.0029933
time: 2.22 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028478, upper bound: 0.0028559
time: 2.14 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002169, 0.0017253, -0.0001346, 0.0016374, -0.0018543, 0.0018599
1: 0.9921112, 0.9970085, 0.9923329, 0.9968008, -0.0046896, 0.0046756
2: -0.0084404, -0.0025674, -0.0080547, -0.0027622, -0.0053401, 0.0051534
3: 0.0025845, 0.0046158, 0.0026706, 0.0045238, -0.0019393, 0.0019452
4: 0.0013761, 0.0050878, 0.0014716, 0.0047830, -0.0034070, 0.0036162
5: 0.0031595, 0.0079193, 0.0033613, 0.0077038, -0.0045443, 0.0045580
6: -0.0020508, 0.0000456, -0.0019170, -0.0000257, -0.0020251, 0.0019626
7: -0.0093144, -0.0058871, -0.0091592, -0.0060324, -0.0032820, 0.0032721
8: -0.0014482, 0.0093412, -0.0009991, 0.0088346, -0.0101865, 0.0102433
9: -0.0057029, 0.0005126, -0.0054215, 0.0002492, -0.0059521, 0.0059341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030132, upper bound: 0.0028643
time: 2.34 seconds

## Relational analysis of IS_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028862, upper bound: 0.0028577
time: 2.07 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0001215, 0.0016348, -0.0002125, 0.0017249, -0.0018463, 0.0018473
1: 0.9923394, 0.9967679, 0.9921125, 0.9969972, -0.0046578, 0.0046554
2: -0.0080434, -0.0027933, -0.0084384, -0.0025778, -0.0051405, 0.0053123
3: 0.0026843, 0.0045211, 0.0025891, 0.0046153, -0.0019310, 0.0019320
4: 0.0014869, 0.0047741, 0.0013812, 0.0050863, -0.0035994, 0.0033929
5: 0.0033934, 0.0076975, 0.0031704, 0.0079181, -0.0045248, 0.0045271
6: -0.0019130, -0.0000370, -0.0020501, 0.0000418, -0.0019548, 0.0020131
7: -0.0091547, -0.0060555, -0.0093136, -0.0058949, -0.0032598, 0.0032581
8: -0.0009276, 0.0088197, -0.0014240, 0.0093387, -0.0101694, 0.0101489
9: -0.0054133, 0.0002072, -0.0057015, 0.0004984, -0.0059117, 0.0059087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029933, upper bound: 0.0028813
time: 2.27 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028559, upper bound: 0.0028763
time: 1.79 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0001346, 0.0016374, -0.0002169, 0.0017253, -0.0018599, 0.0018543
1: 0.9923329, 0.9968008, 0.9921112, 0.9970085, -0.0046756, 0.0046896
2: -0.0080547, -0.0027622, -0.0084404, -0.0025674, -0.0051534, 0.0053401
3: 0.0026706, 0.0045238, 0.0025845, 0.0046158, -0.0019452, 0.0019393
4: 0.0014716, 0.0047830, 0.0013761, 0.0050878, -0.0036162, 0.0034070
5: 0.0033613, 0.0077038, 0.0031595, 0.0079193, -0.0045580, 0.0045443
6: -0.0019170, -0.0000257, -0.0020508, 0.0000456, -0.0019626, 0.0020251
7: -0.0091592, -0.0060324, -0.0093144, -0.0058871, -0.0032721, 0.0032820
8: -0.0009991, 0.0088346, -0.0014482, 0.0093412, -0.0102433, 0.0101865
9: -0.0054215, 0.0002492, -0.0057029, 0.0005126, -0.0059341, 0.0059521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028643, upper bound: 0.0030541
time: 1.98 seconds

## Relational analysis of IS_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028579, upper bound: 0.0028862
time: 2.28 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002174, 0.0017253, -0.0001959, 0.0017232, -0.0019406, 0.0019213
1: 0.9921111, 0.9970097, 0.9921165, 0.9969556, -0.0048445, 0.0048932
2: -0.0084405, -0.0025662, -0.0084312, -0.0026170, -0.0054676, 0.0055099
3: 0.0025840, 0.0046158, 0.0026064, 0.0046136, -0.0020296, 0.0020094
4: 0.0013755, 0.0050880, 0.0014004, 0.0050806, -0.0037051, 0.0036876
5: 0.0031583, 0.0079194, 0.0032109, 0.0079141, -0.0047558, 0.0047085
6: -0.0020509, 0.0000460, -0.0020476, 0.0000274, -0.0020783, 0.0020936
7: -0.0093145, -0.0058863, -0.0093107, -0.0059241, -0.0033903, 0.0034244
8: -0.0014508, 0.0093415, -0.0013338, 0.0093292, -0.0106783, 0.0105736
9: -0.0057030, 0.0005141, -0.0056962, 0.0004455, -0.0061485, 0.0062103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030507, upper bound: 0.0028536
time: 2.11 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030541, upper bound: 0.0028643
time: 2.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.65 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.65
Output dim: 1, lower bound: -0.0028536, upper bound: 0.0030095
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.65
Output dim: 1, lower bound: -0.0028644, upper bound: 0.0030132
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.65
Output dim: 1, lower bound: -0.0028813, upper bound: 0.0029933
IS_B1_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 5.65
Output dim: 1, lower bound: -0.0028478, upper bound: 0.0028559
IS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.65
Output dim: 1, lower bound: -0.0030132, upper bound: 0.0028643
IS_B1_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 5.65
Output dim: 1, lower bound: -0.0028862, upper bound: 0.0028577
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.65
Output dim: 1, lower bound: -0.0029933, upper bound: 0.0028813
IS_B2_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 5.65
Output dim: 1, lower bound: -0.0028559, upper bound: 0.0028763
IS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.65
Output dim: 1, lower bound: -0.0028643, upper bound: 0.0030541
IS_B2_A1_A2_A2, status: Status.VERIFIED, split count: 4, time: 5.65
Output dim: 1, lower bound: -0.0028579, upper bound: 0.0028862
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.65
Output dim: 1, lower bound: -0.0030507, upper bound: 0.0028536
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.65
Output dim: 1, lower bound: -0.0030541, upper bound: 0.0028643

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001427, 0.0016375, -0.0001215, 0.0016348, -0.0017775, 0.0017590
1: 0.9923328, 0.9968213, 0.9923394, 0.9967679, -0.0044351, 0.0044819
2: -0.0080552, -0.0027431, -0.0080434, -0.0027933, -0.0049215, 0.0049680
3: 0.0026622, 0.0045239, 0.0026843, 0.0045211, -0.0018590, 0.0018396
4: 0.0014623, 0.0047834, 0.0014869, 0.0047741, -0.0033118, 0.0032965
5: 0.0033415, 0.0077040, 0.0033934, 0.0076975, -0.0043560, 0.0043106
6: -0.0019171, -0.0000187, -0.0019130, -0.0000370, -0.0018801, 0.0018943
7: -0.0091594, -0.0060182, -0.0091547, -0.0060555, -0.0031039, 0.0031365
8: -0.0010432, 0.0088352, -0.0009276, 0.0088197, -0.0097657, 0.0096643
9: -0.0054219, 0.0002750, -0.0054133, 0.0002072, -0.0056291, 0.0056883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028060, upper bound: 0.0029612
time: 2.54 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028128, upper bound: 0.0029630
time: 2.01 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001471, 0.0016379, -0.0001346, 0.0016374, -0.0017845, 0.0017725
1: 0.9923315, 0.9968325, 0.9923329, 0.9968008, -0.0044693, 0.0044996
2: -0.0080571, -0.0027326, -0.0080547, -0.0027622, -0.0049487, 0.0049810
3: 0.0026575, 0.0045244, 0.0026706, 0.0045238, -0.0018663, 0.0018538
4: 0.0014571, 0.0047849, 0.0014716, 0.0047830, -0.0033259, 0.0033133
5: 0.0033305, 0.0077051, 0.0033613, 0.0077038, -0.0043732, 0.0043439
6: -0.0019178, -0.0000148, -0.0019170, -0.0000257, -0.0018921, 0.0019021
7: -0.0091602, -0.0060103, -0.0091592, -0.0060324, -0.0031278, 0.0031490
8: -0.0010675, 0.0088377, -0.0009991, 0.0088346, -0.0098034, 0.0097379
9: -0.0054233, 0.0002893, -0.0054215, 0.0002492, -0.0056725, 0.0057108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B1_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028162, upper bound: 0.0029644
time: 2.18 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028227, upper bound: 0.0029675
time: 2.22 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001910, 0.0017227, -0.0001215, 0.0016348, -0.0018258, 0.0018442
1: 0.9921176, 0.9969432, 0.9923394, 0.9967679, -0.0046502, 0.0046038
2: -0.0084291, -0.0026286, -0.0080434, -0.0027933, -0.0053029, 0.0050882
3: 0.0026116, 0.0046131, 0.0026843, 0.0045211, -0.0019095, 0.0019288
4: 0.0014061, 0.0050789, 0.0014869, 0.0047741, -0.0033680, 0.0035920
5: 0.0032230, 0.0079130, 0.0033934, 0.0076975, -0.0044745, 0.0045196
6: -0.0020469, 0.0000232, -0.0019130, -0.0000370, -0.0020099, 0.0019362
7: -0.0093099, -0.0059328, -0.0091547, -0.0060555, -0.0032543, 0.0032219
8: -0.0013070, 0.0093264, -0.0009276, 0.0088197, -0.0100319, 0.0101572
9: -0.0056947, 0.0004298, -0.0054133, 0.0002072, -0.0059019, 0.0058430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028172, upper bound: 0.0029353
time: 2.77 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028401, upper bound: 0.0029484
time: 2.32 seconds

## BFS IS instance: IS_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0002169, 0.0017253, -0.0001139, 0.0016353, -0.0018522, 0.0018392
1: 0.9921112, 0.9970085, 0.9923382, 0.9967489, -0.0046377, 0.0046703
2: -0.0084404, -0.0025674, -0.0080457, -0.0028112, -0.0052894, 0.0051443
3: 0.0025845, 0.0046158, 0.0026923, 0.0045217, -0.0019371, 0.0019235
4: 0.0013761, 0.0050878, 0.0014957, 0.0047759, -0.0033998, 0.0035921
5: 0.0031595, 0.0079193, 0.0034120, 0.0076987, -0.0045392, 0.0045073
6: -0.0020508, 0.0000456, -0.0019138, -0.0000436, -0.0020072, 0.0019594
7: -0.0093144, -0.0058871, -0.0091556, -0.0060689, -0.0032455, 0.0032685
8: -0.0014482, 0.0093412, -0.0008862, 0.0088226, -0.0101746, 0.0101297
9: -0.0057029, 0.0005126, -0.0054149, 0.0001829, -0.0058858, 0.0059275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029853, upper bound: 0.0028020
time: 1.87 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2

### Relational analysis result of IS_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029982, upper bound: 0.0028224
time: 2.63 seconds

## BFS IS instance: IS_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001215, 0.0016348, -0.0001910, 0.0017227, -0.0018442, 0.0018258
1: 0.9923394, 0.9967679, 0.9921176, 0.9969432, -0.0046038, 0.0046502
2: -0.0080434, -0.0027933, -0.0084291, -0.0026286, -0.0050882, 0.0053029
3: 0.0026843, 0.0045211, 0.0026116, 0.0046131, -0.0019288, 0.0019095
4: 0.0014869, 0.0047741, 0.0014061, 0.0050789, -0.0035920, 0.0033680
5: 0.0033934, 0.0076975, 0.0032230, 0.0079130, -0.0045196, 0.0044745
6: -0.0019130, -0.0000370, -0.0020469, 0.0000232, -0.0019362, 0.0020099
7: -0.0091547, -0.0060555, -0.0093099, -0.0059328, -0.0032219, 0.0032543
8: -0.0009276, 0.0088197, -0.0013070, 0.0093264, -0.0101572, 0.0100319
9: -0.0054133, 0.0002072, -0.0056947, 0.0004298, -0.0058430, 0.0059019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029353, upper bound: 0.0028172
time: 1.90 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2

### Relational analysis result of IS_B2_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029484, upper bound: 0.0028401
time: 1.81 seconds

## BFS IS instance: IS_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0001139, 0.0016353, -0.0002169, 0.0017253, -0.0018392, 0.0018522
1: 0.9923382, 0.9967489, 0.9921112, 0.9970085, -0.0046703, 0.0046377
2: -0.0080457, -0.0028112, -0.0084404, -0.0025674, -0.0051443, 0.0052894
3: 0.0026923, 0.0045217, 0.0025845, 0.0046158, -0.0019235, 0.0019371
4: 0.0014957, 0.0047759, 0.0013761, 0.0050878, -0.0035921, 0.0033998
5: 0.0034120, 0.0076987, 0.0031595, 0.0079193, -0.0045073, 0.0045392
6: -0.0019138, -0.0000436, -0.0020508, 0.0000456, -0.0019594, 0.0020072
7: -0.0091556, -0.0060689, -0.0093144, -0.0058871, -0.0032685, 0.0032455
8: -0.0008862, 0.0088226, -0.0014482, 0.0093412, -0.0101297, 0.0101746
9: -0.0054149, 0.0001829, -0.0057029, 0.0005126, -0.0059275, 0.0058858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_A2_A1_B1

### Relational analysis result of IS_B2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028020, upper bound: 0.0029853
time: 2.45 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2

### Relational analysis result of IS_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028224, upper bound: 0.0030097
time: 2.28 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001700, 0.0017210, -0.0001910, 0.0017227, -0.0018927, 0.0019120
1: 0.9921221, 0.9968902, 0.9921176, 0.9969432, -0.0048211, 0.0047726
2: -0.0084214, -0.0026784, -0.0084291, -0.0026286, -0.0054425, 0.0053965
3: 0.0026336, 0.0046112, 0.0026116, 0.0046131, -0.0019795, 0.0019997
4: 0.0014305, 0.0050728, 0.0014061, 0.0050789, -0.0036484, 0.0036667
5: 0.0032744, 0.0079087, 0.0032230, 0.0079130, -0.0046385, 0.0046857
6: -0.0020442, 0.0000050, -0.0020469, 0.0000232, -0.0020674, 0.0020519
7: -0.0093068, -0.0059699, -0.0093099, -0.0059328, -0.0033740, 0.0033400
8: -0.0011924, 0.0093163, -0.0013070, 0.0093264, -0.0104167, 0.0105232
9: -0.0056891, 0.0003625, -0.0056947, 0.0004298, -0.0061188, 0.0060572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030003, upper bound: 0.0028059
time: 2.22 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030061, upper bound: 0.0028127
time: 2.19 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001830, 0.0017227, -0.0001955, 0.0017232, -0.0019062, 0.0019181
1: 0.9921179, 0.9969229, 0.9921166, 0.9969543, -0.0048364, 0.0048063
2: -0.0084287, -0.0026476, -0.0084310, -0.0026181, -0.0054548, 0.0054222
3: 0.0026200, 0.0046130, 0.0026069, 0.0046135, -0.0019936, 0.0020061
4: 0.0014154, 0.0050786, 0.0014010, 0.0050804, -0.0036650, 0.0036777
5: 0.0032426, 0.0079127, 0.0032121, 0.0079140, -0.0046714, 0.0047007
6: -0.0020468, 0.0000162, -0.0020476, 0.0000270, -0.0020738, 0.0020638
7: -0.0093097, -0.0059470, -0.0093106, -0.0059250, -0.0033848, 0.0033637
8: -0.0012632, 0.0093260, -0.0013312, 0.0093290, -0.0104895, 0.0105554
9: -0.0056944, 0.0004041, -0.0056961, 0.0004440, -0.0061384, 0.0061002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030035, upper bound: 0.0028161
time: 2.60 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030097, upper bound: 0.0028224
time: 2.41 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.16 seconds
IS_B1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0028060, upper bound: 0.0029612
IS_B1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0028128, upper bound: 0.0029630
IS_B1_A1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0028162, upper bound: 0.0029644
IS_B1_A1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0028227, upper bound: 0.0029675
IS_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0028172, upper bound: 0.0029353
IS_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0028401, upper bound: 0.0029484
IS_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0029853, upper bound: 0.0028020
IS_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0029982, upper bound: 0.0028224
IS_B2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0029353, upper bound: 0.0028172
IS_B2_A1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0029484, upper bound: 0.0028401
IS_B2_A1_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0028020, upper bound: 0.0029853
IS_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0028224, upper bound: 0.0030097
IS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0030003, upper bound: 0.0028059
IS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0030061, upper bound: 0.0028127
IS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0030035, upper bound: 0.0028161
IS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.16
Output dim: 1, lower bound: -0.0030097, upper bound: 0.0028224

## BFS IS instance: IS_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001720, 0.0017240, -0.0001127, 0.0016353, -0.0018073, 0.0018367
1: 0.9921145, 0.9968951, 0.9923383, 0.9967458, -0.0046313, 0.0045568
2: -0.0084348, -0.0026737, -0.0080455, -0.0028140, -0.0052810, 0.0050415
3: 0.0026315, 0.0046144, 0.0026935, 0.0045216, -0.0018901, 0.0019210
4: 0.0014282, 0.0050834, 0.0014970, 0.0047757, -0.0033475, 0.0035864
5: 0.0032696, 0.0079161, 0.0034149, 0.0076986, -0.0044290, 0.0045013
6: -0.0020489, 0.0000067, -0.0019138, -0.0000446, -0.0020042, 0.0019204
7: -0.0093121, -0.0059664, -0.0091555, -0.0060710, -0.0032411, 0.0031891
8: -0.0012031, 0.0093339, -0.0008797, 0.0088224, -0.0099310, 0.0101159
9: -0.0056988, 0.0003688, -0.0054148, 0.0001791, -0.0058780, 0.0057836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B1_A2_B2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029637, upper bound: 0.0028162
time: 2.47 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030033, upper bound: 0.0028224
time: 2.54 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001127, 0.0016353, -0.0001720, 0.0017240, -0.0018367, 0.0018073
1: 0.9923383, 0.9967458, 0.9921145, 0.9968951, -0.0045568, 0.0046313
2: -0.0080455, -0.0028140, -0.0084348, -0.0026737, -0.0050415, 0.0052810
3: 0.0026935, 0.0045216, 0.0026315, 0.0046144, -0.0019210, 0.0018901
4: 0.0014970, 0.0047757, 0.0014282, 0.0050834, -0.0035864, 0.0033475
5: 0.0034149, 0.0076986, 0.0032696, 0.0079161, -0.0045013, 0.0044290
6: -0.0019138, -0.0000446, -0.0020489, 0.0000067, -0.0019204, 0.0020042
7: -0.0091555, -0.0060710, -0.0093121, -0.0059664, -0.0031891, 0.0032411
8: -0.0008797, 0.0088224, -0.0012031, 0.0093339, -0.0101159, 0.0099310
9: -0.0054148, 0.0001791, -0.0056988, 0.0003688, -0.0057836, 0.0058780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_A2_A1_B2_A1

### Relational analysis result of IS_B2_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028162, upper bound: 0.0030033
time: 1.74 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_A2

### Relational analysis result of IS_B2_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028162, upper bound: 0.0030097
time: 2.23 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001653, 0.0017208, -0.0001476, 0.0017364, -0.0019017, 0.0018684
1: 0.9921227, 0.9968784, 0.9920833, 0.9968339, -0.0047112, 0.0047952
2: -0.0084205, -0.0026895, -0.0084890, -0.0027314, -0.0053373, 0.0054457
3: 0.0026385, 0.0046110, 0.0026570, 0.0046274, -0.0019889, 0.0019540
4: 0.0014360, 0.0050721, 0.0014565, 0.0051263, -0.0036903, 0.0036156
5: 0.0032860, 0.0079082, 0.0033293, 0.0079464, -0.0046604, 0.0045788
6: -0.0020439, 0.0000009, -0.0020677, -0.0000144, -0.0020295, 0.0020686
7: -0.0093064, -0.0059782, -0.0093340, -0.0060094, -0.0032970, 0.0033557
8: -0.0011666, 0.0093151, -0.0010702, 0.0094051, -0.0104687, 0.0102846
9: -0.0056884, 0.0003474, -0.0057384, 0.0002909, -0.0059793, 0.0060858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029494, upper bound: 0.0027605
time: 2.19 seconds

## Relational analysis of IS_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029496, upper bound: 0.0027540
time: 3.06 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001688, 0.0017209, -0.0001460, 0.0017214, -0.0018902, 0.0018670
1: 0.9921224, 0.9968873, 0.9921210, 0.9968298, -0.0047075, 0.0047663
2: -0.0084213, -0.0026812, -0.0084233, -0.0027351, -0.0053402, 0.0053879
3: 0.0026348, 0.0046112, 0.0026586, 0.0046117, -0.0019769, 0.0019526
4: 0.0014319, 0.0050727, 0.0014584, 0.0050743, -0.0036424, 0.0036144
5: 0.0032773, 0.0079086, 0.0033332, 0.0079097, -0.0046324, 0.0045754
6: -0.0020442, 0.0000039, -0.0020449, -0.0000158, -0.0020284, 0.0020488
7: -0.0093067, -0.0059720, -0.0093075, -0.0060122, -0.0032945, 0.0033356
8: -0.0011859, 0.0093161, -0.0010616, 0.0093188, -0.0104026, 0.0102789
9: -0.0056890, 0.0003587, -0.0056904, 0.0002858, -0.0059748, 0.0060492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029822, upper bound: 0.0027910
time: 3.03 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029822, upper bound: 0.0028127
time: 2.31 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001783, 0.0017224, -0.0001521, 0.0017368, -0.0019151, 0.0018745
1: 0.9921185, 0.9969111, 0.9920822, 0.9968451, -0.0047266, 0.0048289
2: -0.0084278, -0.0026588, -0.0084909, -0.0027208, -0.0053497, 0.0054705
3: 0.0026249, 0.0046128, 0.0026523, 0.0046278, -0.0020029, 0.0019605
4: 0.0014209, 0.0050779, 0.0014513, 0.0051277, -0.0037068, 0.0036266
5: 0.0032542, 0.0079123, 0.0033183, 0.0079475, -0.0046933, 0.0045939
6: -0.0020465, 0.0000121, -0.0020683, -0.0000105, -0.0020359, 0.0020805
7: -0.0093094, -0.0059553, -0.0093347, -0.0060015, -0.0033079, 0.0033794
8: -0.0012374, 0.0093248, -0.0010946, 0.0094076, -0.0105412, 0.0103172
9: -0.0056938, 0.0003889, -0.0057398, 0.0003052, -0.0059990, 0.0061287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029522, upper bound: 0.0027709
time: 2.29 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029496, upper bound: 0.0027608
time: 2.66 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001818, 0.0017226, -0.0001505, 0.0017218, -0.0019037, 0.0018731
1: 0.9921181, 0.9969199, 0.9921200, 0.9968411, -0.0047230, 0.0047999
2: -0.0084286, -0.0026504, -0.0084252, -0.0027245, -0.0053537, 0.0054135
3: 0.0026212, 0.0046130, 0.0026540, 0.0046122, -0.0019909, 0.0019590
4: 0.0014168, 0.0050785, 0.0014531, 0.0050758, -0.0036590, 0.0036254
5: 0.0032455, 0.0079127, 0.0033222, 0.0079108, -0.0046652, 0.0045904
6: -0.0020467, 0.0000152, -0.0020456, -0.0000119, -0.0020348, 0.0020608
7: -0.0093097, -0.0059491, -0.0093083, -0.0060043, -0.0033054, 0.0033592
8: -0.0012567, 0.0093258, -0.0010860, 0.0093213, -0.0104754, 0.0103114
9: -0.0056943, 0.0004003, -0.0056918, 0.0003001, -0.0059944, 0.0060921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029822, upper bound: 0.0028020
time: 2.85 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029853, upper bound: 0.0028223
time: 2.52 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.55 seconds
IS_B1_A2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.55
Output dim: 1, lower bound: -0.0029637, upper bound: 0.0028162
IS_B1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.55
Output dim: 1, lower bound: -0.0030033, upper bound: 0.0028224
IS_B2_A1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.55
Output dim: 1, lower bound: -0.0028162, upper bound: 0.0030033
IS_B2_A1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.55
Output dim: 1, lower bound: -0.0028162, upper bound: 0.0030097
IS_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.55
Output dim: 1, lower bound: -0.0029494, upper bound: 0.0027605
IS_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 6.55
Output dim: 1, lower bound: -0.0029496, upper bound: 0.0027540
IS_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 6.55
Output dim: 1, lower bound: -0.0029822, upper bound: 0.0027910
IS_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 6.55
Output dim: 1, lower bound: -0.0029822, upper bound: 0.0028127
IS_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.55
Output dim: 1, lower bound: -0.0029522, upper bound: 0.0027709
IS_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 6.55
Output dim: 1, lower bound: -0.0029496, upper bound: 0.0027608
IS_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 6.55
Output dim: 1, lower bound: -0.0029822, upper bound: 0.0028020
IS_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 6.55
Output dim: 1, lower bound: -0.0029853, upper bound: 0.0028223

## BFS IS instance: IS_B1_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001720, 0.0017240, -0.0000695, 0.0016340, -0.0018060, 0.0017935
1: 0.9921145, 0.9968951, 0.9923414, 0.9966368, -0.0045223, 0.0045537
2: -0.0084348, -0.0026737, -0.0080399, -0.0029163, -0.0051839, 0.0050357
3: 0.0026315, 0.0046144, 0.0027387, 0.0045203, -0.0018888, 0.0018758
4: 0.0014282, 0.0050834, 0.0015472, 0.0047713, -0.0033431, 0.0035362
5: 0.0032696, 0.0079161, 0.0035207, 0.0076955, -0.0044259, 0.0043954
6: -0.0020489, 0.0000067, -0.0019118, -0.0000820, -0.0019668, 0.0019185
7: -0.0093121, -0.0059664, -0.0091533, -0.0061472, -0.0031649, 0.0031869
8: -0.0012031, 0.0093339, -0.0006441, 0.0088150, -0.0099235, 0.0098814
9: -0.0056988, 0.0003688, -0.0054107, 0.0000409, -0.0057397, 0.0057795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_B1_A2_B2_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029911, upper bound: 0.0028127
time: 2.40 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029911, upper bound: 0.0028224
time: 2.27 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0000704, 0.0016485, -0.0001720, 0.0017240, -0.0017945, 0.0018205
1: 0.9923050, 0.9966390, 0.9921145, 0.9968951, -0.0045901, 0.0045245
2: -0.0081034, -0.0029141, -0.0084348, -0.0026737, -0.0050958, 0.0051787
3: 0.0027377, 0.0045354, 0.0026315, 0.0046144, -0.0018767, 0.0019040
4: 0.0015461, 0.0048215, 0.0014282, 0.0050834, -0.0035373, 0.0033933
5: 0.0035185, 0.0077310, 0.0032696, 0.0079161, -0.0043976, 0.0044614
6: -0.0019339, -0.0000812, -0.0020489, 0.0000067, -0.0019405, 0.0019676
7: -0.0091788, -0.0061456, -0.0093121, -0.0059664, -0.0032124, 0.0031665
8: -0.0006491, 0.0088986, -0.0012031, 0.0093339, -0.0098861, 0.0100052
9: -0.0054571, 0.0000438, -0.0056988, 0.0003688, -0.0058259, 0.0057427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_B2_A1_A2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027902, upper bound: 0.0030002
time: 2.10 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027902, upper bound: 0.0030033
time: 2.22 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0000695, 0.0016340, -0.0001720, 0.0017240, -0.0017935, 0.0018060
1: 0.9923414, 0.9966368, 0.9921145, 0.9968951, -0.0045537, 0.0045223
2: -0.0080399, -0.0029163, -0.0084348, -0.0026737, -0.0050357, 0.0051839
3: 0.0027387, 0.0045203, 0.0026315, 0.0046144, -0.0018758, 0.0018888
4: 0.0015472, 0.0047713, 0.0014282, 0.0050834, -0.0035362, 0.0033431
5: 0.0035207, 0.0076955, 0.0032696, 0.0079161, -0.0043954, 0.0044259
6: -0.0019118, -0.0000820, -0.0020489, 0.0000067, -0.0019185, 0.0019668
7: -0.0091533, -0.0061472, -0.0093121, -0.0059664, -0.0031869, 0.0031649
8: -0.0006441, 0.0088150, -0.0012031, 0.0093339, -0.0098814, 0.0099235
9: -0.0054107, 0.0000409, -0.0056988, 0.0003688, -0.0057795, 0.0057397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_B2_A1_A2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027902, upper bound: 0.0029966
time: 2.39 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027902, upper bound: 0.0029997
time: 2.06 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.63 seconds
IS_B1_A2_B2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.63
Output dim: 1, lower bound: -0.0029911, upper bound: 0.0028127
IS_B1_A2_B2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.63
Output dim: 1, lower bound: -0.0029911, upper bound: 0.0028224
IS_B2_A1_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.63
Output dim: 1, lower bound: -0.0027902, upper bound: 0.0030002
IS_B2_A1_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.63
Output dim: 1, lower bound: -0.0027902, upper bound: 0.0030033
IS_B2_A1_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.63
Output dim: 1, lower bound: -0.0027902, upper bound: 0.0029966
IS_B2_A1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.63
Output dim: 1, lower bound: -0.0027902, upper bound: 0.0029997

## BFS IS instance: IS_B2_A1_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0000704, 0.0016485, -0.0001255, 0.0017198, -0.0017902, 0.0017740
1: 0.9923050, 0.9966390, 0.9921253, 0.9967780, -0.0044730, 0.0045137
2: -0.0081034, -0.0029141, -0.0084161, -0.0027838, -0.0049865, 0.0051741
3: 0.0027377, 0.0045354, 0.0026801, 0.0046100, -0.0018723, 0.0018553
4: 0.0015461, 0.0048215, 0.0014822, 0.0050686, -0.0035225, 0.0033393
5: 0.0035185, 0.0077310, 0.0033836, 0.0079057, -0.0043872, 0.0043474
6: -0.0019339, -0.0000812, -0.0020424, -0.0000336, -0.0019003, 0.0019611
7: -0.0091788, -0.0061456, -0.0093046, -0.0060484, -0.0031304, 0.0031590
8: -0.0006491, 0.0088986, -0.0009495, 0.0093093, -0.0098652, 0.0097511
9: -0.0054571, 0.0000438, -0.0056852, 0.0002200, -0.0056771, 0.0057290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_B2_A1_A2_A1_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_A2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027321, upper bound: 0.0029597
time: 2.18 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_A2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027321, upper bound: 0.0029496
time: 2.29 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0000704, 0.0016485, -0.0001381, 0.0017214, -0.0017918, 0.0017866
1: 0.9923050, 0.9966390, 0.9921212, 0.9968099, -0.0045049, 0.0045178
2: -0.0081034, -0.0029141, -0.0084232, -0.0027539, -0.0050077, 0.0051672
3: 0.0027377, 0.0045354, 0.0026669, 0.0046117, -0.0018740, 0.0018685
4: 0.0015461, 0.0048215, 0.0014676, 0.0050743, -0.0035282, 0.0033540
5: 0.0035185, 0.0077310, 0.0033526, 0.0079097, -0.0043912, 0.0043784
6: -0.0019339, -0.0000812, -0.0020449, -0.0000226, -0.0019112, 0.0019636
7: -0.0091788, -0.0061456, -0.0093075, -0.0060262, -0.0031526, 0.0031619
8: -0.0006491, 0.0088986, -0.0010183, 0.0093187, -0.0098710, 0.0098196
9: -0.0054571, 0.0000438, -0.0056904, 0.0002604, -0.0057175, 0.0057343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_B2_A1_A2_A1_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027902, upper bound: 0.0030035
time: 2.02 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_A2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027902, upper bound: 0.0030035
time: 2.68 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0000695, 0.0016340, -0.0001255, 0.0017198, -0.0017892, 0.0017595
1: 0.9923414, 0.9966368, 0.9921253, 0.9967780, -0.0044366, 0.0045115
2: -0.0080399, -0.0029163, -0.0084161, -0.0027838, -0.0049258, 0.0051788
3: 0.0027387, 0.0045203, 0.0026801, 0.0046100, -0.0018713, 0.0018402
4: 0.0015472, 0.0047713, 0.0014822, 0.0050686, -0.0035214, 0.0032891
5: 0.0035207, 0.0076955, 0.0033836, 0.0079057, -0.0043849, 0.0043119
6: -0.0019118, -0.0000820, -0.0020424, -0.0000336, -0.0018782, 0.0019603
7: -0.0091533, -0.0061472, -0.0093046, -0.0060484, -0.0031048, 0.0031574
8: -0.0006441, 0.0088150, -0.0009495, 0.0093093, -0.0098606, 0.0096690
9: -0.0054107, 0.0000409, -0.0056852, 0.0002200, -0.0056307, 0.0057260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_B2_A1_A2_A1_B2_A2_B1_A1

### Relational analysis result of IS_B2_A1_A2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027627, upper bound: 0.0029543
time: 2.57 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_A2_B1_A2

### Relational analysis result of IS_B2_A1_A2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027627, upper bound: 0.0029484
time: 2.34 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0000695, 0.0016340, -0.0001381, 0.0017214, -0.0017909, 0.0017721
1: 0.9923414, 0.9966368, 0.9921212, 0.9968099, -0.0044685, 0.0045156
2: -0.0080399, -0.0029163, -0.0084232, -0.0027539, -0.0049486, 0.0051726
3: 0.0027387, 0.0045203, 0.0026669, 0.0046117, -0.0018730, 0.0018533
4: 0.0015472, 0.0047713, 0.0014676, 0.0050743, -0.0035271, 0.0033037
5: 0.0035207, 0.0076955, 0.0033526, 0.0079097, -0.0043889, 0.0043428
6: -0.0019118, -0.0000820, -0.0020449, -0.0000226, -0.0018892, 0.0019628
7: -0.0091533, -0.0061472, -0.0093075, -0.0060262, -0.0031271, 0.0031603
8: -0.0006441, 0.0088150, -0.0010183, 0.0093187, -0.0098663, 0.0097379
9: -0.0054107, 0.0000409, -0.0056904, 0.0002604, -0.0056711, 0.0057313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_B2_A1_A2_A1_B2_A2_B2_A1

### Relational analysis result of IS_B2_A1_A2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027627, upper bound: 0.0029572
time: 2.38 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_A2_B2_A2

### Relational analysis result of IS_B2_A1_A2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027627, upper bound: 0.0029507
time: 2.37 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.93 seconds
IS_B2_A1_A2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 1, lower bound: -0.0027321, upper bound: 0.0029597
IS_B2_A1_A2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 1, lower bound: -0.0027321, upper bound: 0.0029496
IS_B2_A1_A2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 1, lower bound: -0.0027902, upper bound: 0.0030035
IS_B2_A1_A2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 1, lower bound: -0.0027902, upper bound: 0.0030035
IS_B2_A1_A2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 1, lower bound: -0.0027627, upper bound: 0.0029543
IS_B2_A1_A2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 1, lower bound: -0.0027627, upper bound: 0.0029484
IS_B2_A1_A2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 1, lower bound: -0.0027627, upper bound: 0.0029572
IS_B2_A1_A2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 1, lower bound: -0.0027627, upper bound: 0.0029507

## BFS IS instance: IS_B2_A1_A2_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0000704, 0.0016485, -0.0001167, 0.0017193, -0.0017897, 0.0017652
1: 0.9923050, 0.9966390, 0.9921265, 0.9967558, -0.0044507, 0.0045125
2: -0.0081034, -0.0029141, -0.0084139, -0.0028046, -0.0049548, 0.0051578
3: 0.0027377, 0.0045354, 0.0026893, 0.0046095, -0.0018718, 0.0018461
4: 0.0015461, 0.0048215, 0.0014924, 0.0050669, -0.0035208, 0.0033291
5: 0.0035185, 0.0077310, 0.0034051, 0.0079045, -0.0043860, 0.0043259
6: -0.0019339, -0.0000812, -0.0020416, -0.0000412, -0.0018927, 0.0019604
7: -0.0091788, -0.0061456, -0.0093038, -0.0060640, -0.0031149, 0.0031582
8: -0.0006491, 0.0088986, -0.0009014, 0.0093065, -0.0098587, 0.0097025
9: -0.0054571, 0.0000438, -0.0056836, 0.0001919, -0.0056489, 0.0057275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_B2_A1_A2_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_A2_A1_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027604, upper bound: 0.0029626
time: 3.04 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_A2_A1_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027603, upper bound: 0.0029523
time: 2.08 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0000704, 0.0016485, -0.0000597, 0.0018008, -0.0018713, 0.0017082
1: 0.9923050, 0.9966390, 0.9919209, 0.9966121, -0.0043070, 0.0047181
2: -0.0081034, -0.0029141, -0.0087717, -0.0029395, -0.0048306, 0.0055339
3: 0.0027377, 0.0045354, 0.0027489, 0.0046948, -0.0019571, 0.0017865
4: 0.0015461, 0.0048215, 0.0015586, 0.0053497, -0.0038036, 0.0032629
5: 0.0035185, 0.0077310, 0.0035448, 0.0081044, -0.0045859, 0.0041862
6: -0.0019339, -0.0000812, -0.0021658, -0.0000905, -0.0018433, 0.0020846
7: -0.0091788, -0.0061456, -0.0094477, -0.0061646, -0.0030143, 0.0033021
8: -0.0006491, 0.0088986, -0.0005906, 0.0097765, -0.0103333, 0.0093908
9: -0.0054571, 0.0000438, -0.0059446, 0.0000095, -0.0054666, 0.0059885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 118

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_B2_A1_A2_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_A2_A1_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027604, upper bound: 0.0029626
time: 2.26 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_A2_A1_B2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027603, upper bound: 0.0029523
time: 2.90 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 6.37 seconds
IS_B2_A1_A2_A1_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 6.37
Output dim: 1, lower bound: -0.0027604, upper bound: 0.0029626
IS_B2_A1_A2_A1_B2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 9, time: 6.37
Output dim: 1, lower bound: -0.0027603, upper bound: 0.0029523
IS_B2_A1_A2_A1_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 6.37
Output dim: 1, lower bound: -0.0027604, upper bound: 0.0029626
IS_B2_A1_A2_A1_B2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 9, time: 6.37
Output dim: 1, lower bound: -0.0027603, upper bound: 0.0029523

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 4.20 + 208.58 = 212.78 seconds
