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
execution time: IAR + RelationalAnalysis = 1.15 + 2.96 = 4.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0032357, upper bound: 0.0032357

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031940
time: 2.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031939
time: 4.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.76 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 6.76
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031940
IS_A2, status: Status.UNKNOWN, split count: 1, time: 6.76
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031939

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0001687, 0.0016401, -0.0002223, 0.0017475, -0.0019162, 0.0018624
1: 0.9923263, 0.9968869, 0.9920551, 0.9970222, -0.0046960, 0.0048318
2: -0.0080667, -0.0026816, -0.0085379, -0.0025545, -0.0051778, 0.0055178
3: 0.0026350, 0.0045267, 0.0025789, 0.0046390, -0.0020041, 0.0019478
4: 0.0014321, 0.0047924, 0.0013698, 0.0051649, -0.0037328, 0.0034227
5: 0.0032778, 0.0077104, 0.0031462, 0.0079738, -0.0046960, 0.0045642
6: -0.0019211, 0.0000038, -0.0020847, 0.0000503, -0.0019714, 0.0020885
7: -0.0091640, -0.0059723, -0.0093537, -0.0058776, -0.0032865, 0.0033814
8: -0.0011850, 0.0088502, -0.0014778, 0.0094694, -0.0105568, 0.0102316
9: -0.0054302, 0.0003582, -0.0057741, 0.0005300, -0.0059602, 0.0061323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031231
time: 2.22 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031940
time: 2.50 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0002174, 0.0017253, -0.0002245, 0.0017627, -0.0019801, 0.0019499
1: 0.9921111, 0.9970097, 0.9920171, 0.9970278, -0.0049167, 0.0049926
2: -0.0084405, -0.0025662, -0.0086045, -0.0025493, -0.0055360, 0.0057072
3: 0.0025840, 0.0046158, 0.0025766, 0.0046549, -0.0020709, 0.0020393
4: 0.0013755, 0.0050880, 0.0013672, 0.0052175, -0.0038420, 0.0037208
5: 0.0031583, 0.0079194, 0.0031408, 0.0080109, -0.0048526, 0.0047785
6: -0.0020509, 0.0000460, -0.0021078, 0.0000522, -0.0021031, 0.0021538
7: -0.0093145, -0.0058863, -0.0093804, -0.0058737, -0.0034408, 0.0034941
8: -0.0014508, 0.0093415, -0.0014898, 0.0095568, -0.0109129, 0.0107295
9: -0.0057030, 0.0005141, -0.0058226, 0.0005370, -0.0062400, 0.0063368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031940, upper bound: 0.0031231
time: 2.28 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031940, upper bound: 0.0031939
time: 2.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.59 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.59
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031231
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.59
Output dim: 1, lower bound: -0.0031231, upper bound: 0.0031940
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.59
Output dim: 1, lower bound: -0.0031940, upper bound: 0.0031231
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.59
Output dim: 1, lower bound: -0.0031940, upper bound: 0.0031939

## BFS IS instance: IS_A1_B1

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029081, upper bound: 0.0030544
time: 2.01 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029019, upper bound: 0.0029021
time: 2.03 seconds

## BFS IS instance: IS_A1_B2

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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029081, upper bound: 0.0030951
time: 2.26 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029019, upper bound: 0.0029304
time: 2.22 seconds

## BFS IS instance: IS_A2_B1

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
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029355, upper bound: 0.0030387
time: 2.02 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029306, upper bound: 0.0029019
time: 2.35 seconds

## BFS IS instance: IS_A2_B2

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029355, upper bound: 0.0030397
time: 2.19 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029306, upper bound: 0.0029019
time: 1.94 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.29 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.29
Output dim: 1, lower bound: -0.0029081, upper bound: 0.0030544
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 5.29
Output dim: 1, lower bound: -0.0029019, upper bound: 0.0029021
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.29
Output dim: 1, lower bound: -0.0029081, upper bound: 0.0030951
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 5.29
Output dim: 1, lower bound: -0.0029019, upper bound: 0.0029304
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.29
Output dim: 1, lower bound: -0.0029355, upper bound: 0.0030387
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 5.29
Output dim: 1, lower bound: -0.0029306, upper bound: 0.0029019
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.29
Output dim: 1, lower bound: -0.0029355, upper bound: 0.0030397
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 5.29
Output dim: 1, lower bound: -0.0029306, upper bound: 0.0029019

## BFS IS instance: IS_A1_B1_A1

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

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029023, upper bound: 0.0029022
time: 1.92 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029023, upper bound: 0.0029022
time: 2.00 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001476, 0.0016380, -0.0002174, 0.0017253, -0.0018729, 0.0018553
1: 0.9923315, 0.9968337, 0.9921111, 0.9970097, -0.0046782, 0.0047226
2: -0.0080573, -0.0027314, -0.0084405, -0.0025662, -0.0051570, 0.0053753
3: 0.0026570, 0.0045244, 0.0025840, 0.0046158, -0.0019588, 0.0019404
4: 0.0014565, 0.0047851, 0.0013755, 0.0050880, -0.0036314, 0.0034096
5: 0.0033294, 0.0077052, 0.0031583, 0.0079194, -0.0045900, 0.0045469
6: -0.0019178, -0.0000144, -0.0020509, 0.0000460, -0.0019639, 0.0020365
7: -0.0091603, -0.0060094, -0.0093145, -0.0058863, -0.0032740, 0.0033050
8: -0.0010701, 0.0088379, -0.0014508, 0.0093415, -0.0103144, 0.0101924
9: -0.0054234, 0.0002908, -0.0057030, 0.0005141, -0.0059375, 0.0059938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029019, upper bound: 0.0029306
time: 1.96 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029019, upper bound: 0.0029306
time: 1.96 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001959, 0.0017232, -0.0001687, 0.0016401, -0.0018360, 0.0018919
1: 0.9921165, 0.9969556, 0.9923263, 0.9968869, -0.0047705, 0.0046294
2: -0.0084312, -0.0026170, -0.0080667, -0.0026816, -0.0054165, 0.0051141
3: 0.0026064, 0.0046136, 0.0026350, 0.0045267, -0.0019202, 0.0019786
4: 0.0014004, 0.0050806, 0.0014321, 0.0047924, -0.0033921, 0.0036485
5: 0.0032109, 0.0079141, 0.0032778, 0.0077104, -0.0044995, 0.0046364
6: -0.0020476, 0.0000274, -0.0019211, 0.0000038, -0.0020515, 0.0019485
7: -0.0093107, -0.0059241, -0.0091640, -0.0059723, -0.0033384, 0.0032399
8: -0.0013338, 0.0093292, -0.0011850, 0.0088502, -0.0100876, 0.0104175
9: -0.0056962, 0.0004455, -0.0054302, 0.0003582, -0.0060544, 0.0058757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029306, upper bound: 0.0029017
time: 2.22 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029306, upper bound: 0.0029019
time: 2.01 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001959, 0.0017232, -0.0002174, 0.0017253, -0.0019213, 0.0019406
1: 0.9921165, 0.9969556, 0.9921111, 0.9970097, -0.0048932, 0.0048445
2: -0.0084312, -0.0026170, -0.0084405, -0.0025662, -0.0055099, 0.0054676
3: 0.0026064, 0.0046136, 0.0025840, 0.0046158, -0.0020094, 0.0020296
4: 0.0014004, 0.0050806, 0.0013755, 0.0050880, -0.0036876, 0.0037051
5: 0.0032109, 0.0079141, 0.0031583, 0.0079194, -0.0047085, 0.0047558
6: -0.0020476, 0.0000274, -0.0020509, 0.0000460, -0.0020936, 0.0020783
7: -0.0093107, -0.0059241, -0.0093145, -0.0058863, -0.0034244, 0.0033903
8: -0.0013338, 0.0093292, -0.0014508, 0.0093415, -0.0105736, 0.0106783
9: -0.0056962, 0.0004455, -0.0057030, 0.0005141, -0.0062103, 0.0061485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029306, upper bound: 0.0029019
time: 2.05 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0029306, upper bound: 0.0029019
time: 2.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.21 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.21
Output dim: 1, lower bound: -0.0029023, upper bound: 0.0029022
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 5.21
Output dim: 1, lower bound: -0.0029023, upper bound: 0.0029022
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.21
Output dim: 1, lower bound: -0.0029019, upper bound: 0.0029306
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 5.21
Output dim: 1, lower bound: -0.0029019, upper bound: 0.0029306
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.21
Output dim: 1, lower bound: -0.0029306, upper bound: 0.0029017
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 5.21
Output dim: 1, lower bound: -0.0029306, upper bound: 0.0029019
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.21
Output dim: 1, lower bound: -0.0029306, upper bound: 0.0029019
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 5.21
Output dim: 1, lower bound: -0.0029306, upper bound: 0.0029019

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 4.12 + 60.55 = 64.67 seconds
