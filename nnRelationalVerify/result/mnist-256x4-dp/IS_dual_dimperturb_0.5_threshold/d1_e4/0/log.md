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
Threshold: 0.00035568


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0046092, -0.0013595, -0.0046092, -0.0013595, -0.0026597, 0.0026597)
1: (-0.0047301, -0.0041476, -0.0047301, -0.0041476, -0.0004827, 0.0004827)
2: (0.0087100, 0.0128162, 0.0087100, 0.0128162, -0.0033498, 0.0033498)
3: (1.0083210, 1.0093396, 1.0083210, 1.0093396, -0.0008919, 0.0008919)
4: (-0.0036898, -0.0030584, -0.0036898, -0.0030584, -0.0005141, 0.0005141)
5: (0.0004245, 0.0029073, 0.0004245, 0.0029073, -0.0020305, 0.0020305)
6: (-0.0025736, -0.0024332, -0.0025736, -0.0024332, -0.0001350, 0.0001350)
7: (-0.0115062, -0.0056517, -0.0115062, -0.0056517, -0.0049390, 0.0049390)
8: (-0.0073817, -0.0008032, -0.0073817, -0.0008032, -0.0053593, 0.0053593)
9: (-0.0037436, -0.0006389, -0.0037436, -0.0006389, -0.0025328, 0.0025328)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.44 + 2.08 = 3.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0004813, upper bound: 0.0004813

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004588, upper bound: 0.0004693
time: 1.25 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004693, upper bound: 0.0004693
time: 1.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.50 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.50
Output dim: 3, lower bound: -0.0004588, upper bound: 0.0004693
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.50
Output dim: 3, lower bound: -0.0004693, upper bound: 0.0004693

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0046053, -0.0014679, -0.0046078, -0.0013980, -0.0026170, 0.0025513
1: -0.0047288, -0.0041712, -0.0047296, -0.0041560, -0.0004734, 0.0004595
2: 0.0087152, 0.0126743, 0.0087118, 0.0127656, -0.0032939, 0.0032070
3: 1.0083517, 1.0093310, 1.0083319, 1.0093367, -0.0008593, 0.0008734
4: -0.0036673, -0.0030593, -0.0036818, -0.0030587, -0.0004913, 0.0005051
5: 0.0004275, 0.0028240, 0.0004256, 0.0028776, -0.0019977, 0.0019472
6: -0.0025735, -0.0024348, -0.0025736, -0.0024338, -0.0001338, 0.0001326
7: -0.0113435, -0.0056569, -0.0114481, -0.0056536, -0.0047735, 0.0048751
8: -0.0071412, -0.0008127, -0.0072957, -0.0008065, -0.0051169, 0.0052639
9: -0.0037389, -0.0007559, -0.0037420, -0.0006803, -0.0024865, 0.0024150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004422, upper bound: 0.0004304
time: 1.02 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004464, upper bound: 0.0004591
time: 1.10 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0046918, -0.0014303, -0.0046061, -0.0013910, -0.0027195, 0.0025768
1: -0.0047497, -0.0041613, -0.0047293, -0.0041537, -0.0004965, 0.0004645
2: 0.0086042, 0.0127238, 0.0087139, 0.0127754, -0.0034272, 0.0032390
3: 1.0083382, 1.0093639, 1.0083302, 1.0093360, -0.0008673, 0.0009066
4: -0.0036752, -0.0030419, -0.0036834, -0.0030590, -0.0004961, 0.0005262
5: 0.0003612, 0.0028529, 0.0004268, 0.0028831, -0.0020763, 0.0019666
6: -0.0025748, -0.0024342, -0.0025735, -0.0024339, -0.0001348, 0.0001337
7: -0.0114011, -0.0055223, -0.0114560, -0.0056564, -0.0048179, 0.0050132
8: -0.0072282, -0.0006286, -0.0073147, -0.0008100, -0.0051661, 0.0054843
9: -0.0038303, -0.0007117, -0.0037403, -0.0006706, -0.0025943, 0.0024388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004536, upper bound: 0.0004304
time: 1.02 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004590, upper bound: 0.0004591
time: 1.06 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.50 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 3, lower bound: -0.0004422, upper bound: 0.0004304
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 3, lower bound: -0.0004464, upper bound: 0.0004591
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 3, lower bound: -0.0004536, upper bound: 0.0004304
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 3, lower bound: -0.0004590, upper bound: 0.0004591

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0046033, -0.0015987, -0.0046494, -0.0017177, -0.0022399, 0.0024041
1: -0.0047278, -0.0041974, -0.0047370, -0.0042249, -0.0003639, 0.0003929
2: 0.0087182, 0.0125046, 0.0086597, 0.0123494, -0.0027862, 0.0029941
3: 1.0083872, 1.0093248, 1.0084236, 1.0093455, -0.0007813, 0.0007378
4: -0.0036408, -0.0030598, -0.0036158, -0.0030508, -0.0004526, 0.0004206
5: 0.0004290, 0.0027236, 0.0003939, 0.0026324, -0.0017072, 0.0018328
6: -0.0025734, -0.0024373, -0.0025746, -0.0024385, -0.0001279, 0.0001298
7: -0.0111349, -0.0056583, -0.0109534, -0.0055646, -0.0045981, 0.0043335
8: -0.0068619, -0.0008188, -0.0065940, -0.0007219, -0.0046625, 0.0043292
9: -0.0037356, -0.0008895, -0.0037842, -0.0010183, -0.0020144, 0.0021703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004286, upper bound: 0.0004204
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004330, upper bound: 0.0004204
time: 0.93 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046053, -0.0014699, -0.0046070, -0.0014410, -0.0023405, 0.0025484
1: -0.0047288, -0.0041716, -0.0047292, -0.0041639, -0.0004079, 0.0004590
2: 0.0087153, 0.0126717, 0.0087130, 0.0127107, -0.0029169, 0.0032041
3: 1.0083523, 1.0093307, 1.0083443, 1.0093334, -0.0008560, 0.0008207
4: -0.0036669, -0.0030593, -0.0036731, -0.0030589, -0.0004909, 0.0004432
5: 0.0004275, 0.0028224, 0.0004262, 0.0028448, -0.0017842, 0.0019454
6: -0.0025735, -0.0024348, -0.0025735, -0.0024352, -0.0001339, 0.0001307
7: -0.0113400, -0.0056570, -0.0113743, -0.0056541, -0.0047414, 0.0045155
8: -0.0071369, -0.0008128, -0.0072034, -0.0008089, -0.0051121, 0.0045915
9: -0.0037388, -0.0007580, -0.0037407, -0.0007239, -0.0021558, 0.0024127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004334, upper bound: 0.0004499
time: 1.18 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004377, upper bound: 0.0004498
time: 1.08 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0046900, -0.0015595, -0.0046480, -0.0017117, -0.0023561, 0.0024304
1: -0.0047488, -0.0041884, -0.0047367, -0.0042230, -0.0004147, 0.0003983
2: 0.0086069, 0.0125550, 0.0086615, 0.0123589, -0.0029512, 0.0030267
3: 1.0083740, 1.0093578, 1.0084225, 1.0093453, -0.0007918, 0.0008027
4: -0.0036489, -0.0030424, -0.0036175, -0.0030510, -0.0004574, 0.0004499
5: 0.0003627, 0.0027537, 0.0003950, 0.0026372, -0.0017975, 0.0018529
6: -0.0025747, -0.0024367, -0.0025745, -0.0024385, -0.0001291, 0.0001300
7: -0.0111914, -0.0055237, -0.0109603, -0.0055670, -0.0046330, 0.0044661
8: -0.0069486, -0.0006343, -0.0066126, -0.0007249, -0.0047130, 0.0046694
9: -0.0038272, -0.0008468, -0.0037827, -0.0010095, -0.0021985, 0.0021948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004411, upper bound: 0.0004204
time: 0.96 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004441, upper bound: 0.0004204
time: 0.96 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046918, -0.0014324, -0.0046054, -0.0014344, -0.0024094, 0.0025739
1: -0.0047497, -0.0041616, -0.0047289, -0.0041615, -0.0004302, 0.0004640
2: 0.0086042, 0.0127211, 0.0087151, 0.0127201, -0.0030129, 0.0032362
3: 1.0083387, 1.0093637, 1.0083425, 1.0093328, -0.0008640, 0.0008522
4: -0.0036748, -0.0030419, -0.0036748, -0.0030593, -0.0004957, 0.0004594
5: 0.0003613, 0.0028513, 0.0004274, 0.0028499, -0.0018376, 0.0019648
6: -0.0025748, -0.0024343, -0.0025735, -0.0024352, -0.0001345, 0.0001319
7: -0.0113975, -0.0055223, -0.0113826, -0.0056570, -0.0047858, 0.0046019
8: -0.0072238, -0.0006287, -0.0072228, -0.0008123, -0.0051614, 0.0047726
9: -0.0038302, -0.0007138, -0.0037391, -0.0007143, -0.0022503, 0.0024366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004499, upper bound: 0.0004469
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004499, upper bound: 0.0004499
time: 1.02 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.45 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 3, lower bound: -0.0004286, upper bound: 0.0004204
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 3, lower bound: -0.0004330, upper bound: 0.0004204
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 3, lower bound: -0.0004334, upper bound: 0.0004499
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 3, lower bound: -0.0004377, upper bound: 0.0004498
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 3, lower bound: -0.0004411, upper bound: 0.0004204
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 3, lower bound: -0.0004441, upper bound: 0.0004204
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 3, lower bound: -0.0004499, upper bound: 0.0004469
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 3, lower bound: -0.0004499, upper bound: 0.0004499

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0045993, -0.0016201, -0.0046391, -0.0017806, -0.0021794, 0.0023739
1: -0.0047270, -0.0041980, -0.0047348, -0.0042266, -0.0003606, 0.0003888
2: 0.0087233, 0.0124809, 0.0086729, 0.0122772, -0.0027174, 0.0029583
3: 1.0083879, 1.0093153, 1.0084260, 1.0093195, -0.0007523, 0.0007256
4: -0.0036376, -0.0030606, -0.0036062, -0.0030528, -0.0004475, 0.0004115
5: 0.0004321, 0.0027075, 0.0004017, 0.0025849, -0.0016617, 0.0018099
6: -0.0025732, -0.0024395, -0.0025740, -0.0024447, -0.0001219, 0.0001271
7: -0.0110791, -0.0056652, -0.0107921, -0.0055816, -0.0045291, 0.0041822
8: -0.0068318, -0.0008273, -0.0065044, -0.0007439, -0.0046122, 0.0042432
9: -0.0037316, -0.0009017, -0.0037735, -0.0010541, -0.0019795, 0.0021480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004078, upper bound: 0.0003727
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004182, upper bound: 0.0004101
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0045978, -0.0016361, -0.0046948, -0.0017868, -0.0022017, 0.0024030
1: -0.0047267, -0.0041984, -0.0047355, -0.0042262, -0.0003626, 0.0003881
2: 0.0087251, 0.0124611, 0.0086168, 0.0122684, -0.0027426, 0.0029862
3: 1.0083884, 1.0093101, 1.0084035, 1.0093193, -0.0007565, 0.0007572
4: -0.0036348, -0.0030609, -0.0036050, -0.0030465, -0.0004504, 0.0004148
5: 0.0004332, 0.0026954, 0.0003603, 0.0025801, -0.0016785, 0.0018313
6: -0.0025731, -0.0024400, -0.0025814, -0.0024432, -0.0001244, 0.0001329
7: -0.0110495, -0.0056677, -0.0107997, -0.0054170, -0.0046462, 0.0042413
8: -0.0068060, -0.0008304, -0.0064907, -0.0006914, -0.0046326, 0.0042745
9: -0.0037301, -0.0009123, -0.0037926, -0.0010607, -0.0019927, 0.0021534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004112, upper bound: 0.0003723
time: 1.04 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004229, upper bound: 0.0004101
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0046011, -0.0014906, -0.0045954, -0.0014994, -0.0022762, 0.0025141
1: -0.0047279, -0.0041722, -0.0047268, -0.0041654, -0.0003993, 0.0004486
2: 0.0087206, 0.0126480, 0.0087277, 0.0126436, -0.0028411, 0.0031620
3: 1.0083530, 1.0093215, 1.0083462, 1.0093069, -0.0008274, 0.0008084
4: -0.0036638, -0.0030601, -0.0036643, -0.0030613, -0.0004841, 0.0004321
5: 0.0004307, 0.0028068, 0.0004350, 0.0028008, -0.0017355, 0.0019193
6: -0.0025733, -0.0024370, -0.0025730, -0.0024413, -0.0001281, 0.0001281
7: -0.0112849, -0.0056639, -0.0112212, -0.0056733, -0.0046715, 0.0043655
8: -0.0071077, -0.0008217, -0.0071219, -0.0008338, -0.0050344, 0.0044788
9: -0.0037346, -0.0007699, -0.0037289, -0.0007568, -0.0021033, 0.0023718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004165, upper bound: 0.0004164
time: 1.11 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004231, upper bound: 0.0004396
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0045996, -0.0015081, -0.0046426, -0.0015115, -0.0023057, 0.0025400
1: -0.0047276, -0.0041726, -0.0047288, -0.0041650, -0.0004072, 0.0004475
2: 0.0087224, 0.0126276, 0.0086785, 0.0126295, -0.0028806, 0.0031842
3: 1.0083534, 1.0093162, 1.0083225, 1.0093126, -0.0008345, 0.0008409
4: -0.0036609, -0.0030604, -0.0036621, -0.0030553, -0.0004859, 0.0004384
5: 0.0004318, 0.0027937, 0.0003999, 0.0027916, -0.0017583, 0.0019381
6: -0.0025732, -0.0024375, -0.0025803, -0.0024390, -0.0001301, 0.0001334
7: -0.0112545, -0.0056664, -0.0112212, -0.0055261, -0.0047764, 0.0044181
8: -0.0070804, -0.0008248, -0.0071004, -0.0007783, -0.0050435, 0.0045438
9: -0.0037332, -0.0007808, -0.0037518, -0.0007662, -0.0021357, 0.0023736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004208, upper bound: 0.0004164
time: 1.23 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004276, upper bound: 0.0004396
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0046857, -0.0015813, -0.0046377, -0.0017744, -0.0022923, 0.0024001
1: -0.0047480, -0.0041889, -0.0047346, -0.0042247, -0.0004061, 0.0003941
2: 0.0086123, 0.0125302, 0.0086749, 0.0122869, -0.0028742, 0.0029909
3: 1.0083748, 1.0093488, 1.0084248, 1.0093192, -0.0007627, 0.0007908
4: -0.0036456, -0.0030433, -0.0036078, -0.0030531, -0.0004523, 0.0004386
5: 0.0003659, 0.0027373, 0.0004029, 0.0025898, -0.0017491, 0.0018299
6: -0.0025745, -0.0024390, -0.0025739, -0.0024448, -0.0001230, 0.0001274
7: -0.0111347, -0.0055309, -0.0107962, -0.0055841, -0.0045632, 0.0043153
8: -0.0069183, -0.0006431, -0.0065216, -0.0007471, -0.0046625, 0.0045532
9: -0.0038230, -0.0008590, -0.0037720, -0.0010457, -0.0021442, 0.0021724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004181, upper bound: 0.0003696
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004306, upper bound: 0.0004101
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0046843, -0.0015966, -0.0046934, -0.0017810, -0.0023241, 0.0024292
1: -0.0047476, -0.0041894, -0.0047353, -0.0042243, -0.0004134, 0.0003934
2: 0.0086141, 0.0125122, 0.0086187, 0.0122776, -0.0029159, 0.0030188
3: 1.0083756, 1.0093428, 1.0084025, 1.0093188, -0.0007670, 0.0008196
4: -0.0036429, -0.0030435, -0.0036067, -0.0030468, -0.0004553, 0.0004451
5: 0.0003671, 0.0027257, 0.0003615, 0.0025846, -0.0017735, 0.0018514
6: -0.0025744, -0.0024394, -0.0025813, -0.0024432, -0.0001255, 0.0001329
7: -0.0111051, -0.0055333, -0.0108061, -0.0054194, -0.0046809, 0.0043746
8: -0.0068928, -0.0006461, -0.0065112, -0.0006945, -0.0046828, 0.0046206
9: -0.0038216, -0.0008697, -0.0037912, -0.0010513, -0.0021755, 0.0021778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004207, upper bound: 0.0003687
time: 0.94 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004337, upper bound: 0.0004101
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0046795, -0.0014924, -0.0046012, -0.0014554, -0.0023757, 0.0025091
1: -0.0047472, -0.0041631, -0.0047281, -0.0041620, -0.0004195, 0.0004553
2: 0.0086198, 0.0126533, 0.0087203, 0.0126960, -0.0029706, 0.0031580
3: 1.0083410, 1.0093379, 1.0083431, 1.0093231, -0.0008516, 0.0008251
4: -0.0036658, -0.0030444, -0.0036716, -0.0030601, -0.0004842, 0.0004526
5: 0.0003707, 0.0028061, 0.0004306, 0.0028341, -0.0018119, 0.0019157
6: -0.0025742, -0.0024403, -0.0025733, -0.0024375, -0.0001319, 0.0001261
7: -0.0112432, -0.0055427, -0.0113278, -0.0056639, -0.0046399, 0.0045319
8: -0.0071409, -0.0006546, -0.0071934, -0.0008213, -0.0050436, 0.0046954
9: -0.0038180, -0.0007472, -0.0037348, -0.0007261, -0.0022096, 0.0023824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0004300
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004396, upper bound: 0.0004365
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0047306, -0.0015012, -0.0045999, -0.0014720, -0.0024076, 0.0025394
1: -0.0047492, -0.0041627, -0.0047277, -0.0041625, -0.0004184, 0.0004625
2: 0.0085657, 0.0126397, 0.0087221, 0.0126766, -0.0030041, 0.0031976
3: 1.0083184, 1.0093414, 1.0083436, 1.0093181, -0.0008851, 0.0008299
4: -0.0036638, -0.0030381, -0.0036689, -0.0030604, -0.0004903, 0.0004559
5: 0.0003324, 0.0027991, 0.0004317, 0.0028216, -0.0018357, 0.0019388
6: -0.0025816, -0.0024380, -0.0025732, -0.0024380, -0.0001379, 0.0001282
7: -0.0112439, -0.0053917, -0.0112953, -0.0056662, -0.0047011, 0.0046486
8: -0.0071194, -0.0005988, -0.0071668, -0.0008241, -0.0051076, 0.0047198
9: -0.0038402, -0.0007568, -0.0037335, -0.0007375, -0.0022169, 0.0024137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0004329
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004396, upper bound: 0.0004396
time: 1.03 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.58 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004078, upper bound: 0.0003727
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004182, upper bound: 0.0004101
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004112, upper bound: 0.0003723
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004229, upper bound: 0.0004101
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004165, upper bound: 0.0004164
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004231, upper bound: 0.0004396
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004208, upper bound: 0.0004164
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004276, upper bound: 0.0004396
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004181, upper bound: 0.0003696
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004306, upper bound: 0.0004101
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004207, upper bound: 0.0003687
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004337, upper bound: 0.0004101
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0004300
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004396, upper bound: 0.0004365
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0004329
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 3, lower bound: -0.0004396, upper bound: 0.0004396

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0045981, -0.0016216, -0.0046305, -0.0017891, -0.0021687, 0.0023600
1: -0.0047044, -0.0041927, -0.0047264, -0.0042272, -0.0003359, 0.0003842
2: 0.0087401, 0.0124888, 0.0086832, 0.0122710, -0.0026896, 0.0029493
3: 1.0083668, 1.0092505, 1.0084288, 1.0092907, -0.0007460, 0.0006604
4: -0.0036400, -0.0030665, -0.0036054, -0.0030552, -0.0004465, 0.0004039
5: 0.0004343, 0.0027072, 0.0004083, 0.0025787, -0.0016525, 0.0018003
6: -0.0025770, -0.0024469, -0.0025733, -0.0024483, -0.0001228, 0.0001194
7: -0.0110065, -0.0055851, -0.0107452, -0.0056009, -0.0044308, 0.0042134
8: -0.0068648, -0.0009127, -0.0064967, -0.0007765, -0.0046015, 0.0041409
9: -0.0036762, -0.0008824, -0.0037525, -0.0010577, -0.0019156, 0.0021415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004079, upper bound: 0.0003727
time: 1.02 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004079, upper bound: 0.0003727
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045945, -0.0016271, -0.0046391, -0.0017806, -0.0021744, 0.0023678
1: -0.0047230, -0.0041982, -0.0047348, -0.0042266, -0.0003490, 0.0003886
2: 0.0087291, 0.0124769, 0.0086729, 0.0122772, -0.0027070, 0.0029520
3: 1.0083889, 1.0093035, 1.0084260, 1.0093195, -0.0007511, 0.0006918
4: -0.0036373, -0.0030619, -0.0036062, -0.0030528, -0.0004468, 0.0004085
5: 0.0004358, 0.0027025, 0.0004017, 0.0025849, -0.0016578, 0.0018055
6: -0.0025729, -0.0024413, -0.0025740, -0.0024447, -0.0001216, 0.0001242
7: -0.0110410, -0.0056751, -0.0107921, -0.0055816, -0.0045019, 0.0041725
8: -0.0068284, -0.0008453, -0.0065044, -0.0007439, -0.0046067, 0.0042025
9: -0.0037203, -0.0009032, -0.0037735, -0.0010541, -0.0019529, 0.0021465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0003953
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0004101
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0045966, -0.0016374, -0.0046863, -0.0017955, -0.0021909, 0.0023891
1: -0.0047040, -0.0041932, -0.0047271, -0.0042268, -0.0003379, 0.0003838
2: 0.0087421, 0.0124694, 0.0086268, 0.0122621, -0.0027146, 0.0029773
3: 1.0083673, 1.0092447, 1.0084058, 1.0092902, -0.0007501, 0.0006905
4: -0.0036372, -0.0030668, -0.0036041, -0.0030488, -0.0004494, 0.0004071
5: 0.0004355, 0.0026952, 0.0003668, 0.0025739, -0.0016693, 0.0018218
6: -0.0025769, -0.0024475, -0.0025807, -0.0024467, -0.0001253, 0.0001252
7: -0.0109754, -0.0055877, -0.0107519, -0.0054362, -0.0045467, 0.0042720
8: -0.0068388, -0.0009159, -0.0064824, -0.0007232, -0.0046223, 0.0041719
9: -0.0036746, -0.0008931, -0.0037719, -0.0010645, -0.0019286, 0.0021466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004112, upper bound: 0.0003723
time: 1.03 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004112, upper bound: 0.0003723
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045929, -0.0016431, -0.0046948, -0.0017868, -0.0021965, 0.0023966
1: -0.0047226, -0.0041987, -0.0047355, -0.0042262, -0.0003508, 0.0003879
2: 0.0087311, 0.0124571, 0.0086168, 0.0122684, -0.0027320, 0.0029800
3: 1.0083892, 1.0092986, 1.0084035, 1.0093193, -0.0007553, 0.0007197
4: -0.0036345, -0.0030622, -0.0036050, -0.0030465, -0.0004497, 0.0004118
5: 0.0004370, 0.0026905, 0.0003603, 0.0025801, -0.0016745, 0.0018268
6: -0.0025728, -0.0024418, -0.0025814, -0.0024432, -0.0001241, 0.0001293
7: -0.0110124, -0.0056778, -0.0107997, -0.0054170, -0.0046073, 0.0042315
8: -0.0068026, -0.0008486, -0.0064907, -0.0006914, -0.0046271, 0.0042326
9: -0.0037188, -0.0009138, -0.0037926, -0.0010607, -0.0019653, 0.0021519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003899, upper bound: 0.0003952
time: 0.96 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003899, upper bound: 0.0004101
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0046010, -0.0014930, -0.0045860, -0.0015080, -0.0022661, 0.0024997
1: -0.0047060, -0.0041666, -0.0047183, -0.0041660, -0.0003767, 0.0004447
2: 0.0087355, 0.0126546, 0.0087388, 0.0126373, -0.0028183, 0.0031535
3: 1.0083317, 1.0092590, 1.0083486, 1.0092779, -0.0008251, 0.0007473
4: -0.0036660, -0.0030656, -0.0036636, -0.0030637, -0.0004834, 0.0004258
5: 0.0004320, 0.0028059, 0.0004421, 0.0027946, -0.0017270, 0.0019091
6: -0.0025771, -0.0024441, -0.0025722, -0.0024449, -0.0001282, 0.0001205
7: -0.0112140, -0.0055840, -0.0111767, -0.0056941, -0.0045778, 0.0043958
8: -0.0071383, -0.0009024, -0.0071149, -0.0008666, -0.0050258, 0.0043898
9: -0.0036818, -0.0007515, -0.0037079, -0.0007600, -0.0020462, 0.0023655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0004011
time: 0.97 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0003981
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045962, -0.0014978, -0.0045954, -0.0014994, -0.0022716, 0.0025131
1: -0.0047239, -0.0041724, -0.0047268, -0.0041654, -0.0003908, 0.0004484
2: 0.0087264, 0.0126438, 0.0087277, 0.0126436, -0.0028350, 0.0031610
3: 1.0083541, 1.0093095, 1.0083462, 1.0093069, -0.0008265, 0.0007797
4: -0.0036634, -0.0030614, -0.0036643, -0.0030613, -0.0004837, 0.0004307
5: 0.0004344, 0.0028018, 0.0004350, 0.0028008, -0.0017320, 0.0019186
6: -0.0025730, -0.0024388, -0.0025730, -0.0024413, -0.0001278, 0.0001246
7: -0.0112470, -0.0056739, -0.0112212, -0.0056733, -0.0046403, 0.0043566
8: -0.0071043, -0.0008397, -0.0071219, -0.0008338, -0.0050310, 0.0044535
9: -0.0037234, -0.0007714, -0.0037289, -0.0007568, -0.0020855, 0.0023703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004045, upper bound: 0.0004337
time: 1.03 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004045, upper bound: 0.0004202
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0045995, -0.0015102, -0.0046331, -0.0015202, -0.0022954, 0.0025249
1: -0.0047056, -0.0041671, -0.0047204, -0.0041656, -0.0003845, 0.0004442
2: 0.0087375, 0.0126348, 0.0086898, 0.0126231, -0.0028576, 0.0031755
3: 1.0083321, 1.0092533, 1.0083249, 1.0092833, -0.0008309, 0.0007779
4: -0.0036632, -0.0030659, -0.0036614, -0.0030576, -0.0004855, 0.0004320
5: 0.0004332, 0.0027929, 0.0004071, 0.0027853, -0.0017498, 0.0019275
6: -0.0025770, -0.0024447, -0.0025796, -0.0024426, -0.0001303, 0.0001259
7: -0.0111796, -0.0055866, -0.0111725, -0.0055463, -0.0046834, 0.0044483
8: -0.0071123, -0.0009057, -0.0070930, -0.0008107, -0.0050380, 0.0044548
9: -0.0036802, -0.0007621, -0.0037310, -0.0007695, -0.0020783, 0.0023687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0004011
time: 0.94 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0003981
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045946, -0.0015153, -0.0046426, -0.0015115, -0.0023009, 0.0025368
1: -0.0047235, -0.0041729, -0.0047288, -0.0041650, -0.0003985, 0.0004473
2: 0.0087284, 0.0126235, 0.0086785, 0.0126295, -0.0028745, 0.0031818
3: 1.0083543, 1.0093048, 1.0083225, 1.0093126, -0.0008336, 0.0008079
4: -0.0036605, -0.0030617, -0.0036621, -0.0030553, -0.0004856, 0.0004369
5: 0.0004356, 0.0027886, 0.0003999, 0.0027916, -0.0017547, 0.0019360
6: -0.0025729, -0.0024393, -0.0025803, -0.0024390, -0.0001298, 0.0001297
7: -0.0112174, -0.0056766, -0.0112212, -0.0055261, -0.0047436, 0.0044089
8: -0.0070770, -0.0008429, -0.0071004, -0.0007783, -0.0050402, 0.0045187
9: -0.0037218, -0.0007823, -0.0037518, -0.0007662, -0.0021171, 0.0023721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004077, upper bound: 0.0004337
time: 1.06 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004077, upper bound: 0.0004202
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0046809, -0.0015827, -0.0046290, -0.0017828, -0.0022793, 0.0023864
1: -0.0047263, -0.0041834, -0.0047261, -0.0042252, -0.0003837, 0.0003892
2: 0.0086311, 0.0125388, 0.0086851, 0.0122807, -0.0028459, 0.0029821
3: 1.0083561, 1.0092852, 1.0084275, 1.0092902, -0.0007535, 0.0007296
4: -0.0036482, -0.0030490, -0.0036070, -0.0030555, -0.0004513, 0.0004316
5: 0.0003707, 0.0027371, 0.0004094, 0.0025837, -0.0017382, 0.0018205
6: -0.0025779, -0.0024460, -0.0025732, -0.0024483, -0.0001234, 0.0001197
7: -0.0110636, -0.0054670, -0.0107509, -0.0056034, -0.0044676, 0.0043338
8: -0.0069529, -0.0007249, -0.0065137, -0.0007796, -0.0046520, 0.0044579
9: -0.0037694, -0.0008389, -0.0037511, -0.0010493, -0.0020836, 0.0021656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003912, upper bound: 0.0003696
time: 0.94 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003912, upper bound: 0.0003696
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0046808, -0.0015884, -0.0046377, -0.0017744, -0.0022872, 0.0023939
1: -0.0047440, -0.0041891, -0.0047346, -0.0042247, -0.0003980, 0.0003939
2: 0.0086183, 0.0125260, 0.0086749, 0.0122869, -0.0028677, 0.0029845
3: 1.0083758, 1.0093364, 1.0084248, 1.0093192, -0.0007616, 0.0007630
4: -0.0036452, -0.0030446, -0.0036078, -0.0030531, -0.0004516, 0.0004371
5: 0.0003697, 0.0027323, 0.0004029, 0.0025898, -0.0017452, 0.0018254
6: -0.0025742, -0.0024407, -0.0025739, -0.0024448, -0.0001227, 0.0001238
7: -0.0110953, -0.0055411, -0.0107962, -0.0055841, -0.0045324, 0.0043053
8: -0.0069149, -0.0006609, -0.0065216, -0.0007471, -0.0046569, 0.0045286
9: -0.0038118, -0.0008605, -0.0037720, -0.0010457, -0.0021263, 0.0021709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003971, upper bound: 0.0003933
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003971, upper bound: 0.0004101
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0046793, -0.0015980, -0.0046848, -0.0017897, -0.0023106, 0.0024156
1: -0.0047260, -0.0041839, -0.0047269, -0.0042249, -0.0003907, 0.0003889
2: 0.0086331, 0.0125211, 0.0086287, 0.0122713, -0.0028872, 0.0030102
3: 1.0083567, 1.0092793, 1.0084050, 1.0092897, -0.0007577, 0.0007562
4: -0.0036456, -0.0030493, -0.0036058, -0.0030491, -0.0004543, 0.0004380
5: 0.0003720, 0.0027255, 0.0003680, 0.0025783, -0.0017623, 0.0018421
6: -0.0025778, -0.0024466, -0.0025807, -0.0024468, -0.0001259, 0.0001254
7: -0.0110338, -0.0054696, -0.0107592, -0.0054387, -0.0045837, 0.0043928
8: -0.0069270, -0.0007282, -0.0065026, -0.0007262, -0.0046727, 0.0045252
9: -0.0037678, -0.0008497, -0.0037704, -0.0010551, -0.0021149, 0.0021709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003930, upper bound: 0.0003687
time: 0.93 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003930, upper bound: 0.0003687
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046793, -0.0016036, -0.0046934, -0.0017810, -0.0023190, 0.0024228
1: -0.0047437, -0.0041896, -0.0047353, -0.0042243, -0.0004053, 0.0003933
2: 0.0086201, 0.0125082, 0.0086187, 0.0122776, -0.0029094, 0.0030126
3: 1.0083764, 1.0093303, 1.0084025, 1.0093188, -0.0007659, 0.0007879
4: -0.0036426, -0.0030448, -0.0036067, -0.0030468, -0.0004546, 0.0004436
5: 0.0003708, 0.0027208, 0.0003615, 0.0025846, -0.0017696, 0.0018468
6: -0.0025741, -0.0024412, -0.0025813, -0.0024432, -0.0001252, 0.0001291
7: -0.0110660, -0.0055436, -0.0108061, -0.0054194, -0.0046410, 0.0043647
8: -0.0068894, -0.0006638, -0.0065112, -0.0006945, -0.0046773, 0.0045955
9: -0.0038104, -0.0008711, -0.0037912, -0.0010513, -0.0021573, 0.0021763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003976, upper bound: 0.0003930
time: 1.03 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003976, upper bound: 0.0004101
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0046697, -0.0015012, -0.0046013, -0.0014572, -0.0023613, 0.0024995
1: -0.0047389, -0.0041637, -0.0047062, -0.0041561, -0.0004166, 0.0004321
2: 0.0086314, 0.0126468, 0.0087351, 0.0127039, -0.0029623, 0.0031337
3: 1.0083433, 1.0093085, 1.0083212, 1.0092611, -0.0007906, 0.0008232
4: -0.0036652, -0.0030468, -0.0036741, -0.0030655, -0.0004774, 0.0004522
5: 0.0003781, 0.0027998, 0.0004318, 0.0028336, -0.0018019, 0.0019074
6: -0.0025735, -0.0024440, -0.0025771, -0.0024444, -0.0001244, 0.0001267
7: -0.0111968, -0.0055643, -0.0112605, -0.0055837, -0.0046663, 0.0044379
8: -0.0071341, -0.0006879, -0.0072263, -0.0009015, -0.0049494, 0.0046896
9: -0.0037967, -0.0007502, -0.0036822, -0.0007064, -0.0022048, 0.0023227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004181
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004118
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046795, -0.0014924, -0.0045963, -0.0014625, -0.0023707, 0.0025039
1: -0.0047472, -0.0041631, -0.0047240, -0.0041623, -0.0004194, 0.0004464
2: 0.0086198, 0.0126533, 0.0087262, 0.0126921, -0.0029675, 0.0031514
3: 1.0083410, 1.0093379, 1.0083441, 1.0093113, -0.0008240, 0.0008241
4: -0.0036658, -0.0030444, -0.0036713, -0.0030614, -0.0004827, 0.0004522
5: 0.0003707, 0.0028061, 0.0004343, 0.0028291, -0.0018084, 0.0019117
6: -0.0025742, -0.0024403, -0.0025730, -0.0024393, -0.0001282, 0.0001258
7: -0.0112432, -0.0055427, -0.0112882, -0.0056739, -0.0046301, 0.0044970
8: -0.0071409, -0.0006546, -0.0071899, -0.0008392, -0.0050181, 0.0046918
9: -0.0038180, -0.0007472, -0.0037236, -0.0007277, -0.0022080, 0.0023640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004101, upper bound: 0.0004306
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004101, upper bound: 0.0004181
time: 1.31 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0047209, -0.0015102, -0.0045998, -0.0014732, -0.0023930, 0.0025295
1: -0.0047408, -0.0041633, -0.0047058, -0.0041566, -0.0004163, 0.0004392
2: 0.0085772, 0.0126331, 0.0087370, 0.0126847, -0.0029961, 0.0031730
3: 1.0083208, 1.0093120, 1.0083218, 1.0092552, -0.0008223, 0.0008262
4: -0.0036630, -0.0030404, -0.0036713, -0.0030658, -0.0004835, 0.0004557
5: 0.0003398, 0.0027927, 0.0004329, 0.0028214, -0.0018256, 0.0019305
6: -0.0025809, -0.0024417, -0.0025770, -0.0024451, -0.0001305, 0.0001288
7: -0.0111969, -0.0054130, -0.0112284, -0.0055863, -0.0047273, 0.0045538
8: -0.0071122, -0.0006310, -0.0071994, -0.0009046, -0.0050132, 0.0047159
9: -0.0038193, -0.0007601, -0.0036807, -0.0007177, -0.0022140, 0.0023539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004207
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004138
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0047306, -0.0015012, -0.0045949, -0.0014790, -0.0024040, 0.0025341
1: -0.0047492, -0.0041627, -0.0047237, -0.0041627, -0.0004182, 0.0004535
2: 0.0085657, 0.0126397, 0.0087280, 0.0126727, -0.0030016, 0.0031910
3: 1.0083184, 1.0093414, 1.0083445, 1.0093064, -0.0008534, 0.0008290
4: -0.0036638, -0.0030381, -0.0036685, -0.0030617, -0.0004888, 0.0004556
5: 0.0003324, 0.0027991, 0.0004354, 0.0028166, -0.0018333, 0.0019348
6: -0.0025816, -0.0024380, -0.0025729, -0.0024397, -0.0001340, 0.0001279
7: -0.0112439, -0.0053917, -0.0112572, -0.0056763, -0.0046913, 0.0046097
8: -0.0071194, -0.0005988, -0.0071633, -0.0008421, -0.0050818, 0.0047162
9: -0.0038402, -0.0007568, -0.0037223, -0.0007390, -0.0022153, 0.0023950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004101, upper bound: 0.0004337
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004101, upper bound: 0.0004202
time: 1.26 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.79 seconds
IS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0004079, upper bound: 0.0003727
IS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0004079, upper bound: 0.0003727
IS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0003953
IS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0004101
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0004112, upper bound: 0.0003723
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0004112, upper bound: 0.0003723
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003899, upper bound: 0.0003952
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003899, upper bound: 0.0004101
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0004011
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0003981
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0004045, upper bound: 0.0004337
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0004045, upper bound: 0.0004202
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0004011
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0003981
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0004077, upper bound: 0.0004337
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0004077, upper bound: 0.0004202
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003912, upper bound: 0.0003696
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003912, upper bound: 0.0003696
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003971, upper bound: 0.0003933
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003971, upper bound: 0.0004101
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003930, upper bound: 0.0003687
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003930, upper bound: 0.0003687
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003976, upper bound: 0.0003930
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003976, upper bound: 0.0004101
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004181
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004118
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0004101, upper bound: 0.0004306
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0004101, upper bound: 0.0004181
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004207
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004138
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0004101, upper bound: 0.0004337
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 3, lower bound: -0.0004101, upper bound: 0.0004202

## BFS IS instance: IS_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0045981, -0.0016216, -0.0046285, -0.0018583, -0.0021055, 0.0023578
1: -0.0047044, -0.0041927, -0.0047258, -0.0042416, -0.0003218, 0.0003836
2: 0.0087401, 0.0124888, 0.0086859, 0.0121803, -0.0026070, 0.0029465
3: 1.0083668, 1.0092505, 1.0084473, 1.0092860, -0.0007420, 0.0006408
4: -0.0036400, -0.0030665, -0.0035910, -0.0030556, -0.0004460, 0.0003906
5: 0.0004343, 0.0027072, 0.0004099, 0.0025256, -0.0016040, 0.0017986
6: -0.0025770, -0.0024469, -0.0025732, -0.0024493, -0.0001218, 0.0001193
7: -0.0110065, -0.0055851, -0.0106427, -0.0056043, -0.0044272, 0.0041155
8: -0.0068648, -0.0009127, -0.0063415, -0.0007808, -0.0045968, 0.0040000
9: -0.0036762, -0.0008824, -0.0037505, -0.0011326, -0.0018466, 0.0021393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0003727
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0003727
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0045981, -0.0016216, -0.0047075, -0.0018219, -0.0021342, 0.0024207
1: -0.0047044, -0.0041927, -0.0047444, -0.0042334, -0.0003316, 0.0004051
2: 0.0087401, 0.0124888, 0.0085839, 0.0122300, -0.0026449, 0.0030314
3: 1.0083668, 1.0092505, 1.0084360, 1.0093144, -0.0007791, 0.0006555
4: -0.0036400, -0.0030665, -0.0035990, -0.0030396, -0.0004603, 0.0003969
5: 0.0004343, 0.0027072, 0.0003494, 0.0025538, -0.0016260, 0.0018471
6: -0.0025770, -0.0024469, -0.0025744, -0.0024485, -0.0001224, 0.0001204
7: -0.0110065, -0.0055851, -0.0106926, -0.0054846, -0.0045220, 0.0041552
8: -0.0068648, -0.0009127, -0.0064273, -0.0006112, -0.0047538, 0.0040693
9: -0.0036762, -0.0008824, -0.0038327, -0.0010918, -0.0018838, 0.0022210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0003727
time: 0.88 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0003727
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045945, -0.0016271, -0.0046377, -0.0017816, -0.0021708, 0.0023634
1: -0.0047230, -0.0041982, -0.0047123, -0.0042209, -0.0003617, 0.0003667
2: 0.0087291, 0.0124769, 0.0086896, 0.0122862, -0.0027139, 0.0029328
3: 1.0083889, 1.0093035, 1.0084065, 1.0092545, -0.0006909, 0.0007339
4: -0.0036373, -0.0030619, -0.0036086, -0.0030586, -0.0004408, 0.0004120
5: 0.0004358, 0.0027025, 0.0004040, 0.0025850, -0.0016560, 0.0018012
6: -0.0025729, -0.0024413, -0.0025775, -0.0024520, -0.0001148, 0.0001295
7: -0.0110410, -0.0056751, -0.0107231, -0.0055107, -0.0045643, 0.0041016
8: -0.0068284, -0.0008453, -0.0065361, -0.0008276, -0.0045223, 0.0042540
9: -0.0037203, -0.0009032, -0.0037185, -0.0010359, -0.0019852, 0.0020923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0003953
time: 1.11 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0003953
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045945, -0.0016271, -0.0046345, -0.0017868, -0.0021697, 0.0023631
1: -0.0047230, -0.0041982, -0.0047308, -0.0042268, -0.0003488, 0.0003776
2: 0.0087291, 0.0124769, 0.0086786, 0.0122740, -0.0027017, 0.0029429
3: 1.0083889, 1.0093035, 1.0084270, 1.0093070, -0.0007201, 0.0006904
4: -0.0036373, -0.0030619, -0.0036058, -0.0030541, -0.0004439, 0.0004080
5: 0.0004358, 0.0027025, 0.0004053, 0.0025805, -0.0016542, 0.0018019
6: -0.0025729, -0.0024413, -0.0025737, -0.0024465, -0.0001189, 0.0001238
7: -0.0110410, -0.0056751, -0.0107555, -0.0055911, -0.0044924, 0.0041483
8: -0.0068284, -0.0008453, -0.0065006, -0.0007612, -0.0045658, 0.0041983
9: -0.0037203, -0.0009032, -0.0037624, -0.0010558, -0.0019511, 0.0021205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0004094
time: 1.00 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0004101
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0045966, -0.0016374, -0.0046841, -0.0018644, -0.0021274, 0.0023868
1: -0.0047040, -0.0041932, -0.0047266, -0.0042413, -0.0003238, 0.0003832
2: 0.0087421, 0.0124694, 0.0086296, 0.0121719, -0.0026316, 0.0029744
3: 1.0083673, 1.0092447, 1.0084245, 1.0092853, -0.0007460, 0.0006722
4: -0.0036372, -0.0030668, -0.0035894, -0.0030492, -0.0004489, 0.0003939
5: 0.0004355, 0.0026952, 0.0003685, 0.0025210, -0.0016205, 0.0018200
6: -0.0025769, -0.0024475, -0.0025806, -0.0024477, -0.0001243, 0.0001251
7: -0.0109754, -0.0055877, -0.0106467, -0.0054399, -0.0045429, 0.0041737
8: -0.0068388, -0.0009159, -0.0063263, -0.0007276, -0.0046175, 0.0040301
9: -0.0036746, -0.0008931, -0.0037697, -0.0011398, -0.0018593, 0.0021443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0003723
time: 0.97 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0003723
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0045966, -0.0016374, -0.0047666, -0.0018273, -0.0021569, 0.0024493
1: -0.0047040, -0.0041932, -0.0047452, -0.0042330, -0.0003335, 0.0004049
2: 0.0087421, 0.0124694, 0.0085218, 0.0122219, -0.0026707, 0.0030611
3: 1.0083673, 1.0092447, 1.0084136, 1.0093130, -0.0007829, 0.0006833
4: -0.0036372, -0.0030668, -0.0035978, -0.0030322, -0.0004635, 0.0004002
5: 0.0004355, 0.0026952, 0.0003052, 0.0025495, -0.0016432, 0.0018684
6: -0.0025769, -0.0024475, -0.0025820, -0.0024469, -0.0001250, 0.0001263
7: -0.0109754, -0.0055877, -0.0107034, -0.0053168, -0.0046311, 0.0042154
8: -0.0068388, -0.0009159, -0.0064174, -0.0005469, -0.0047790, 0.0041001
9: -0.0036746, -0.0008931, -0.0038576, -0.0010963, -0.0018968, 0.0022288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0003723
time: 1.05 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0003723
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045929, -0.0016431, -0.0046902, -0.0017898, -0.0021916, 0.0023907
1: -0.0047226, -0.0041987, -0.0047130, -0.0042208, -0.0003632, 0.0003661
2: 0.0087311, 0.0124571, 0.0086350, 0.0122753, -0.0027375, 0.0029598
3: 1.0083892, 1.0092986, 1.0083857, 1.0092523, -0.0006939, 0.0007622
4: -0.0036345, -0.0030622, -0.0036072, -0.0030522, -0.0004437, 0.0004151
5: 0.0004370, 0.0026905, 0.0003649, 0.0025787, -0.0016717, 0.0018215
6: -0.0025728, -0.0024418, -0.0025840, -0.0024505, -0.0001172, 0.0001342
7: -0.0110124, -0.0056778, -0.0107247, -0.0053561, -0.0046579, 0.0041562
8: -0.0068026, -0.0008486, -0.0065209, -0.0007739, -0.0045429, 0.0042831
9: -0.0037188, -0.0009138, -0.0037381, -0.0010437, -0.0019971, 0.0020974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003899, upper bound: 0.0003952
time: 1.10 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003899, upper bound: 0.0003952
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045929, -0.0016431, -0.0046901, -0.0017929, -0.0021920, 0.0023918
1: -0.0047226, -0.0041987, -0.0047315, -0.0042265, -0.0003506, 0.0003772
2: 0.0087311, 0.0124571, 0.0086225, 0.0122654, -0.0027269, 0.0029710
3: 1.0083892, 1.0092986, 1.0084044, 1.0093067, -0.0007234, 0.0007184
4: -0.0036345, -0.0030622, -0.0036045, -0.0030478, -0.0004468, 0.0004112
5: 0.0004370, 0.0026905, 0.0003640, 0.0025759, -0.0016711, 0.0018232
6: -0.0025728, -0.0024418, -0.0025811, -0.0024449, -0.0001214, 0.0001290
7: -0.0110124, -0.0056778, -0.0107643, -0.0054268, -0.0045975, 0.0042091
8: -0.0068026, -0.0008486, -0.0064868, -0.0007090, -0.0045861, 0.0042287
9: -0.0037188, -0.0009138, -0.0037815, -0.0010624, -0.0019635, 0.0021257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003899, upper bound: 0.0004094
time: 1.12 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003899, upper bound: 0.0004101
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0046428, -0.0018112, -0.0045860, -0.0015080, -0.0025164, 0.0021497
1: -0.0047132, -0.0042342, -0.0047183, -0.0041660, -0.0004091, 0.0003664
2: 0.0086831, 0.0122407, 0.0087388, 0.0126373, -0.0031425, 0.0026951
3: 1.0084243, 1.0092663, 1.0083486, 1.0092779, -0.0007279, 0.0007503
4: -0.0036002, -0.0030576, -0.0036636, -0.0030637, -0.0004100, 0.0004757
5: 0.0004001, 0.0025617, 0.0004421, 0.0027946, -0.0019193, 0.0016404
6: -0.0025778, -0.0024490, -0.0025722, -0.0024449, -0.0001253, 0.0001165
7: -0.0107217, -0.0055021, -0.0111767, -0.0056941, -0.0040578, 0.0047420
8: -0.0064392, -0.0008170, -0.0071149, -0.0008666, -0.0042394, 0.0049025
9: -0.0037236, -0.0010870, -0.0037079, -0.0007600, -0.0022805, 0.0019829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0004010
time: 0.93 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0004011
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0046004, -0.0015329, -0.0045860, -0.0015080, -0.0022645, 0.0022320
1: -0.0047056, -0.0041742, -0.0047183, -0.0041660, -0.0003764, 0.0003806
2: 0.0087365, 0.0126031, 0.0087388, 0.0126373, -0.0028172, 0.0027886
3: 1.0083437, 1.0092559, 1.0083486, 1.0092779, -0.0007715, 0.0007446
4: -0.0036578, -0.0030658, -0.0036636, -0.0030637, -0.0004233, 0.0004256
5: 0.0004325, 0.0027753, 0.0004421, 0.0027946, -0.0017261, 0.0017024
6: -0.0025771, -0.0024454, -0.0025722, -0.0024449, -0.0001272, 0.0001216
7: -0.0111451, -0.0055844, -0.0111767, -0.0056941, -0.0042405, 0.0043755
8: -0.0070503, -0.0009045, -0.0071149, -0.0008666, -0.0043712, 0.0043879
9: -0.0036806, -0.0007928, -0.0037079, -0.0007600, -0.0020452, 0.0020438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0003949
time: 0.98 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0003981
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0046393, -0.0018158, -0.0045954, -0.0014994, -0.0025237, 0.0021619
1: -0.0047317, -0.0042401, -0.0047268, -0.0041654, -0.0004215, 0.0003691
2: 0.0086724, 0.0122292, 0.0087277, 0.0126436, -0.0031603, 0.0027008
3: 1.0084442, 1.0093193, 1.0083462, 1.0093069, -0.0007289, 0.0007848
4: -0.0035975, -0.0030531, -0.0036643, -0.0030613, -0.0004100, 0.0004799
5: 0.0004016, 0.0025578, 0.0004350, 0.0028008, -0.0019256, 0.0016490
6: -0.0025740, -0.0024435, -0.0025730, -0.0024413, -0.0001246, 0.0001213
7: -0.0107565, -0.0055833, -0.0112212, -0.0056733, -0.0041250, 0.0047065
8: -0.0064029, -0.0007510, -0.0071219, -0.0008338, -0.0042425, 0.0049572
9: -0.0037673, -0.0011076, -0.0037289, -0.0007568, -0.0023139, 0.0019864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004217
time: 1.09 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004337
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0045955, -0.0015378, -0.0045954, -0.0014994, -0.0022703, 0.0022412
1: -0.0047235, -0.0041801, -0.0047268, -0.0041654, -0.0003905, 0.0003833
2: 0.0087274, 0.0125920, 0.0087277, 0.0126436, -0.0028339, 0.0027925
3: 1.0083663, 1.0093064, 1.0083462, 1.0093069, -0.0007732, 0.0007769
4: -0.0036551, -0.0030616, -0.0036643, -0.0030613, -0.0004231, 0.0004305
5: 0.0004349, 0.0027712, 0.0004350, 0.0028008, -0.0017312, 0.0017088
6: -0.0025730, -0.0024402, -0.0025730, -0.0024413, -0.0001267, 0.0001254
7: -0.0111759, -0.0056744, -0.0112212, -0.0056733, -0.0043018, 0.0043368
8: -0.0070164, -0.0008418, -0.0071219, -0.0008338, -0.0043721, 0.0044516
9: -0.0037222, -0.0008125, -0.0037289, -0.0007568, -0.0020845, 0.0020458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004138
time: 1.15 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004202
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0046413, -0.0018258, -0.0046331, -0.0015202, -0.0025405, 0.0021748
1: -0.0047128, -0.0042347, -0.0047204, -0.0041656, -0.0004110, 0.0003656
2: 0.0086850, 0.0122234, 0.0086898, 0.0126231, -0.0031697, 0.0027154
3: 1.0084249, 1.0092599, 1.0083249, 1.0092833, -0.0007337, 0.0007817
4: -0.0035977, -0.0030578, -0.0036614, -0.0030576, -0.0004118, 0.0004793
5: 0.0004013, 0.0025507, 0.0004071, 0.0027853, -0.0019375, 0.0016586
6: -0.0025777, -0.0024495, -0.0025796, -0.0024426, -0.0001275, 0.0001221
7: -0.0106931, -0.0055046, -0.0111725, -0.0055463, -0.0041650, 0.0048039
8: -0.0064146, -0.0008201, -0.0070930, -0.0008107, -0.0042509, 0.0049352
9: -0.0037221, -0.0010975, -0.0037310, -0.0007695, -0.0022942, 0.0019861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0004010
time: 1.03 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0004011
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0045989, -0.0015514, -0.0046331, -0.0015202, -0.0022938, 0.0022619
1: -0.0047053, -0.0041747, -0.0047204, -0.0041656, -0.0003841, 0.0003795
2: 0.0087385, 0.0125821, 0.0086898, 0.0126231, -0.0028566, 0.0028159
3: 1.0083443, 1.0092503, 1.0083249, 1.0092833, -0.0007784, 0.0007752
4: -0.0036549, -0.0030661, -0.0036614, -0.0030576, -0.0004258, 0.0004319
5: 0.0004337, 0.0027614, 0.0004071, 0.0027853, -0.0017489, 0.0017243
6: -0.0025770, -0.0024461, -0.0025796, -0.0024426, -0.0001292, 0.0001276
7: -0.0111079, -0.0055870, -0.0111725, -0.0055463, -0.0043656, 0.0044280
8: -0.0070232, -0.0009078, -0.0070930, -0.0008107, -0.0043885, 0.0044529
9: -0.0036791, -0.0008041, -0.0037310, -0.0007695, -0.0020773, 0.0020470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0003950
time: 1.10 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0003981
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0046378, -0.0018301, -0.0046426, -0.0015115, -0.0025478, 0.0021855
1: -0.0047314, -0.0042406, -0.0047288, -0.0041650, -0.0004231, 0.0003678
2: 0.0086743, 0.0122121, 0.0086785, 0.0126295, -0.0031874, 0.0027205
3: 1.0084447, 1.0093133, 1.0083225, 1.0093126, -0.0007359, 0.0008102
4: -0.0035949, -0.0030534, -0.0036621, -0.0030553, -0.0004116, 0.0004834
5: 0.0004027, 0.0025469, 0.0003999, 0.0027916, -0.0019438, 0.0016661
6: -0.0025739, -0.0024439, -0.0025803, -0.0024390, -0.0001269, 0.0001263
7: -0.0107306, -0.0055858, -0.0112212, -0.0055261, -0.0042284, 0.0047688
8: -0.0063789, -0.0007542, -0.0071004, -0.0007783, -0.0042506, 0.0049893
9: -0.0037658, -0.0011179, -0.0037518, -0.0007662, -0.0023269, 0.0019872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004217
time: 1.58 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004337
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0045940, -0.0015563, -0.0046426, -0.0015115, -0.0022996, 0.0022706
1: -0.0047231, -0.0041806, -0.0047288, -0.0041650, -0.0003982, 0.0003816
2: 0.0087294, 0.0125710, 0.0086785, 0.0126295, -0.0028734, 0.0028198
3: 1.0083666, 1.0093013, 1.0083225, 1.0093126, -0.0007809, 0.0008052
4: -0.0036522, -0.0030619, -0.0036621, -0.0030553, -0.0004256, 0.0004367
5: 0.0004361, 0.0027571, 0.0003999, 0.0027916, -0.0017539, 0.0017304
6: -0.0025728, -0.0024406, -0.0025803, -0.0024390, -0.0001288, 0.0001311
7: -0.0111447, -0.0056771, -0.0112212, -0.0055261, -0.0044168, 0.0043891
8: -0.0069886, -0.0008451, -0.0071004, -0.0007783, -0.0043867, 0.0045167
9: -0.0037207, -0.0008239, -0.0037518, -0.0007662, -0.0021161, 0.0020475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004138
time: 1.04 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004202
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0047192, -0.0017746, -0.0046290, -0.0017828, -0.0022523, 0.0021821
1: -0.0047316, -0.0042263, -0.0047261, -0.0042252, -0.0003542, 0.0003410
2: 0.0085831, 0.0122909, 0.0086851, 0.0122807, -0.0027982, 0.0027119
3: 1.0084136, 1.0092936, 1.0084275, 1.0092902, -0.0006957, 0.0006981
4: -0.0036085, -0.0030417, -0.0036070, -0.0030555, -0.0004079, 0.0004210
5: 0.0003415, 0.0025900, 0.0004094, 0.0025837, -0.0017167, 0.0016634
6: -0.0025789, -0.0024483, -0.0025732, -0.0024483, -0.0001242, 0.0001187
7: -0.0107724, -0.0053878, -0.0107509, -0.0056034, -0.0041891, 0.0043479
8: -0.0065265, -0.0006478, -0.0065137, -0.0007796, -0.0041867, 0.0043202
9: -0.0038060, -0.0010461, -0.0037511, -0.0010493, -0.0020011, 0.0019378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003912, upper bound: 0.0003693
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003912, upper bound: 0.0003693
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0046819, -0.0014962, -0.0046290, -0.0017828, -0.0022794, 0.0025017
1: -0.0047268, -0.0041641, -0.0047261, -0.0042252, -0.0003841, 0.0004170
2: 0.0086296, 0.0126528, 0.0086851, 0.0122807, -0.0028475, 0.0031334
3: 1.0083325, 1.0092883, 1.0084275, 1.0092902, -0.0007852, 0.0007321
4: -0.0036660, -0.0030487, -0.0036070, -0.0030555, -0.0004758, 0.0004318
5: 0.0003699, 0.0028036, 0.0004094, 0.0025837, -0.0017387, 0.0019090
6: -0.0025780, -0.0024448, -0.0025732, -0.0024483, -0.0001231, 0.0001204
7: -0.0112048, -0.0054663, -0.0107509, -0.0056034, -0.0046430, 0.0043154
8: -0.0071412, -0.0007216, -0.0065137, -0.0007796, -0.0049156, 0.0044609
9: -0.0037712, -0.0007474, -0.0037511, -0.0010493, -0.0020852, 0.0022944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003912, upper bound: 0.0003693
time: 1.04 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003912, upper bound: 0.0003693
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0046808, -0.0015884, -0.0046361, -0.0017753, -0.0022827, 0.0023894
1: -0.0047440, -0.0041891, -0.0047120, -0.0042193, -0.0004079, 0.0003721
2: 0.0086183, 0.0125260, 0.0086916, 0.0122957, -0.0028708, 0.0029651
3: 1.0083758, 1.0093364, 1.0084052, 1.0092539, -0.0007012, 0.0008029
4: -0.0036452, -0.0030446, -0.0036102, -0.0030589, -0.0004456, 0.0004392
5: 0.0003697, 0.0027323, 0.0004052, 0.0025899, -0.0017425, 0.0018211
6: -0.0025742, -0.0024407, -0.0025774, -0.0024520, -0.0001159, 0.0001298
7: -0.0110953, -0.0055411, -0.0107279, -0.0055135, -0.0045986, 0.0042330
8: -0.0069149, -0.0006609, -0.0065529, -0.0008309, -0.0045724, 0.0045622
9: -0.0038118, -0.0008605, -0.0037169, -0.0010280, -0.0021503, 0.0021166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003933
time: 0.92 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003933
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046808, -0.0015884, -0.0046330, -0.0017807, -0.0022855, 0.0023891
1: -0.0047440, -0.0041891, -0.0047306, -0.0042249, -0.0003978, 0.0003826
2: 0.0086183, 0.0125260, 0.0086805, 0.0122836, -0.0028654, 0.0029753
3: 1.0083758, 1.0093364, 1.0084260, 1.0093067, -0.0007307, 0.0007620
4: -0.0036452, -0.0030446, -0.0036074, -0.0030544, -0.0004487, 0.0004367
5: 0.0003697, 0.0027323, 0.0004064, 0.0025854, -0.0017439, 0.0018218
6: -0.0025742, -0.0024407, -0.0025736, -0.0024466, -0.0001201, 0.0001235
7: -0.0110953, -0.0055411, -0.0107585, -0.0055936, -0.0045228, 0.0042831
8: -0.0069149, -0.0006609, -0.0065177, -0.0007644, -0.0046159, 0.0045246
9: -0.0038118, -0.0008605, -0.0037609, -0.0010474, -0.0021246, 0.0021451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0004101
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0004101
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0047177, -0.0017896, -0.0046848, -0.0017897, -0.0022748, 0.0022122
1: -0.0047312, -0.0042268, -0.0047269, -0.0042249, -0.0003561, 0.0003405
2: 0.0085850, 0.0122729, 0.0086287, 0.0122713, -0.0028237, 0.0027410
3: 1.0084143, 1.0092871, 1.0084050, 1.0092897, -0.0006998, 0.0007275
4: -0.0036060, -0.0030420, -0.0036058, -0.0030491, -0.0004110, 0.0004244
5: 0.0003427, 0.0025785, 0.0003680, 0.0025783, -0.0017336, 0.0016856
6: -0.0025788, -0.0024488, -0.0025807, -0.0024468, -0.0001267, 0.0001245
7: -0.0107469, -0.0053904, -0.0107592, -0.0054387, -0.0043080, 0.0044071
8: -0.0065039, -0.0006510, -0.0065026, -0.0007262, -0.0042077, 0.0043513
9: -0.0038045, -0.0010556, -0.0037704, -0.0010551, -0.0020143, 0.0019428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003930, upper bound: 0.0003684
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003930, upper bound: 0.0003684
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0046803, -0.0015126, -0.0046848, -0.0017897, -0.0023107, 0.0025317
1: -0.0047265, -0.0041645, -0.0047269, -0.0042249, -0.0003912, 0.0004167
2: 0.0086316, 0.0126325, 0.0086287, 0.0122713, -0.0028887, 0.0031629
3: 1.0083328, 1.0092824, 1.0084050, 1.0092897, -0.0007895, 0.0007587
4: -0.0036632, -0.0030490, -0.0036058, -0.0030491, -0.0004790, 0.0004383
5: 0.0003712, 0.0027911, 0.0003680, 0.0025783, -0.0017627, 0.0019312
6: -0.0025778, -0.0024455, -0.0025807, -0.0024468, -0.0001256, 0.0001260
7: -0.0111742, -0.0054689, -0.0107592, -0.0054387, -0.0047610, 0.0043744
8: -0.0071136, -0.0007249, -0.0065026, -0.0007262, -0.0049381, 0.0045283
9: -0.0037696, -0.0007589, -0.0037704, -0.0010551, -0.0021165, 0.0023005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003930, upper bound: 0.0003684
time: 1.09 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003930, upper bound: 0.0003684
time: 1.51 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0046793, -0.0016036, -0.0046887, -0.0017839, -0.0023130, 0.0024169
1: -0.0047437, -0.0041896, -0.0047128, -0.0042192, -0.0004148, 0.0003715
2: 0.0086201, 0.0125082, 0.0086369, 0.0122847, -0.0029108, 0.0029923
3: 1.0083764, 1.0093303, 1.0083853, 1.0092518, -0.0007044, 0.0008254
4: -0.0036426, -0.0030448, -0.0036088, -0.0030525, -0.0004485, 0.0004455
5: 0.0003708, 0.0027208, 0.0003661, 0.0025832, -0.0017658, 0.0018414
6: -0.0025741, -0.0024412, -0.0025840, -0.0024505, -0.0001183, 0.0001345
7: -0.0110660, -0.0055436, -0.0107307, -0.0053587, -0.0046932, 0.0042889
8: -0.0068894, -0.0006638, -0.0065397, -0.0007770, -0.0045929, 0.0046275
9: -0.0038104, -0.0008711, -0.0037366, -0.0010343, -0.0021804, 0.0021217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003930
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003930
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046793, -0.0016036, -0.0046886, -0.0017870, -0.0023174, 0.0024180
1: -0.0047437, -0.0041896, -0.0047312, -0.0042246, -0.0004051, 0.0003823
2: 0.0086201, 0.0125082, 0.0086243, 0.0122746, -0.0029073, 0.0030036
3: 1.0083764, 1.0093303, 1.0084034, 1.0093064, -0.0007343, 0.0007869
4: -0.0036426, -0.0030448, -0.0036063, -0.0030481, -0.0004517, 0.0004432
5: 0.0003708, 0.0027208, 0.0003651, 0.0025803, -0.0017684, 0.0018432
6: -0.0025741, -0.0024412, -0.0025810, -0.0024450, -0.0001226, 0.0001288
7: -0.0110660, -0.0055436, -0.0107697, -0.0054292, -0.0046312, 0.0043451
8: -0.0068894, -0.0006638, -0.0065072, -0.0007120, -0.0046365, 0.0045915
9: -0.0038104, -0.0008711, -0.0037801, -0.0010531, -0.0021556, 0.0021505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004101
time: 1.39 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004101
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0047075, -0.0018219, -0.0046013, -0.0014572, -0.0026151, 0.0021503
1: -0.0047444, -0.0042334, -0.0047062, -0.0041561, -0.0004519, 0.0003524
2: 0.0085839, 0.0122300, 0.0087351, 0.0127039, -0.0032885, 0.0026759
3: 1.0084360, 1.0093144, 1.0083212, 1.0092611, -0.0006922, 0.0008349
4: -0.0035990, -0.0030396, -0.0036741, -0.0030655, -0.0004039, 0.0005020
5: 0.0003494, 0.0025538, 0.0004318, 0.0028336, -0.0019966, 0.0016393
6: -0.0025744, -0.0024485, -0.0025771, -0.0024444, -0.0001214, 0.0001226
7: -0.0106926, -0.0054846, -0.0112605, -0.0055837, -0.0041520, 0.0048067
8: -0.0064273, -0.0006112, -0.0072263, -0.0009015, -0.0041623, 0.0052036
9: -0.0038327, -0.0010918, -0.0036822, -0.0007064, -0.0024410, 0.0019391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004079
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004079
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0046691, -0.0015412, -0.0046013, -0.0014572, -0.0023599, 0.0022377
1: -0.0047385, -0.0041709, -0.0047062, -0.0041561, -0.0004163, 0.0003699
2: 0.0086323, 0.0125955, 0.0087351, 0.0127039, -0.0029614, 0.0027806
3: 1.0083545, 1.0093052, 1.0083212, 1.0092611, -0.0007391, 0.0008201
4: -0.0036570, -0.0030470, -0.0036741, -0.0030655, -0.0004195, 0.0004521
5: 0.0003786, 0.0027692, 0.0004318, 0.0028336, -0.0018012, 0.0017055
6: -0.0025734, -0.0024454, -0.0025771, -0.0024444, -0.0001233, 0.0001275
7: -0.0111285, -0.0055647, -0.0112605, -0.0055837, -0.0043400, 0.0044163
8: -0.0070474, -0.0006898, -0.0072263, -0.0009015, -0.0043226, 0.0046877
9: -0.0037957, -0.0007909, -0.0036822, -0.0007064, -0.0022038, 0.0020146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004060
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004060
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0047166, -0.0018133, -0.0045963, -0.0014625, -0.0026222, 0.0021548
1: -0.0047529, -0.0042328, -0.0047240, -0.0041623, -0.0004549, 0.0003671
2: 0.0085732, 0.0122363, 0.0087262, 0.0126921, -0.0032922, 0.0026941
3: 1.0084335, 1.0093439, 1.0083441, 1.0093113, -0.0007259, 0.0008358
4: -0.0035997, -0.0030372, -0.0036713, -0.0030614, -0.0004093, 0.0005023
5: 0.0003424, 0.0025600, 0.0004343, 0.0028291, -0.0020012, 0.0016435
6: -0.0025751, -0.0024450, -0.0025730, -0.0024393, -0.0001257, 0.0001215
7: -0.0107382, -0.0054641, -0.0112882, -0.0056739, -0.0041142, 0.0048667
8: -0.0064350, -0.0005787, -0.0071899, -0.0008392, -0.0042317, 0.0052065
9: -0.0038537, -0.0010883, -0.0037236, -0.0007277, -0.0024439, 0.0019804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0003971
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004306
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046789, -0.0015325, -0.0045963, -0.0014625, -0.0023693, 0.0022436
1: -0.0047469, -0.0041703, -0.0047240, -0.0041623, -0.0004190, 0.0003838
2: 0.0086207, 0.0126020, 0.0087262, 0.0126921, -0.0029666, 0.0027968
3: 1.0083523, 1.0093344, 1.0083441, 1.0093113, -0.0007717, 0.0008211
4: -0.0036577, -0.0030446, -0.0036713, -0.0030614, -0.0004243, 0.0004520
5: 0.0003711, 0.0027755, 0.0004343, 0.0028291, -0.0018076, 0.0017106
6: -0.0025742, -0.0024417, -0.0025730, -0.0024393, -0.0001271, 0.0001270
7: -0.0111735, -0.0055431, -0.0112882, -0.0056739, -0.0043014, 0.0044757
8: -0.0070542, -0.0006564, -0.0071899, -0.0008392, -0.0043858, 0.0046898
9: -0.0038170, -0.0007878, -0.0037236, -0.0007277, -0.0022070, 0.0020528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0003970
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004181
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0047680, -0.0018273, -0.0045998, -0.0014732, -0.0026673, 0.0021812
1: -0.0047458, -0.0042330, -0.0047058, -0.0041566, -0.0004714, 0.0003600
2: 0.0085195, 0.0122219, 0.0087370, 0.0126847, -0.0033569, 0.0027170
3: 1.0084136, 1.0093156, 1.0083218, 1.0092552, -0.0007230, 0.0008683
4: -0.0035978, -0.0030317, -0.0036713, -0.0030658, -0.0004105, 0.0005129
5: 0.0003040, 0.0025495, 0.0004329, 0.0028214, -0.0020366, 0.0016631
6: -0.0025820, -0.0024466, -0.0025770, -0.0024451, -0.0001272, 0.0001235
7: -0.0107054, -0.0053168, -0.0112284, -0.0055863, -0.0041993, 0.0049166
8: -0.0064174, -0.0005417, -0.0071994, -0.0009046, -0.0042316, 0.0053209
9: -0.0038602, -0.0010963, -0.0036807, -0.0007177, -0.0025026, 0.0019723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004112
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004112
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0047202, -0.0015520, -0.0045998, -0.0014732, -0.0023912, 0.0022652
1: -0.0047405, -0.0041705, -0.0047058, -0.0041566, -0.0004159, 0.0003777
2: 0.0085782, 0.0125804, 0.0087370, 0.0126847, -0.0029951, 0.0028171
3: 1.0083320, 1.0093088, 1.0083218, 1.0092552, -0.0007710, 0.0008235
4: -0.0036548, -0.0030406, -0.0036713, -0.0030658, -0.0004254, 0.0004556
5: 0.0003403, 0.0027607, 0.0004329, 0.0028214, -0.0018245, 0.0017267
6: -0.0025809, -0.0024431, -0.0025770, -0.0024451, -0.0001295, 0.0001294
7: -0.0111252, -0.0054135, -0.0112284, -0.0055863, -0.0043910, 0.0045330
8: -0.0070239, -0.0006330, -0.0071994, -0.0009046, -0.0043837, 0.0047138
9: -0.0038182, -0.0008011, -0.0036807, -0.0007177, -0.0022129, 0.0020440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004090
time: 1.25 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004090
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0047769, -0.0018184, -0.0045949, -0.0014790, -0.0026777, 0.0021858
1: -0.0047543, -0.0042324, -0.0047237, -0.0041627, -0.0004731, 0.0003750
2: 0.0085089, 0.0122283, 0.0087280, 0.0126727, -0.0033616, 0.0027358
3: 1.0084112, 1.0093449, 1.0083445, 1.0093064, -0.0007550, 0.0008690
4: -0.0035987, -0.0030295, -0.0036685, -0.0030617, -0.0004159, 0.0005126
5: 0.0002973, 0.0025559, 0.0004354, 0.0028166, -0.0020439, 0.0016676
6: -0.0025827, -0.0024430, -0.0025729, -0.0024397, -0.0001310, 0.0001226
7: -0.0107519, -0.0052969, -0.0112572, -0.0056763, -0.0041619, 0.0049732
8: -0.0064256, -0.0005104, -0.0071633, -0.0008421, -0.0043016, 0.0053194
9: -0.0038805, -0.0010925, -0.0037223, -0.0007390, -0.0025038, 0.0020143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003976
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004337
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0047299, -0.0015433, -0.0045949, -0.0014790, -0.0024023, 0.0022711
1: -0.0047488, -0.0041699, -0.0047237, -0.0041627, -0.0004179, 0.0003918
2: 0.0085667, 0.0125868, 0.0087280, 0.0126727, -0.0030005, 0.0028337
3: 1.0083297, 1.0093383, 1.0083445, 1.0093064, -0.0008011, 0.0008262
4: -0.0036555, -0.0030383, -0.0036685, -0.0030617, -0.0004302, 0.0004554
5: 0.0003330, 0.0027669, 0.0004354, 0.0028166, -0.0018323, 0.0017319
6: -0.0025816, -0.0024394, -0.0025729, -0.0024397, -0.0001330, 0.0001290
7: -0.0111719, -0.0053923, -0.0112572, -0.0056763, -0.0043527, 0.0045890
8: -0.0070310, -0.0006008, -0.0071633, -0.0008421, -0.0044479, 0.0047141
9: -0.0038392, -0.0007978, -0.0037223, -0.0007390, -0.0022142, 0.0020830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003981
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004202
time: 1.10 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.59 seconds
IS_A1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0003727
IS_A1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0003727
IS_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0003727
IS_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0003727
IS_A1_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0003953
IS_A1_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0003953
IS_A1_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0004094
IS_A1_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0004101
IS_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0003723
IS_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0003723
IS_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0003723
IS_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0003723
IS_A1_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003899, upper bound: 0.0003952
IS_A1_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003899, upper bound: 0.0003952
IS_A1_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003899, upper bound: 0.0004094
IS_A1_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003899, upper bound: 0.0004101
IS_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0004010
IS_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0004011
IS_A1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0003949
IS_A1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003902, upper bound: 0.0003981
IS_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004217
IS_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004337
IS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004138
IS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004202
IS_A1_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0004010
IS_A1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0004011
IS_A1_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0003950
IS_A1_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003924, upper bound: 0.0003981
IS_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004217
IS_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004337
IS_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004138
IS_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004202
IS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003912, upper bound: 0.0003693
IS_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003912, upper bound: 0.0003693
IS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003912, upper bound: 0.0003693
IS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003912, upper bound: 0.0003693
IS_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003933
IS_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003933
IS_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0004101
IS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0004101
IS_A2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003930, upper bound: 0.0003684
IS_A2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003930, upper bound: 0.0003684
IS_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003930, upper bound: 0.0003684
IS_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003930, upper bound: 0.0003684
IS_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003930
IS_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003930
IS_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004101
IS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004101
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004079
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004079
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004060
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004060
IS_A2_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0003971
IS_A2_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004306
IS_A2_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0003970
IS_A2_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004181
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004112
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004112
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004090
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004090
IS_A2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003976
IS_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004337
IS_A2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003981
IS_A2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004202

## BFS IS instance: IS_A1_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0046428, -0.0018112, -0.0046285, -0.0018583, -0.0021255, 0.0021527
1: -0.0047132, -0.0042342, -0.0047258, -0.0042416, -0.0003175, 0.0003353
2: 0.0086831, 0.0122407, 0.0086859, 0.0121803, -0.0026276, 0.0026755
3: 1.0084243, 1.0092663, 1.0084473, 1.0092860, -0.0006825, 0.0006409
4: -0.0036002, -0.0030576, -0.0035910, -0.0030556, -0.0004026, 0.0003927
5: 0.0004001, 0.0025617, 0.0004099, 0.0025256, -0.0016190, 0.0016410
6: -0.0025778, -0.0024490, -0.0025732, -0.0024493, -0.0001223, 0.0001176
7: -0.0107217, -0.0055021, -0.0106427, -0.0056043, -0.0041364, 0.0041743
8: -0.0064392, -0.0008170, -0.0063415, -0.0007808, -0.0041315, 0.0040116
9: -0.0037236, -0.0010870, -0.0037505, -0.0011326, -0.0018454, 0.0019117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003527, upper bound: 0.0002969
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003920, upper bound: 0.0003756
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045985, -0.0015329, -0.0046285, -0.0018583, -0.0021022, 0.0024733
1: -0.0047046, -0.0041742, -0.0047258, -0.0042416, -0.0003220, 0.0004113
2: 0.0087396, 0.0126031, 0.0086859, 0.0121803, -0.0026052, 0.0030976
3: 1.0083437, 1.0092524, 1.0084473, 1.0092860, -0.0007739, 0.0006427
4: -0.0036578, -0.0030664, -0.0035910, -0.0030556, -0.0004704, 0.0003907
5: 0.0004340, 0.0027753, 0.0004099, 0.0025256, -0.0016017, 0.0018873
6: -0.0025771, -0.0024460, -0.0025732, -0.0024493, -0.0001215, 0.0001202
7: -0.0111419, -0.0055844, -0.0106427, -0.0056043, -0.0046015, 0.0040981
8: -0.0070503, -0.0009117, -0.0063415, -0.0007808, -0.0048593, 0.0040008
9: -0.0036767, -0.0007928, -0.0037505, -0.0011326, -0.0018470, 0.0022675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003527, upper bound: 0.0003005
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003920, upper bound: 0.0003756
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0046428, -0.0018112, -0.0047075, -0.0018219, -0.0021542, 0.0022157
1: -0.0047132, -0.0042342, -0.0047444, -0.0042334, -0.0003273, 0.0003568
2: 0.0086831, 0.0122407, 0.0085839, 0.0122300, -0.0026655, 0.0027605
3: 1.0084243, 1.0092663, 1.0084360, 1.0093144, -0.0007196, 0.0006556
4: -0.0036002, -0.0030576, -0.0035990, -0.0030396, -0.0004168, 0.0003989
5: 0.0004001, 0.0025617, 0.0003494, 0.0025538, -0.0016410, 0.0016894
6: -0.0025778, -0.0024490, -0.0025744, -0.0024485, -0.0001230, 0.0001187
7: -0.0107217, -0.0055021, -0.0106926, -0.0054846, -0.0042312, 0.0042140
8: -0.0064392, -0.0008170, -0.0064273, -0.0006112, -0.0042885, 0.0040808
9: -0.0037236, -0.0010870, -0.0038327, -0.0010918, -0.0018826, 0.0019934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003727
time: 0.93 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003727
time: 1.37 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045985, -0.0015329, -0.0047075, -0.0018219, -0.0021309, 0.0025362
1: -0.0047046, -0.0041742, -0.0047444, -0.0042334, -0.0003318, 0.0004328
2: 0.0087396, 0.0126031, 0.0085839, 0.0122300, -0.0026432, 0.0031825
3: 1.0083437, 1.0092524, 1.0084360, 1.0093144, -0.0008109, 0.0006574
4: -0.0036578, -0.0030664, -0.0035990, -0.0030396, -0.0004846, 0.0003969
5: 0.0004340, 0.0027753, 0.0003494, 0.0025538, -0.0016238, 0.0019357
6: -0.0025771, -0.0024460, -0.0025744, -0.0024485, -0.0001222, 0.0001214
7: -0.0111419, -0.0055844, -0.0106926, -0.0054846, -0.0046963, 0.0041378
8: -0.0070503, -0.0009117, -0.0064273, -0.0006112, -0.0050162, 0.0040701
9: -0.0036767, -0.0007928, -0.0038327, -0.0010918, -0.0018842, 0.0023492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003727
time: 1.05 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003727
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0045945, -0.0016271, -0.0046357, -0.0018514, -0.0021076, 0.0023612
1: -0.0047230, -0.0041982, -0.0047118, -0.0042353, -0.0003478, 0.0003662
2: 0.0087291, 0.0124769, 0.0086922, 0.0121953, -0.0026311, 0.0029300
3: 1.0083889, 1.0093035, 1.0084258, 1.0092499, -0.0006872, 0.0007163
4: -0.0036373, -0.0030619, -0.0035942, -0.0030590, -0.0004403, 0.0003986
5: 0.0004358, 0.0027025, 0.0004056, 0.0025314, -0.0016075, 0.0017995
6: -0.0025729, -0.0024413, -0.0025774, -0.0024529, -0.0001138, 0.0001294
7: -0.0110410, -0.0056751, -0.0106195, -0.0055142, -0.0045607, 0.0040032
8: -0.0068284, -0.0008453, -0.0063824, -0.0008319, -0.0045178, 0.0041118
9: -0.0037203, -0.0009032, -0.0037164, -0.0011099, -0.0019162, 0.0020901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003953
time: 0.94 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003953
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0045945, -0.0016271, -0.0047089, -0.0018145, -0.0021360, 0.0024180
1: -0.0047230, -0.0041982, -0.0047301, -0.0042277, -0.0003570, 0.0003867
2: 0.0087291, 0.0124769, 0.0085950, 0.0122448, -0.0026693, 0.0030097
3: 1.0083889, 1.0093035, 1.0084177, 1.0092773, -0.0007236, 0.0007245
4: -0.0036373, -0.0030619, -0.0036022, -0.0030433, -0.0004543, 0.0004049
5: 0.0004358, 0.0027025, 0.0003493, 0.0025599, -0.0016293, 0.0018435
6: -0.0025729, -0.0024413, -0.0025779, -0.0024522, -0.0001144, 0.0001297
7: -0.0110410, -0.0056751, -0.0106690, -0.0054125, -0.0046250, 0.0040396
8: -0.0068284, -0.0008453, -0.0064680, -0.0006632, -0.0046737, 0.0041807
9: -0.0037203, -0.0009032, -0.0037988, -0.0010699, -0.0019525, 0.0021705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003953
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003953
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0045945, -0.0016271, -0.0046325, -0.0018561, -0.0021062, 0.0023609
1: -0.0047230, -0.0041982, -0.0047303, -0.0042412, -0.0003348, 0.0003771
2: 0.0087291, 0.0124769, 0.0086812, 0.0121830, -0.0026191, 0.0029401
3: 1.0083889, 1.0093035, 1.0084459, 1.0093026, -0.0007162, 0.0006716
4: -0.0036373, -0.0030619, -0.0035914, -0.0030545, -0.0004435, 0.0003946
5: 0.0004358, 0.0027025, 0.0004068, 0.0025273, -0.0016055, 0.0018002
6: -0.0025729, -0.0024413, -0.0025736, -0.0024474, -0.0001179, 0.0001238
7: -0.0110410, -0.0056751, -0.0106528, -0.0055945, -0.0044887, 0.0040502
8: -0.0068284, -0.0008453, -0.0063453, -0.0007655, -0.0045612, 0.0040555
9: -0.0037203, -0.0009032, -0.0037604, -0.0011308, -0.0018818, 0.0021183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003947, upper bound: 0.0004094
time: 0.96 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003947, upper bound: 0.0004094
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0045945, -0.0016271, -0.0047117, -0.0018198, -0.0021351, 0.0024237
1: -0.0047230, -0.0041982, -0.0047489, -0.0042330, -0.0003448, 0.0003977
2: 0.0087291, 0.0124769, 0.0085791, 0.0122328, -0.0026574, 0.0030256
3: 1.0083889, 1.0093035, 1.0084344, 1.0093322, -0.0007530, 0.0006869
4: -0.0036373, -0.0030619, -0.0035994, -0.0030385, -0.0004578, 0.0004009
5: 0.0004358, 0.0027025, 0.0003462, 0.0025554, -0.0016277, 0.0018486
6: -0.0025729, -0.0024413, -0.0025748, -0.0024468, -0.0001185, 0.0001251
7: -0.0110410, -0.0056751, -0.0107003, -0.0054740, -0.0045772, 0.0040894
8: -0.0068284, -0.0008453, -0.0064312, -0.0005965, -0.0047185, 0.0041248
9: -0.0037203, -0.0009032, -0.0038429, -0.0010900, -0.0019180, 0.0021993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003947, upper bound: 0.0004101
time: 1.03 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003947, upper bound: 0.0004101
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0046413, -0.0018258, -0.0046841, -0.0018644, -0.0021476, 0.0021826
1: -0.0047128, -0.0042347, -0.0047266, -0.0042413, -0.0003194, 0.0003349
2: 0.0086850, 0.0122234, 0.0086296, 0.0121719, -0.0026524, 0.0027044
3: 1.0084249, 1.0092599, 1.0084245, 1.0092853, -0.0006864, 0.0006761
4: -0.0035977, -0.0030578, -0.0035894, -0.0030492, -0.0004055, 0.0003959
5: 0.0004013, 0.0025507, 0.0003685, 0.0025210, -0.0016356, 0.0016630
6: -0.0025777, -0.0024495, -0.0025806, -0.0024477, -0.0001248, 0.0001235
7: -0.0106931, -0.0055046, -0.0106467, -0.0054399, -0.0042547, 0.0042328
8: -0.0064146, -0.0008201, -0.0063263, -0.0007276, -0.0041522, 0.0040420
9: -0.0037221, -0.0010975, -0.0037697, -0.0011398, -0.0018582, 0.0019166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003525, upper bound: 0.0002917
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003941, upper bound: 0.0003754
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045970, -0.0015514, -0.0046841, -0.0018644, -0.0021241, 0.0025032
1: -0.0047042, -0.0041747, -0.0047266, -0.0042413, -0.0003239, 0.0004110
2: 0.0087416, 0.0125821, 0.0086296, 0.0121719, -0.0026299, 0.0031271
3: 1.0083443, 1.0092466, 1.0084245, 1.0092853, -0.0007779, 0.0006741
4: -0.0036549, -0.0030667, -0.0035894, -0.0030492, -0.0004736, 0.0003939
5: 0.0004352, 0.0027614, 0.0003685, 0.0025210, -0.0016182, 0.0019094
6: -0.0025770, -0.0024466, -0.0025806, -0.0024477, -0.0001240, 0.0001260
7: -0.0111048, -0.0055870, -0.0106467, -0.0054399, -0.0047194, 0.0041563
8: -0.0070232, -0.0009150, -0.0063263, -0.0007276, -0.0048821, 0.0040309
9: -0.0036752, -0.0008041, -0.0037697, -0.0011398, -0.0018597, 0.0022736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003525, upper bound: 0.0002956
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003941, upper bound: 0.0003754
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0046413, -0.0018258, -0.0047666, -0.0018273, -0.0021771, 0.0022451
1: -0.0047128, -0.0042347, -0.0047452, -0.0042330, -0.0003291, 0.0003566
2: 0.0086850, 0.0122234, 0.0085218, 0.0122219, -0.0026915, 0.0027911
3: 1.0084249, 1.0092599, 1.0084136, 1.0093130, -0.0007233, 0.0006872
4: -0.0035977, -0.0030578, -0.0035978, -0.0030322, -0.0004201, 0.0004023
5: 0.0004013, 0.0025507, 0.0003052, 0.0025495, -0.0016582, 0.0017113
6: -0.0025777, -0.0024495, -0.0025820, -0.0024469, -0.0001255, 0.0001246
7: -0.0106931, -0.0055046, -0.0107034, -0.0053168, -0.0043430, 0.0042745
8: -0.0064146, -0.0008201, -0.0064174, -0.0005469, -0.0043138, 0.0041121
9: -0.0037221, -0.0010975, -0.0038576, -0.0010963, -0.0018958, 0.0020011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003723
time: 0.92 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003723
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045970, -0.0015514, -0.0047666, -0.0018273, -0.0021537, 0.0025657
1: -0.0047042, -0.0041747, -0.0047452, -0.0042330, -0.0003336, 0.0004327
2: 0.0087416, 0.0125821, 0.0085218, 0.0122219, -0.0026689, 0.0032138
3: 1.0083443, 1.0092466, 1.0084136, 1.0093130, -0.0008148, 0.0006852
4: -0.0036549, -0.0030667, -0.0035978, -0.0030322, -0.0004882, 0.0004003
5: 0.0004352, 0.0027614, 0.0003052, 0.0025495, -0.0016409, 0.0019577
6: -0.0025770, -0.0024466, -0.0025820, -0.0024469, -0.0001247, 0.0001272
7: -0.0111048, -0.0055870, -0.0107034, -0.0053168, -0.0048077, 0.0041980
8: -0.0070232, -0.0009150, -0.0064174, -0.0005469, -0.0050436, 0.0041009
9: -0.0036752, -0.0008041, -0.0038576, -0.0010963, -0.0018972, 0.0023581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003723
time: 1.03 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003723
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0045929, -0.0016431, -0.0046881, -0.0018588, -0.0021283, 0.0023885
1: -0.0047226, -0.0041987, -0.0047125, -0.0042352, -0.0003493, 0.0003656
2: 0.0087311, 0.0124571, 0.0086376, 0.0121849, -0.0026544, 0.0029570
3: 1.0083892, 1.0092986, 1.0084050, 1.0092478, -0.0006901, 0.0007450
4: -0.0036345, -0.0030622, -0.0035925, -0.0030526, -0.0004432, 0.0004017
5: 0.0004370, 0.0026905, 0.0003665, 0.0025258, -0.0016231, 0.0018198
6: -0.0025728, -0.0024418, -0.0025840, -0.0024515, -0.0001162, 0.0001342
7: -0.0110124, -0.0056778, -0.0106186, -0.0053596, -0.0046541, 0.0040581
8: -0.0068026, -0.0008486, -0.0063647, -0.0007783, -0.0045382, 0.0041404
9: -0.0037188, -0.0009138, -0.0037360, -0.0011183, -0.0019280, 0.0020951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003952
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003952
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0045929, -0.0016431, -0.0047704, -0.0018213, -0.0021576, 0.0024518
1: -0.0047226, -0.0041987, -0.0047309, -0.0042270, -0.0003591, 0.0003865
2: 0.0087311, 0.0124571, 0.0085307, 0.0122351, -0.0026938, 0.0030428
3: 1.0083892, 1.0092986, 1.0083946, 1.0092751, -0.0007268, 0.0007535
4: -0.0036345, -0.0030622, -0.0036008, -0.0030357, -0.0004576, 0.0004081
5: 0.0004370, 0.0026905, 0.0003035, 0.0025546, -0.0016456, 0.0018684
6: -0.0025728, -0.0024418, -0.0025855, -0.0024507, -0.0001169, 0.0001355
7: -0.0110124, -0.0056778, -0.0106745, -0.0052338, -0.0047481, 0.0040969
8: -0.0068026, -0.0008486, -0.0064551, -0.0005976, -0.0046973, 0.0042102
9: -0.0037188, -0.0009138, -0.0038239, -0.0010751, -0.0019656, 0.0021781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003952
time: 0.93 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003952
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0045929, -0.0016431, -0.0046878, -0.0018618, -0.0021283, 0.0023895
1: -0.0047226, -0.0041987, -0.0047309, -0.0042409, -0.0003367, 0.0003766
2: 0.0087311, 0.0124571, 0.0086253, 0.0121750, -0.0026439, 0.0029681
3: 1.0083892, 1.0092986, 1.0084231, 1.0093020, -0.0007193, 0.0007010
4: -0.0036345, -0.0030622, -0.0035899, -0.0030482, -0.0004464, 0.0003978
5: 0.0004370, 0.0026905, 0.0003657, 0.0025230, -0.0016222, 0.0018214
6: -0.0025728, -0.0024418, -0.0025810, -0.0024458, -0.0001203, 0.0001289
7: -0.0110124, -0.0056778, -0.0106611, -0.0054306, -0.0045934, 0.0041104
8: -0.0068026, -0.0008486, -0.0063305, -0.0007135, -0.0045813, 0.0040853
9: -0.0037188, -0.0009138, -0.0037794, -0.0011378, -0.0018940, 0.0021234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003969, upper bound: 0.0004094
time: 1.01 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003969, upper bound: 0.0004094
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0045929, -0.0016431, -0.0047706, -0.0018247, -0.0021581, 0.0024531
1: -0.0047226, -0.0041987, -0.0047497, -0.0042326, -0.0003464, 0.0003976
2: 0.0087311, 0.0124571, 0.0085171, 0.0122251, -0.0026833, 0.0030551
3: 1.0083892, 1.0092986, 1.0084121, 1.0093312, -0.0007560, 0.0007114
4: -0.0036345, -0.0030622, -0.0035983, -0.0030312, -0.0004609, 0.0004043
5: 0.0004370, 0.0026905, 0.0003021, 0.0025515, -0.0016450, 0.0018704
6: -0.0025728, -0.0024418, -0.0025824, -0.0024450, -0.0001211, 0.0001301
7: -0.0110124, -0.0056778, -0.0107129, -0.0053068, -0.0046839, 0.0041519
8: -0.0068026, -0.0008486, -0.0064217, -0.0005334, -0.0047422, 0.0041557
9: -0.0037188, -0.0009138, -0.0038666, -0.0010942, -0.0019309, 0.0022077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003969, upper bound: 0.0004101
time: 0.96 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003969, upper bound: 0.0004101
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0046428, -0.0018112, -0.0045835, -0.0015766, -0.0024475, 0.0021469
1: -0.0047132, -0.0042342, -0.0047175, -0.0041814, -0.0003937, 0.0003656
2: 0.0086831, 0.0122407, 0.0087422, 0.0125470, -0.0030504, 0.0026914
3: 1.0084243, 1.0092663, 1.0083688, 1.0092720, -0.0007228, 0.0007304
4: -0.0036002, -0.0030576, -0.0036491, -0.0030642, -0.0004094, 0.0004609
5: 0.0004001, 0.0025617, 0.0004441, 0.0027418, -0.0018663, 0.0016382
6: -0.0025778, -0.0024490, -0.0025722, -0.0024459, -0.0001237, 0.0001164
7: -0.0107217, -0.0055021, -0.0110713, -0.0056977, -0.0040542, 0.0046340
8: -0.0064392, -0.0008170, -0.0069599, -0.0008728, -0.0042329, 0.0047435
9: -0.0037236, -0.0010870, -0.0037049, -0.0008358, -0.0022032, 0.0019797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0002927
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003804, upper bound: 0.0003918
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046428, -0.0018112, -0.0046691, -0.0015412, -0.0024880, 0.0022407
1: -0.0047132, -0.0042342, -0.0047385, -0.0041709, -0.0004043, 0.0003872
2: 0.0086831, 0.0122407, 0.0086323, 0.0125955, -0.0031068, 0.0028130
3: 1.0084243, 1.0092663, 1.0083545, 1.0093052, -0.0007570, 0.0007468
4: -0.0036002, -0.0030576, -0.0036570, -0.0030470, -0.0004286, 0.0004701
5: 0.0004001, 0.0025617, 0.0003786, 0.0027692, -0.0018977, 0.0017101
6: -0.0025778, -0.0024490, -0.0025734, -0.0024454, -0.0001242, 0.0001177
7: -0.0107217, -0.0055021, -0.0111285, -0.0055647, -0.0041824, 0.0046879
8: -0.0064392, -0.0008170, -0.0070474, -0.0006898, -0.0044336, 0.0048446
9: -0.0037236, -0.0010870, -0.0037957, -0.0007909, -0.0022544, 0.0020773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0002927
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003804, upper bound: 0.0003918
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0046004, -0.0015329, -0.0045835, -0.0015766, -0.0022021, 0.0022294
1: -0.0047056, -0.0041742, -0.0047175, -0.0041814, -0.0003623, 0.0003798
2: 0.0087365, 0.0126031, 0.0087422, 0.0125470, -0.0027341, 0.0027850
3: 1.0083437, 1.0092559, 1.0083688, 1.0092720, -0.0007665, 0.0007251
4: -0.0036578, -0.0030658, -0.0036491, -0.0030642, -0.0004227, 0.0004122
5: 0.0004325, 0.0027753, 0.0004441, 0.0027418, -0.0016782, 0.0017004
6: -0.0025771, -0.0024454, -0.0025722, -0.0024459, -0.0001257, 0.0001213
7: -0.0111451, -0.0055844, -0.0110713, -0.0056977, -0.0042372, 0.0042790
8: -0.0070503, -0.0009045, -0.0069599, -0.0008728, -0.0043649, 0.0042442
9: -0.0036806, -0.0007928, -0.0037049, -0.0008358, -0.0019764, 0.0020407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003600
time: 0.90 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004075, upper bound: 0.0003853
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046004, -0.0015329, -0.0046691, -0.0015412, -0.0022305, 0.0022934
1: -0.0047056, -0.0041742, -0.0047385, -0.0041709, -0.0003708, 0.0004001
2: 0.0087365, 0.0126031, 0.0086323, 0.0125955, -0.0027734, 0.0028725
3: 1.0083437, 1.0092559, 1.0083545, 1.0093052, -0.0007996, 0.0007366
4: -0.0036578, -0.0030658, -0.0036570, -0.0030470, -0.0004375, 0.0004190
5: 0.0004325, 0.0027753, 0.0003786, 0.0027692, -0.0017001, 0.0017498
6: -0.0025771, -0.0024454, -0.0025734, -0.0024454, -0.0001262, 0.0001222
7: -0.0111451, -0.0055844, -0.0111285, -0.0055647, -0.0043246, 0.0043066
8: -0.0070503, -0.0009045, -0.0070474, -0.0006898, -0.0045289, 0.0043207
9: -0.0036806, -0.0007928, -0.0037957, -0.0007909, -0.0020154, 0.0021256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003605
time: 0.91 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004075, upper bound: 0.0003885
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0046393, -0.0018158, -0.0045950, -0.0015010, -0.0025194, 0.0021553
1: -0.0047317, -0.0042401, -0.0047047, -0.0041597, -0.0004331, 0.0003466
2: 0.0086724, 0.0122292, 0.0087433, 0.0126512, -0.0031650, 0.0026792
3: 1.0084442, 1.0093193, 1.0083244, 1.0092454, -0.0006714, 0.0008293
4: -0.0035975, -0.0030531, -0.0036666, -0.0030669, -0.0004040, 0.0004826
5: 0.0004016, 0.0025578, 0.0004367, 0.0028004, -0.0019231, 0.0016427
6: -0.0025740, -0.0024435, -0.0025767, -0.0024480, -0.0001179, 0.0001269
7: -0.0107565, -0.0055833, -0.0111567, -0.0055940, -0.0041929, 0.0046360
8: -0.0064029, -0.0007510, -0.0071541, -0.0009155, -0.0041553, 0.0049995
9: -0.0037673, -0.0011076, -0.0036755, -0.0007377, -0.0023415, 0.0019300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004206
time: 0.94 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004217
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046393, -0.0018158, -0.0045905, -0.0015065, -0.0025178, 0.0021566
1: -0.0047317, -0.0042401, -0.0047227, -0.0041656, -0.0004212, 0.0003610
2: 0.0086724, 0.0122292, 0.0087337, 0.0126397, -0.0031541, 0.0026941
3: 1.0084442, 1.0093193, 1.0083472, 1.0092951, -0.0007017, 0.0007837
4: -0.0035975, -0.0030531, -0.0036639, -0.0030626, -0.0004086, 0.0004792
5: 0.0004016, 0.0025578, 0.0004388, 0.0027958, -0.0019212, 0.0016449
6: -0.0025740, -0.0024435, -0.0025726, -0.0024431, -0.0001213, 0.0001210
7: -0.0107565, -0.0055833, -0.0111818, -0.0056834, -0.0041148, 0.0046800
8: -0.0064029, -0.0007510, -0.0071183, -0.0008519, -0.0042174, 0.0049519
9: -0.0037673, -0.0011076, -0.0037176, -0.0007584, -0.0023122, 0.0019680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004300
time: 1.28 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004337
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045955, -0.0015378, -0.0045950, -0.0015010, -0.0022657, 0.0022370
1: -0.0047235, -0.0041801, -0.0047047, -0.0041597, -0.0004007, 0.0003614
2: 0.0087274, 0.0125920, 0.0087433, 0.0126512, -0.0028375, 0.0027745
3: 1.0083663, 1.0093064, 1.0083244, 1.0092454, -0.0007161, 0.0008197
4: -0.0036551, -0.0030616, -0.0036666, -0.0030669, -0.0004177, 0.0004326
5: 0.0004349, 0.0027712, 0.0004367, 0.0028004, -0.0017286, 0.0017045
6: -0.0025730, -0.0024402, -0.0025767, -0.0024480, -0.0001200, 0.0001315
7: -0.0111759, -0.0056744, -0.0111567, -0.0055940, -0.0043749, 0.0042649
8: -0.0070164, -0.0008418, -0.0071541, -0.0009155, -0.0042915, 0.0044862
9: -0.0037222, -0.0008125, -0.0036755, -0.0007377, -0.0021083, 0.0019925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004032, upper bound: 0.0004102
time: 1.23 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004032, upper bound: 0.0004138
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045955, -0.0015378, -0.0045905, -0.0015065, -0.0022662, 0.0022364
1: -0.0047235, -0.0041801, -0.0047227, -0.0041656, -0.0003903, 0.0003748
2: 0.0087274, 0.0125920, 0.0087337, 0.0126397, -0.0028312, 0.0027865
3: 1.0083663, 1.0093064, 1.0083472, 1.0092951, -0.0007447, 0.0007760
4: -0.0036551, -0.0030616, -0.0036639, -0.0030626, -0.0004217, 0.0004301
5: 0.0004349, 0.0027712, 0.0004388, 0.0027958, -0.0017284, 0.0017052
6: -0.0025730, -0.0024402, -0.0025726, -0.0024431, -0.0001230, 0.0001251
7: -0.0111759, -0.0056744, -0.0111818, -0.0056834, -0.0042924, 0.0043074
8: -0.0070164, -0.0008418, -0.0071183, -0.0008519, -0.0043470, 0.0044479
9: -0.0037222, -0.0008125, -0.0037176, -0.0007584, -0.0020829, 0.0020279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004032, upper bound: 0.0004166
time: 1.27 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004032, upper bound: 0.0004202
time: 1.40 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0046413, -0.0018258, -0.0046304, -0.0015910, -0.0024715, 0.0021718
1: -0.0047128, -0.0042347, -0.0047196, -0.0041811, -0.0003956, 0.0003648
2: 0.0086850, 0.0122234, 0.0086935, 0.0125304, -0.0030774, 0.0027115
3: 1.0084249, 1.0092599, 1.0083448, 1.0092778, -0.0007283, 0.0007624
4: -0.0035977, -0.0030578, -0.0036467, -0.0030582, -0.0004112, 0.0004644
5: 0.0004013, 0.0025507, 0.0004092, 0.0027310, -0.0018844, 0.0016563
6: -0.0025777, -0.0024495, -0.0025795, -0.0024436, -0.0001260, 0.0001221
7: -0.0106931, -0.0055046, -0.0110662, -0.0055501, -0.0041611, 0.0046961
8: -0.0064146, -0.0008201, -0.0069370, -0.0008171, -0.0042441, 0.0047761
9: -0.0037221, -0.0010975, -0.0037279, -0.0008451, -0.0022168, 0.0019828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003124, upper bound: 0.0002897
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003826, upper bound: 0.0003918
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046413, -0.0018258, -0.0047202, -0.0015520, -0.0025117, 0.0022622
1: -0.0047128, -0.0042347, -0.0047405, -0.0041705, -0.0004061, 0.0003866
2: 0.0086850, 0.0122234, 0.0085782, 0.0125804, -0.0031336, 0.0028312
3: 1.0084249, 1.0092599, 1.0083320, 1.0093088, -0.0007598, 0.0007760
4: -0.0035977, -0.0030578, -0.0036548, -0.0030406, -0.0004302, 0.0004736
5: 0.0004013, 0.0025507, 0.0003403, 0.0027607, -0.0019156, 0.0017260
6: -0.0025777, -0.0024495, -0.0025809, -0.0024431, -0.0001264, 0.0001233
7: -0.0106931, -0.0055046, -0.0111252, -0.0054135, -0.0042803, 0.0047485
8: -0.0064146, -0.0008201, -0.0070239, -0.0006330, -0.0044468, 0.0048768
9: -0.0037221, -0.0010975, -0.0038182, -0.0008011, -0.0022678, 0.0020816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003124, upper bound: 0.0002897
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003826, upper bound: 0.0003918
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045989, -0.0015514, -0.0046304, -0.0015910, -0.0022318, 0.0022591
1: -0.0047053, -0.0041747, -0.0047196, -0.0041811, -0.0003701, 0.0003787
2: 0.0087385, 0.0125821, 0.0086935, 0.0125304, -0.0027736, 0.0028121
3: 1.0083443, 1.0092503, 1.0083448, 1.0092778, -0.0007732, 0.0007567
4: -0.0036549, -0.0030661, -0.0036467, -0.0030582, -0.0004252, 0.0004185
5: 0.0004337, 0.0027614, 0.0004092, 0.0027310, -0.0017011, 0.0017222
6: -0.0025770, -0.0024461, -0.0025795, -0.0024436, -0.0001278, 0.0001273
7: -0.0111079, -0.0055870, -0.0110662, -0.0055501, -0.0043621, 0.0043313
8: -0.0070232, -0.0009078, -0.0069370, -0.0008171, -0.0043817, 0.0043103
9: -0.0036791, -0.0008041, -0.0037279, -0.0008451, -0.0020080, 0.0020437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003715, upper bound: 0.0003596
time: 1.00 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004114, upper bound: 0.0003853
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045989, -0.0015514, -0.0047202, -0.0015520, -0.0022588, 0.0023229
1: -0.0047053, -0.0041747, -0.0047405, -0.0041705, -0.0003781, 0.0003993
2: 0.0087385, 0.0125821, 0.0085782, 0.0125804, -0.0028104, 0.0029028
3: 1.0083443, 1.0092503, 1.0083320, 1.0093088, -0.0008030, 0.0007651
4: -0.0036549, -0.0030661, -0.0036548, -0.0030406, -0.0004403, 0.0004246
5: 0.0004337, 0.0027614, 0.0003403, 0.0027607, -0.0017219, 0.0017718
6: -0.0025770, -0.0024461, -0.0025809, -0.0024431, -0.0001282, 0.0001283
7: -0.0111079, -0.0055870, -0.0111252, -0.0054135, -0.0044408, 0.0043586
8: -0.0070232, -0.0009078, -0.0070239, -0.0006330, -0.0045494, 0.0043797
9: -0.0036791, -0.0008041, -0.0038182, -0.0008011, -0.0020445, 0.0021315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003715, upper bound: 0.0003603
time: 0.98 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004114, upper bound: 0.0003885
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0046378, -0.0018301, -0.0046394, -0.0015144, -0.0025419, 0.0021770
1: -0.0047314, -0.0042406, -0.0047069, -0.0041597, -0.0004342, 0.0003454
2: 0.0086743, 0.0122121, 0.0086962, 0.0126361, -0.0031905, 0.0026978
3: 1.0084447, 1.0093133, 1.0083063, 1.0092493, -0.0006747, 0.0008503
4: -0.0035949, -0.0030534, -0.0036644, -0.0030609, -0.0004058, 0.0004860
5: 0.0004027, 0.0025469, 0.0004034, 0.0027903, -0.0019401, 0.0016587
6: -0.0025739, -0.0024439, -0.0025831, -0.0024462, -0.0001200, 0.0001313
7: -0.0107306, -0.0055858, -0.0111468, -0.0054633, -0.0042869, 0.0046941
8: -0.0063789, -0.0007542, -0.0071307, -0.0008594, -0.0041671, 0.0050304
9: -0.0037658, -0.0011179, -0.0036985, -0.0007485, -0.0023539, 0.0019325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004206
time: 1.01 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004217
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046378, -0.0018301, -0.0046375, -0.0015183, -0.0025421, 0.0021801
1: -0.0047314, -0.0042406, -0.0047248, -0.0041653, -0.0004228, 0.0003599
2: 0.0086743, 0.0122121, 0.0086848, 0.0126259, -0.0031815, 0.0027138
3: 1.0084447, 1.0093133, 1.0083234, 1.0093008, -0.0007073, 0.0008090
4: -0.0035949, -0.0030534, -0.0036618, -0.0030566, -0.0004102, 0.0004828
5: 0.0004027, 0.0025469, 0.0004038, 0.0027868, -0.0019396, 0.0016620
6: -0.0025739, -0.0024439, -0.0025800, -0.0024407, -0.0001235, 0.0001260
7: -0.0107306, -0.0055858, -0.0111847, -0.0055363, -0.0042178, 0.0047442
8: -0.0063789, -0.0007542, -0.0070968, -0.0007957, -0.0042262, 0.0049844
9: -0.0037658, -0.0011179, -0.0037407, -0.0007678, -0.0023252, 0.0019703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004300
time: 1.08 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004337
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045940, -0.0015563, -0.0046394, -0.0015144, -0.0022939, 0.0022640
1: -0.0047231, -0.0041806, -0.0047069, -0.0041597, -0.0004078, 0.0003600
2: 0.0087294, 0.0125710, 0.0086962, 0.0126361, -0.0028756, 0.0028001
3: 1.0083666, 1.0093013, 1.0083063, 1.0092493, -0.0007197, 0.0008440
4: -0.0036522, -0.0030619, -0.0036644, -0.0030609, -0.0004201, 0.0004385
5: 0.0004361, 0.0027571, 0.0004034, 0.0027903, -0.0017506, 0.0017244
6: -0.0025728, -0.0024406, -0.0025831, -0.0024462, -0.0001219, 0.0001363
7: -0.0111447, -0.0056771, -0.0111468, -0.0054633, -0.0044731, 0.0043140
8: -0.0069886, -0.0008451, -0.0071307, -0.0008594, -0.0043070, 0.0045478
9: -0.0037207, -0.0008239, -0.0036985, -0.0007485, -0.0021387, 0.0019956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004055, upper bound: 0.0004102
time: 1.11 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004055, upper bound: 0.0004138
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045940, -0.0015563, -0.0046375, -0.0015183, -0.0022959, 0.0022657
1: -0.0047231, -0.0041806, -0.0047248, -0.0041653, -0.0003980, 0.0003733
2: 0.0087294, 0.0125710, 0.0086848, 0.0126259, -0.0028709, 0.0028136
3: 1.0083666, 1.0093013, 1.0083234, 1.0093008, -0.0007513, 0.0008042
4: -0.0036522, -0.0030619, -0.0036618, -0.0030566, -0.0004242, 0.0004363
5: 0.0004361, 0.0027571, 0.0004038, 0.0027868, -0.0017515, 0.0017267
6: -0.0025728, -0.0024406, -0.0025800, -0.0024407, -0.0001250, 0.0001308
7: -0.0111447, -0.0056771, -0.0111847, -0.0055363, -0.0044072, 0.0043622
8: -0.0069886, -0.0008451, -0.0070968, -0.0007957, -0.0043632, 0.0045130
9: -0.0037207, -0.0008239, -0.0037407, -0.0007678, -0.0021145, 0.0020309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004055, upper bound: 0.0004166
time: 1.14 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004055, upper bound: 0.0004202
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0047192, -0.0017746, -0.0046285, -0.0018583, -0.0021851, 0.0021810
1: -0.0047316, -0.0042263, -0.0047258, -0.0042416, -0.0003378, 0.0003452
2: 0.0085831, 0.0122909, 0.0086859, 0.0121803, -0.0027099, 0.0027145
3: 1.0084136, 1.0092936, 1.0084473, 1.0092860, -0.0006940, 0.0006764
4: -0.0036085, -0.0030417, -0.0035910, -0.0030556, -0.0004090, 0.0004068
5: 0.0003415, 0.0025900, 0.0004099, 0.0025256, -0.0016651, 0.0016628
6: -0.0025789, -0.0024483, -0.0025732, -0.0024493, -0.0001233, 0.0001182
7: -0.0107724, -0.0053878, -0.0106427, -0.0056043, -0.0041734, 0.0042500
8: -0.0065265, -0.0006478, -0.0063415, -0.0007808, -0.0042024, 0.0041680
9: -0.0038060, -0.0010461, -0.0037505, -0.0011326, -0.0019260, 0.0019493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003693
time: 0.98 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003693
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0047192, -0.0017746, -0.0047075, -0.0018219, -0.0021555, 0.0021835
1: -0.0047316, -0.0042263, -0.0047444, -0.0042334, -0.0003236, 0.0003413
2: 0.0085831, 0.0122909, 0.0085839, 0.0122300, -0.0026638, 0.0027121
3: 1.0084136, 1.0092936, 1.0084360, 1.0093144, -0.0006986, 0.0006571
4: -0.0036085, -0.0030417, -0.0035990, -0.0030396, -0.0004077, 0.0003979
5: 0.0003415, 0.0025900, 0.0003494, 0.0025538, -0.0016418, 0.0016644
6: -0.0025789, -0.0024483, -0.0025744, -0.0024485, -0.0001241, 0.0001196
7: -0.0107724, -0.0053878, -0.0106926, -0.0054846, -0.0042013, 0.0042349
8: -0.0065265, -0.0006478, -0.0064273, -0.0006112, -0.0041832, 0.0040646
9: -0.0038060, -0.0010461, -0.0038327, -0.0010918, -0.0018709, 0.0019365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003693
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003693
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0046819, -0.0014962, -0.0046285, -0.0018583, -0.0022118, 0.0025162
1: -0.0047268, -0.0041641, -0.0047258, -0.0042416, -0.0003692, 0.0004233
2: 0.0086296, 0.0126528, 0.0086859, 0.0121803, -0.0027591, 0.0031567
3: 1.0083325, 1.0092883, 1.0084473, 1.0092860, -0.0007892, 0.0007127
4: -0.0036660, -0.0030487, -0.0035910, -0.0030556, -0.0004804, 0.0004178
5: 0.0003699, 0.0028036, 0.0004099, 0.0025256, -0.0016867, 0.0019205
6: -0.0025780, -0.0024448, -0.0025732, -0.0024493, -0.0001222, 0.0001196
7: -0.0112048, -0.0054663, -0.0106427, -0.0056043, -0.0046525, 0.0042175
8: -0.0071412, -0.0007216, -0.0063415, -0.0007808, -0.0049689, 0.0043100
9: -0.0037712, -0.0007474, -0.0037505, -0.0011326, -0.0020109, 0.0023226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003210, upper bound: 0.0002412
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004085, upper bound: 0.0003601
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046819, -0.0014962, -0.0047075, -0.0018219, -0.0021533, 0.0025031
1: -0.0047268, -0.0041641, -0.0047444, -0.0042334, -0.0003545, 0.0004173
2: 0.0086296, 0.0126528, 0.0085839, 0.0122300, -0.0026804, 0.0031336
3: 1.0083325, 1.0092883, 1.0084360, 1.0093144, -0.0007881, 0.0006972
4: -0.0036660, -0.0030487, -0.0035990, -0.0030396, -0.0004756, 0.0004047
5: 0.0003699, 0.0028036, 0.0003494, 0.0025538, -0.0016416, 0.0019100
6: -0.0025780, -0.0024448, -0.0025744, -0.0024485, -0.0001231, 0.0001210
7: -0.0112048, -0.0054663, -0.0106926, -0.0054846, -0.0046536, 0.0041557
8: -0.0071412, -0.0007216, -0.0064273, -0.0006112, -0.0049121, 0.0041704
9: -0.0037712, -0.0007474, -0.0038327, -0.0010918, -0.0019437, 0.0022931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003210, upper bound: 0.0002412
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004085, upper bound: 0.0003601
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0047189, -0.0017794, -0.0046361, -0.0017753, -0.0022597, 0.0021853
1: -0.0047503, -0.0042320, -0.0047120, -0.0042193, -0.0003793, 0.0003227
2: 0.0085700, 0.0122793, 0.0086916, 0.0122957, -0.0028250, 0.0026972
3: 1.0084327, 1.0093486, 1.0084052, 1.0092539, -0.0006410, 0.0007700
4: -0.0036056, -0.0030371, -0.0036102, -0.0030589, -0.0004023, 0.0004285
5: 0.0003406, 0.0025860, 0.0004052, 0.0025899, -0.0017237, 0.0016642
6: -0.0025752, -0.0024427, -0.0025774, -0.0024520, -0.0001168, 0.0001287
7: -0.0108066, -0.0054619, -0.0107279, -0.0055135, -0.0043209, 0.0042622
8: -0.0064906, -0.0005819, -0.0065529, -0.0008309, -0.0041047, 0.0044251
9: -0.0038499, -0.0010663, -0.0037169, -0.0010280, -0.0020669, 0.0018863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003926
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003926
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0046818, -0.0015019, -0.0046361, -0.0017753, -0.0022832, 0.0025040
1: -0.0047445, -0.0041696, -0.0047120, -0.0042193, -0.0004083, 0.0003992
2: 0.0086168, 0.0126408, 0.0086916, 0.0122957, -0.0028722, 0.0031178
3: 1.0083517, 1.0093387, 1.0084052, 1.0092539, -0.0007306, 0.0008052
4: -0.0036631, -0.0030443, -0.0036102, -0.0030589, -0.0004703, 0.0004394
5: 0.0003689, 0.0027988, 0.0004052, 0.0025899, -0.0017432, 0.0019092
6: -0.0025743, -0.0024396, -0.0025774, -0.0024520, -0.0001158, 0.0001302
7: -0.0112314, -0.0055404, -0.0107279, -0.0055135, -0.0047650, 0.0042155
8: -0.0071041, -0.0006577, -0.0065529, -0.0008309, -0.0048358, 0.0045650
9: -0.0038136, -0.0007679, -0.0037169, -0.0010280, -0.0021517, 0.0022444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003926
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003926
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0047189, -0.0017794, -0.0046330, -0.0017807, -0.0022579, 0.0021853
1: -0.0047503, -0.0042320, -0.0047306, -0.0042249, -0.0003666, 0.0003342
2: 0.0085700, 0.0122793, 0.0086805, 0.0122836, -0.0028143, 0.0027072
3: 1.0084327, 1.0093486, 1.0084260, 1.0093067, -0.0006699, 0.0007275
4: -0.0036056, -0.0030371, -0.0036074, -0.0030544, -0.0004056, 0.0004251
5: 0.0003406, 0.0025860, 0.0004064, 0.0025854, -0.0017217, 0.0016651
6: -0.0025752, -0.0024427, -0.0025736, -0.0024466, -0.0001210, 0.0001231
7: -0.0108066, -0.0054619, -0.0107585, -0.0055936, -0.0042479, 0.0043024
8: -0.0064906, -0.0005819, -0.0065177, -0.0007644, -0.0041530, 0.0043737
9: -0.0038499, -0.0010663, -0.0037609, -0.0010474, -0.0020342, 0.0019179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003981, upper bound: 0.0004077
time: 1.15 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003981, upper bound: 0.0004077
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046818, -0.0015019, -0.0046330, -0.0017807, -0.0022860, 0.0025041
1: -0.0047445, -0.0041696, -0.0047306, -0.0042249, -0.0003982, 0.0004103
2: 0.0086168, 0.0126408, 0.0086805, 0.0122836, -0.0028669, 0.0031278
3: 1.0083517, 1.0093387, 1.0084260, 1.0093067, -0.0007614, 0.0007644
4: -0.0036631, -0.0030443, -0.0036074, -0.0030544, -0.0004735, 0.0004369
5: 0.0003689, 0.0027988, 0.0004064, 0.0025854, -0.0017446, 0.0019101
6: -0.0025743, -0.0024396, -0.0025736, -0.0024466, -0.0001200, 0.0001242
7: -0.0112314, -0.0055404, -0.0107585, -0.0055936, -0.0046946, 0.0042649
8: -0.0071041, -0.0006577, -0.0065177, -0.0007644, -0.0048820, 0.0045274
9: -0.0038136, -0.0007679, -0.0037609, -0.0010474, -0.0021261, 0.0022745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003981, upper bound: 0.0004077
time: 1.16 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003981, upper bound: 0.0004077
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0047177, -0.0017896, -0.0046841, -0.0018644, -0.0022071, 0.0022127
1: -0.0047312, -0.0042268, -0.0047266, -0.0042413, -0.0003398, 0.0003448
2: 0.0085850, 0.0122729, 0.0086296, 0.0121719, -0.0027346, 0.0027463
3: 1.0084143, 1.0092871, 1.0084245, 1.0092853, -0.0006979, 0.0007088
4: -0.0036060, -0.0030420, -0.0035894, -0.0030492, -0.0004126, 0.0004100
5: 0.0003427, 0.0025785, 0.0003685, 0.0025210, -0.0016817, 0.0016863
6: -0.0025788, -0.0024488, -0.0025806, -0.0024477, -0.0001258, 0.0001241
7: -0.0107469, -0.0053904, -0.0106467, -0.0054399, -0.0042944, 0.0043084
8: -0.0065039, -0.0006510, -0.0063263, -0.0007276, -0.0042298, 0.0041984
9: -0.0038045, -0.0010556, -0.0037697, -0.0011398, -0.0019388, 0.0019576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003684
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003684
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0047177, -0.0017896, -0.0047666, -0.0018273, -0.0021769, 0.0022142
1: -0.0047312, -0.0042268, -0.0047452, -0.0042330, -0.0003257, 0.0003408
2: 0.0085850, 0.0122729, 0.0085218, 0.0122219, -0.0026882, 0.0027422
3: 1.0084143, 1.0092871, 1.0084136, 1.0093130, -0.0007029, 0.0006928
4: -0.0036060, -0.0030420, -0.0035978, -0.0030322, -0.0004109, 0.0004013
5: 0.0003427, 0.0025785, 0.0003052, 0.0025495, -0.0016579, 0.0016870
6: -0.0025788, -0.0024488, -0.0025820, -0.0024469, -0.0001267, 0.0001255
7: -0.0107469, -0.0053904, -0.0107034, -0.0053168, -0.0043213, 0.0042955
8: -0.0065039, -0.0006510, -0.0064174, -0.0005469, -0.0042051, 0.0040974
9: -0.0038045, -0.0010556, -0.0038576, -0.0010963, -0.0018850, 0.0019419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003684
time: 1.06 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003684
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0046803, -0.0015126, -0.0046841, -0.0018644, -0.0022436, 0.0025475
1: -0.0047265, -0.0041645, -0.0047266, -0.0042413, -0.0003764, 0.0004229
2: 0.0086316, 0.0126325, 0.0086296, 0.0121719, -0.0028013, 0.0031876
3: 1.0083328, 1.0092824, 1.0084245, 1.0092853, -0.0007932, 0.0007412
4: -0.0036632, -0.0030490, -0.0035894, -0.0030492, -0.0004837, 0.0004244
5: 0.0003712, 0.0027911, 0.0003685, 0.0025210, -0.0017112, 0.0019437
6: -0.0025778, -0.0024455, -0.0025806, -0.0024477, -0.0001247, 0.0001253
7: -0.0111742, -0.0054689, -0.0106467, -0.0054399, -0.0047704, 0.0042756
8: -0.0071136, -0.0007249, -0.0063263, -0.0007276, -0.0049935, 0.0043789
9: -0.0037696, -0.0007589, -0.0037697, -0.0011398, -0.0020434, 0.0023294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003171, upper bound: 0.0002324
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004110, upper bound: 0.0003592
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046803, -0.0015126, -0.0047666, -0.0018273, -0.0021841, 0.0025337
1: -0.0047265, -0.0041645, -0.0047452, -0.0042330, -0.0003621, 0.0004170
2: 0.0086316, 0.0126325, 0.0085218, 0.0122219, -0.0027215, 0.0031640
3: 1.0083328, 1.0092824, 1.0084136, 1.0093130, -0.0007925, 0.0007275
4: -0.0036632, -0.0030490, -0.0035978, -0.0030322, -0.0004789, 0.0004113
5: 0.0003712, 0.0027911, 0.0003052, 0.0025495, -0.0016653, 0.0019326
6: -0.0025778, -0.0024455, -0.0025820, -0.0024469, -0.0001256, 0.0001266
7: -0.0111742, -0.0054689, -0.0107034, -0.0053168, -0.0047734, 0.0042160
8: -0.0071136, -0.0007249, -0.0064174, -0.0005469, -0.0049355, 0.0042395
9: -0.0037696, -0.0007589, -0.0038576, -0.0010963, -0.0019768, 0.0022996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003171, upper bound: 0.0002324
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004110, upper bound: 0.0003592
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0047176, -0.0017941, -0.0046887, -0.0017839, -0.0022811, 0.0022133
1: -0.0047499, -0.0042324, -0.0047128, -0.0042192, -0.0003808, 0.0003221
2: 0.0085716, 0.0122613, 0.0086369, 0.0122847, -0.0028494, 0.0027254
3: 1.0084333, 1.0093427, 1.0083853, 1.0092518, -0.0006441, 0.0007967
4: -0.0036033, -0.0030374, -0.0036088, -0.0030525, -0.0004053, 0.0004317
5: 0.0003416, 0.0025746, 0.0003661, 0.0025832, -0.0017399, 0.0016849
6: -0.0025751, -0.0024432, -0.0025840, -0.0024505, -0.0001192, 0.0001335
7: -0.0107815, -0.0054642, -0.0107307, -0.0053587, -0.0044182, 0.0043183
8: -0.0064684, -0.0005846, -0.0065397, -0.0007770, -0.0041265, 0.0044551
9: -0.0038486, -0.0010756, -0.0037366, -0.0010343, -0.0020793, 0.0018918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003923
time: 1.04 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003923
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0046803, -0.0015182, -0.0046887, -0.0017839, -0.0023135, 0.0025325
1: -0.0047441, -0.0041700, -0.0047128, -0.0042192, -0.0004152, 0.0003988
2: 0.0086186, 0.0126204, 0.0086369, 0.0122847, -0.0029123, 0.0031460
3: 1.0083523, 1.0093330, 1.0083853, 1.0092518, -0.0007338, 0.0008276
4: -0.0036603, -0.0030446, -0.0036088, -0.0030525, -0.0004733, 0.0004457
5: 0.0003700, 0.0027864, 0.0003661, 0.0025832, -0.0017665, 0.0019304
6: -0.0025741, -0.0024401, -0.0025840, -0.0024505, -0.0001182, 0.0001349
7: -0.0111989, -0.0055428, -0.0107307, -0.0053587, -0.0048583, 0.0042713
8: -0.0070768, -0.0006606, -0.0065397, -0.0007770, -0.0048567, 0.0046304
9: -0.0038122, -0.0007795, -0.0037366, -0.0010343, -0.0021819, 0.0022496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003923
time: 1.38 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003923
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0047176, -0.0017941, -0.0046886, -0.0017870, -0.0022807, 0.0022148
1: -0.0047499, -0.0042324, -0.0047312, -0.0042246, -0.0003683, 0.0003337
2: 0.0085716, 0.0122613, 0.0086243, 0.0122746, -0.0028402, 0.0027365
3: 1.0084333, 1.0093427, 1.0084034, 1.0093064, -0.0006734, 0.0007545
4: -0.0036033, -0.0030374, -0.0036063, -0.0030481, -0.0004087, 0.0004285
5: 0.0003416, 0.0025746, 0.0003651, 0.0025803, -0.0017390, 0.0016869
6: -0.0025751, -0.0024432, -0.0025810, -0.0024450, -0.0001235, 0.0001283
7: -0.0107815, -0.0054642, -0.0107697, -0.0054292, -0.0043540, 0.0043647
8: -0.0064684, -0.0005846, -0.0065072, -0.0007120, -0.0041748, 0.0044049
9: -0.0038486, -0.0010756, -0.0037801, -0.0010531, -0.0020471, 0.0019236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003995, upper bound: 0.0004077
time: 1.05 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003995, upper bound: 0.0004077
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046803, -0.0015182, -0.0046886, -0.0017870, -0.0023179, 0.0025339
1: -0.0047441, -0.0041700, -0.0047312, -0.0042246, -0.0004054, 0.0004100
2: 0.0086186, 0.0126204, 0.0086243, 0.0122746, -0.0029088, 0.0031574
3: 1.0083523, 1.0093330, 1.0084034, 1.0093064, -0.0007651, 0.0007892
4: -0.0036603, -0.0030446, -0.0036063, -0.0030481, -0.0004766, 0.0004434
5: 0.0003700, 0.0027864, 0.0003651, 0.0025803, -0.0017691, 0.0019322
6: -0.0025741, -0.0024401, -0.0025810, -0.0024450, -0.0001225, 0.0001294
7: -0.0111989, -0.0055428, -0.0107697, -0.0054292, -0.0048026, 0.0043269
8: -0.0070768, -0.0006606, -0.0065072, -0.0007120, -0.0049044, 0.0045944
9: -0.0038122, -0.0007795, -0.0037801, -0.0010531, -0.0021571, 0.0022807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003995, upper bound: 0.0004077
time: 1.37 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003995, upper bound: 0.0004077
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0047075, -0.0018219, -0.0046004, -0.0015329, -0.0025362, 0.0021507
1: -0.0047444, -0.0042334, -0.0047056, -0.0041742, -0.0004328, 0.0003549
2: 0.0085839, 0.0122300, 0.0087365, 0.0126031, -0.0031825, 0.0026781
3: 1.0084360, 1.0093144, 1.0083437, 1.0092559, -0.0006907, 0.0008109
4: -0.0035990, -0.0030396, -0.0036578, -0.0030658, -0.0004050, 0.0004846
5: 0.0003494, 0.0025538, 0.0004325, 0.0027753, -0.0019357, 0.0016397
6: -0.0025744, -0.0024485, -0.0025771, -0.0024454, -0.0001201, 0.0001222
7: -0.0106926, -0.0054846, -0.0111451, -0.0055844, -0.0041378, 0.0046892
8: -0.0064273, -0.0006112, -0.0070503, -0.0009045, -0.0041799, 0.0050162
9: -0.0038327, -0.0010918, -0.0036806, -0.0007928, -0.0023492, 0.0019501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002412, upper bound: 0.0003206
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003604, upper bound: 0.0003984
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0047075, -0.0018219, -0.0046803, -0.0014962, -0.0025031, 0.0021518
1: -0.0047444, -0.0042334, -0.0047268, -0.0041643, -0.0004171, 0.0003545
2: 0.0085839, 0.0122300, 0.0086309, 0.0126528, -0.0031336, 0.0026792
3: 1.0084360, 1.0093144, 1.0083337, 1.0092883, -0.0006972, 0.0007864
4: -0.0035990, -0.0030396, -0.0036660, -0.0030488, -0.0004046, 0.0004756
5: 0.0003494, 0.0025538, 0.0003711, 0.0028036, -0.0019100, 0.0016406
6: -0.0025744, -0.0024485, -0.0025776, -0.0024448, -0.0001210, 0.0001227
7: -0.0106926, -0.0054846, -0.0112048, -0.0054732, -0.0041493, 0.0046536
8: -0.0064273, -0.0006112, -0.0071412, -0.0007216, -0.0041704, 0.0049121
9: -0.0038327, -0.0010918, -0.0037712, -0.0007476, -0.0022929, 0.0019437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002412, upper bound: 0.0003206
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003604, upper bound: 0.0003984
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0046691, -0.0015412, -0.0046004, -0.0015329, -0.0022934, 0.0022305
1: -0.0047385, -0.0041709, -0.0047056, -0.0041742, -0.0004001, 0.0003708
2: 0.0086323, 0.0125955, 0.0087365, 0.0126031, -0.0028725, 0.0027734
3: 1.0083545, 1.0093052, 1.0083437, 1.0092559, -0.0007366, 0.0007996
4: -0.0036570, -0.0030470, -0.0036578, -0.0030658, -0.0004190, 0.0004375
5: 0.0003786, 0.0027692, 0.0004325, 0.0027753, -0.0017498, 0.0017001
6: -0.0025734, -0.0024454, -0.0025771, -0.0024454, -0.0001222, 0.0001262
7: -0.0111285, -0.0055647, -0.0111451, -0.0055844, -0.0043066, 0.0043246
8: -0.0070474, -0.0006898, -0.0070503, -0.0009045, -0.0043207, 0.0045289
9: -0.0037957, -0.0007909, -0.0036806, -0.0007928, -0.0021256, 0.0020154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003630
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004069, upper bound: 0.0003964
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046691, -0.0015412, -0.0046803, -0.0014962, -0.0022675, 0.0022362
1: -0.0047385, -0.0041709, -0.0047268, -0.0041643, -0.0003884, 0.0003714
2: 0.0086323, 0.0125955, 0.0086309, 0.0126528, -0.0028344, 0.0027798
3: 1.0083545, 1.0093052, 1.0083337, 1.0092883, -0.0007443, 0.0007815
4: -0.0036570, -0.0030470, -0.0036660, -0.0030488, -0.0004196, 0.0004303
5: 0.0003786, 0.0027692, 0.0003711, 0.0028036, -0.0017295, 0.0017044
6: -0.0025734, -0.0024454, -0.0025776, -0.0024448, -0.0001236, 0.0001273
7: -0.0111285, -0.0055647, -0.0112048, -0.0054732, -0.0043346, 0.0043094
8: -0.0070474, -0.0006898, -0.0071412, -0.0007216, -0.0043239, 0.0044456
9: -0.0037957, -0.0007909, -0.0037712, -0.0007476, -0.0020799, 0.0020154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003630
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004069, upper bound: 0.0003964
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0047118, -0.0018145, -0.0045963, -0.0014625, -0.0026136, 0.0021505
1: -0.0047301, -0.0042274, -0.0047240, -0.0041623, -0.0004317, 0.0003771
2: 0.0085926, 0.0122448, 0.0087262, 0.0126921, -0.0032694, 0.0026975
3: 1.0084151, 1.0092773, 1.0083441, 1.0093113, -0.0007658, 0.0007756
4: -0.0036022, -0.0030431, -0.0036713, -0.0030614, -0.0004113, 0.0004960
5: 0.0003472, 0.0025599, 0.0004343, 0.0028291, -0.0019939, 0.0016412
6: -0.0025785, -0.0024522, -0.0025730, -0.0024393, -0.0001310, 0.0001148
7: -0.0106690, -0.0054006, -0.0112882, -0.0056739, -0.0040401, 0.0049079
8: -0.0064680, -0.0006632, -0.0071899, -0.0008392, -0.0042656, 0.0051190
9: -0.0037988, -0.0010695, -0.0037236, -0.0007277, -0.0023871, 0.0020040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0003894
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0003894
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0047117, -0.0018198, -0.0045963, -0.0014625, -0.0026172, 0.0021538
1: -0.0047489, -0.0042330, -0.0047240, -0.0041623, -0.0004442, 0.0003669
2: 0.0085791, 0.0122328, 0.0087262, 0.0126921, -0.0032830, 0.0026929
3: 1.0084344, 1.0093322, 1.0083441, 1.0093113, -0.0007249, 0.0008073
4: -0.0035994, -0.0030385, -0.0036713, -0.0030614, -0.0004089, 0.0004997
5: 0.0003462, 0.0025554, 0.0004343, 0.0028291, -0.0019974, 0.0016430
6: -0.0025748, -0.0024468, -0.0025730, -0.0024393, -0.0001254, 0.0001188
7: -0.0107003, -0.0054740, -0.0112882, -0.0056739, -0.0040862, 0.0048569
8: -0.0064312, -0.0005965, -0.0071899, -0.0008392, -0.0042280, 0.0051687
9: -0.0038429, -0.0010900, -0.0037236, -0.0007277, -0.0024194, 0.0019788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004182
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004182
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0046735, -0.0015341, -0.0045963, -0.0014625, -0.0023602, 0.0022385
1: -0.0047251, -0.0041650, -0.0047240, -0.0041623, -0.0003968, 0.0003938
2: 0.0086403, 0.0126103, 0.0087262, 0.0126921, -0.0029446, 0.0028006
3: 1.0083336, 1.0092717, 1.0083441, 1.0093113, -0.0008118, 0.0007638
4: -0.0036603, -0.0030504, -0.0036713, -0.0030614, -0.0004265, 0.0004462
5: 0.0003764, 0.0027751, 0.0004343, 0.0028291, -0.0017999, 0.0017077
6: -0.0025775, -0.0024485, -0.0025730, -0.0024393, -0.0001327, 0.0001202
7: -0.0111082, -0.0054798, -0.0112882, -0.0056739, -0.0042297, 0.0045248
8: -0.0070883, -0.0007392, -0.0071899, -0.0008392, -0.0044215, 0.0046063
9: -0.0037628, -0.0007688, -0.0037236, -0.0007277, -0.0021520, 0.0020769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0003926
time: 1.46 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0003926
time: 1.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0046737, -0.0015396, -0.0045963, -0.0014625, -0.0023643, 0.0022388
1: -0.0047429, -0.0041705, -0.0047240, -0.0041623, -0.0004105, 0.0003836
2: 0.0086270, 0.0125978, 0.0087262, 0.0126921, -0.0029601, 0.0027942
3: 1.0083532, 1.0093220, 1.0083441, 1.0093113, -0.0007707, 0.0007942
4: -0.0036574, -0.0030459, -0.0036713, -0.0030614, -0.0004240, 0.0004506
5: 0.0003751, 0.0027704, 0.0004343, 0.0028291, -0.0018038, 0.0017074
6: -0.0025738, -0.0024435, -0.0025730, -0.0024393, -0.0001268, 0.0001233
7: -0.0111327, -0.0055537, -0.0112882, -0.0056739, -0.0042703, 0.0044661
8: -0.0070507, -0.0006743, -0.0071899, -0.0008392, -0.0043823, 0.0046648
9: -0.0038057, -0.0007894, -0.0037236, -0.0007277, -0.0021894, 0.0020513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0004122
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0004122
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0047680, -0.0018273, -0.0045989, -0.0015514, -0.0025868, 0.0021825
1: -0.0047458, -0.0042330, -0.0047053, -0.0041747, -0.0004529, 0.0003620
2: 0.0085195, 0.0122219, 0.0087385, 0.0125821, -0.0032506, 0.0027192
3: 1.0084136, 1.0093156, 1.0083443, 1.0092503, -0.0007193, 0.0008457
4: -0.0035978, -0.0030317, -0.0036549, -0.0030661, -0.0004112, 0.0004956
5: 0.0003040, 0.0025495, 0.0004337, 0.0027614, -0.0019747, 0.0016641
6: -0.0025820, -0.0024466, -0.0025770, -0.0024461, -0.0001258, 0.0001226
7: -0.0107054, -0.0053168, -0.0111079, -0.0055870, -0.0041836, 0.0048004
8: -0.0064174, -0.0005417, -0.0070232, -0.0009078, -0.0042433, 0.0051361
9: -0.0038602, -0.0010963, -0.0036791, -0.0008041, -0.0024132, 0.0019800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002324, upper bound: 0.0003168
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003596, upper bound: 0.0004017
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0047680, -0.0018273, -0.0046787, -0.0015126, -0.0025529, 0.0021827
1: -0.0047458, -0.0042330, -0.0047265, -0.0041647, -0.0004397, 0.0003621
2: 0.0085195, 0.0122219, 0.0086329, 0.0126325, -0.0031999, 0.0027203
3: 1.0084136, 1.0093156, 1.0083340, 1.0092824, -0.0007275, 0.0008320
4: -0.0035978, -0.0030317, -0.0036632, -0.0030491, -0.0004112, 0.0004869
5: 0.0003040, 0.0025495, 0.0003723, 0.0027911, -0.0019481, 0.0016643
6: -0.0025820, -0.0024466, -0.0025775, -0.0024455, -0.0001266, 0.0001232
7: -0.0107054, -0.0053168, -0.0111742, -0.0054758, -0.0041950, 0.0047734
8: -0.0064174, -0.0005417, -0.0071136, -0.0007249, -0.0042395, 0.0050358
9: -0.0038602, -0.0010963, -0.0037696, -0.0007591, -0.0023606, 0.0019768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002324, upper bound: 0.0003168
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003596, upper bound: 0.0004017
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0047202, -0.0015520, -0.0045989, -0.0015514, -0.0023229, 0.0022588
1: -0.0047405, -0.0041705, -0.0047053, -0.0041747, -0.0003993, 0.0003781
2: 0.0085782, 0.0125804, 0.0087385, 0.0125821, -0.0029028, 0.0028104
3: 1.0083320, 1.0093088, 1.0083443, 1.0092503, -0.0007651, 0.0008030
4: -0.0036548, -0.0030406, -0.0036549, -0.0030661, -0.0004246, 0.0004403
5: 0.0003403, 0.0027607, 0.0004337, 0.0027614, -0.0017718, 0.0017219
6: -0.0025809, -0.0024431, -0.0025770, -0.0024461, -0.0001283, 0.0001282
7: -0.0111252, -0.0054135, -0.0111079, -0.0055870, -0.0043586, 0.0044408
8: -0.0070239, -0.0006330, -0.0070232, -0.0009078, -0.0043797, 0.0045495
9: -0.0038182, -0.0008011, -0.0036791, -0.0008041, -0.0021315, 0.0020445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003755, upper bound: 0.0003646
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004069, upper bound: 0.0003994
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0047202, -0.0015520, -0.0046787, -0.0015126, -0.0022977, 0.0022636
1: -0.0047405, -0.0041705, -0.0047265, -0.0041647, -0.0003875, 0.0003793
2: 0.0085782, 0.0125804, 0.0086329, 0.0126325, -0.0028622, 0.0028162
3: 1.0083320, 1.0093088, 1.0083340, 1.0092824, -0.0007757, 0.0007872
4: -0.0036548, -0.0030406, -0.0036632, -0.0030491, -0.0004255, 0.0004329
5: 0.0003403, 0.0027607, 0.0003723, 0.0027911, -0.0017517, 0.0017255
6: -0.0025809, -0.0024431, -0.0025775, -0.0024455, -0.0001295, 0.0001292
7: -0.0111252, -0.0054135, -0.0111742, -0.0054758, -0.0043856, 0.0044358
8: -0.0070239, -0.0006330, -0.0071136, -0.0007249, -0.0043852, 0.0044634
9: -0.0038182, -0.0008011, -0.0037696, -0.0007591, -0.0020833, 0.0020446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003755, upper bound: 0.0003646
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004069, upper bound: 0.0003994
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0047731, -0.0018213, -0.0045949, -0.0014790, -0.0026694, 0.0021800
1: -0.0047322, -0.0042270, -0.0047237, -0.0041627, -0.0004514, 0.0003849
2: 0.0085263, 0.0122351, 0.0087280, 0.0126727, -0.0033403, 0.0027371
3: 1.0083946, 1.0092797, 1.0083445, 1.0093064, -0.0007934, 0.0008076
4: -0.0036008, -0.0030349, -0.0036685, -0.0030617, -0.0004177, 0.0005069
5: 0.0003013, 0.0025546, 0.0004354, 0.0028166, -0.0020366, 0.0016640
6: -0.0025855, -0.0024501, -0.0025729, -0.0024397, -0.0001361, 0.0001156
7: -0.0106780, -0.0052338, -0.0112572, -0.0056763, -0.0040864, 0.0050236
8: -0.0064551, -0.0005880, -0.0071633, -0.0008421, -0.0043328, 0.0052363
9: -0.0038291, -0.0010751, -0.0037223, -0.0007390, -0.0024499, 0.0020367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003899
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003899
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0047721, -0.0018247, -0.0045949, -0.0014790, -0.0026726, 0.0021853
1: -0.0047503, -0.0042326, -0.0047237, -0.0041627, -0.0004655, 0.0003748
2: 0.0085148, 0.0122251, 0.0087280, 0.0126727, -0.0033552, 0.0027344
3: 1.0084121, 1.0093336, 1.0083445, 1.0093064, -0.0007540, 0.0008408
4: -0.0035983, -0.0030308, -0.0036685, -0.0030617, -0.0004155, 0.0005112
5: 0.0003009, 0.0025515, 0.0004354, 0.0028166, -0.0020400, 0.0016672
6: -0.0025824, -0.0024447, -0.0025729, -0.0024397, -0.0001307, 0.0001190
7: -0.0107148, -0.0053068, -0.0112572, -0.0056763, -0.0041342, 0.0049635
8: -0.0064217, -0.0005283, -0.0071633, -0.0008421, -0.0042977, 0.0052961
9: -0.0038692, -0.0010942, -0.0037223, -0.0007390, -0.0024871, 0.0020126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004229
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004229
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0047252, -0.0015456, -0.0045949, -0.0014790, -0.0023943, 0.0022650
1: -0.0047271, -0.0041646, -0.0047237, -0.0041627, -0.0003960, 0.0004015
2: 0.0085854, 0.0125936, 0.0087280, 0.0126727, -0.0029789, 0.0028359
3: 1.0083133, 1.0092742, 1.0083445, 1.0093064, -0.0008392, 0.0007652
4: -0.0036578, -0.0030439, -0.0036685, -0.0030617, -0.0004321, 0.0004497
5: 0.0003377, 0.0027661, 0.0004354, 0.0028166, -0.0018252, 0.0017281
6: -0.0025846, -0.0024467, -0.0025729, -0.0024397, -0.0001381, 0.0001220
7: -0.0111029, -0.0053310, -0.0112572, -0.0056763, -0.0042779, 0.0046430
8: -0.0070624, -0.0006816, -0.0071633, -0.0008421, -0.0044803, 0.0046319
9: -0.0037860, -0.0007793, -0.0037223, -0.0007390, -0.0021605, 0.0021057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0003944
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0003944
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0047248, -0.0015503, -0.0045949, -0.0014790, -0.0023971, 0.0022666
1: -0.0047449, -0.0041701, -0.0047237, -0.0041627, -0.0004098, 0.0003916
2: 0.0085729, 0.0125829, 0.0087280, 0.0126727, -0.0029941, 0.0028310
3: 1.0083306, 1.0093260, 1.0083445, 1.0093064, -0.0008002, 0.0007973
4: -0.0036551, -0.0030396, -0.0036685, -0.0030617, -0.0004298, 0.0004540
5: 0.0003369, 0.0027619, 0.0004354, 0.0028166, -0.0018284, 0.0017288
6: -0.0025813, -0.0024411, -0.0025729, -0.0024397, -0.0001327, 0.0001252
7: -0.0111333, -0.0054026, -0.0112572, -0.0056763, -0.0043220, 0.0045795
8: -0.0070275, -0.0006184, -0.0071633, -0.0008421, -0.0044444, 0.0046901
9: -0.0038282, -0.0007994, -0.0037223, -0.0007390, -0.0021974, 0.0020815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0004154
time: 1.36 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0004153
time: 1.19 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.09 seconds
IS_A1_B1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003527, upper bound: 0.0002969
IS_A1_B1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003920, upper bound: 0.0003756
IS_A1_B1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003527, upper bound: 0.0003005
IS_A1_B1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003920, upper bound: 0.0003756
IS_A1_B1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003727
IS_A1_B1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003727
IS_A1_B1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003727
IS_A1_B1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003727
IS_A1_B1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003953
IS_A1_B1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003953
IS_A1_B1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003953
IS_A1_B1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003953
IS_A1_B1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003947, upper bound: 0.0004094
IS_A1_B1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003947, upper bound: 0.0004094
IS_A1_B1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003947, upper bound: 0.0004101
IS_A1_B1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003947, upper bound: 0.0004101
IS_A1_B1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003525, upper bound: 0.0002917
IS_A1_B1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003941, upper bound: 0.0003754
IS_A1_B1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003525, upper bound: 0.0002956
IS_A1_B1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003941, upper bound: 0.0003754
IS_A1_B1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003723
IS_A1_B1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003723
IS_A1_B1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003723
IS_A1_B1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003723
IS_A1_B1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003952
IS_A1_B1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003952
IS_A1_B1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003952
IS_A1_B1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003952
IS_A1_B1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003969, upper bound: 0.0004094
IS_A1_B1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003969, upper bound: 0.0004094
IS_A1_B1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003969, upper bound: 0.0004101
IS_A1_B1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003969, upper bound: 0.0004101
IS_A1_B2_B1_A1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0002927
IS_A1_B2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003804, upper bound: 0.0003918
IS_A1_B2_B1_A1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0002927
IS_A1_B2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003804, upper bound: 0.0003918
IS_A1_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003600
IS_A1_B2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004075, upper bound: 0.0003853
IS_A1_B2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003605
IS_A1_B2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004075, upper bound: 0.0003885
IS_A1_B2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004206
IS_A1_B2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004217
IS_A1_B2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004300
IS_A1_B2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004337
IS_A1_B2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004032, upper bound: 0.0004102
IS_A1_B2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004032, upper bound: 0.0004138
IS_A1_B2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004032, upper bound: 0.0004166
IS_A1_B2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004032, upper bound: 0.0004202
IS_A1_B2_B2_A1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003124, upper bound: 0.0002897
IS_A1_B2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003826, upper bound: 0.0003918
IS_A1_B2_B2_A1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003124, upper bound: 0.0002897
IS_A1_B2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003826, upper bound: 0.0003918
IS_A1_B2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003715, upper bound: 0.0003596
IS_A1_B2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004114, upper bound: 0.0003853
IS_A1_B2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003715, upper bound: 0.0003603
IS_A1_B2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004114, upper bound: 0.0003885
IS_A1_B2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004206
IS_A1_B2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004217
IS_A1_B2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004300
IS_A1_B2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0004337
IS_A1_B2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004055, upper bound: 0.0004102
IS_A1_B2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004055, upper bound: 0.0004138
IS_A1_B2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004055, upper bound: 0.0004166
IS_A1_B2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004055, upper bound: 0.0004202
IS_A2_B1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003693
IS_A2_B1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003693
IS_A2_B1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003693
IS_A2_B1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003693
IS_A2_B1_B1_A1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003210, upper bound: 0.0002412
IS_A2_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004085, upper bound: 0.0003601
IS_A2_B1_B1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003210, upper bound: 0.0002412
IS_A2_B1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004085, upper bound: 0.0003601
IS_A2_B1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003926
IS_A2_B1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003926
IS_A2_B1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003926
IS_A2_B1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003693, upper bound: 0.0003926
IS_A2_B1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003981, upper bound: 0.0004077
IS_A2_B1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003981, upper bound: 0.0004077
IS_A2_B1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003981, upper bound: 0.0004077
IS_A2_B1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003981, upper bound: 0.0004077
IS_A2_B1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003684
IS_A2_B1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003684
IS_A2_B1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003684
IS_A2_B1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003684
IS_A2_B1_B2_A1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003171, upper bound: 0.0002324
IS_A2_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004110, upper bound: 0.0003592
IS_A2_B1_B2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003171, upper bound: 0.0002324
IS_A2_B1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004110, upper bound: 0.0003592
IS_A2_B1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003923
IS_A2_B1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003923
IS_A2_B1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003923
IS_A2_B1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003923
IS_A2_B1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003995, upper bound: 0.0004077
IS_A2_B1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003995, upper bound: 0.0004077
IS_A2_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003995, upper bound: 0.0004077
IS_A2_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003995, upper bound: 0.0004077
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0002412, upper bound: 0.0003206
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003604, upper bound: 0.0003984
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0002412, upper bound: 0.0003206
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003604, upper bound: 0.0003984
IS_A2_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003630
IS_A2_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004069, upper bound: 0.0003964
IS_A2_B2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003630
IS_A2_B2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004069, upper bound: 0.0003964
IS_A2_B2_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0003894
IS_A2_B2_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0003894
IS_A2_B2_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004182
IS_A2_B2_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003696, upper bound: 0.0004182
IS_A2_B2_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0003926
IS_A2_B2_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0003926
IS_A2_B2_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0004122
IS_A2_B2_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0004122
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0002324, upper bound: 0.0003168
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003596, upper bound: 0.0004017
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0002324, upper bound: 0.0003168
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003596, upper bound: 0.0004017
IS_A2_B2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003755, upper bound: 0.0003646
IS_A2_B2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004069, upper bound: 0.0003994
IS_A2_B2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003755, upper bound: 0.0003646
IS_A2_B2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004069, upper bound: 0.0003994
IS_A2_B2_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003899
IS_A2_B2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003899
IS_A2_B2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004229
IS_A2_B2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0004229
IS_A2_B2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0003944
IS_A2_B2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0003944
IS_A2_B2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0004154
IS_A2_B2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.09
Output dim: 3, lower bound: -0.0004164, upper bound: 0.0004153

## BFS IS instance: IS_A1_B1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046428, -0.0018112, -0.0046282, -0.0018739, -0.0020784, 0.0021524
1: -0.0047132, -0.0042342, -0.0047257, -0.0042447, -0.0003038, 0.0003352
2: 0.0086831, 0.0122407, 0.0086862, 0.0121600, -0.0025623, 0.0026751
3: 1.0084243, 1.0092663, 1.0084536, 1.0092839, -0.0006802, 0.0006223
4: -0.0036002, -0.0030576, -0.0035878, -0.0030556, -0.0004025, 0.0003818
5: 0.0004001, 0.0025617, 0.0004101, 0.0025137, -0.0015825, 0.0016407
6: -0.0025778, -0.0024490, -0.0025729, -0.0024497, -0.0001218, 0.0001175
7: -0.0107217, -0.0055021, -0.0106175, -0.0056047, -0.0041360, 0.0041212
8: -0.0064392, -0.0008170, -0.0063080, -0.0007816, -0.0041306, 0.0038930
9: -0.0037236, -0.0010870, -0.0037501, -0.0011488, -0.0017865, 0.0019112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003752, upper bound: 0.0003756
time: 1.01 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003752, upper bound: 0.0003756
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045985, -0.0015329, -0.0046282, -0.0018739, -0.0020509, 0.0024730
1: -0.0047046, -0.0041742, -0.0047257, -0.0042447, -0.0003079, 0.0004111
2: 0.0087396, 0.0126031, 0.0086862, 0.0121600, -0.0025351, 0.0030972
3: 1.0083437, 1.0092524, 1.0084536, 1.0092839, -0.0007716, 0.0006220
4: -0.0036578, -0.0030664, -0.0035878, -0.0030556, -0.0004703, 0.0003793
5: 0.0004340, 0.0027753, 0.0004101, 0.0025137, -0.0015620, 0.0018870
6: -0.0025771, -0.0024460, -0.0025729, -0.0024497, -0.0001210, 0.0001201
7: -0.0111419, -0.0055844, -0.0106175, -0.0056047, -0.0046010, 0.0040337
8: -0.0070503, -0.0009117, -0.0063080, -0.0007816, -0.0048584, 0.0038763
9: -0.0036767, -0.0007928, -0.0037501, -0.0011488, -0.0017840, 0.0022670

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003960, upper bound: 0.0003756
time: 1.03 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003960, upper bound: 0.0003756
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0046428, -0.0018112, -0.0047118, -0.0018145, -0.0021592, 0.0022172
1: -0.0047132, -0.0042342, -0.0047301, -0.0042274, -0.0003327, 0.0003426
2: 0.0086831, 0.0122407, 0.0085926, 0.0122448, -0.0026777, 0.0027509
3: 1.0084243, 1.0092663, 1.0084151, 1.0092773, -0.0006879, 0.0006780
4: -0.0036002, -0.0030576, -0.0036022, -0.0030431, -0.0004133, 0.0004018
5: 0.0004001, 0.0025617, 0.0003472, 0.0025599, -0.0016453, 0.0016896
6: -0.0025778, -0.0024490, -0.0025785, -0.0024522, -0.0001198, 0.0001233
7: -0.0107217, -0.0055021, -0.0106690, -0.0054006, -0.0043026, 0.0041868
8: -0.0064392, -0.0008170, -0.0064680, -0.0006632, -0.0042369, 0.0041165
9: -0.0037236, -0.0010870, -0.0037988, -0.0010695, -0.0019024, 0.0019586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003727
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003727
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046428, -0.0018112, -0.0047117, -0.0018198, -0.0021568, 0.0022199
1: -0.0047132, -0.0042342, -0.0047489, -0.0042330, -0.0003276, 0.0003619
2: 0.0086831, 0.0122407, 0.0085791, 0.0122328, -0.0026684, 0.0027661
3: 1.0084243, 1.0092663, 1.0084344, 1.0093322, -0.0007364, 0.0006574
4: -0.0036002, -0.0030576, -0.0035994, -0.0030385, -0.0004180, 0.0003993
5: 0.0004001, 0.0025617, 0.0003462, 0.0025554, -0.0016430, 0.0016926
6: -0.0025778, -0.0024490, -0.0025748, -0.0024468, -0.0001249, 0.0001192
7: -0.0107217, -0.0055021, -0.0107003, -0.0054740, -0.0042412, 0.0042253
8: -0.0064392, -0.0008170, -0.0064312, -0.0005965, -0.0043058, 0.0040847
9: -0.0037236, -0.0010870, -0.0038429, -0.0010900, -0.0018844, 0.0020046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003727
time: 0.97 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003727
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045985, -0.0015329, -0.0047118, -0.0018145, -0.0021359, 0.0025377
1: -0.0047046, -0.0041742, -0.0047301, -0.0042274, -0.0003372, 0.0004186
2: 0.0087396, 0.0126031, 0.0085926, 0.0122448, -0.0026554, 0.0031730
3: 1.0083437, 1.0092524, 1.0084151, 1.0092773, -0.0007792, 0.0006798
4: -0.0036578, -0.0030664, -0.0036022, -0.0030431, -0.0004811, 0.0003998
5: 0.0004340, 0.0027753, 0.0003472, 0.0025599, -0.0016281, 0.0019359
6: -0.0025771, -0.0024460, -0.0025785, -0.0024522, -0.0001190, 0.0001259
7: -0.0111419, -0.0055844, -0.0106690, -0.0054006, -0.0047677, 0.0041106
8: -0.0070503, -0.0009117, -0.0064680, -0.0006632, -0.0049647, 0.0041057
9: -0.0036767, -0.0007928, -0.0037988, -0.0010695, -0.0019040, 0.0023144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0003727
time: 1.02 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0003727
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045985, -0.0015329, -0.0047117, -0.0018198, -0.0021335, 0.0025404
1: -0.0047046, -0.0041742, -0.0047489, -0.0042330, -0.0003321, 0.0004379
2: 0.0087396, 0.0126031, 0.0085791, 0.0122328, -0.0026461, 0.0031882
3: 1.0083437, 1.0092524, 1.0084344, 1.0093322, -0.0008278, 0.0006592
4: -0.0036578, -0.0030664, -0.0035994, -0.0030385, -0.0004858, 0.0003973
5: 0.0004340, 0.0027753, 0.0003462, 0.0025554, -0.0016257, 0.0019389
6: -0.0025771, -0.0024460, -0.0025748, -0.0024468, -0.0001241, 0.0001218
7: -0.0111419, -0.0055844, -0.0107003, -0.0054740, -0.0047063, 0.0041491
8: -0.0070503, -0.0009117, -0.0064312, -0.0005965, -0.0050335, 0.0040739
9: -0.0036767, -0.0007928, -0.0038429, -0.0010900, -0.0018860, 0.0023604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0003727
time: 1.10 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003894, upper bound: 0.0003727
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0046393, -0.0018158, -0.0046357, -0.0018514, -0.0021292, 0.0021555
1: -0.0047317, -0.0042401, -0.0047118, -0.0042353, -0.0003412, 0.0003168
2: 0.0086724, 0.0122292, 0.0086922, 0.0121953, -0.0026512, 0.0026602
3: 1.0084442, 1.0093193, 1.0084258, 1.0092499, -0.0006274, 0.0007166
4: -0.0035975, -0.0030531, -0.0035942, -0.0030590, -0.0003968, 0.0003999
5: 0.0004016, 0.0025578, 0.0004056, 0.0025314, -0.0016234, 0.0016414
6: -0.0025740, -0.0024435, -0.0025774, -0.0024529, -0.0001149, 0.0001276
7: -0.0107565, -0.0055833, -0.0106195, -0.0055142, -0.0042668, 0.0040683
8: -0.0064029, -0.0007510, -0.0063824, -0.0008319, -0.0040486, 0.0041125
9: -0.0037673, -0.0011076, -0.0037164, -0.0011099, -0.0019082, 0.0018594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003040, upper bound: 0.0003259
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003752, upper bound: 0.0003939
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045949, -0.0015378, -0.0046357, -0.0018514, -0.0021045, 0.0024761
1: -0.0047231, -0.0041801, -0.0047118, -0.0042353, -0.0003479, 0.0003931
2: 0.0087285, 0.0125920, 0.0086922, 0.0121953, -0.0026297, 0.0030827
3: 1.0083663, 1.0093052, 1.0084258, 1.0092499, -0.0007167, 0.0007180
4: -0.0036551, -0.0030618, -0.0035942, -0.0030590, -0.0004650, 0.0003987
5: 0.0004354, 0.0027712, 0.0004056, 0.0025314, -0.0016053, 0.0018878
6: -0.0025730, -0.0024403, -0.0025774, -0.0024529, -0.0001137, 0.0001300
7: -0.0111748, -0.0056744, -0.0106195, -0.0055142, -0.0047278, 0.0039881
8: -0.0070164, -0.0008442, -0.0063824, -0.0008319, -0.0047805, 0.0041126
9: -0.0037209, -0.0008125, -0.0037164, -0.0011099, -0.0019166, 0.0022174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003040, upper bound: 0.0003259
time: 0.78 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003752, upper bound: 0.0003939
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0046393, -0.0018158, -0.0047089, -0.0018145, -0.0021576, 0.0022122
1: -0.0047317, -0.0042401, -0.0047301, -0.0042277, -0.0003505, 0.0003374
2: 0.0086724, 0.0122292, 0.0085950, 0.0122448, -0.0026894, 0.0027400
3: 1.0084442, 1.0093193, 1.0084177, 1.0092773, -0.0006639, 0.0007249
4: -0.0035975, -0.0030531, -0.0036022, -0.0030433, -0.0004107, 0.0004062
5: 0.0004016, 0.0025578, 0.0003493, 0.0025599, -0.0016452, 0.0016854
6: -0.0025740, -0.0024435, -0.0025779, -0.0024522, -0.0001155, 0.0001279
7: -0.0107565, -0.0055833, -0.0106690, -0.0054125, -0.0043311, 0.0041047
8: -0.0064029, -0.0007510, -0.0064680, -0.0006632, -0.0042045, 0.0041815
9: -0.0037673, -0.0011076, -0.0037988, -0.0010699, -0.0019446, 0.0019399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002200, upper bound: 0.0002978
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003595, upper bound: 0.0003853
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045949, -0.0015378, -0.0047089, -0.0018145, -0.0021329, 0.0025329
1: -0.0047231, -0.0041801, -0.0047301, -0.0042277, -0.0003572, 0.0004137
2: 0.0087285, 0.0125920, 0.0085950, 0.0122448, -0.0026679, 0.0031624
3: 1.0083663, 1.0093052, 1.0084177, 1.0092773, -0.0007531, 0.0007263
4: -0.0036551, -0.0030618, -0.0036022, -0.0030433, -0.0004789, 0.0004050
5: 0.0004354, 0.0027712, 0.0003493, 0.0025599, -0.0016272, 0.0019318
6: -0.0025730, -0.0024403, -0.0025779, -0.0024522, -0.0001143, 0.0001303
7: -0.0111748, -0.0056744, -0.0106690, -0.0054125, -0.0047921, 0.0040245
8: -0.0070164, -0.0008442, -0.0064680, -0.0006632, -0.0049364, 0.0041816
9: -0.0037209, -0.0008125, -0.0037988, -0.0010699, -0.0019530, 0.0022979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002200, upper bound: 0.0003055
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003595, upper bound: 0.0003853
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0046393, -0.0018158, -0.0046325, -0.0018561, -0.0021274, 0.0021554
1: -0.0047317, -0.0042401, -0.0047303, -0.0042412, -0.0003296, 0.0003287
2: 0.0086724, 0.0122292, 0.0086812, 0.0121830, -0.0026399, 0.0026703
3: 1.0084442, 1.0093193, 1.0084459, 1.0093026, -0.0006554, 0.0006709
4: -0.0035975, -0.0030531, -0.0035914, -0.0030545, -0.0004002, 0.0003963
5: 0.0004016, 0.0025578, 0.0004068, 0.0025273, -0.0016212, 0.0016422
6: -0.0025740, -0.0024435, -0.0025736, -0.0024474, -0.0001189, 0.0001220
7: -0.0107565, -0.0055833, -0.0106528, -0.0055945, -0.0041945, 0.0041153
8: -0.0064029, -0.0007510, -0.0063453, -0.0007655, -0.0040970, 0.0040626
9: -0.0037673, -0.0011076, -0.0037604, -0.0011308, -0.0018781, 0.0018911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003823, upper bound: 0.0003804
time: 0.92 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003935, upper bound: 0.0004051
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045949, -0.0015378, -0.0046325, -0.0018561, -0.0021032, 0.0024762
1: -0.0047231, -0.0041801, -0.0047303, -0.0042412, -0.0003350, 0.0004047
2: 0.0087285, 0.0125920, 0.0086812, 0.0121830, -0.0026177, 0.0030927
3: 1.0083663, 1.0093052, 1.0084459, 1.0093026, -0.0007472, 0.0006734
4: -0.0036551, -0.0030618, -0.0035914, -0.0030545, -0.0004682, 0.0003947
5: 0.0004354, 0.0027712, 0.0004068, 0.0025273, -0.0016034, 0.0018887
6: -0.0025730, -0.0024403, -0.0025736, -0.0024474, -0.0001177, 0.0001246
7: -0.0111748, -0.0056744, -0.0106528, -0.0055945, -0.0046611, 0.0040333
8: -0.0070164, -0.0008442, -0.0063453, -0.0007655, -0.0048265, 0.0040564
9: -0.0037209, -0.0008125, -0.0037604, -0.0011308, -0.0018823, 0.0022474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003823, upper bound: 0.0003804
time: 1.11 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003935, upper bound: 0.0004051
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0046393, -0.0018158, -0.0047117, -0.0018198, -0.0021562, 0.0022182
1: -0.0047317, -0.0042401, -0.0047489, -0.0042330, -0.0003395, 0.0003494
2: 0.0086724, 0.0122292, 0.0085791, 0.0122328, -0.0026782, 0.0027557
3: 1.0084442, 1.0093193, 1.0084344, 1.0093322, -0.0006922, 0.0006861
4: -0.0035975, -0.0030531, -0.0035994, -0.0030385, -0.0004145, 0.0004027
5: 0.0004016, 0.0025578, 0.0003462, 0.0025554, -0.0016434, 0.0016907
6: -0.0025740, -0.0024435, -0.0025748, -0.0024468, -0.0001195, 0.0001233
7: -0.0107565, -0.0055833, -0.0107003, -0.0054740, -0.0042829, 0.0041545
8: -0.0064029, -0.0007510, -0.0064312, -0.0005965, -0.0042542, 0.0041318
9: -0.0037673, -0.0011076, -0.0038429, -0.0010900, -0.0019143, 0.0019721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003484, upper bound: 0.0003805
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003855, upper bound: 0.0004002
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045949, -0.0015378, -0.0047117, -0.0018198, -0.0021321, 0.0025390
1: -0.0047231, -0.0041801, -0.0047489, -0.0042330, -0.0003449, 0.0004253
2: 0.0087285, 0.0125920, 0.0085791, 0.0122328, -0.0026560, 0.0031782
3: 1.0083663, 1.0093052, 1.0084344, 1.0093322, -0.0007839, 0.0006886
4: -0.0036551, -0.0030618, -0.0035994, -0.0030385, -0.0004826, 0.0004010
5: 0.0004354, 0.0027712, 0.0003462, 0.0025554, -0.0016256, 0.0019371
6: -0.0025730, -0.0024403, -0.0025748, -0.0024468, -0.0001184, 0.0001259
7: -0.0111748, -0.0056744, -0.0107003, -0.0054740, -0.0047495, 0.0040725
8: -0.0070164, -0.0008442, -0.0064312, -0.0005965, -0.0049838, 0.0041256
9: -0.0037209, -0.0008125, -0.0038429, -0.0010900, -0.0019185, 0.0023283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1_A2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003484, upper bound: 0.0003805
time: 0.98 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003855, upper bound: 0.0004002
time: 1.60 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046413, -0.0018258, -0.0046838, -0.0018802, -0.0021014, 0.0021823
1: -0.0047128, -0.0042347, -0.0047264, -0.0042444, -0.0003059, 0.0003347
2: 0.0086850, 0.0122234, 0.0086300, 0.0121513, -0.0025882, 0.0027040
3: 1.0084249, 1.0092599, 1.0084313, 1.0092833, -0.0006841, 0.0006564
4: -0.0035977, -0.0030578, -0.0035862, -0.0030493, -0.0004055, 0.0003852
5: 0.0004013, 0.0025507, 0.0003687, 0.0025089, -0.0015998, 0.0016627
6: -0.0025777, -0.0024495, -0.0025803, -0.0024482, -0.0001243, 0.0001235
7: -0.0106931, -0.0055046, -0.0106215, -0.0054404, -0.0042542, 0.0041795
8: -0.0064146, -0.0008201, -0.0062916, -0.0007284, -0.0041514, 0.0039253
9: -0.0037221, -0.0010975, -0.0037693, -0.0011559, -0.0018001, 0.0019161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003754, upper bound: 0.0003754
time: 1.00 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003754, upper bound: 0.0003754
time: 1.42 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045970, -0.0015514, -0.0046838, -0.0018802, -0.0020737, 0.0025029
1: -0.0047042, -0.0041747, -0.0047264, -0.0042444, -0.0003099, 0.0004108
2: 0.0087416, 0.0125821, 0.0086300, 0.0121513, -0.0025608, 0.0031266
3: 1.0083443, 1.0092466, 1.0084313, 1.0092833, -0.0007756, 0.0006520
4: -0.0036549, -0.0030667, -0.0035862, -0.0030493, -0.0004735, 0.0003827
5: 0.0004352, 0.0027614, 0.0003687, 0.0025089, -0.0015792, 0.0019091
6: -0.0025770, -0.0024466, -0.0025803, -0.0024482, -0.0001235, 0.0001261
7: -0.0111048, -0.0055870, -0.0106215, -0.0054404, -0.0047189, 0.0040919
8: -0.0070232, -0.0009150, -0.0062916, -0.0007284, -0.0048813, 0.0039084
9: -0.0036752, -0.0008041, -0.0037693, -0.0011559, -0.0017976, 0.0022732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003973, upper bound: 0.0003754
time: 1.00 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003973, upper bound: 0.0003754
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0046413, -0.0018258, -0.0047704, -0.0018213, -0.0021811, 0.0022490
1: -0.0047128, -0.0042347, -0.0047309, -0.0042270, -0.0003343, 0.0003425
2: 0.0086850, 0.0122234, 0.0085307, 0.0122351, -0.0027023, 0.0027826
3: 1.0084249, 1.0092599, 1.0083946, 1.0092751, -0.0006910, 0.0007057
4: -0.0035977, -0.0030578, -0.0036008, -0.0030357, -0.0004166, 0.0004050
5: 0.0004013, 0.0025507, 0.0003035, 0.0025546, -0.0016618, 0.0017131
6: -0.0025777, -0.0024495, -0.0025855, -0.0024507, -0.0001222, 0.0001284
7: -0.0106931, -0.0055046, -0.0106745, -0.0052338, -0.0044150, 0.0042444
8: -0.0064146, -0.0008201, -0.0064551, -0.0005976, -0.0042607, 0.0041463
9: -0.0037221, -0.0010975, -0.0038239, -0.0010751, -0.0019151, 0.0019660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.52 + 598.24 = 601.76 seconds
