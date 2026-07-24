## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00372996


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0055734, 0.0100193, 0.0055734, 0.0100193, -0.0042068, 0.0042068)
1: (0.0014744, 0.0057097, 0.0014744, 0.0057097, -0.0041603, 0.0041603)
2: (-0.0215303, -0.0109305, -0.0215303, -0.0109305, -0.0075060, 0.0075060)
3: (-0.0048905, 0.0043076, -0.0048905, 0.0043076, -0.0081778, 0.0081778)
4: (0.0146226, 0.0160674, 0.0146226, 0.0160674, -0.0014447, 0.0014447)
5: (-0.0081362, 0.0048168, -0.0081362, 0.0048168, -0.0119655, 0.0119655)
6: (0.9919707, 1.0006981, 0.9919707, 1.0006981, -0.0076799, 0.0076799)
7: (0.0130867, 0.0173991, 0.0130867, 0.0173991, -0.0027001, 0.0027001)
8: (0.0033968, 0.0073235, 0.0033968, 0.0073235, -0.0039268, 0.0039268)
9: (-0.0240450, -0.0151404, -0.0240450, -0.0151404, -0.0070012, 0.0070012)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.63 + 1.57 = 3.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0038762, upper bound: 0.0038762

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038091, upper bound: 0.0038022
time: 0.82 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038280, upper bound: 0.0038280
time: 0.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.65 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 6, lower bound: -0.0038091, upper bound: 0.0038022
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 6, lower bound: -0.0038280, upper bound: 0.0038280

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0057054, 0.0100745, 0.0056275, 0.0100129, -0.0041363, 0.0041709
1: 0.0016001, 0.0057623, 0.0015260, 0.0057036, -0.0040985, 0.0041250
2: -0.0212155, -0.0107988, -0.0214012, -0.0109458, -0.0071975, 0.0074097
3: -0.0050048, 0.0040345, -0.0048772, 0.0041956, -0.0081001, 0.0080063
4: 0.0145882, 0.0160771, 0.0146266, 0.0160662, -0.0014780, 0.0014504
5: -0.0082971, 0.0044321, -0.0081175, 0.0046589, -0.0118586, 0.0117475
6: 0.9918622, 1.0004389, 0.9919833, 1.0005916, -0.0076064, 0.0075134
7: 0.0130243, 0.0172854, 0.0130939, 0.0173525, -0.0026773, 0.0025659
8: 0.0035134, 0.0073723, 0.0034446, 0.0073179, -0.0038045, 0.0039277
9: -0.0237805, -0.0150297, -0.0239365, -0.0151533, -0.0067857, 0.0069212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037448, upper bound: 0.0037724
time: 0.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037792, upper bound: 0.0037724
time: 0.72 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0056276, 0.0100153, 0.0055848, 0.0100184, -0.0041095, 0.0041910
1: 0.0015261, 0.0057059, 0.0014853, 0.0057089, -0.0040683, 0.0041449
2: -0.0214008, -0.0109400, -0.0215030, -0.0109325, -0.0072346, 0.0074727
3: -0.0048823, 0.0041953, -0.0048888, 0.0042840, -0.0081454, 0.0079682
4: 0.0146251, 0.0160667, 0.0146232, 0.0160672, -0.0014421, 0.0014435
5: -0.0081246, 0.0046585, -0.0081337, 0.0047834, -0.0119194, 0.0116790
6: 0.9919784, 1.0005915, 0.9919723, 1.0006756, -0.0076492, 0.0074806
7: 0.0130911, 0.0173524, 0.0130876, 0.0173893, -0.0026875, 0.0026015
8: 0.0034447, 0.0073200, 0.0034068, 0.0073228, -0.0038781, 0.0039132
9: -0.0239362, -0.0151483, -0.0240221, -0.0151421, -0.0067825, 0.0069723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037628, upper bound: 0.0037959
time: 0.80 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037959, upper bound: 0.0037959
time: 0.84 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.29 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 6, lower bound: -0.0037448, upper bound: 0.0037724
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 6, lower bound: -0.0037792, upper bound: 0.0037724
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 6, lower bound: -0.0037628, upper bound: 0.0037959
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 6, lower bound: -0.0037959, upper bound: 0.0037959

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0057300, 0.0100628, 0.0056914, 0.0099812, -0.0040655, 0.0040658
1: 0.0016236, 0.0057511, 0.0015868, 0.0056735, -0.0040303, 0.0040217
2: -0.0211568, -0.0108268, -0.0212489, -0.0110212, -0.0070437, 0.0071885
3: -0.0049805, 0.0039835, -0.0048118, 0.0040634, -0.0078966, 0.0078662
4: 0.0145955, 0.0160750, 0.0146464, 0.0160607, -0.0014651, 0.0014286
5: -0.0082629, 0.0043603, -0.0080253, 0.0044728, -0.0115604, 0.0115442
6: 0.9918853, 1.0003905, 0.9920453, 1.0004663, -0.0074150, 0.0073817
7: 0.0130376, 0.0172642, 0.0131296, 0.0172975, -0.0025976, 0.0025009
8: 0.0035351, 0.0073619, 0.0035010, 0.0072899, -0.0037548, 0.0038610
9: -0.0237312, -0.0150533, -0.0238085, -0.0152166, -0.0066583, 0.0067412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036817, upper bound: 0.0037005
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036832, upper bound: 0.0037044
time: 0.82 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0057241, 0.0100617, 0.0056716, 0.0100426, -0.0040833, 0.0042000
1: 0.0016180, 0.0057501, 0.0015680, 0.0057319, -0.0040475, 0.0041570
2: -0.0211709, -0.0108293, -0.0212959, -0.0108750, -0.0071023, 0.0073538
3: -0.0049783, 0.0039957, -0.0049387, 0.0041043, -0.0081483, 0.0078996
4: 0.0145962, 0.0160748, 0.0146081, 0.0160715, -0.0014753, 0.0014667
5: -0.0082598, 0.0043775, -0.0082040, 0.0045304, -0.0119378, 0.0115947
6: 0.9918873, 1.0004021, 0.9919250, 1.0005052, -0.0076503, 0.0074130
7: 0.0130388, 0.0172693, 0.0130604, 0.0173145, -0.0026379, 0.0025459
8: 0.0035299, 0.0073610, 0.0034835, 0.0073441, -0.0038142, 0.0038775
9: -0.0237430, -0.0150554, -0.0238481, -0.0150938, -0.0066922, 0.0069381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037252, upper bound: 0.0037009
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037112, upper bound: 0.0037049
time: 0.80 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0056521, 0.0100036, 0.0056479, 0.0099869, -0.0040355, 0.0040885
1: 0.0015494, 0.0056948, 0.0015454, 0.0056788, -0.0039972, 0.0040426
2: -0.0213426, -0.0109679, -0.0213526, -0.0110078, -0.0070738, 0.0072488
3: -0.0048581, 0.0041447, -0.0048234, 0.0041535, -0.0079439, 0.0078215
4: 0.0146324, 0.0160646, 0.0146429, 0.0160617, -0.0014292, 0.0014217
5: -0.0080905, 0.0045874, -0.0080417, 0.0045997, -0.0116268, 0.0114659
6: 0.9920014, 1.0005437, 0.9920343, 1.0005518, -0.0074594, 0.0073423
7: 0.0131043, 0.0173313, 0.0131232, 0.0173349, -0.0026064, 0.0025343
8: 0.0034663, 0.0073097, 0.0034625, 0.0072949, -0.0038286, 0.0038471
9: -0.0238873, -0.0151717, -0.0238957, -0.0152053, -0.0066490, 0.0067910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036983, upper bound: 0.0037258
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036995, upper bound: 0.0037283
time: 0.91 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0056469, 0.0100026, 0.0056286, 0.0100483, -0.0040398, 0.0042211
1: 0.0015444, 0.0056938, 0.0015270, 0.0057373, -0.0040004, 0.0041668
2: -0.0213549, -0.0109703, -0.0213985, -0.0108614, -0.0070968, 0.0074163
3: -0.0048560, 0.0041555, -0.0049505, 0.0041932, -0.0081949, 0.0078268
4: 0.0146331, 0.0160644, 0.0146046, 0.0160725, -0.0014394, 0.0014599
5: -0.0080875, 0.0046025, -0.0082206, 0.0046557, -0.0120003, 0.0114774
6: 0.9920034, 1.0005537, 0.9919137, 1.0005896, -0.0076945, 0.0073462
7: 0.0131055, 0.0173358, 0.0130539, 0.0173515, -0.0026454, 0.0025764
8: 0.0034617, 0.0073088, 0.0034455, 0.0073491, -0.0038874, 0.0038632
9: -0.0238977, -0.0151738, -0.0239342, -0.0150823, -0.0066542, 0.0069872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037413, upper bound: 0.0037258
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037284, upper bound: 0.0037284
time: 0.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.12 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.12
Output dim: 6, lower bound: -0.0036817, upper bound: 0.0037005
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.12
Output dim: 6, lower bound: -0.0036832, upper bound: 0.0037044
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.12
Output dim: 6, lower bound: -0.0037252, upper bound: 0.0037009
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.12
Output dim: 6, lower bound: -0.0037112, upper bound: 0.0037049
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.12
Output dim: 6, lower bound: -0.0036983, upper bound: 0.0037258
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.12
Output dim: 6, lower bound: -0.0036995, upper bound: 0.0037283
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 6, lower bound: -0.0037413, upper bound: 0.0037258
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.12
Output dim: 6, lower bound: -0.0037284, upper bound: 0.0037284

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0056495, 0.0099398, 0.0056289, 0.0100410, -0.0039716, 0.0041255
1: 0.0015469, 0.0056340, 0.0015273, 0.0057304, -0.0039292, 0.0040849
2: -0.0213487, -0.0111201, -0.0213978, -0.0108787, -0.0070319, 0.0071997
3: -0.0047260, 0.0041501, -0.0049354, 0.0041926, -0.0079993, 0.0077053
4: 0.0146722, 0.0160534, 0.0146091, 0.0160712, -0.0013990, 0.0014443
5: -0.0079045, 0.0045949, -0.0081994, 0.0046548, -0.0117230, 0.0112894
6: 0.9921268, 1.0005486, 0.9919280, 1.0005888, -0.0075093, 0.0072338
7: 0.0131764, 0.0173336, 0.0130621, 0.0173513, -0.0025599, 0.0025645
8: 0.0034640, 0.0072533, 0.0034458, 0.0073427, -0.0038787, 0.0038075
9: -0.0238925, -0.0152996, -0.0239336, -0.0150969, -0.0065718, 0.0068020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037169, upper bound: 0.0037095
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037169, upper bound: 0.0037258
time: 0.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.02 seconds
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 6, lower bound: -0.0037169, upper bound: 0.0037095
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 6, lower bound: -0.0037169, upper bound: 0.0037258

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.20 + 23.60 = 26.80 seconds
