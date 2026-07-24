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
Threshold: 0.0055854


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005995, 0.0005995)
1: (0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033195, 0.0033195)
2: (0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0074161, 0.0074161)
3: (0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031252, 0.0031252)
4: (1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0121244, 0.0121244)
5: (0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023587, 0.0023587)
6: (-0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030695, 0.0030695)
7: (-0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003915, 0.0003915)
8: (-0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021208, 0.0021208)
9: (-0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0106170, 0.0106170)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.16 + 2.11 = 3.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0062060, upper bound: 0.0062060

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057839, upper bound: 0.0059414
time: 1.26 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059416, upper bound: 0.0059416
time: 1.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.60 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.60
Output dim: 4, lower bound: -0.0057839, upper bound: 0.0059414
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.60
Output dim: 4, lower bound: -0.0059416, upper bound: 0.0059416

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0044778, -0.0038874, -0.0044783, -0.0038595, -0.0005881, 0.0005632
1: 0.0012727, 0.0045417, 0.0011183, 0.0045441, -0.0031185, 0.0032565
2: 0.0048194, 0.0121228, 0.0048141, 0.0124678, -0.0072754, 0.0069670
3: 0.0022258, 0.0053034, 0.0020804, 0.0053057, -0.0029359, 0.0030659
4: 1.0053854, 1.0173256, 1.0048213, 1.0173343, -0.0113902, 0.0118944
5: 0.0032891, 0.0056119, 0.0031793, 0.0056136, -0.0022159, 0.0023139
6: -0.0130460, -0.0100232, -0.0130483, -0.0098804, -0.0030112, 0.0028836
7: -0.0104675, -0.0100819, -0.0104678, -0.0100637, -0.0003841, 0.0003678
8: -0.0039289, -0.0018403, -0.0040275, -0.0018388, -0.0019923, 0.0020805
9: -0.0089578, 0.0014980, -0.0089654, 0.0019919, -0.0104156, 0.0099742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057839, upper bound: 0.0057839
time: 1.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057839, upper bound: 0.0059414
time: 1.25 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0044937, -0.0038744, -0.0044783, -0.0038562, -0.0006034, 0.0005705
1: 0.0012005, 0.0046294, 0.0010995, 0.0045443, -0.0031591, 0.0033410
2: 0.0046236, 0.0122842, 0.0048137, 0.0125098, -0.0074641, 0.0070577
3: 0.0021578, 0.0053860, 0.0020627, 0.0053058, -0.0029742, 0.0031454
4: 1.0051216, 1.0176458, 1.0047528, 1.0173348, -0.0115386, 0.0122029
5: 0.0032378, 0.0056742, 0.0031660, 0.0056137, -0.0022447, 0.0023739
6: -0.0131271, -0.0099564, -0.0130484, -0.0098630, -0.0030893, 0.0029212
7: -0.0104778, -0.0100734, -0.0104678, -0.0100615, -0.0003941, 0.0003726
8: -0.0039750, -0.0017843, -0.0040395, -0.0018387, -0.0020183, 0.0021345
9: -0.0092381, 0.0017290, -0.0089659, 0.0020520, -0.0106858, 0.0101040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059414, upper bound: 0.0057839
time: 1.15 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059414, upper bound: 0.0059416
time: 1.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.90 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.90
Output dim: 4, lower bound: -0.0057839, upper bound: 0.0057839
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.90
Output dim: 4, lower bound: -0.0057839, upper bound: 0.0059414
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.90
Output dim: 4, lower bound: -0.0059414, upper bound: 0.0057839
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.90
Output dim: 4, lower bound: -0.0059414, upper bound: 0.0059416

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0044778, -0.0038874, -0.0044778, -0.0038874, -0.0005627, 0.0005627
1: 0.0012727, 0.0045417, 0.0012727, 0.0045417, -0.0031157, 0.0031157
2: 0.0048194, 0.0121228, 0.0048194, 0.0121228, -0.0069608, 0.0069608
3: 0.0022258, 0.0053034, 0.0022258, 0.0053034, -0.0029333, 0.0029333
4: 1.0053854, 1.0173256, 1.0053854, 1.0173256, -0.0113801, 0.0113801
5: 0.0032891, 0.0056119, 0.0032891, 0.0056119, -0.0022139, 0.0022139
6: -0.0130460, -0.0100232, -0.0130460, -0.0100232, -0.0028810, 0.0028810
7: -0.0104675, -0.0100819, -0.0104675, -0.0100819, -0.0003675, 0.0003675
8: -0.0039289, -0.0018403, -0.0039289, -0.0018403, -0.0019906, 0.0019906
9: -0.0089578, 0.0014980, -0.0089578, 0.0014980, -0.0099652, 0.0099652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055426, upper bound: 0.0053645
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055862, upper bound: 0.0055891
time: 1.38 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0044778, -0.0038874, -0.0044937, -0.0038744, -0.0005737, 0.0005751
1: 0.0012727, 0.0045417, 0.0012005, 0.0046294, -0.0031843, 0.0031766
2: 0.0048194, 0.0121228, 0.0046236, 0.0122842, -0.0070968, 0.0071141
3: 0.0022258, 0.0053034, 0.0021578, 0.0053860, -0.0029979, 0.0029906
4: 1.0053854, 1.0173256, 1.0051216, 1.0176458, -0.0116307, 0.0116025
5: 0.0032891, 0.0056119, 0.0032378, 0.0056742, -0.0022626, 0.0022571
6: -0.0130460, -0.0100232, -0.0131271, -0.0099564, -0.0029373, 0.0029445
7: -0.0104675, -0.0100819, -0.0104778, -0.0100734, -0.0003747, 0.0003756
8: -0.0039289, -0.0018403, -0.0039750, -0.0017843, -0.0020344, 0.0020295
9: -0.0089578, 0.0014980, -0.0092381, 0.0017290, -0.0101600, 0.0101847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055426, upper bound: 0.0055066
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055862, upper bound: 0.0057496
time: 1.17 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0044937, -0.0038744, -0.0044778, -0.0038874, -0.0005751, 0.0005737
1: 0.0012005, 0.0046294, 0.0012727, 0.0045417, -0.0031766, 0.0031843
2: 0.0046236, 0.0122842, 0.0048194, 0.0121228, -0.0071141, 0.0070968
3: 0.0021578, 0.0053860, 0.0022258, 0.0053034, -0.0029906, 0.0029979
4: 1.0051216, 1.0176458, 1.0053854, 1.0173256, -0.0116025, 0.0116307
5: 0.0032378, 0.0056742, 0.0032891, 0.0056119, -0.0022571, 0.0022626
6: -0.0131271, -0.0099564, -0.0130460, -0.0100232, -0.0029445, 0.0029373
7: -0.0104778, -0.0100734, -0.0104675, -0.0100819, -0.0003756, 0.0003747
8: -0.0039750, -0.0017843, -0.0039289, -0.0018403, -0.0020295, 0.0020344
9: -0.0092381, 0.0017290, -0.0089578, 0.0014980, -0.0101847, 0.0101600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057142, upper bound: 0.0053622
time: 1.33 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057496, upper bound: 0.0055862
time: 1.25 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0044937, -0.0038744, -0.0044937, -0.0038744, -0.0005691, 0.0005691
1: 0.0012005, 0.0046294, 0.0012005, 0.0046294, -0.0031512, 0.0031512
2: 0.0046236, 0.0122842, 0.0046236, 0.0122842, -0.0070401, 0.0070401
3: 0.0021578, 0.0053860, 0.0021578, 0.0053860, -0.0029667, 0.0029667
4: 1.0051216, 1.0176458, 1.0051216, 1.0176458, -0.0115098, 0.0115098
5: 0.0032378, 0.0056742, 0.0032378, 0.0056742, -0.0022391, 0.0022391
6: -0.0131271, -0.0099564, -0.0131271, -0.0099564, -0.0029139, 0.0029139
7: -0.0104778, -0.0100734, -0.0104778, -0.0100734, -0.0003717, 0.0003717
8: -0.0039750, -0.0017843, -0.0039750, -0.0017843, -0.0020132, 0.0020132
9: -0.0092381, 0.0017290, -0.0092381, 0.0017290, -0.0100788, 0.0100788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057142, upper bound: 0.0053680
time: 1.63 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057496, upper bound: 0.0055900
time: 1.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.53 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.53
Output dim: 4, lower bound: -0.0055426, upper bound: 0.0053645
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 4, lower bound: -0.0055862, upper bound: 0.0055891
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.53
Output dim: 4, lower bound: -0.0055426, upper bound: 0.0055066
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 4, lower bound: -0.0055862, upper bound: 0.0057496
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 4, lower bound: -0.0057142, upper bound: 0.0053622
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 4, lower bound: -0.0057496, upper bound: 0.0055862
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 4, lower bound: -0.0057142, upper bound: 0.0053680
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 4, lower bound: -0.0057496, upper bound: 0.0055900

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0044671, -0.0038917, -0.0044770, -0.0038878, -0.0005416, 0.0005480
1: 0.0012963, 0.0044822, 0.0012745, 0.0045370, -0.0030341, 0.0029986
2: 0.0049524, 0.0120701, 0.0048300, 0.0121188, -0.0066993, 0.0067785
3: 0.0022480, 0.0052474, 0.0022275, 0.0052990, -0.0028565, 0.0028231
4: 1.0054716, 1.0171083, 1.0053920, 1.0173082, -0.0110820, 0.0109525
5: 0.0033058, 0.0055696, 0.0032904, 0.0056085, -0.0021559, 0.0021307
6: -0.0129910, -0.0100450, -0.0130417, -0.0100249, -0.0027728, 0.0028056
7: -0.0104605, -0.0100847, -0.0104669, -0.0100821, -0.0003537, 0.0003579
8: -0.0039138, -0.0018784, -0.0039277, -0.0018434, -0.0019384, 0.0019158
9: -0.0087674, 0.0014226, -0.0089426, 0.0014922, -0.0095909, 0.0097042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0053645, upper bound: 0.0055474
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053645, upper bound: 0.0055891
time: 1.53 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0044671, -0.0038917, -0.0044929, -0.0038747, -0.0005517, 0.0005623
1: 0.0012963, 0.0044822, 0.0012022, 0.0046249, -0.0031133, 0.0030548
2: 0.0049524, 0.0120701, 0.0046335, 0.0122802, -0.0068247, 0.0069554
3: 0.0022480, 0.0052474, 0.0021594, 0.0053818, -0.0029310, 0.0028759
4: 1.0054716, 1.0171083, 1.0051280, 1.0176295, -0.0113713, 0.0111575
5: 0.0033058, 0.0055696, 0.0032390, 0.0056710, -0.0022122, 0.0021706
6: -0.0129910, -0.0100450, -0.0131230, -0.0099581, -0.0028247, 0.0028788
7: -0.0104605, -0.0100847, -0.0104773, -0.0100736, -0.0003603, 0.0003672
8: -0.0039138, -0.0018784, -0.0039739, -0.0017872, -0.0019890, 0.0019516
9: -0.0087674, 0.0014226, -0.0092238, 0.0017233, -0.0097704, 0.0099576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053622, upper bound: 0.0057142
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053622, upper bound: 0.0057496
time: 1.55 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0044476, -0.0038802, -0.0044631, -0.0038914, -0.0005210, 0.0005030
1: 0.0012328, 0.0043744, 0.0012945, 0.0044601, -0.0027852, 0.0028849
2: 0.0051932, 0.0122119, 0.0050019, 0.0120741, -0.0064451, 0.0062225
3: 0.0021882, 0.0051459, 0.0022463, 0.0052265, -0.0026222, 0.0027160
4: 1.0052398, 1.0167146, 1.0054650, 1.0170273, -0.0101731, 0.0105370
5: 0.0032607, 0.0054930, 0.0033046, 0.0055539, -0.0019791, 0.0020499
6: -0.0128913, -0.0099863, -0.0129705, -0.0100434, -0.0026676, 0.0025755
7: -0.0104478, -0.0100772, -0.0104579, -0.0100845, -0.0003403, 0.0003285
8: -0.0039543, -0.0019472, -0.0039149, -0.0018925, -0.0017794, 0.0018431
9: -0.0084227, 0.0016255, -0.0086965, 0.0014283, -0.0092269, 0.0089083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0054771, upper bound: 0.0051933
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055251, upper bound: 0.0051933
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0044833, -0.0038784, -0.0044770, -0.0038878, -0.0005540, 0.0005605
1: 0.0012226, 0.0045721, 0.0012745, 0.0045370, -0.0031037, 0.0030673
2: 0.0047515, 0.0122348, 0.0048300, 0.0121188, -0.0068528, 0.0069340
3: 0.0021786, 0.0053321, 0.0022275, 0.0052990, -0.0029220, 0.0028878
4: 1.0052024, 1.0174366, 1.0053920, 1.0173082, -0.0113363, 0.0112034
5: 0.0032535, 0.0056335, 0.0032904, 0.0056085, -0.0022054, 0.0021795
6: -0.0130742, -0.0099769, -0.0130417, -0.0100249, -0.0028363, 0.0028699
7: -0.0104711, -0.0100760, -0.0104669, -0.0100821, -0.0003618, 0.0003661
8: -0.0039609, -0.0018209, -0.0039277, -0.0018434, -0.0019829, 0.0019597
9: -0.0090550, 0.0016582, -0.0089426, 0.0014922, -0.0098106, 0.0099269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055066, upper bound: 0.0055426
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055066, upper bound: 0.0055862
time: 1.44 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0044476, -0.0038802, -0.0044785, -0.0038781, -0.0005152, 0.0004976
1: 0.0012328, 0.0043744, 0.0012208, 0.0045453, -0.0027550, 0.0028527
2: 0.0051932, 0.0122119, 0.0048114, 0.0122387, -0.0063732, 0.0061549
3: 0.0021882, 0.0051459, 0.0021769, 0.0053068, -0.0025937, 0.0026857
4: 1.0052398, 1.0167146, 1.0051960, 1.0173386, -0.0100625, 0.0104194
5: 0.0032607, 0.0054930, 0.0032522, 0.0056145, -0.0019576, 0.0020270
6: -0.0128913, -0.0099863, -0.0130493, -0.0099753, -0.0026378, 0.0025475
7: -0.0104478, -0.0100772, -0.0104679, -0.0100758, -0.0003365, 0.0003250
8: -0.0039543, -0.0019472, -0.0039620, -0.0018381, -0.0017601, 0.0018225
9: -0.0084227, 0.0016255, -0.0089692, 0.0016638, -0.0091240, 0.0088115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0054771, upper bound: 0.0052005
time: 1.21 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055253, upper bound: 0.0052005
time: 1.37 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0044833, -0.0038784, -0.0044929, -0.0038747, -0.0005479, 0.0005555
1: 0.0012226, 0.0045721, 0.0012022, 0.0046249, -0.0030756, 0.0030340
2: 0.0047515, 0.0122348, 0.0046335, 0.0122802, -0.0067782, 0.0068712
3: 0.0021786, 0.0053321, 0.0021594, 0.0053818, -0.0028955, 0.0028564
4: 1.0052024, 1.0174366, 1.0051280, 1.0176295, -0.0112336, 0.0110816
5: 0.0032535, 0.0056335, 0.0032390, 0.0056710, -0.0021854, 0.0021558
6: -0.0130742, -0.0099769, -0.0131230, -0.0099581, -0.0028055, 0.0028439
7: -0.0104711, -0.0100760, -0.0104773, -0.0100736, -0.0003579, 0.0003628
8: -0.0039609, -0.0018209, -0.0039739, -0.0017872, -0.0019649, 0.0019383
9: -0.0090550, 0.0016582, -0.0092238, 0.0017233, -0.0097038, 0.0098370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055071, upper bound: 0.0055477
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055071, upper bound: 0.0055899
time: 1.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.17 seconds
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.17
Output dim: 4, lower bound: -0.0053645, upper bound: 0.0055474
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 4, lower bound: -0.0053645, upper bound: 0.0055891
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 4, lower bound: -0.0053622, upper bound: 0.0057142
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 4, lower bound: -0.0053622, upper bound: 0.0057496
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.17
Output dim: 4, lower bound: -0.0054771, upper bound: 0.0051933
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.17
Output dim: 4, lower bound: -0.0055251, upper bound: 0.0051933
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.17
Output dim: 4, lower bound: -0.0055066, upper bound: 0.0055426
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 4, lower bound: -0.0055066, upper bound: 0.0055862
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.17
Output dim: 4, lower bound: -0.0054771, upper bound: 0.0052005
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.17
Output dim: 4, lower bound: -0.0055253, upper bound: 0.0052005
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.17
Output dim: 4, lower bound: -0.0055071, upper bound: 0.0055477
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 4, lower bound: -0.0055071, upper bound: 0.0055899

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0044671, -0.0038917, -0.0044671, -0.0038917, -0.0005269, 0.0005269
1: 0.0012963, 0.0044822, 0.0012963, 0.0044822, -0.0029173, 0.0029173
2: 0.0049524, 0.0120701, 0.0049524, 0.0120701, -0.0065176, 0.0065176
3: 0.0022480, 0.0052474, 0.0022480, 0.0052474, -0.0027466, 0.0027466
4: 1.0054716, 1.0171083, 1.0054716, 1.0171083, -0.0106556, 0.0106556
5: 0.0033058, 0.0055696, 0.0033058, 0.0055696, -0.0020729, 0.0020729
6: -0.0129910, -0.0100450, -0.0129910, -0.0100450, -0.0026976, 0.0026976
7: -0.0104605, -0.0100847, -0.0104605, -0.0100847, -0.0003441, 0.0003441
8: -0.0039138, -0.0018784, -0.0039138, -0.0018784, -0.0018638, 0.0018638
9: -0.0087674, 0.0014226, -0.0087674, 0.0014226, -0.0093308, 0.0093308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051941, upper bound: 0.0053054
time: 1.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051941, upper bound: 0.0054260
time: 1.53 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0044671, -0.0038917, -0.0044476, -0.0038802, -0.0005089, 0.0005210
1: 0.0012963, 0.0044822, 0.0012328, 0.0043744, -0.0028849, 0.0028179
2: 0.0049524, 0.0120701, 0.0051932, 0.0122119, -0.0062954, 0.0064453
3: 0.0022480, 0.0052474, 0.0021882, 0.0051459, -0.0027160, 0.0026529
4: 1.0054716, 1.0171083, 1.0052398, 1.0167146, -0.0105372, 0.0102923
5: 0.0033058, 0.0055696, 0.0032607, 0.0054930, -0.0020499, 0.0020023
6: -0.0129910, -0.0100450, -0.0128913, -0.0099863, -0.0026056, 0.0026677
7: -0.0104605, -0.0100847, -0.0104478, -0.0100772, -0.0003324, 0.0003403
8: -0.0039138, -0.0018784, -0.0039543, -0.0019472, -0.0018431, 0.0018003
9: -0.0087674, 0.0014226, -0.0084227, 0.0016255, -0.0090127, 0.0092272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051933, upper bound: 0.0054771
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051933, upper bound: 0.0055251
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0044671, -0.0038917, -0.0044833, -0.0038784, -0.0005382, 0.0005411
1: 0.0012963, 0.0044822, 0.0012226, 0.0045721, -0.0029963, 0.0029799
2: 0.0049524, 0.0120701, 0.0047515, 0.0122348, -0.0066574, 0.0066941
3: 0.0022480, 0.0052474, 0.0021786, 0.0053321, -0.0028209, 0.0028055
4: 1.0054716, 1.0171083, 1.0052024, 1.0174366, -0.0109440, 0.0108841
5: 0.0033058, 0.0055696, 0.0032535, 0.0056335, -0.0021290, 0.0021174
6: -0.0129910, -0.0100450, -0.0130742, -0.0099769, -0.0027555, 0.0027706
7: -0.0104605, -0.0100847, -0.0104711, -0.0100760, -0.0003515, 0.0003534
8: -0.0039138, -0.0018784, -0.0039609, -0.0018209, -0.0019143, 0.0019038
9: -0.0087674, 0.0014226, -0.0090550, 0.0016582, -0.0095309, 0.0095834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051933, upper bound: 0.0055078
time: 1.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051933, upper bound: 0.0055598
time: 1.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0044833, -0.0038784, -0.0044671, -0.0038917, -0.0005411, 0.0005382
1: 0.0012226, 0.0045721, 0.0012963, 0.0044822, -0.0029799, 0.0029963
2: 0.0047515, 0.0122348, 0.0049524, 0.0120701, -0.0066941, 0.0066574
3: 0.0021786, 0.0053321, 0.0022480, 0.0052474, -0.0028055, 0.0028209
4: 1.0052024, 1.0174366, 1.0054716, 1.0171083, -0.0108841, 0.0109440
5: 0.0032535, 0.0056335, 0.0033058, 0.0055696, -0.0021174, 0.0021290
6: -0.0130742, -0.0099769, -0.0129910, -0.0100450, -0.0027706, 0.0027555
7: -0.0104711, -0.0100760, -0.0104605, -0.0100847, -0.0003534, 0.0003515
8: -0.0039609, -0.0018209, -0.0039138, -0.0018784, -0.0019038, 0.0019143
9: -0.0090550, 0.0016582, -0.0087674, 0.0014226, -0.0095834, 0.0095309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0053167, upper bound: 0.0052827
time: 1.52 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0053167, upper bound: 0.0054246
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0044833, -0.0038784, -0.0044833, -0.0038784, -0.0005342, 0.0005342
1: 0.0012226, 0.0045721, 0.0012226, 0.0045721, -0.0029581, 0.0029581
2: 0.0047515, 0.0122348, 0.0047515, 0.0122348, -0.0066088, 0.0066088
3: 0.0021786, 0.0053321, 0.0021786, 0.0053321, -0.0027849, 0.0027849
4: 1.0052024, 1.0174366, 1.0052024, 1.0174366, -0.0108045, 0.0108045
5: 0.0032535, 0.0056335, 0.0032535, 0.0056335, -0.0021019, 0.0021019
6: -0.0130742, -0.0099769, -0.0130742, -0.0099769, -0.0027353, 0.0027353
7: -0.0104711, -0.0100760, -0.0104711, -0.0100760, -0.0003489, 0.0003489
8: -0.0039609, -0.0018209, -0.0039609, -0.0018209, -0.0018899, 0.0018899
9: -0.0090550, 0.0016582, -0.0090550, 0.0016582, -0.0094613, 0.0094613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0053177, upper bound: 0.0052865
time: 1.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0053177, upper bound: 0.0054292
time: 1.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.27 seconds
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.27
Output dim: 4, lower bound: -0.0051941, upper bound: 0.0053054
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.27
Output dim: 4, lower bound: -0.0051941, upper bound: 0.0054260
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.27
Output dim: 4, lower bound: -0.0051933, upper bound: 0.0054771
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.27
Output dim: 4, lower bound: -0.0051933, upper bound: 0.0055251
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.27
Output dim: 4, lower bound: -0.0051933, upper bound: 0.0055078
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.27
Output dim: 4, lower bound: -0.0051933, upper bound: 0.0055598
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.27
Output dim: 4, lower bound: -0.0053167, upper bound: 0.0052827
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.27
Output dim: 4, lower bound: -0.0053167, upper bound: 0.0054246
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.27
Output dim: 4, lower bound: -0.0053177, upper bound: 0.0052865
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.27
Output dim: 4, lower bound: -0.0053177, upper bound: 0.0054292

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.28 + 71.28 = 74.56 seconds
