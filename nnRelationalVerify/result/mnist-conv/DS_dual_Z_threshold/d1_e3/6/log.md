## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.200980818


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (9.4635420, 10.3109207, 9.4635420, 10.3109207, -0.3757746, 0.3757749)
1: (-18.0093117, -16.7421856, -18.0093117, -16.7421856, -0.4540387, 0.4540386)
2: (-3.0654144, -2.1132352, -3.0654144, -2.1132352, -0.3724043, 0.3724043)
3: (-10.2060471, -9.0677299, -10.2060471, -9.0677299, -0.4265924, 0.4265924)
4: (-21.1053066, -19.9463177, -21.1053066, -19.9463177, -0.3780588, 0.3780588)
5: (-0.6191039, 0.2850968, -0.6191039, 0.2850968, -0.3018680, 0.3018680)
6: (-5.3789139, -4.5177712, -5.3789139, -4.5177712, -0.2712505, 0.2712505)
7: (-4.0503621, -3.0631983, -4.0503621, -3.0631983, -0.3652309, 0.3652310)
8: (1.1103096, 1.8106346, 1.1103096, 1.8106346, -0.3119166, 0.3119164)
9: (-7.7042732, -6.6410213, -7.7042732, -6.6410213, -0.3552496, 0.3552498)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.23 + 34.56 = 55.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.2011819, upper bound: 0.2011820

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 6158
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 5821
type: DSZ, layer: 1, pos: 5834

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011811, upper bound: 0.1999706
time: 4.45 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1999706, upper bound: 0.2011811
time: 5.58 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.19 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.19
Output dim: 0, lower bound: -0.2011811, upper bound: 0.1999706
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.19
Output dim: 0, lower bound: -0.1999706, upper bound: 0.2011811

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 9.4635420, 10.3109207, 9.4635420, 10.3109207, -0.3713584, 0.3704777
1: -18.0093117, -16.7421856, -18.0093117, -16.7421856, -0.4545951, 0.4538059
2: -3.0654144, -2.1132352, -3.0654144, -2.1132352, -0.3708603, 0.3705528
3: -10.2060471, -9.0677299, -10.2060471, -9.0677299, -0.4159646, 0.4177306
4: -21.1053066, -19.9463177, -21.1053066, -19.9463177, -0.3668728, 0.3645201
5: -0.6191039, 0.2850968, -0.6191039, 0.2850968, -0.2898775, 0.2918746
6: -5.3789139, -4.5177712, -5.3789139, -4.5177712, -0.2679520, 0.2670659
7: -4.0503621, -3.0631983, -4.0503621, -3.0631983, -0.3561372, 0.3579102
8: 1.1103096, 1.8106346, 1.1103096, 1.8106346, -0.3046198, 0.3031672
9: -7.7042732, -6.6410213, -7.7042732, -6.6410213, -0.3541136, 0.3525964

Time for backsubstitution: 19.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 6158
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 5821
type: DSZ, layer: 1, pos: 5834

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011810, upper bound: 0.1999195
time: 7.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011300, upper bound: 0.1999705
time: 19.05 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 9.4635420, 10.3109207, 9.4635420, 10.3109207, -0.3704774, 0.3713584
1: -18.0093117, -16.7421856, -18.0093117, -16.7421856, -0.4538057, 0.4545951
2: -3.0654144, -2.1132352, -3.0654144, -2.1132352, -0.3705528, 0.3708603
3: -10.2060471, -9.0677299, -10.2060471, -9.0677299, -0.4177308, 0.4159644
4: -21.1053066, -19.9463177, -21.1053066, -19.9463177, -0.3645201, 0.3668729
5: -0.6191039, 0.2850968, -0.6191039, 0.2850968, -0.2918748, 0.2898777
6: -5.3789139, -4.5177712, -5.3789139, -4.5177712, -0.2670659, 0.2679518
7: -4.0503621, -3.0631983, -4.0503621, -3.0631983, -0.3579103, 0.3561373
8: 1.1103096, 1.8106346, 1.1103096, 1.8106346, -0.3031673, 0.3046197
9: -7.7042732, -6.6410213, -7.7042732, -6.6410213, -0.3525963, 0.3541137

Time for backsubstitution: 19.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 6158
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 5821
type: DSZ, layer: 1, pos: 5834

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1999705, upper bound: 0.2011301
time: 7.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1999195, upper bound: 0.2011809
time: 8.88 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 36.12 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 36.12
Output dim: 0, lower bound: -0.2011810, upper bound: 0.1999195
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 36.12
Output dim: 0, lower bound: -0.2011300, upper bound: 0.1999705
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 36.12
Output dim: 0, lower bound: -0.1999705, upper bound: 0.2011301
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 36.12
Output dim: 0, lower bound: -0.1999195, upper bound: 0.2011809

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 9.4635420, 10.3109207, 9.4635420, 10.3109207, -0.3713586, 0.3704815
1: -18.0093117, -16.7421856, -18.0093117, -16.7421856, -0.4545949, 0.4536614
2: -3.0654144, -2.1132352, -3.0654144, -2.1132352, -0.3705642, 0.3701510
3: -10.2060471, -9.0677299, -10.2060471, -9.0677299, -0.4161959, 0.4177294
4: -21.1053066, -19.9463177, -21.1053066, -19.9463177, -0.3660975, 0.3633174
5: -0.6191039, 0.2850968, -0.6191039, 0.2850968, -0.2895026, 0.2912929
6: -5.3789139, -4.5177712, -5.3789139, -4.5177712, -0.2677919, 0.2668177
7: -4.0503621, -3.0631983, -4.0503621, -3.0631983, -0.3561348, 0.3585858
8: 1.1103096, 1.8106346, 1.1103096, 1.8106346, -0.3040800, 0.3028196
9: -7.7042732, -6.6410213, -7.7042732, -6.6410213, -0.3521299, 0.3495206

Time for backsubstitution: 20.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 6158
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 5821
type: DSZ, layer: 1, pos: 5834

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 915

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2004640, upper bound: 0.1998232
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1998750, upper bound: 0.1998231
time: 7.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 9.4635420, 10.3109207, 9.4635420, 10.3109207, -0.3713622, 0.3704779
1: -18.0093117, -16.7421856, -18.0093117, -16.7421856, -0.4544508, 0.4538057
2: -3.0654144, -2.1132352, -3.0654144, -2.1132352, -0.3704586, 0.3702564
3: -10.2060471, -9.0677299, -10.2060471, -9.0677299, -0.4159632, 0.4179618
4: -21.1053066, -19.9463177, -21.1053066, -19.9463177, -0.3656702, 0.3637447
5: -0.6191039, 0.2850968, -0.6191039, 0.2850968, -0.2892959, 0.2914996
6: -5.3789139, -4.5177712, -5.3789139, -4.5177712, -0.2677039, 0.2669057
7: -4.0503621, -3.0631983, -4.0503621, -3.0631983, -0.3568128, 0.3579078
8: 1.1103096, 1.8106346, 1.1103096, 1.8106346, -0.3042722, 0.3026277
9: -7.7042732, -6.6410213, -7.7042732, -6.6410213, -0.3510377, 0.3506126

Time for backsubstitution: 20.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 6158
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 5821
type: DSZ, layer: 1, pos: 5834

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 915

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2004126, upper bound: 0.1998740
time: 6.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1998240, upper bound: 0.1998742
time: 5.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 9.4635420, 10.3109207, 9.4635420, 10.3109207, -0.3704779, 0.3713622
1: -18.0093117, -16.7421856, -18.0093117, -16.7421856, -0.4538057, 0.4544508
2: -3.0654144, -2.1132352, -3.0654144, -2.1132352, -0.3702564, 0.3704586
3: -10.2060471, -9.0677299, -10.2060471, -9.0677299, -0.4179621, 0.4159632
4: -21.1053066, -19.9463177, -21.1053066, -19.9463177, -0.3637447, 0.3656702
5: -0.6191039, 0.2850968, -0.6191039, 0.2850968, -0.2914996, 0.2892959
6: -5.3789139, -4.5177712, -5.3789139, -4.5177712, -0.2669058, 0.2677039
7: -4.0503621, -3.0631983, -4.0503621, -3.0631983, -0.3579077, 0.3568130
8: 1.1103096, 1.8106346, 1.1103096, 1.8106346, -0.3026276, 0.3042721
9: -7.7042732, -6.6410213, -7.7042732, -6.6410213, -0.3506126, 0.3510379

Time for backsubstitution: 20.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 6158
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 5821
type: DSZ, layer: 1, pos: 5834

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 915

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1998741, upper bound: 0.1998241
time: 4.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1998741, upper bound: 0.2004126
time: 4.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 9.4635420, 10.3109207, 9.4635420, 10.3109207, -0.3704815, 0.3713586
1: -18.0093117, -16.7421856, -18.0093117, -16.7421856, -0.4536617, 0.4545949
2: -3.0654144, -2.1132352, -3.0654144, -2.1132352, -0.3701510, 0.3705640
3: -10.2060471, -9.0677299, -10.2060471, -9.0677299, -0.4177294, 0.4161956
4: -21.1053066, -19.9463177, -21.1053066, -19.9463177, -0.3633175, 0.3660976
5: -0.6191039, 0.2850968, -0.6191039, 0.2850968, -0.2912929, 0.2895026
6: -5.3789139, -4.5177712, -5.3789139, -4.5177712, -0.2668178, 0.2677919
7: -4.0503621, -3.0631983, -4.0503621, -3.0631983, -0.3585860, 0.3561347
8: 1.1103096, 1.8106346, 1.1103096, 1.8106346, -0.3028197, 0.3040801
9: -7.7042732, -6.6410213, -7.7042732, -6.6410213, -0.3495204, 0.3521299

Time for backsubstitution: 20.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 6158
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 5821
type: DSZ, layer: 1, pos: 5834

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 915

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1998232, upper bound: 0.1998750
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1998232, upper bound: 0.2004640
time: 4.90 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.38 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 30.38
Output dim: 0, lower bound: -0.2004640, upper bound: 0.1998232
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 30.38
Output dim: 0, lower bound: -0.1998750, upper bound: 0.1998231
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 30.38
Output dim: 0, lower bound: -0.2004126, upper bound: 0.1998740
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 30.38
Output dim: 0, lower bound: -0.1998240, upper bound: 0.1998742
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 30.38
Output dim: 0, lower bound: -0.1998741, upper bound: 0.1998241
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 30.38
Output dim: 0, lower bound: -0.1998741, upper bound: 0.2004126
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 30.38
Output dim: 0, lower bound: -0.1998232, upper bound: 0.1998750
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 30.38
Output dim: 0, lower bound: -0.1998232, upper bound: 0.2004640

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 55.79 + 216.99 = 272.78 seconds
