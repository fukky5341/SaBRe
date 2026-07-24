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
execution time: IAR + RelationalAnalysis = 21.43 + 35.03 = 56.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.2011819, upper bound: 0.2011820

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5834
type: B, layer: 1, pos: 5834
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 6158
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5821

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2003800, upper bound: 0.2011808
time: 5.08 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011806, upper bound: 0.2011807
time: 7.16 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.41 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.41
Output dim: 0, lower bound: -0.2003800, upper bound: 0.2011808
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.41
Output dim: 0, lower bound: -0.2011806, upper bound: 0.2011807

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 9.4675369, 10.3100147, 9.4651976, 10.3109198, -0.3718033, 0.3731747
1: -18.0085487, -16.7450600, -18.0092773, -16.7433758, -0.4521160, 0.4511561
2: -3.0641062, -2.1183579, -3.0653806, -2.1153669, -0.3689878, 0.3672657
3: -10.2058897, -9.0678663, -10.2060204, -9.0677881, -0.4263947, 0.4264424
4: -21.1035786, -19.9540787, -21.1053066, -19.9495392, -0.3730817, 0.3703110
5: -0.6190092, 0.2848809, -0.6190937, 0.2850063, -0.3016834, 0.3016411
6: -5.3782730, -4.5202599, -5.3788967, -4.5188055, -0.2695900, 0.2687554
7: -4.0453310, -3.0643194, -4.0482798, -3.0631983, -0.3602213, 0.3619800
8: 1.1129742, 1.8099394, 1.1114168, 1.8106189, -0.3092399, 0.3101137
9: -7.7038288, -6.6430001, -7.7042732, -6.6418386, -0.3539562, 0.3532761

Time for backsubstitution: 20.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5834
type: A, layer: 1, pos: 5834
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 6158
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5834

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2003783, upper bound: 0.2006242
time: 7.94 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5834

### Candidate
type: B, layer: 1, pos: 871

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2003794, upper bound: 0.2009506
time: 6.44 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2001600, upper bound: 0.2009608
time: 5.80 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 9.4635429, 10.3109207, 9.4635420, 10.3109217, -0.3717899, 0.3757746
1: -18.0093117, -16.7421913, -18.0093117, -16.7421837, -0.4540386, 0.4511390
2: -3.0654125, -2.1132360, -3.0654128, -2.1132352, -0.3724040, 0.3671787
3: -10.2060480, -9.0677319, -10.2060461, -9.0677309, -0.4265933, 0.4264486
4: -21.1053066, -19.9463177, -21.1053066, -19.9463196, -0.3780590, 0.3702124
5: -0.6191040, 0.2850980, -0.6191039, 0.2850962, -0.3018678, 0.3016412
6: -5.3789129, -4.5177736, -5.3789129, -4.5177722, -0.2712506, 0.2687140
7: -4.0503607, -3.0631983, -4.0503607, -3.0631983, -0.3602036, 0.3652308
8: 1.1103110, 1.8106346, 1.1103106, 1.8106351, -0.3091984, 0.3119161
9: -7.7042732, -6.6410193, -7.7042732, -6.6410189, -0.3552501, 0.3532920

Time for backsubstitution: 20.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 5834
type: A, layer: 1, pos: 5834
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 6158
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5821

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011807, upper bound: 0.2003797
time: 8.59 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011807, upper bound: 0.2011807
time: 4.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 34.16 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 34.16
Output dim: 0, lower bound: -0.2003794, upper bound: 0.2009506
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 34.16
Output dim: 0, lower bound: -0.2001600, upper bound: 0.2009608
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 34.16
Output dim: 0, lower bound: -0.2011807, upper bound: 0.2003797
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 34.16
Output dim: 0, lower bound: -0.2011807, upper bound: 0.2011807

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 9.4635429, 10.3109207, 9.4675369, 10.3100147, -0.3748631, 0.3718038
1: -18.0093117, -16.7421913, -18.0085487, -16.7450600, -0.4511619, 0.4533188
2: -3.0654125, -2.1132360, -3.0641062, -2.1183579, -0.3672775, 0.3711183
3: -10.2060480, -9.0677319, -10.2058897, -9.0678663, -0.4264531, 0.4264526
4: -21.1053066, -19.9463177, -21.1035786, -19.9540787, -0.3703110, 0.3742869
5: -0.6191040, 0.2850980, -0.6190092, 0.2848809, -0.3016465, 0.3017753
6: -5.3789129, -4.5177736, -5.3782730, -4.5202599, -0.2687630, 0.2706236
7: -4.0503607, -3.0631983, -4.0453310, -3.0643194, -0.3640972, 0.3602211
8: 1.1103110, 1.8106346, 1.1129742, 1.8099394, -0.3112216, 0.3092499
9: -7.7042732, -6.6410193, -7.7038288, -6.6430001, -0.3532760, 0.3547997

Time for backsubstitution: 20.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5834
type: B, layer: 1, pos: 5834
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 6158
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5834

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2006239, upper bound: 0.2003781
time: 5.70 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011784, upper bound: 0.2003780
time: 5.45 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 9.4635429, 10.3109207, 9.4635429, 10.3109207, -0.3717902, 0.3717902
1: -18.0093117, -16.7421913, -18.0093117, -16.7421913, -0.4511391, 0.4511392
2: -3.0654125, -2.1132360, -3.0654125, -2.1132360, -0.3671783, 0.3671782
3: -10.2060480, -9.0677319, -10.2060480, -9.0677319, -0.4264488, 0.4264488
4: -21.1053066, -19.9463177, -21.1053066, -19.9463177, -0.3702124, 0.3702124
5: -0.6191040, 0.2850980, -0.6191040, 0.2850980, -0.3016412, 0.3016412
6: -5.3789129, -4.5177736, -5.3789129, -4.5177736, -0.2687139, 0.2687140
7: -4.0503607, -3.0631983, -4.0503607, -3.0631983, -0.3602036, 0.3602037
8: 1.1103110, 1.8106346, 1.1103110, 1.8106346, -0.3091984, 0.3091984
9: -7.7042732, -6.6410193, -7.7042732, -6.6410193, -0.3532920, 0.3532920

Time for backsubstitution: 20.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5834
type: B, layer: 1, pos: 5834
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 6158
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5834

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2006242, upper bound: 0.2003779
time: 8.66 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011787, upper bound: 0.2003780
time: 3.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.52 seconds
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 33.52
Output dim: 0, lower bound: -0.2006239, upper bound: 0.2003781
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.52
Output dim: 0, lower bound: -0.2011784, upper bound: 0.2003780
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 33.52
Output dim: 0, lower bound: -0.2006242, upper bound: 0.2003779
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.52
Output dim: 0, lower bound: -0.2011787, upper bound: 0.2003780

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 9.4635210, 10.3141403, 9.4675388, 10.3100147, -0.3750153, 0.3750343
1: -18.0093327, -16.7373905, -18.0085487, -16.7450619, -0.4513540, 0.4571263
2: -3.0674739, -2.1132383, -3.0641062, -2.1183581, -0.3694078, 0.3711574
3: -10.2077017, -9.0677338, -10.2058907, -9.0678663, -0.4281669, 0.4264817
4: -21.1053028, -19.9409637, -21.1035748, -19.9540787, -0.3704181, 0.3743342
5: -0.6206995, 0.2850950, -0.6190093, 0.2848779, -0.3032591, 0.3018056
6: -5.3818731, -4.5177774, -5.3782754, -4.5202594, -0.2717530, 0.2706816
7: -4.0504193, -3.0586193, -4.0453305, -3.0643194, -0.3643264, 0.3647964
8: 1.1103134, 1.8133082, 1.1129751, 1.8099384, -0.3112733, 0.3119559
9: -7.7042909, -6.6409607, -7.7038288, -6.6430016, -0.3532917, 0.3548931

Time for backsubstitution: 20.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 5834
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 6158
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 871

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2009483, upper bound: 0.2003778
time: 8.31 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2009585, upper bound: 0.2001583
time: 7.77 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 9.4635210, 10.3141403, 9.4635458, 10.3109207, -0.3719397, 0.3750234
1: -18.0093327, -16.7373905, -18.0093117, -16.7421875, -0.4513335, 0.4559476
2: -3.0674739, -2.1132383, -3.0654128, -2.1132362, -0.3693115, 0.3672152
3: -10.2077017, -9.0677338, -10.2060461, -9.0677319, -0.4281631, 0.4264779
4: -21.1053028, -19.9409637, -21.1053028, -19.9463158, -0.3703191, 0.3743484
5: -0.6206995, 0.2850950, -0.6191041, 0.2850960, -0.3032537, 0.3016709
6: -5.3818731, -4.5177774, -5.3789139, -4.5177755, -0.2717057, 0.2687703
7: -4.0504193, -3.0586193, -4.0503569, -3.0631983, -0.3604301, 0.3647833
8: 1.1103134, 1.8133082, 1.1103125, 1.8106346, -0.3092492, 0.3119056
9: -7.7042909, -6.6409607, -7.7042732, -6.6410203, -0.3533094, 0.3533844

Time for backsubstitution: 20.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5834
type: B, layer: 1, pos: 6158
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 871

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011781, upper bound: 0.2001479
time: 5.90 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2009586, upper bound: 0.2001581
time: 4.65 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.67 seconds
NS_A2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 31.67
Output dim: 0, lower bound: -0.2009483, upper bound: 0.2003778
NS_A2_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 31.67
Output dim: 0, lower bound: -0.2009585, upper bound: 0.2001583
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 0, lower bound: -0.2011781, upper bound: 0.2001479
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 31.67
Output dim: 0, lower bound: -0.2009586, upper bound: 0.2001581

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 9.4635315, 10.3141317, 9.4635677, 10.3109016, -0.3714707, 0.3747022
1: -18.0093212, -16.7375507, -18.0092812, -16.7425156, -0.4507358, 0.4556458
2: -3.0674670, -2.1132908, -3.0653949, -2.1133471, -0.3688277, 0.3669679
3: -10.2076950, -9.0678186, -10.2060328, -9.0679035, -0.4279389, 0.4263606
4: -21.1052952, -19.9412804, -21.1052990, -19.9469662, -0.3698370, 0.3738820
5: -0.6204135, 0.2850812, -0.6185313, 0.2850678, -0.3030252, 0.3014531
6: -5.3818245, -4.5180101, -5.3788204, -4.5182476, -0.2711789, 0.2682067
7: -4.0497808, -3.0587745, -4.0490723, -3.0635118, -0.3596772, 0.3641982
8: 1.1103387, 1.8133025, 1.1103554, 1.8106241, -0.3089600, 0.3113365
9: -7.7042060, -6.6414571, -7.7040968, -6.6420217, -0.3522673, 0.3524139

Time for backsubstitution: 21.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5834
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 6158
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1998756, upper bound: 0.2000661
time: 8.13 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011768, upper bound: 0.2001472
time: 5.43 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 34.80 seconds
NS_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 34.80
Output dim: 0, lower bound: -0.1998756, upper bound: 0.2000661
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 34.80
Output dim: 0, lower bound: -0.2011768, upper bound: 0.2001472

## BFS NS instance: NS_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: 9.4635315, 10.3141317, 9.4635696, 10.3109026, -0.3670545, 0.3747025
1: -18.0093212, -16.7375507, -18.0092812, -16.7425117, -0.4512925, 0.4554915
2: -3.0674670, -2.1132908, -3.0653934, -2.1133466, -0.3672835, 0.3669682
3: -10.2076950, -9.0678186, -10.2060318, -9.0679035, -0.4279389, 0.4174993
4: -21.1052952, -19.9412804, -21.1052990, -19.9469681, -0.3586510, 0.3731980
5: -0.6204135, 0.2850812, -0.6185309, 0.2850651, -0.3030252, 0.2914598
6: -5.3818245, -4.5180101, -5.3788195, -4.5182447, -0.2678803, 0.2681620
7: -4.0497808, -3.0587745, -4.0490713, -3.0635118, -0.3596004, 0.3568769
8: 1.1103387, 1.8133025, 1.1103554, 1.8106236, -0.3016627, 0.3113366
9: -7.7042060, -6.6414571, -7.7040968, -6.6420240, -0.3511307, 0.3522146

Time for backsubstitution: 20.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 6158
type: B, layer: 1, pos: 5834
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 915

## Relational analysis of NS_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011753, upper bound: 0.1988411
time: 3.92 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2004591, upper bound: 0.1994290
time: 8.08 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 33.11 seconds
NS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 33.11
Output dim: 0, lower bound: -0.2011753, upper bound: 0.1988411
NS_A2_B2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 33.11
Output dim: 0, lower bound: -0.2004591, upper bound: 0.1994290

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 9.4635725, 10.3139353, 9.4635696, 10.3109026, -0.3670189, 0.3745070
1: -18.0093002, -16.7375717, -18.0092812, -16.7425117, -0.4511869, 0.4554505
2: -3.0674205, -2.1134579, -3.0653934, -2.1133466, -0.3672587, 0.3668015
3: -10.2072639, -9.0678339, -10.2060318, -9.0679035, -0.4274983, 0.4174957
4: -21.1052952, -19.9416142, -21.1052990, -19.9469681, -0.3586401, 0.3728592
5: -0.6200978, 0.2850673, -0.6185309, 0.2850651, -0.3027027, 0.2914590
6: -5.3818150, -4.5180759, -5.3788195, -4.5182447, -0.2678446, 0.2680931
7: -4.0494699, -3.0587745, -4.0490713, -3.0635118, -0.3592771, 0.3568501
8: 1.1103668, 1.8130317, 1.1103554, 1.8106236, -0.3016613, 0.3110596
9: -7.7042060, -6.6416569, -7.7040968, -6.6420240, -0.3510606, 0.3519876

Time for backsubstitution: 20.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6158
type: B, layer: 1, pos: 5834
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6158

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011741, upper bound: 0.1970788
time: 5.95 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011741, upper bound: 0.1988398
time: 4.13 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 31.16 seconds
NS_A2_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 31.16
Output dim: 0, lower bound: -0.2011741, upper bound: 0.1970788
NS_A2_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 31.16
Output dim: 0, lower bound: -0.2011741, upper bound: 0.1988398

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 9.4647427, 10.3139286, 9.4740648, 10.3105841, -0.3627875, 0.3639047
1: -18.0084782, -16.7375908, -18.0019283, -16.7428322, -0.4501221, 0.4479215
2: -3.0669241, -2.1134608, -3.0609450, -2.1134810, -0.3666430, 0.3623223
3: -10.2072392, -9.0682278, -10.2056847, -9.0714321, -0.4239397, 0.4167550
4: -21.1040497, -19.9416409, -21.0941315, -19.9474602, -0.3526976, 0.3616660
5: -0.6200846, 0.2835920, -0.6181040, 0.2718415, -0.2894484, 0.2852926
6: -5.3817968, -4.5180798, -5.3786316, -4.5182819, -0.2677886, 0.2678930
7: -4.0494523, -3.0590751, -4.0488377, -3.0662086, -0.3566149, 0.3562243
8: 1.1103706, 1.8120861, 1.1106014, 1.8021345, -0.2931876, 0.3059072
9: -7.7035418, -6.6416759, -7.6981483, -6.6423597, -0.3500593, 0.3460091

Time for backsubstitution: 20.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5834
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5834

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011741, upper bound: 0.1965234
time: 6.05 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011744, upper bound: 0.1965234
time: 4.99 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 9.4635725, 10.3139353, 9.4635696, 10.3109026, -0.3670192, 0.3660493
1: -18.0093002, -16.7375717, -18.0092831, -16.7425137, -0.4511871, 0.4497013
2: -3.0674205, -2.1134579, -3.0653949, -2.1133466, -0.3672587, 0.3631947
3: -10.2072639, -9.0678339, -10.2060318, -9.0679035, -0.4245744, 0.4174953
4: -21.1052952, -19.9416142, -21.1052971, -19.9469662, -0.3586402, 0.3634872
5: -0.6200978, 0.2850673, -0.6185299, 0.2850664, -0.2914567, 0.2914588
6: -5.3818150, -4.5180759, -5.3788204, -4.5182447, -0.2678446, 0.2681681
7: -4.0494699, -3.0587745, -4.0490718, -3.0635121, -0.3573833, 0.3568411
8: 1.1103668, 1.8130317, 1.1103554, 1.8106236, -0.2946901, 0.3110595
9: -7.7042060, -6.6416569, -7.7040977, -6.6420245, -0.3510604, 0.3470471

Time for backsubstitution: 21.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5834
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5834

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011741, upper bound: 0.1982844
time: 3.91 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011744, upper bound: 0.1982843
time: 6.69 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 31.87 seconds
NS_A2_B2_A2_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 31.87
Output dim: 0, lower bound: -0.2011741, upper bound: 0.1965234
NS_A2_B2_A2_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 31.87
Output dim: 0, lower bound: -0.2011744, upper bound: 0.1965234
NS_A2_B2_A2_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 31.87
Output dim: 0, lower bound: -0.2011741, upper bound: 0.1982844
NS_A2_B2_A2_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 31.87
Output dim: 0, lower bound: -0.2011744, upper bound: 0.1982843

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 9.4647427, 10.3139286, 9.4746189, 10.3105850, -0.3627441, 0.3633413
1: -18.0084782, -16.7375908, -18.0011044, -16.7428322, -0.4499665, 0.4470918
2: -3.0669241, -2.1134608, -3.0609350, -2.1137929, -0.3663307, 0.3622830
3: -10.2072392, -9.0682278, -10.2056770, -9.0716887, -0.4236798, 0.4167228
4: -21.1040497, -19.9416409, -21.0932064, -19.9474640, -0.3526959, 0.3607379
5: -0.6200846, 0.2835920, -0.6180993, 0.2715756, -0.2891805, 0.2852797
6: -5.3817968, -4.5180798, -5.3786283, -4.5187759, -0.2672894, 0.2678347
7: -4.0494523, -3.0590751, -4.0480433, -3.0662086, -0.3564562, 0.3554206
8: 1.1103706, 1.8120861, 1.1110363, 1.8021297, -0.2931347, 0.3054572
9: -7.7035418, -6.6416759, -7.6981483, -6.6423612, -0.3500547, 0.3459882

Time for backsubstitution: 21.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6158

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1994131, upper bound: 0.1965234
time: 6.90 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1994130, upper bound: 0.1965235
time: 8.35 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 9.4647427, 10.3139286, 9.4740372, 10.3138046, -0.3627691, 0.3608451
1: -18.0084782, -16.7375908, -18.0019493, -16.7380314, -0.4502029, 0.4433882
2: -3.0669241, -2.1134608, -3.0630069, -2.1134794, -0.3649005, 0.3626788
3: -10.2072392, -9.0682278, -10.2073421, -9.0714331, -0.4224858, 0.4169865
4: -21.1040497, -19.9416409, -21.0941296, -19.9420986, -0.3527786, 0.3584627
5: -0.6200846, 0.2835920, -0.6196992, 0.2718397, -0.2879531, 0.2853408
6: -5.3817968, -4.5180798, -5.3815932, -4.5182838, -0.2649995, 0.2680398
7: -4.0494523, -3.0590751, -4.0488997, -3.0616305, -0.3566149, 0.3518720
8: 1.1103706, 1.8120861, 1.1106024, 1.8048091, -0.2933986, 0.3060156
9: -7.7035418, -6.6416759, -7.6981659, -6.6422963, -0.3501351, 0.3460091

Time for backsubstitution: 20.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6158

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1994134, upper bound: 0.1965235
time: 7.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1994133, upper bound: 0.1965235
time: 7.45 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 9.4635725, 10.3139353, 9.4641247, 10.3109007, -0.3669057, 0.3654859
1: -18.0093002, -16.7375717, -18.0084591, -16.7425117, -0.4510306, 0.4488714
2: -3.0674205, -2.1134579, -3.0653868, -2.1136580, -0.3669467, 0.3631554
3: -10.2072639, -9.0678339, -10.2060280, -9.0681610, -0.4243143, 0.4174631
4: -21.1052952, -19.9416142, -21.1043701, -19.9469719, -0.3585314, 0.3625591
5: -0.6200978, 0.2850673, -0.6185280, 0.2848014, -0.2911892, 0.2914276
6: -5.3818150, -4.5180759, -5.3788137, -4.5187407, -0.2673457, 0.2681097
7: -4.0494699, -3.0587745, -4.0482740, -3.0635121, -0.3572245, 0.3560367
8: 1.1103668, 1.8130317, 1.1107922, 1.8106189, -0.2946382, 0.3106216
9: -7.7042060, -6.6416569, -7.7040977, -6.6420283, -0.3510556, 0.3470263

Time for backsubstitution: 20.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6158

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1994131, upper bound: 0.1982837
time: 8.09 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1994131, upper bound: 0.1965234
time: 6.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 9.4635725, 10.3139353, 9.4635439, 10.3141222, -0.3670425, 0.3629894
1: -18.0093002, -16.7375717, -18.0093040, -16.7377129, -0.4512672, 0.4451673
2: -3.0674205, -2.1134579, -3.0674574, -2.1133475, -0.3655164, 0.3635497
3: -10.2072639, -9.0678339, -10.2076874, -9.0679064, -0.4231207, 0.4177265
4: -21.1052952, -19.9416142, -21.1052914, -19.9416103, -0.3587210, 0.3602952
5: -0.6200978, 0.2850673, -0.6201260, 0.2850664, -0.2899623, 0.2915469
6: -5.3818150, -4.5180759, -5.3817797, -4.5182481, -0.2650557, 0.2683144
7: -4.0494699, -3.0587745, -4.0491285, -3.0589347, -0.3573833, 0.3524888
8: 1.1103668, 1.8130317, 1.1103573, 1.8132977, -0.2949011, 0.3086132
9: -7.7042060, -6.6416569, -7.7041149, -6.6419649, -0.3511357, 0.3470471

Time for backsubstitution: 20.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6158
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6158

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1994134, upper bound: 0.1982839
time: 4.47 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1994133, upper bound: 0.1965235
time: 7.29 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 32.89 seconds
NS_A2_B2_A2_B1_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 9, time: 32.89
Output dim: 0, lower bound: -0.1994131, upper bound: 0.1965234
NS_A2_B2_A2_B1_B2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 9, time: 32.89
Output dim: 0, lower bound: -0.1994130, upper bound: 0.1965235
NS_A2_B2_A2_B1_B2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 32.89
Output dim: 0, lower bound: -0.1994134, upper bound: 0.1965235
NS_A2_B2_A2_B1_B2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 9, time: 32.89
Output dim: 0, lower bound: -0.1994133, upper bound: 0.1965235
NS_A2_B2_A2_B1_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 32.89
Output dim: 0, lower bound: -0.1994131, upper bound: 0.1982837
NS_A2_B2_A2_B1_B2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 9, time: 32.89
Output dim: 0, lower bound: -0.1994131, upper bound: 0.1965234
NS_A2_B2_A2_B1_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 32.89
Output dim: 0, lower bound: -0.1994134, upper bound: 0.1982839
NS_A2_B2_A2_B1_B2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 9, time: 32.89
Output dim: 0, lower bound: -0.1994133, upper bound: 0.1965235

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 56.46 + 535.95 = 592.41 seconds
