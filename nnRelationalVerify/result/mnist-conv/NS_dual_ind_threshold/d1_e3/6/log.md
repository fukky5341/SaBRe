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
execution time: IAR + RelationalAnalysis = 21.26 + 35.10 = 56.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.2011819, upper bound: 0.2011820

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 5834
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 6158
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5821

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2003800, upper bound: 0.2011808
time: 5.03 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011806, upper bound: 0.2011807
time: 7.18 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.36 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.36
Output dim: 0, lower bound: -0.2003800, upper bound: 0.2011808
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.36
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

Time for backsubstitution: 19.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5834
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 6158
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5834

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2003783, upper bound: 0.2006242
time: 7.96 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1990773, upper bound: 0.2010989
time: 5.28 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5821

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2003797, upper bound: 0.2003797
time: 8.34 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2003797, upper bound: 0.2011808
time: 4.86 seconds

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

Time for backsubstitution: 20.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 5834
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 6158
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5821

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011807, upper bound: 0.2003797
time: 8.66 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011807, upper bound: 0.2011807
time: 4.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 34.04 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 34.04
Output dim: 0, lower bound: -0.2003797, upper bound: 0.2003797
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 34.04
Output dim: 0, lower bound: -0.2003797, upper bound: 0.2011808
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 34.04
Output dim: 0, lower bound: -0.2011807, upper bound: 0.2003797
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 34.04
Output dim: 0, lower bound: -0.2011807, upper bound: 0.2011807

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 9.4675369, 10.3100147, 9.4635429, 10.3109207, -0.3718038, 0.3748629
1: -18.0085487, -16.7450600, -18.0093117, -16.7421913, -0.4533190, 0.4511619
2: -3.0641062, -2.1183579, -3.0654125, -2.1132360, -0.3711183, 0.3672774
3: -10.2058897, -9.0678663, -10.2060480, -9.0677319, -0.4264526, 0.4264531
4: -21.1035786, -19.9540787, -21.1053066, -19.9463177, -0.3742870, 0.3703110
5: -0.6190092, 0.2848809, -0.6191040, 0.2850980, -0.3017752, 0.3016465
6: -5.3782730, -4.5202599, -5.3789129, -4.5177736, -0.2706236, 0.2687629
7: -4.0453310, -3.0643194, -4.0503607, -3.0631983, -0.3602213, 0.3640974
8: 1.1129742, 1.8099394, 1.1103110, 1.8106346, -0.3092496, 0.3112215
9: -7.7038288, -6.6430001, -7.7042732, -6.6410193, -0.3547997, 0.3532761

Time for backsubstitution: 20.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5834
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 6158
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5834

### Candidate
type: A, layer: 1, pos: 5829

### Candidate
type: A, layer: 1, pos: 871

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2001500, upper bound: 0.2011802
time: 5.88 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2001601, upper bound: 0.2009607
time: 11.84 seconds

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

Time for backsubstitution: 20.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5834
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 6158
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5834

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2006239, upper bound: 0.2003781
time: 5.60 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011784, upper bound: 0.2003780
time: 5.36 seconds

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

Time for backsubstitution: 20.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5834
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 6158
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5834

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2006242, upper bound: 0.2003779
time: 8.42 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011787, upper bound: 0.2003780
time: 3.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 32.99 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.99
Output dim: 0, lower bound: -0.2001500, upper bound: 0.2011802
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 32.99
Output dim: 0, lower bound: -0.2001601, upper bound: 0.2009607
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 32.99
Output dim: 0, lower bound: -0.2006239, upper bound: 0.2003781
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.99
Output dim: 0, lower bound: -0.2011784, upper bound: 0.2003780
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 32.99
Output dim: 0, lower bound: -0.2006242, upper bound: 0.2003779
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.99
Output dim: 0, lower bound: -0.2011787, upper bound: 0.2003780

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 9.4675608, 10.3099976, 9.4635582, 10.3109112, -0.3714833, 0.3743932
1: -18.0085258, -16.7453861, -18.0092964, -16.7423458, -0.4530170, 0.4505634
2: -3.0640886, -2.1184688, -3.0654032, -2.1132894, -0.3708707, 0.3667936
3: -10.2058773, -9.0680380, -10.2060404, -9.0678167, -0.4263356, 0.4262285
4: -21.1035690, -19.9547234, -21.1053047, -19.9466438, -0.3738203, 0.3698294
5: -0.6184354, 0.2848478, -0.6188198, 0.2850823, -0.3015575, 0.3014177
6: -5.3781791, -4.5207291, -5.3788671, -4.5180063, -0.2700599, 0.2682358
7: -4.0440454, -3.0646348, -4.0497236, -3.0633526, -0.3596363, 0.3633437
8: 1.1130190, 1.8099289, 1.1103344, 1.8106289, -0.3086808, 0.3109328
9: -7.7036543, -6.6440072, -7.7041855, -6.6415195, -0.3538289, 0.3522329

Time for backsubstitution: 20.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5834
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6158
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5834

### Candidate
type: B, layer: 1, pos: 5829

### Candidate
type: B, layer: 1, pos: 6158

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2001488, upper bound: 0.1994177
time: 3.82 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2001487, upper bound: 0.2011786
time: 4.29 seconds

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

Time for backsubstitution: 21.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 5834
type: B, layer: 1, pos: 6158
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1998757, upper bound: 0.2002965
time: 8.18 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011770, upper bound: 0.2003775
time: 6.31 seconds

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

Time for backsubstitution: 20.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 5834
type: B, layer: 1, pos: 6158
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1998760, upper bound: 0.2002962
time: 4.49 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011773, upper bound: 0.2003772
time: 3.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.90 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 28.90
Output dim: 0, lower bound: -0.2001488, upper bound: 0.1994177
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.90
Output dim: 0, lower bound: -0.2001487, upper bound: 0.2011786
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 28.90
Output dim: 0, lower bound: -0.1998757, upper bound: 0.2002965
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.90
Output dim: 0, lower bound: -0.2011770, upper bound: 0.2003775
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 28.90
Output dim: 0, lower bound: -0.1998760, upper bound: 0.2002962
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.90
Output dim: 0, lower bound: -0.2011773, upper bound: 0.2003772

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 9.4675608, 10.3099976, 9.4635563, 10.3109121, -0.3714833, 0.3659360
1: -18.0085258, -16.7453861, -18.0092964, -16.7423477, -0.4530164, 0.4448152
2: -3.0640886, -2.1184688, -3.0654049, -2.1132903, -0.3708707, 0.3631873
3: -10.2058773, -9.0680380, -10.2060432, -9.0678167, -0.4234121, 0.4262280
4: -21.1035690, -19.9547234, -21.1053009, -19.9466400, -0.3735254, 0.3604734
5: -0.6184354, 0.2848478, -0.6188180, 0.2850817, -0.2903121, 0.3014178
6: -5.3781791, -4.5207291, -5.3788662, -4.5180063, -0.2700601, 0.2683109
7: -4.0440454, -3.0646348, -4.0497236, -3.0633521, -0.3577428, 0.3633349
8: 1.1130190, 1.8099289, 1.1103344, 1.8106289, -0.3017097, 0.3109326
9: -7.7036543, -6.6440072, -7.7041860, -6.6415167, -0.3538296, 0.3472927

Time for backsubstitution: 21.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5834
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6158
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5834

### Candidate
type: A, layer: 1, pos: 5829

### Candidate
type: A, layer: 1, pos: 6158

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1983877, upper bound: 0.2011785
time: 4.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1983877, upper bound: 0.2011787
time: 3.47 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 9.4635210, 10.3141403, 9.4675388, 10.3100166, -0.3705990, 0.3750343
1: -18.0093327, -16.7373905, -18.0085487, -16.7450619, -0.4519113, 0.4568976
2: -3.0674739, -2.1132383, -3.0641041, -2.1183584, -0.3678634, 0.3711567
3: -10.2077017, -9.0677338, -10.2058897, -9.0678654, -0.4281671, 0.4176190
4: -21.1053028, -19.9409637, -21.1035748, -19.9540768, -0.3592319, 0.3736503
5: -0.6206995, 0.2850950, -0.6190079, 0.2848800, -0.3032591, 0.2918121
6: -5.3818731, -4.5177774, -5.3782763, -4.5202599, -0.2684547, 0.2706366
7: -4.0504193, -3.0586193, -4.0453310, -3.0643194, -0.3642497, 0.3574753
8: 1.1103134, 1.8133082, 1.1129761, 1.8099375, -0.3039756, 0.3119562
9: -7.7042909, -6.6409607, -7.7038288, -6.6430020, -0.3521554, 0.3546939

Time for backsubstitution: 21.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 6158
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2010965, upper bound: 0.1990756
time: 5.63 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2010965, upper bound: 0.1990755
time: 8.58 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 9.4635210, 10.3141403, 9.4635448, 10.3109207, -0.3675239, 0.3750231
1: -18.0093327, -16.7373905, -18.0093079, -16.7421875, -0.4518905, 0.4557929
2: -3.0674739, -2.1132383, -3.0654125, -2.1132367, -0.3677673, 0.3672149
3: -10.2077017, -9.0677338, -10.2060471, -9.0677309, -0.4281633, 0.4176152
4: -21.1053028, -19.9409637, -21.1053028, -19.9463234, -0.3591326, 0.3736645
5: -0.6206995, 0.2850950, -0.6191013, 0.2850953, -0.3032537, 0.2916777
6: -5.3818731, -4.5177774, -5.3789148, -4.5177746, -0.2684072, 0.2687254
7: -4.0504193, -3.0586193, -4.0503573, -3.0631983, -0.3603537, 0.3574619
8: 1.1103134, 1.8133082, 1.1103125, 1.8106337, -0.3019516, 0.3119059
9: -7.7042909, -6.6409607, -7.7042732, -6.6410213, -0.3521731, 0.3531852

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 6158
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 173

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2010968, upper bound: 0.1990754
time: 4.02 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2010968, upper bound: 0.1990754
time: 8.52 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 34.48 seconds
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 34.48
Output dim: 0, lower bound: -0.1983877, upper bound: 0.2011785
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.48
Output dim: 0, lower bound: -0.1983877, upper bound: 0.2011787
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 34.48
Output dim: 0, lower bound: -0.2010965, upper bound: 0.1990756
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.48
Output dim: 0, lower bound: -0.2010965, upper bound: 0.1990755
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 34.48
Output dim: 0, lower bound: -0.2010968, upper bound: 0.1990754
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.48
Output dim: 0, lower bound: -0.2010968, upper bound: 0.1990754

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 9.4780560, 10.3096809, 9.4635563, 10.3109121, -0.3608825, 0.3671608
1: -18.0011673, -16.7457027, -18.0092964, -16.7423477, -0.4454929, 0.4496770
2: -3.0596380, -2.1186025, -3.0654049, -2.1132903, -0.3663925, 0.3666804
3: -10.2055283, -9.0715685, -10.2060432, -9.0678167, -0.4259899, 0.4226975
4: -21.0924072, -19.9552155, -21.1053009, -19.9466400, -0.3623487, 0.3622794
5: -0.6180074, 0.2716245, -0.6188180, 0.2850817, -0.2952498, 0.2881731
6: -5.3779945, -4.5207629, -5.3788662, -4.5180063, -0.2698715, 0.2682002
7: -4.0438118, -3.0673318, -4.0497236, -3.0633521, -0.3593228, 0.3606994
8: 1.1132636, 1.8014388, 1.1103344, 1.8106289, -0.3059008, 0.3024609
9: -7.6977067, -6.6443396, -7.7041860, -6.6415167, -0.3478718, 0.3515581

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5834
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5834

### Candidate
type: B, layer: 1, pos: 5829

### Candidate
type: B, layer: 1, pos: 871

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1983877, upper bound: 0.2011779
time: 8.04 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1983877, upper bound: 0.2011779
time: 7.70 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 9.4675617, 10.3100004, 9.4635563, 10.3109121, -0.3630261, 0.3659360
1: -18.0085220, -16.7453823, -18.0092964, -16.7423477, -0.4472684, 0.4448154
2: -3.0640879, -2.1184688, -3.0654049, -2.1132903, -0.3672640, 0.3631873
3: -10.2058764, -9.0680399, -10.2060432, -9.0678167, -0.4234126, 0.4233046
4: -21.1035671, -19.9547234, -21.1053009, -19.9466400, -0.3644447, 0.3604736
5: -0.6184379, 0.2848483, -0.6188180, 0.2850817, -0.2903123, 0.2901727
6: -5.3781805, -4.5207295, -5.3788662, -4.5180063, -0.2701356, 0.2683107
7: -4.0440454, -3.0646350, -4.0497236, -3.0633521, -0.3577428, 0.3614504
8: 1.1130185, 1.8099289, 1.1103344, 1.8106289, -0.3017097, 0.3039615
9: -7.7036543, -6.6440072, -7.7041860, -6.6415167, -0.3488889, 0.3472927

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5834
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5834

### Candidate
type: B, layer: 1, pos: 5829

### Candidate
type: B, layer: 1, pos: 871

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 915

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1971041, upper bound: 0.1994152
time: 5.77 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1976696, upper bound: 0.1987003
time: 5.27 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 9.4662247, 10.3026142, 9.4675388, 10.3100166, -0.3652240, 0.3635058
1: -18.0085011, -16.7413788, -18.0085487, -16.7450619, -0.4494493, 0.4534183
2: -3.0645821, -2.1190996, -3.0641041, -2.1183584, -0.3670029, 0.3652906
3: -10.1843700, -9.0698576, -10.2058897, -9.0678654, -0.4042876, 0.4103459
4: -21.1046867, -19.9550991, -21.1035748, -19.9540768, -0.3608532, 0.3594586
5: -0.6030051, 0.2833520, -0.6190079, 0.2848800, -0.2853794, 0.2918850
6: -5.3812475, -4.5227566, -5.3782763, -4.5202599, -0.2685877, 0.2657511
7: -4.0395479, -3.0590360, -4.0453310, -3.0643194, -0.3528929, 0.3586414
8: 1.1128149, 1.7982960, 1.1129761, 1.8099375, -0.3013833, 0.2966449
9: -7.7040696, -6.6489563, -7.7038288, -6.6430020, -0.3526287, 0.3476890

Time for backsubstitution: 21.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 5834
type: B, layer: 1, pos: 6158
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 871

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1998752, upper bound: 0.1988456
time: 3.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1996557, upper bound: 0.1988558
time: 13.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 9.4635191, 10.3141403, 9.4675388, 10.3100166, -0.3705988, 0.3706181
1: -18.0093327, -16.7373867, -18.0085487, -16.7450619, -0.4519109, 0.4575789
2: -3.0674763, -2.1132338, -3.0641041, -2.1183584, -0.3678632, 0.3696122
3: -10.2077007, -9.0677299, -10.2058897, -9.0678654, -0.4193034, 0.4176190
4: -21.1053028, -19.9409637, -21.1035748, -19.9540768, -0.3592319, 0.3631230
5: -0.6206974, 0.2850943, -0.6190079, 0.2848800, -0.2932661, 0.2918119
6: -5.3818736, -4.5177765, -5.3782763, -4.5202599, -0.2684546, 0.2673835
7: -4.0504179, -3.0586193, -4.0453310, -3.0643194, -0.3570045, 0.3574755
8: 1.1103125, 1.8133059, 1.1129761, 1.8099375, -0.3039753, 0.3046584
9: -7.7042909, -6.6409612, -7.7038288, -6.6430020, -0.3521554, 0.3537564

Time for backsubstitution: 21.75 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.35 + 557.34 = 613.70 seconds
