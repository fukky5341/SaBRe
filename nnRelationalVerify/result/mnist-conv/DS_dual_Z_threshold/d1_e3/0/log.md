## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.14146995799999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2816482, 0.2816482)
1: (-12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3865297, 0.3865297)
2: (-2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3325222, 0.3325224)
3: (-10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4566672, 0.4566669)
4: (-6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4396453, 0.4396451)
5: (-8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3318262, 0.3318262)
6: (-3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3250239, 0.3250240)
7: (-10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2660110, 0.2660111)
8: (-2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3044350, 0.3044350)
9: (-3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3198457, 0.3198457)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.43 + 32.96 = 55.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.1443570, upper bound: 0.1443571

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4596
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 6112
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 4596

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443540, upper bound: 0.1424735
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424734, upper bound: 0.1443542
time: 2.82 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.88 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.88
Output dim: 0, lower bound: -0.1443540, upper bound: 0.1424735
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.88
Output dim: 0, lower bound: -0.1424734, upper bound: 0.1443542

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2799909, 0.2790952
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3854406, 0.3848474
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3308246, 0.3314168
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4543931, 0.4551868
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4390531, 0.4387343
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3303797, 0.3308849
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3241963, 0.3237535
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2645504, 0.2637553
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3037858, 0.3040121
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3193148, 0.3190274

Time for backsubstitution: 21.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 6112
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 469

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1435886, upper bound: 0.1424728
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443533, upper bound: 0.1417082
time: 3.05 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2790952, 0.2799909
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3848474, 0.3854406
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3314168, 0.3308244
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4551871, 0.4543934
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4387341, 0.4390533
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3308849, 0.3303797
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3237534, 0.3241962
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2637553, 0.2645504
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3040121, 0.3037858
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3190275, 0.3193147

Time for backsubstitution: 21.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 6112
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 469

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1417081, upper bound: 0.1443534
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424727, upper bound: 0.1435888
time: 2.94 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.31 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.31
Output dim: 0, lower bound: -0.1435886, upper bound: 0.1424728
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.31
Output dim: 0, lower bound: -0.1443533, upper bound: 0.1417082
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.31
Output dim: 0, lower bound: -0.1417081, upper bound: 0.1443534
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.31
Output dim: 0, lower bound: -0.1424727, upper bound: 0.1435888

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2792873, 0.2794447
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3862901, 0.3831429
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3306968, 0.3314800
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4550052, 0.4539566
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4385519, 0.4389815
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3308759, 0.3298824
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3246393, 0.3228608
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2648071, 0.2632363
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3039227, 0.3037348
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3189133, 0.3192270

Time for backsubstitution: 21.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6112
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1435871, upper bound: 0.1413804
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424962, upper bound: 0.1424712
time: 2.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2799909, 0.2783918
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3837359, 0.3848474
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3308246, 0.3312891
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4531631, 0.4551868
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4390531, 0.4382329
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3293772, 0.3308849
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3233037, 0.3237535
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2640315, 0.2637553
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3035083, 0.3040121
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3193148, 0.3186259

Time for backsubstitution: 21.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6112
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 6112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443517, upper bound: 0.1406156
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1432608, upper bound: 0.1417064
time: 5.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2783918, 0.2803402
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3856969, 0.3837359
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3312891, 0.3308876
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4557991, 0.4531631
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4382329, 0.4393005
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3313811, 0.3293772
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3241968, 0.3233036
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2640120, 0.2640314
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3041492, 0.3035083
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3186260, 0.3195143

Time for backsubstitution: 21.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6112
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1417064, upper bound: 0.1432610
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1406156, upper bound: 0.1443519
time: 2.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2790952, 0.2792873
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3831429, 0.3854406
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3314168, 0.3306968
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4539566, 0.4543934
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4387341, 0.4385519
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3298824, 0.3303797
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3228607, 0.3241962
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2632362, 0.2645504
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3037348, 0.3037858
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3190275, 0.3189132

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6112
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 6112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424711, upper bound: 0.1424963
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413802, upper bound: 0.1435872
time: 3.10 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.05 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.05
Output dim: 0, lower bound: -0.1435871, upper bound: 0.1413804
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.05
Output dim: 0, lower bound: -0.1424962, upper bound: 0.1424712
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.05
Output dim: 0, lower bound: -0.1443517, upper bound: 0.1406156
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.05
Output dim: 0, lower bound: -0.1432608, upper bound: 0.1417064
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.05
Output dim: 0, lower bound: -0.1417064, upper bound: 0.1432610
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.05
Output dim: 0, lower bound: -0.1406156, upper bound: 0.1443519
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.05
Output dim: 0, lower bound: -0.1424711, upper bound: 0.1424963
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.05
Output dim: 0, lower bound: -0.1413802, upper bound: 0.1435872

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2773035, 0.2763972
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3831980, 0.3811290
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3300679, 0.3310685
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4534025, 0.4514966
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4344525, 0.4363084
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3305202, 0.3293357
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3214135, 0.3207582
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2639534, 0.2619289
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3036606, 0.3033330
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3180461, 0.3178945

Time for backsubstitution: 21.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1435870, upper bound: 0.1413505
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1435572, upper bound: 0.1413803
time: 2.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2762399, 0.2774608
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3842762, 0.3800509
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3302853, 0.3308511
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4525452, 0.4523542
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4358788, 0.4348822
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3303292, 0.3295267
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3225369, 0.3196348
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2634997, 0.2623824
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3035209, 0.3034728
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3175809, 0.3183599

Time for backsubstitution: 20.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424962, upper bound: 0.1424414
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424664, upper bound: 0.1424712
time: 2.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2780070, 0.2753444
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3806438, 0.3828337
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3301957, 0.3308778
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4515605, 0.4527273
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4349532, 0.4355597
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3290215, 0.3303378
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3200774, 0.3216511
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2631776, 0.2624478
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3032465, 0.3036098
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3184476, 0.3172936

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443516, upper bound: 0.1405859
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443218, upper bound: 0.1406156
time: 3.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2769434, 0.2764077
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3817222, 0.3817556
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3304131, 0.3306601
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4507031, 0.4535847
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4363799, 0.4341331
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3288305, 0.3305287
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3212008, 0.3205274
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2627242, 0.2629013
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3031065, 0.3037496
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3179824, 0.3177588

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1432608, upper bound: 0.1416768
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1432310, upper bound: 0.1417066
time: 2.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2764077, 0.2772927
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3826051, 0.3817222
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3306601, 0.3304763
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4541965, 0.4507029
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4341331, 0.4366274
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3310254, 0.3288305
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3209705, 0.3212010
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2631581, 0.2627242
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3038871, 0.3031068
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3177588, 0.3181820

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1417064, upper bound: 0.1432312
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1416766, upper bound: 0.1432609
time: 3.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2753444, 0.2783563
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3836832, 0.3806438
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3308775, 0.3302586
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4533391, 0.4515605
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4355597, 0.4352012
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3308344, 0.3290215
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3220944, 0.3200775
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2627046, 0.2631776
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3037474, 0.3032465
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3172936, 0.3186471

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1406155, upper bound: 0.1443220
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1405858, upper bound: 0.1443518
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2771113, 0.2762399
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3800509, 0.3834267
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3307879, 0.3302853
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4523540, 0.4519336
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4346342, 0.4358788
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3295267, 0.3298326
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3196349, 0.3220938
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2623825, 0.2632430
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3034728, 0.3033836
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3181603, 0.3175809

Time for backsubstitution: 21.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424710, upper bound: 0.1424665
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424412, upper bound: 0.1424963
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2760477, 0.2773035
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3811293, 0.3823485
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3310053, 0.3300679
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4514966, 0.4527910
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4360609, 0.4344525
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3293357, 0.3300235
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3207583, 0.3209702
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2619288, 0.2636964
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3033330, 0.3035233
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3176951, 0.3180461

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413802, upper bound: 0.1435574
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413504, upper bound: 0.1435872
time: 3.43 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.63 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1435870, upper bound: 0.1413505
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1435572, upper bound: 0.1413803
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1424962, upper bound: 0.1424414
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1424664, upper bound: 0.1424712
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1443516, upper bound: 0.1405859
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1443218, upper bound: 0.1406156
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1432608, upper bound: 0.1416768
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1432310, upper bound: 0.1417066
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1417064, upper bound: 0.1432312
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1416766, upper bound: 0.1432609
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1406155, upper bound: 0.1443220
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1405858, upper bound: 0.1443518
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1424710, upper bound: 0.1424665
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1424412, upper bound: 0.1424963
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1413802, upper bound: 0.1435574
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.63
Output dim: 0, lower bound: -0.1413504, upper bound: 0.1435872

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2774150, 0.2764924
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3827627, 0.3810532
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3297839, 0.3307278
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4552717, 0.4538641
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4367194, 0.4379005
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3316841, 0.3302166
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3189125, 0.3186730
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2639509, 0.2619282
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3036065, 0.3035722
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3162081, 0.3156900

Time for backsubstitution: 21.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 563

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1435868, upper bound: 0.1411491
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1433856, upper bound: 0.1413503
time: 3.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2773986, 0.2765088
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3831220, 0.3806939
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3297272, 0.3307846
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4557705, 0.4533658
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4360442, 0.4385753
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3314011, 0.3304999
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3193283, 0.3182577
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2639526, 0.2619265
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3038998, 0.3032789
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3158417, 0.3160565

Time for backsubstitution: 21.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 563

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1435570, upper bound: 0.1411789
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1433558, upper bound: 0.1413800
time: 3.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2763515, 0.2775559
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3838410, 0.3799751
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3300014, 0.3305101
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4544144, 0.4547217
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4381456, 0.4364738
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3314931, 0.3304076
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3200364, 0.3175496
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2634975, 0.2623817
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3034668, 0.3037119
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3157430, 0.3161552

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 563

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424959, upper bound: 0.1422400
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1422948, upper bound: 0.1424411
time: 3.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2763350, 0.2775724
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3842001, 0.3796155
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3299446, 0.3305669
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4549127, 0.4542232
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4374709, 0.4371490
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3312101, 0.3306909
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3204517, 0.3171343
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2634991, 0.2623800
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3037601, 0.3034186
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3153765, 0.3165216

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 563

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424661, upper bound: 0.1422698
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1422650, upper bound: 0.1424709
time: 3.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2781186, 0.2754395
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3802087, 0.3827581
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3299122, 0.3305368
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4534292, 0.4550946
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4372206, 0.4371519
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3301857, 0.3312190
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3175769, 0.3195657
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2631754, 0.2624471
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3031924, 0.3038490
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3166094, 0.3150890

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 563

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 563

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443514, upper bound: 0.1403845
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1441502, upper bound: 0.1405857
time: 3.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2781022, 0.2754560
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3805680, 0.3823986
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3298550, 0.3305936
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4539280, 0.4545960
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4365454, 0.4378266
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3299024, 0.3315020
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3179922, 0.3191503
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2631768, 0.2624454
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3034854, 0.3035557
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3162427, 0.3154554

Time for backsubstitution: 21.90 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.39 + 561.06 = 616.44 seconds
