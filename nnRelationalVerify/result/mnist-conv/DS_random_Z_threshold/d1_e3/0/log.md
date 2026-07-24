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
execution time: IAR + RelationalAnalysis = 23.65 + 32.67 = 56.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.1443570, upper bound: 0.1443571

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 563
type: DSZ, layer: 1, pos: 4596
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 6112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 563

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443567, upper bound: 0.1441557
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1441556, upper bound: 0.1443569
time: 2.85 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.76 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.76
Output dim: 0, lower bound: -0.1443567, upper bound: 0.1441557
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.76
Output dim: 0, lower bound: -0.1441556, upper bound: 0.1443569

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2806668, 0.2804658
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3795998, 0.3781857
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3284976, 0.3276930
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4494741, 0.4480367
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4281011, 0.4300232
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3308358, 0.3304460
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3212731, 0.3218971
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2635220, 0.2639443
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3039088, 0.3040662
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3158075, 0.3165455

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 4596
type: DSZ, layer: 1, pos: 6112
type: DSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 469

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1435914, upper bound: 0.1441550
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443560, upper bound: 0.1433904
time: 3.22 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2804658, 0.2806668
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3781857, 0.3795998
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3276927, 0.3284976
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4480369, 0.4494739
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4300237, 0.4281006
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3304460, 0.3308358
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3218973, 0.3212732
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2639443, 0.2635221
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3040662, 0.3039091
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3165454, 0.3158073

Time for backsubstitution: 22.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 6112
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 4596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1441555, upper bound: 0.1443270
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1441257, upper bound: 0.1443568
time: 3.05 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.72 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.72
Output dim: 0, lower bound: -0.1435914, upper bound: 0.1441550
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.72
Output dim: 0, lower bound: -0.1443560, upper bound: 0.1433904
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.72
Output dim: 0, lower bound: -0.1441555, upper bound: 0.1443270
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.72
Output dim: 0, lower bound: -0.1441257, upper bound: 0.1443568

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2799633, 0.2808151
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3804491, 0.3764808
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3283699, 0.3277559
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4500856, 0.4468064
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4275994, 0.4302711
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3313322, 0.3294437
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3217158, 0.3210038
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2637789, 0.2634254
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3040464, 0.3037894
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3154062, 0.3167453

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6112
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1435898, upper bound: 0.1430626
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424989, upper bound: 0.1441534
time: 2.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2806668, 0.2797620
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3778949, 0.3781857
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3284976, 0.3275652
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4482436, 0.4480367
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4281011, 0.4295225
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3298335, 0.3304460
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3203797, 0.3218971
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2630031, 0.2639443
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3036320, 0.3040662
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3158075, 0.3161442

Time for backsubstitution: 22.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4596
type: DSZ, layer: 1, pos: 6112
type: DSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4596

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443530, upper bound: 0.1415067
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424724, upper bound: 0.1433874
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2805774, 0.2807622
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3777509, 0.3795245
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3274093, 0.3281574
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4499059, 0.4518418
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4322896, 0.4296918
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3316102, 0.3317170
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3193960, 0.3191875
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2639419, 0.2635216
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3040116, 0.3041475
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3147068, 0.3136024

Time for backsubstitution: 23.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4596
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 6112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4596

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1441525, upper bound: 0.1424434
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1422719, upper bound: 0.1443240
time: 3.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2805610, 0.2807784
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3781104, 0.3791652
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3273525, 0.3282142
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4504046, 0.4513435
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4316144, 0.4303670
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3313272, 0.3320000
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3198113, 0.3187720
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2639438, 0.2635199
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3043048, 0.3038542
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3143406, 0.3139687

Time for backsubstitution: 22.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6112
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 4596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1441241, upper bound: 0.1432644
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1430333, upper bound: 0.1443552
time: 2.92 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.17 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 0, lower bound: -0.1435898, upper bound: 0.1430626
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 0, lower bound: -0.1424989, upper bound: 0.1441534
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 0, lower bound: -0.1443530, upper bound: 0.1415067
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 0, lower bound: -0.1424724, upper bound: 0.1433874
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 0, lower bound: -0.1441525, upper bound: 0.1424434
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 0, lower bound: -0.1422719, upper bound: 0.1443240
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 0, lower bound: -0.1441241, upper bound: 0.1432644
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 0, lower bound: -0.1430333, upper bound: 0.1443552

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2779789, 0.2777674
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3773572, 0.3744671
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3277414, 0.3273451
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4484832, 0.4443460
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4235001, 0.4275975
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3309765, 0.3288970
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3184896, 0.3189012
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2629249, 0.2621179
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3037841, 0.3033874
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3145390, 0.3154128

Time for backsubstitution: 22.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4596
type: DSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4596

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1435868, upper bound: 0.1411790
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1417062, upper bound: 0.1430596
time: 2.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2769156, 0.2788308
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3784354, 0.3733888
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3279593, 0.3271277
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4476254, 0.4452033
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4249263, 0.4261713
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3307855, 0.3290880
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3196130, 0.3177778
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2624714, 0.2625713
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3036444, 0.3035271
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3140738, 0.3158779

Time for backsubstitution: 22.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424989, upper bound: 0.1441236
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424691, upper bound: 0.1441534
time: 2.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2790091, 0.2772088
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3768058, 0.3765035
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3268001, 0.3264596
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4459696, 0.4465561
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4275084, 0.4286108
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3283873, 0.3295050
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3195529, 0.3206276
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2615426, 0.2616882
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3029826, 0.3036430
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3152764, 0.3153260

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 6112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443529, upper bound: 0.1414769
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443231, upper bound: 0.1415067
time: 3.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2781136, 0.2781043
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3762128, 0.3770964
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3273928, 0.3258672
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4467630, 0.4457626
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4271894, 0.4289298
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3288925, 0.3289998
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3191099, 0.3210701
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2607474, 0.2624834
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3032088, 0.3034165
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3149891, 0.3156133

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 6112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424723, upper bound: 0.1433576
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424425, upper bound: 0.1433873
time: 3.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2789202, 0.2782092
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3766615, 0.3778419
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3257113, 0.3270519
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4476314, 0.4503608
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4316978, 0.4287815
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3301637, 0.3307757
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3185689, 0.3179177
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2624812, 0.2612656
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3033624, 0.3037245
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3141767, 0.3127847

Time for backsubstitution: 22.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6112
type: DSZ, layer: 1, pos: 469

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1441510, upper bound: 0.1413510
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1430601, upper bound: 0.1424419
time: 3.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2780244, 0.2791047
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3760684, 0.3784349
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3263040, 0.3264594
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4484248, 0.4495673
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4313793, 0.4291005
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3306689, 0.3302705
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3181264, 0.3183604
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2616861, 0.2620609
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3035889, 0.3034980
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3138891, 0.3130717

Time for backsubstitution: 22.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 6112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 469

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1415066, upper bound: 0.1443233
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1422712, upper bound: 0.1435586
time: 3.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2785771, 0.2777309
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3750184, 0.3771513
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3267231, 0.3278022
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4488022, 0.4488835
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4275150, 0.4276934
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3309715, 0.3314533
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3165855, 0.3166696
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2630897, 0.2622124
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3040428, 0.3034527
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3134735, 0.3126364

Time for backsubstitution: 22.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 4596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 469

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1433588, upper bound: 0.1432637
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1441234, upper bound: 0.1424991
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2775135, 0.2787945
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3760965, 0.3760731
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3269405, 0.3275847
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4479449, 0.4497409
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4289412, 0.4262671
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3307805, 0.3316443
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3177090, 0.3155459
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2626362, 0.2626660
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3039029, 0.3035924
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3130083, 0.3131018

Time for backsubstitution: 22.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4596
type: DSZ, layer: 1, pos: 469

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4596

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1430303, upper bound: 0.1424716
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1411497, upper bound: 0.1443523
time: 2.97 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.49 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1435868, upper bound: 0.1411790
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1417062, upper bound: 0.1430596
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1424989, upper bound: 0.1441236
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1424691, upper bound: 0.1441534
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1443529, upper bound: 0.1414769
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1443231, upper bound: 0.1415067
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1424723, upper bound: 0.1433576
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1424425, upper bound: 0.1433873
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1441510, upper bound: 0.1413510
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1430601, upper bound: 0.1424419
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1415066, upper bound: 0.1443233
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1422712, upper bound: 0.1435586
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1433588, upper bound: 0.1432637
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1441234, upper bound: 0.1424991
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1430303, upper bound: 0.1424716
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 0, lower bound: -0.1411497, upper bound: 0.1443523

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2763216, 0.2752144
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3762679, 0.3727849
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3260434, 0.3262396
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4462087, 0.4428654
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4229078, 0.4266863
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3295302, 0.3279560
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3176627, 0.3176316
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2614644, 0.2598619
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3031347, 0.3029642
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3140081, 0.3145945

Time for backsubstitution: 22.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1435868, upper bound: 0.1411491
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1435570, upper bound: 0.1411789
time: 3.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2754261, 0.2761099
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3756750, 0.3733778
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3266361, 0.3256474
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4470026, 0.4420719
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4225883, 0.4270053
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3300354, 0.3274508
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3172197, 0.3180743
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2606692, 0.2606572
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3033609, 0.3027380
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3137206, 0.3148818

Time for backsubstitution: 22.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1417062, upper bound: 0.1430297
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1416763, upper bound: 0.1430595
time: 3.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2770276, 0.2789266
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3780007, 0.3733137
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3276749, 0.3267868
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4494951, 0.4475718
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4271927, 0.4277625
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3319497, 0.3299692
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3171122, 0.3156923
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2624691, 0.2625707
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3035896, 0.3037655
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3122354, 0.3136733

Time for backsubstitution: 22.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4596

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424959, upper bound: 0.1422400
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1406153, upper bound: 0.1441205
time: 4.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2770112, 0.2789428
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3783600, 0.3729544
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3276181, 0.3268435
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4499938, 0.4470730
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4265175, 0.4284372
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3316667, 0.3302524
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3175275, 0.3152767
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2624710, 0.2625691
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3038826, 0.3034723
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3118691, 0.3140397

Time for backsubstitution: 22.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4596

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424661, upper bound: 0.1422698
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1405855, upper bound: 0.1441504
time: 3.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2791212, 0.2773046
1: -12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3763707, 0.3764279
2: -2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3265162, 0.3261192
3: -10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4478385, 0.4489236
4: -6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4297752, 0.4302032
5: -8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3295517, 0.3303859
6: -3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3170519, 0.3185419
7: -10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2615401, 0.2616878
8: -2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3029282, 0.3038819
9: -3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3134385, 0.3131216

Time for backsubstitution: 22.77 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.32 + 546.48 = 602.80 seconds
