## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.872541919


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8442516, 1.8442519)
1: (-17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7918735, 2.7918735)
2: (-3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3578615, 2.3578615)
3: (-10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.6013856, 2.6013861)
4: (-12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7610822, 2.7610822)
5: (-4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0475702, 2.0475702)
6: (-3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2829180, 2.2829187)
7: (-9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1810565, 3.1810570)
8: (-2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0486369, 2.0486372)
9: (-4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.3078995, 2.3078992)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.05 + 40.55 = 63.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.8742898, upper bound: 0.8742899

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 5778

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742886, upper bound: 0.8733164
time: 5.22 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8733167, upper bound: 0.8742893
time: 5.06 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.38 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.38
Output dim: 0, lower bound: -0.8742886, upper bound: 0.8733164
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.38
Output dim: 0, lower bound: -0.8733167, upper bound: 0.8742893

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8438129, 1.8439796
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7894311, 2.7903647
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3533096, 2.3504939
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5931783, 2.5963163
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7595897, 2.7586677
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0417767, 2.0439901
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2812147, 2.2818644
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1783309, 3.1793728
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0478249, 2.0473304
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.3060527, 2.3049107

Time for backsubstitution: 21.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5773

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8713675, upper bound: 0.8733158
time: 8.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742847, upper bound: 0.8703952
time: 4.92 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8439798, 1.8438132
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7903638, 2.7894311
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3504939, 2.3533096
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5963159, 2.5931792
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7586675, 2.7595899
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0439901, 2.0417769
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2818642, 2.2812147
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1793733, 3.1783304
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0473304, 2.0478249
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.3049107, 2.3060524

Time for backsubstitution: 21.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5773
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5773

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8703955, upper bound: 0.8742852
time: 7.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8733128, upper bound: 0.8713682
time: 5.10 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 33.71 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 33.71
Output dim: 0, lower bound: -0.8713675, upper bound: 0.8733158
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 33.71
Output dim: 0, lower bound: -0.8742847, upper bound: 0.8703952
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 33.71
Output dim: 0, lower bound: -0.8703955, upper bound: 0.8742852
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 33.71
Output dim: 0, lower bound: -0.8733128, upper bound: 0.8713682

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8374751, 1.8384337
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7822771, 2.7881360
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3368630, 2.3316970
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.6027355, 2.6077766
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7527876, 2.7527151
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0438790, 2.0475817
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2673488, 2.2697299
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1771488, 3.1780233
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0255141, 2.0278046
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2869201, 2.2881696

Time for backsubstitution: 21.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 471

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8713635, upper bound: 0.8687372
time: 8.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8669564, upper bound: 0.8733089
time: 7.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8382671, 1.8376415
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7872028, 2.7832108
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3345127, 2.3340473
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.6046391, 2.6058726
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7536373, 2.7518654
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0453687, 2.0460920
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2690806, 2.2679977
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1769810, 3.1781912
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0282989, 2.0250196
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2893114, 2.2857783

Time for backsubstitution: 21.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 471

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742808, upper bound: 0.8659139
time: 4.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8698103, upper bound: 0.8703923
time: 10.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8376410, 1.8382673
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7832108, 2.7872028
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3340473, 2.3345127
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.6058722, 2.6046391
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7518654, 2.7536373
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0460920, 2.0453684
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2679982, 2.2690804
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1781912, 3.1769805
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0250196, 2.0282989
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2857780, 2.2893114

Time for backsubstitution: 21.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 471

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8703916, upper bound: 0.8698101
time: 5.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8659143, upper bound: 0.8742814
time: 6.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8384335, 1.8374751
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7881365, 2.7822771
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3316970, 2.3368630
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.6077766, 2.6027355
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7527142, 2.7527876
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0475821, 2.0438786
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2697301, 2.2673481
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1780233, 3.1771488
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0278044, 2.0255141
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2881699, 2.2869201

Time for backsubstitution: 21.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 471

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8733089, upper bound: 0.8669565
time: 5.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8687356, upper bound: 0.8713639
time: 7.16 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 33.34 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 33.34
Output dim: 0, lower bound: -0.8713635, upper bound: 0.8687372
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.34
Output dim: 0, lower bound: -0.8669564, upper bound: 0.8733089
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.34
Output dim: 0, lower bound: -0.8742808, upper bound: 0.8659139
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 33.34
Output dim: 0, lower bound: -0.8698103, upper bound: 0.8703923
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 33.34
Output dim: 0, lower bound: -0.8703916, upper bound: 0.8698101
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.34
Output dim: 0, lower bound: -0.8659143, upper bound: 0.8742814
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.34
Output dim: 0, lower bound: -0.8733089, upper bound: 0.8669565
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 33.34
Output dim: 0, lower bound: -0.8687356, upper bound: 0.8713639

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8333762, 1.8354797
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7796555, 2.7862468
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3324947, 2.3256330
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.6022234, 2.6070666
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7333050, 2.7386901
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0410991, 2.0437241
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2554321, 2.2531843
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1682262, 3.1715937
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0252466, 2.0276117
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2648973, 2.2723110

Time for backsubstitution: 21.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8669526, upper bound: 0.8733082
time: 13.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8669548, upper bound: 0.8733054
time: 8.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8353136, 1.8335421
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7853136, 2.7805891
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3284483, 2.3296797
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.6039305, 2.6053615
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7396126, 2.7323828
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0415111, 2.0433121
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2525349, 2.2560825
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1705503, 3.1692686
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0281057, 2.0247521
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2734528, 2.2637556

Time for backsubstitution: 21.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742781, upper bound: 0.8659141
time: 16.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742793, upper bound: 0.8659138
time: 7.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8335421, 1.8353133
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7805891, 2.7853136
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3296795, 2.3284488
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.6053610, 2.6039295
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7323828, 2.7396123
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0433116, 2.0415108
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2560825, 2.2525349
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1692686, 3.1705503
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0247521, 2.0281062
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2637558, 2.2734530

Time for backsubstitution: 21.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8659135, upper bound: 0.8742787
time: 5.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8659135, upper bound: 0.8742775
time: 5.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8354800, 1.8333757
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7862473, 2.7796559
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3256330, 2.3324950
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.6070662, 2.6022239
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7386904, 2.7333050
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0437236, 2.0410988
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2531843, 2.2554324
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1715937, 3.1682258
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0276117, 2.0252466
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2723112, 2.2648973

Time for backsubstitution: 21.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8733055, upper bound: 0.8669578
time: 9.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8733076, upper bound: 0.8669520
time: 5.13 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 36.35 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.35
Output dim: 0, lower bound: -0.8669526, upper bound: 0.8733082
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.35
Output dim: 0, lower bound: -0.8669548, upper bound: 0.8733054
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.35
Output dim: 0, lower bound: -0.8742781, upper bound: 0.8659141
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.35
Output dim: 0, lower bound: -0.8742793, upper bound: 0.8659138
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.35
Output dim: 0, lower bound: -0.8659135, upper bound: 0.8742787
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.35
Output dim: 0, lower bound: -0.8659135, upper bound: 0.8742775
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.35
Output dim: 0, lower bound: -0.8733055, upper bound: 0.8669578
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.35
Output dim: 0, lower bound: -0.8733076, upper bound: 0.8669520

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8331144, 1.8350120
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7749281, 2.7777677
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3322802, 2.3252511
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5977812, 2.5990977
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7303486, 2.7370391
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0395174, 2.0408864
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2548223, 2.2528441
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1603842, 3.1672277
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0252047, 2.0275898
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2597160, 2.2694194

Time for backsubstitution: 21.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 864

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8663646, upper bound: 0.8726987
time: 5.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8652840, upper bound: 0.8727058
time: 7.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8329084, 1.8352180
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7711763, 2.7815180
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3321128, 2.3254182
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5942545, 2.6026239
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7316542, 2.7357335
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0382605, 2.0421429
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2550921, 2.2525740
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1638594, 3.1637516
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0252247, 2.0275698
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2620058, 2.2671297

Time for backsubstitution: 20.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 864

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8663680, upper bound: 0.8726971
time: 6.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8652840, upper bound: 0.8727018
time: 6.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8350523, 1.8330743
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7805853, 2.7721100
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3282337, 2.3292978
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5994864, 2.5973921
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7366562, 2.7307317
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0399294, 2.0404744
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2519250, 2.2557423
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1627092, 3.1649027
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0280643, 2.0247307
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2682714, 2.2608640

Time for backsubstitution: 21.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 864

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8737233, upper bound: 0.8652856
time: 6.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726944, upper bound: 0.8652867
time: 4.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8348458, 1.8332806
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7768354, 2.7758603
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3280668, 2.3294647
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.5959606, 2.6009188
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7379618, 2.7294261
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0386734, 2.0417309
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2521949, 2.2554722
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1661844, 3.1614265
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0280843, 2.0247107
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2705617, 2.2585742

Time for backsubstitution: 22.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 864

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8737239, upper bound: 0.8652848
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726967, upper bound: 0.8652872
time: 4.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8332808, 1.8348455
1: -17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7758608, 2.7768345
2: -3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3294644, 2.3280668
3: -10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.6009188, 2.5959606
4: -12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7294264, 2.7379615
5: -4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0417309, 2.0386732
6: -3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2554727, 2.2521946
7: -9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1614265, 3.1661849
8: -2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0247107, 2.0280843
9: -4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.2585745, 2.2705617

Time for backsubstitution: 21.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 4603
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 864

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8652874, upper bound: 0.8726971
time: 14.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8652853, upper bound: 0.8737236
time: 5.58 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 41.92 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 41.92
Output dim: 0, lower bound: -0.8663646, upper bound: 0.8726987
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 41.92
Output dim: 0, lower bound: -0.8652840, upper bound: 0.8727058
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 41.92
Output dim: 0, lower bound: -0.8663680, upper bound: 0.8726971
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 41.92
Output dim: 0, lower bound: -0.8652840, upper bound: 0.8727018
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 41.92
Output dim: 0, lower bound: -0.8737233, upper bound: 0.8652856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 41.92
Output dim: 0, lower bound: -0.8726944, upper bound: 0.8652867
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 41.92
Output dim: 0, lower bound: -0.8737239, upper bound: 0.8652848
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 41.92
Output dim: 0, lower bound: -0.8726967, upper bound: 0.8652872
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 41.92
Output dim: 0, lower bound: -0.8652874, upper bound: 0.8726971
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 41.92
Output dim: 0, lower bound: -0.8652853, upper bound: 0.8737236
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.92
Output dim: 0, lower bound: -0.8659135, upper bound: 0.8742775
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.92
Output dim: 0, lower bound: -0.8733055, upper bound: 0.8669578
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.92
Output dim: 0, lower bound: -0.8733076, upper bound: 0.8669520

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 63.60 + 551.74 = 615.34 seconds
