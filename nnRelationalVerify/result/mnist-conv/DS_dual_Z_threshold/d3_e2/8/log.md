## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.370427409


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.2011700, 1.2011700)
1: (-13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8999453, 0.8999455)
2: (-5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0896006, 1.0896006)
3: (-8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.8056455, 0.8056452)
4: (-11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9431391, 0.9431391)
5: (0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9501886, 0.9501886)
6: (-4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8429096, 0.8429098)
7: (-11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0050030, 1.0050030)
8: (6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6467774, 0.6467774)
9: (-5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6523471, 0.6523471)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.58 + 33.42 = 57.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.3741690, upper bound: 0.3741700

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 6210
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 5745

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3741686, upper bound: 0.3736827
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3736818, upper bound: 0.3741695
time: 3.04 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.45 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.45
Output dim: 8, lower bound: -0.3741686, upper bound: 0.3736827
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.45
Output dim: 8, lower bound: -0.3736818, upper bound: 0.3741695

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1950855, 1.1988492
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8986712, 0.8994603
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0880623, 1.0855742
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.8032966, 0.8047516
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9399624, 0.9347985
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9472404, 0.9424567
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8273299, 0.8369768
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0009055, 0.9942694
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6466532, 0.6464505
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6477013, 0.6505767

Time for backsubstitution: 20.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6210
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 6210

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3725340, upper bound: 0.3736824
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3741682, upper bound: 0.3720482
time: 3.18 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1988487, 1.1950855
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8994608, 0.8986712
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0855742, 1.0880623
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.8047514, 0.8032966
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9347982, 0.9399626
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9424567, 0.9472404
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8369768, 0.8273301
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -0.9942694, 1.0009055
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6464508, 0.6466532
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6505766, 0.6477011

Time for backsubstitution: 20.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6210
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6210

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3720472, upper bound: 0.3741692
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3736815, upper bound: 0.3725349
time: 3.38 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.46 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.46
Output dim: 8, lower bound: -0.3725340, upper bound: 0.3736824
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.46
Output dim: 8, lower bound: -0.3741682, upper bound: 0.3720482
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.46
Output dim: 8, lower bound: -0.3720472, upper bound: 0.3741692
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.46
Output dim: 8, lower bound: -0.3736815, upper bound: 0.3725349

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1945782, 1.1989679
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9011335, 0.9002995
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0838180, 1.0799170
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7953844, 0.7942073
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9435740, 0.9376314
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9506569, 0.9469638
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8188918, 0.8257203
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0020342, 0.9951549
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6422250, 0.6443768
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6423485, 0.6465603

Time for backsubstitution: 20.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 945

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3711025, upper bound: 0.3719845
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3708360, upper bound: 0.3722509
time: 3.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1952038, 1.1983418
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8995104, 0.9019229
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0824056, 1.0813293
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7927523, 0.7968395
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9427958, 0.9384093
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9517474, 0.9458733
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8160737, 0.8285379
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0017915, 0.9953980
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6445792, 0.6420226
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6436849, 0.6452241

Time for backsubstitution: 21.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 945

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3727367, upper bound: 0.3703502
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3724703, upper bound: 0.3706167
time: 3.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1983414, 1.1952038
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9019232, 0.8995106
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0813298, 1.0824056
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7968397, 0.7927523
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9384089, 0.9427958
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9458737, 0.9517474
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8285382, 0.8160737
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -0.9953980, 1.0017915
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6420226, 0.6445794
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6452241, 0.6436847

Time for backsubstitution: 21.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 945

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3706157, upper bound: 0.3724712
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3703493, upper bound: 0.3727377
time: 3.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1989679, 1.1945782
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9003000, 0.9011340
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0799170, 1.0838180
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7942076, 0.7953844
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9376316, 0.9435735
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9469643, 0.9506569
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8257201, 0.8188913
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -0.9951549, 1.0020342
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6443768, 0.6422253
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6465602, 0.6423485

Time for backsubstitution: 20.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 945

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3722500, upper bound: 0.3708370
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3719835, upper bound: 0.3711034
time: 3.04 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 26.89 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 8, lower bound: -0.3711025, upper bound: 0.3719845
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 8, lower bound: -0.3708360, upper bound: 0.3722509
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 8, lower bound: -0.3727367, upper bound: 0.3703502
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 8, lower bound: -0.3724703, upper bound: 0.3706167
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 8, lower bound: -0.3706157, upper bound: 0.3724712
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 8, lower bound: -0.3703493, upper bound: 0.3727377
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 8, lower bound: -0.3722500, upper bound: 0.3708370
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 8, lower bound: -0.3719835, upper bound: 0.3711034

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1944580, 1.1995339
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9019108, 0.9001334
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0837865, 1.0800605
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7951717, 0.7952085
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9436569, 0.9376142
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9506068, 0.9471974
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8187678, 0.8263028
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0020423, 0.9951539
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6425941, 0.6442976
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6425867, 0.6465091

Time for backsubstitution: 20.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 5859

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3711004, upper bound: 0.3718339
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3706850, upper bound: 0.3718380
time: 3.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1945782, 1.1988473
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9009676, 0.9002995
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0838180, 1.0798855
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7953844, 0.7939944
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9435558, 0.9376314
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9506569, 0.9469142
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8188918, 0.8255968
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0020332, 0.9951549
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6421459, 0.6443768
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6422975, 0.6465603

Time for backsubstitution: 21.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 5859

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3706896, upper bound: 0.3718335
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3706854, upper bound: 0.3722488
time: 2.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1950836, 1.1989079
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9002872, 0.9017568
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0823741, 1.0814729
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7925396, 0.7978406
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9428787, 0.9383922
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9516978, 0.9461064
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8159502, 0.8291204
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0017991, 0.9953966
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6449482, 0.6419435
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6439230, 0.6451728

Time for backsubstitution: 21.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 5859

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3727346, upper bound: 0.3701995
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723193, upper bound: 0.3702038
time: 3.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1952038, 1.1982212
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8993440, 0.9019229
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0824056, 1.0812984
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7927523, 0.7966266
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9427786, 0.9384093
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9517474, 0.9458237
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8160737, 0.8284144
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0017905, 0.9953980
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6445000, 0.6420226
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6436336, 0.6452241

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 5859

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723239, upper bound: 0.3701993
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723197, upper bound: 0.3706146
time: 3.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1982212, 1.1957703
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9026999, 0.8993444
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0812984, 1.0825486
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7966266, 0.7937531
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9384928, 0.9427786
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9458237, 0.9519806
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8284142, 0.8166561
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -0.9954057, 1.0017900
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6423917, 0.6445003
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6454623, 0.6436335

Time for backsubstitution: 21.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 5859

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3706136, upper bound: 0.3723206
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3701983, upper bound: 0.3723248
time: 3.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1983414, 1.1950836
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9017568, 0.8995106
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0813298, 1.0823741
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7968397, 0.7925394
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9383917, 0.9427958
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9458737, 0.9516978
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8285382, 0.8159502
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -0.9953966, 1.0017915
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6419435, 0.6445794
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6451728, 0.6436847

Time for backsubstitution: 22.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 5859

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3702029, upper bound: 0.3723203
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3701987, upper bound: 0.3727356
time: 3.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1988468, 1.1951442
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9010763, 0.9009678
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0798855, 1.0839610
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7939944, 0.7963853
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9377146, 0.9435563
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9469142, 0.9508901
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8255966, 0.8194737
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -0.9951630, 1.0020328
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6447458, 0.6421461
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6467984, 0.6422973

Time for backsubstitution: 22.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 5859

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3722479, upper bound: 0.3706864
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718326, upper bound: 0.3706906
time: 3.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1989679, 1.1944575
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9001336, 0.9011340
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0799170, 1.0837865
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7942076, 0.7951715
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9376144, 0.9435735
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9469643, 0.9506068
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8257201, 0.8187678
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -0.9951539, 1.0020342
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6442976, 0.6422253
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6465089, 0.6423485

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5859

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 5859

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718371, upper bound: 0.3706860
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718329, upper bound: 0.3711013
time: 3.18 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.96 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3711004, upper bound: 0.3718339
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3706850, upper bound: 0.3718380
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3706896, upper bound: 0.3718335
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3706854, upper bound: 0.3722488
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3727346, upper bound: 0.3701995
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3723193, upper bound: 0.3702038
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3723239, upper bound: 0.3701993
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3723197, upper bound: 0.3706146
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3706136, upper bound: 0.3723206
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3701983, upper bound: 0.3723248
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3702029, upper bound: 0.3723203
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3701987, upper bound: 0.3727356
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3722479, upper bound: 0.3706864
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3718326, upper bound: 0.3706906
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3718371, upper bound: 0.3706860
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 8, lower bound: -0.3718329, upper bound: 0.3711013

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1903191, 1.1974845
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9002538, 0.8967969
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0829916, 1.0784616
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7816520, 0.7885046
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9413056, 0.9328780
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9475322, 0.9456711
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8116663, 0.8227751
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0007405, 0.9925365
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6413898, 0.6418679
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6411819, 0.6436827

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 164
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2529
type: DSZ, layer: 3, pos: 1696
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 2641
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 1115
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 551
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1201
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2584
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2823
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1474

Time for candidate selection: 0.55 seconds

### Candidate
type: DSZ, layer: 3, pos: 1501

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3693840, upper bound: 0.3713666
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3706323, upper bound: 0.3702234
time: 3.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1924076, 1.1953950
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8985744, 0.8984716
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0821877, 1.0789108
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7884679, 0.7816887
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9389205, 0.9352634
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9490809, 0.9441228
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8152401, 0.8192012
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -0.9994245, 0.9938526
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6401644, 0.6430898
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6397605, 0.6451043

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 164
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2529
type: DSZ, layer: 3, pos: 1696
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 2641
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 1115
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 551
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1201
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2584
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2823
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1474

Time for candidate selection: 0.63 seconds

### Candidate
type: DSZ, layer: 3, pos: 1501

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3689687, upper bound: 0.3713709
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3702169, upper bound: 0.3702276
time: 3.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1904392, 1.1967974
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8993058, 0.8969634
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0830231, 1.0782871
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7818646, 0.7872908
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9412055, 0.9328952
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9475822, 0.9453878
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8117902, 0.8220692
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0007319, 0.9925385
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6409383, 0.6419470
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6408925, 0.6437337

Time for backsubstitution: 23.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 164
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2529
type: DSZ, layer: 3, pos: 1696
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 2641
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 1115
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 551
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1201
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2584
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2823
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1474

Time for candidate selection: 0.62 seconds

### Candidate
type: DSZ, layer: 3, pos: 1501

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3689733, upper bound: 0.3713664
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3702215, upper bound: 0.3702230
time: 3.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1925278, 1.1947083
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8976312, 0.8986380
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0822191, 1.0790906
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7886810, 0.7804747
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9388199, 0.9352806
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9491305, 0.9438396
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8153641, 0.8184953
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -0.9994159, 0.9938545
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6397161, 0.6431689
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6394711, 0.6451553

Time for backsubstitution: 22.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 164
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2529
type: DSZ, layer: 3, pos: 1696
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 2641
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 1115
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 551
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1201
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2584
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2823
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1474

Time for candidate selection: 0.56 seconds

### Candidate
type: DSZ, layer: 3, pos: 1501

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3689691, upper bound: 0.3717817
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3702173, upper bound: 0.3706383
time: 3.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1909447, 1.1968579
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8986306, 0.8984203
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0815792, 1.0798740
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7790198, 0.7911367
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9405279, 0.9336560
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9486232, 0.9445806
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8088486, 0.8255928
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0004978, 0.9927793
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6437440, 0.6395137
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6425180, 0.6423465

Time for backsubstitution: 23.12 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.00 + 543.47 = 600.47 seconds
