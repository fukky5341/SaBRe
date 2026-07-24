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
execution time: IAR + RelationalAnalysis = 22.99 + 32.74 = 55.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.3741690, upper bound: 0.3741700

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6210
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6210

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3725344, upper bound: 0.3741697
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3741687, upper bound: 0.3725354
time: 3.16 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.11 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.11
Output dim: 8, lower bound: -0.3725344, upper bound: 0.3741697
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.11
Output dim: 8, lower bound: -0.3741687, upper bound: 0.3725354

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.2006607, 1.2012877
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9024086, 0.9007850
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0853558, 1.0839434
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7977333, 0.7951014
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9467502, 0.9459724
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9536042, 0.9546947
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8344700, 0.8316529
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0061316, 1.0058889
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6423495, 0.6447036
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6469946, 0.6483309

Time for backsubstitution: 21.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5859

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3725338, upper bound: 0.3724348
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3707995, upper bound: 0.3741690
time: 3.04 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.2012882, 1.2006617
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9007850, 0.9024086
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0839434, 1.0853558
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7951012, 0.7977335
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9459724, 0.9467502
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9546947, 0.9536042
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8316529, 0.8344705
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0058889, 1.0061316
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6447036, 0.6423495
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6483307, 0.6469947

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5859

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3741681, upper bound: 0.3708005
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3724338, upper bound: 0.3725348
time: 3.20 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.36 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.36
Output dim: 8, lower bound: -0.3725338, upper bound: 0.3724348
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.36
Output dim: 8, lower bound: -0.3707995, upper bound: 0.3741690
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.36
Output dim: 8, lower bound: -0.3741681, upper bound: 0.3708005
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.36
Output dim: 8, lower bound: -0.3724338, upper bound: 0.3725348

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1965237, 1.1992388
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9007468, 0.8974488
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0845604, 1.0823441
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7842135, 0.7883976
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9443994, 0.9412360
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9505296, 0.9531689
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8273690, 0.8281255
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0048308, 1.0032721
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6411419, 0.6422739
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6455898, 0.6455044

Time for backsubstitution: 22.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5745

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3725333, upper bound: 0.3719475
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3720466, upper bound: 0.3724343
time: 2.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1986122, 1.1971498
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8990722, 0.8991237
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0837564, 1.0831480
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7910299, 0.7815814
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9420137, 0.9436214
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9520779, 0.9516201
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8309429, 0.8245516
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0035148, 1.0045881
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6399198, 0.6434960
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6441681, 0.6469259

Time for backsubstitution: 21.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5745

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3707991, upper bound: 0.3736818
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3703123, upper bound: 0.3741685
time: 3.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1971493, 1.1986127
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8991237, 0.8990722
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0831480, 1.0837564
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7815814, 0.7910297
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9436212, 0.9420137
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9516201, 0.9520779
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8245513, 0.8309431
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0045881, 1.0035148
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6434960, 0.6399198
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6469259, 0.6441681

Time for backsubstitution: 21.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 945

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3727351, upper bound: 0.3706869
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723244, upper bound: 0.3706865
time: 3.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1992388, 1.1965241
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8974490, 0.9007471
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0823441, 1.0845604
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7883978, 0.7842135
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9412360, 0.9443991
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9531689, 0.9505296
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8281252, 0.8273692
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0032721, 1.0048308
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6422739, 0.6411419
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6455045, 0.6455898

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 945

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723198, upper bound: 0.3706910
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723202, upper bound: 0.3711018
time: 3.07 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.30 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.30
Output dim: 8, lower bound: -0.3725333, upper bound: 0.3719475
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.30
Output dim: 8, lower bound: -0.3720466, upper bound: 0.3724343
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.30
Output dim: 8, lower bound: -0.3707991, upper bound: 0.3736818
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.30
Output dim: 8, lower bound: -0.3703123, upper bound: 0.3741685
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.30
Output dim: 8, lower bound: -0.3727351, upper bound: 0.3706869
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.30
Output dim: 8, lower bound: -0.3723244, upper bound: 0.3706865
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.30
Output dim: 8, lower bound: -0.3723198, upper bound: 0.3706910
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.30
Output dim: 8, lower bound: -0.3723202, upper bound: 0.3711018

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1904392, 1.1969185
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8994727, 0.8969634
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0830231, 1.0783181
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7818646, 0.7875037
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9412227, 0.9328952
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9475822, 0.9454379
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8117902, 0.8221931
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0007339, 0.9925385
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6410174, 0.6419470
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6409435, 0.6437337

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 945

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3711004, upper bound: 0.3718339
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3706896, upper bound: 0.3718335
time: 3.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1942034, 1.1931548
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9002614, 0.8961742
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0805345, 1.0808067
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7833200, 0.7860489
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9360585, 0.9380593
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9427986, 0.9502215
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8214371, 0.8125465
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -0.9940972, 0.9991746
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6408148, 0.6421497
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6438193, 0.6408582

Time for backsubstitution: 22.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 945

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3706136, upper bound: 0.3723206
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3702029, upper bound: 0.3723203
time: 2.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1925278, 1.1948295
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8977971, 0.8986380
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0822191, 1.0791221
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7886810, 0.7806878
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9388375, 0.9352806
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9491305, 0.9438896
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8153641, 0.8186193
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -0.9994178, 0.9938545
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6397953, 0.6431689
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6395221, 0.6451553

Time for backsubstitution: 22.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 945

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3706850, upper bound: 0.3718380
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3706854, upper bound: 0.3722488
time: 2.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1962929, 1.1910658
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8985868, 0.8978488
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0797310, 1.0816107
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7901359, 0.7792327
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9336729, 0.9404447
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9443474, 0.9486728
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8250105, 0.8089726
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -0.9927812, 1.0004907
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6395929, 0.6433713
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6423974, 0.6422799

Time for backsubstitution: 22.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 945

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3701983, upper bound: 0.3723248
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3701987, upper bound: 0.3727356
time: 3.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1970291, 1.1991792
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8999047, 0.8989058
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0831165, 1.0839000
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7813687, 0.7920308
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9437046, 0.9419966
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9515705, 0.9523110
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8244278, 0.8315253
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0045958, 1.0035133
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6438687, 0.6398408
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6471643, 0.6441171

Time for backsubstitution: 22.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5745

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3727346, upper bound: 0.3701995
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3722479, upper bound: 0.3706864
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1971493, 1.1984925
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8989573, 0.8990722
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0831480, 1.0837250
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7815814, 0.7908170
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9436040, 0.9420137
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9516201, 0.9520283
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8245513, 0.8308191
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0045862, 1.0035148
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6434171, 0.6399198
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6468749, 0.6441681

Time for backsubstitution: 22.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5745

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723239, upper bound: 0.3701993
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718371, upper bound: 0.3706860
time: 2.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1991186, 1.1970906
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8982253, 0.9005806
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0823126, 1.0843492
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7881846, 0.7852147
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9413190, 0.9443820
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9531188, 0.9507627
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8280017, 0.8279514
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0032797, 1.0048294
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6426432, 0.6410630
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6457429, 0.6455387

Time for backsubstitution: 22.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5745

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723193, upper bound: 0.3702038
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718326, upper bound: 0.3706906
time: 3.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1992388, 1.1964040
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8972826, 0.9007471
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0823441, 1.0845289
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7883978, 0.7840009
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9412189, 0.9443991
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9531689, 0.9504795
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8281252, 0.8272452
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0032701, 1.0048308
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6421950, 0.6411419
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6454535, 0.6455898

Time for backsubstitution: 22.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5745

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723197, upper bound: 0.3706146
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718329, upper bound: 0.3711013
time: 2.89 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.77 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3711004, upper bound: 0.3718339
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3706896, upper bound: 0.3718335
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3706136, upper bound: 0.3723206
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3702029, upper bound: 0.3723203
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3706850, upper bound: 0.3718380
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3706854, upper bound: 0.3722488
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3701983, upper bound: 0.3723248
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3701987, upper bound: 0.3727356
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3727346, upper bound: 0.3701995
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3722479, upper bound: 0.3706864
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3723239, upper bound: 0.3701993
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3718371, upper bound: 0.3706860
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3723193, upper bound: 0.3702038
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3718326, upper bound: 0.3706906
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.77
Output dim: 8, lower bound: -0.3723197, upper bound: 0.3706146
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.77
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

Time for backsubstitution: 22.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2823
type: DSZ, layer: 3, pos: 551
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1115
type: DSZ, layer: 3, pos: 2641
type: DSZ, layer: 3, pos: 1696
type: DSZ, layer: 3, pos: 1201
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 1474
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 2529
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2584
type: DSZ, layer: 3, pos: 164
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1851

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2823

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3675901, upper bound: 0.3685408
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3678073, upper bound: 0.3683236
time: 3.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 22.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1696
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1474
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2823
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1201
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2529
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 164
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 551
type: DSZ, layer: 3, pos: 1115
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2641
type: DSZ, layer: 3, pos: 2584
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 564

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1696

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3670194, upper bound: 0.3684182
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3672843, upper bound: 0.3681533
time: 2.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1940823, 1.1937203
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9010425, 0.8960078
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0805030, 1.0809498
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7831068, 0.7870498
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9361415, 0.9380424
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9427490, 0.9504542
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8213127, 0.8131285
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -0.9941044, 0.9991727
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6411872, 0.6420703
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6440578, 0.6408072

Time for backsubstitution: 22.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1696
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2584
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 2823
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 2529
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1115
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1201
type: DSZ, layer: 3, pos: 164
type: DSZ, layer: 3, pos: 551
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2641
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1474
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 3105

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 403

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3677709, upper bound: 0.3704430
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3687359, upper bound: 0.3694781
time: 3.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.1942034, 1.1930337
1: -13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.9000955, 0.8961742
2: -5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0805345, 1.0807753
3: -8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.7833200, 0.7858357
4: -11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9360409, 0.9380593
5: 0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9427986, 0.9501715
6: -4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8214371, 0.8124225
7: -11.3278637, -9.9057770, -11.3278637, -9.9057770, -0.9940953, 0.9991746
8: 6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6407356, 0.6421497
9: -5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6437683, 0.6408582

Time for backsubstitution: 22.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1696
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 551
type: DSZ, layer: 3, pos: 1474
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 2584
type: DSZ, layer: 3, pos: 1115
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2529
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 164
type: DSZ, layer: 3, pos: 1201
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2641
type: DSZ, layer: 3, pos: 2823
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 556

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1417

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3686777, upper bound: 0.3715897
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3694725, upper bound: 0.3686753
time: 5.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 22.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2529
type: DSZ, layer: 3, pos: 2823
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1474
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 551
type: DSZ, layer: 3, pos: 2584
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 164
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 1696
type: DSZ, layer: 3, pos: 1201
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 2641
type: DSZ, layer: 3, pos: 1115
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2321

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3682000, upper bound: 0.3692582
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3681051, upper bound: 0.3693529
time: 3.29 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 29.15 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.15
Output dim: 8, lower bound: -0.3675901, upper bound: 0.3685408
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.15
Output dim: 8, lower bound: -0.3678073, upper bound: 0.3683236
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.15
Output dim: 8, lower bound: -0.3670194, upper bound: 0.3684182
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.15
Output dim: 8, lower bound: -0.3672843, upper bound: 0.3681533
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 29.15
Output dim: 8, lower bound: -0.3677709, upper bound: 0.3704430
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.15
Output dim: 8, lower bound: -0.3687359, upper bound: 0.3694781
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 29.15
Output dim: 8, lower bound: -0.3686777, upper bound: 0.3715897
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.15
Output dim: 8, lower bound: -0.3694725, upper bound: 0.3686753
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.15
Output dim: 8, lower bound: -0.3682000, upper bound: 0.3692582
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.15
Output dim: 8, lower bound: -0.3681051, upper bound: 0.3693529
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 8, lower bound: -0.3706854, upper bound: 0.3722488
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 8, lower bound: -0.3701983, upper bound: 0.3723248
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 8, lower bound: -0.3701987, upper bound: 0.3727356
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 8, lower bound: -0.3727346, upper bound: 0.3701995
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 8, lower bound: -0.3722479, upper bound: 0.3706864
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 8, lower bound: -0.3723239, upper bound: 0.3701993
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 8, lower bound: -0.3718371, upper bound: 0.3706860
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 8, lower bound: -0.3723193, upper bound: 0.3702038
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 8, lower bound: -0.3718326, upper bound: 0.3706906
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 8, lower bound: -0.3723197, upper bound: 0.3706146
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 8, lower bound: -0.3718329, upper bound: 0.3711013

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.73 + 550.17 = 605.90 seconds
