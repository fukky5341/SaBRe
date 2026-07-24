## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.761135001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3250217, 1.3250217)
1: (-4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9551978, 0.9551978)
2: (-5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.2082760, 1.2082758)
3: (5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1771312, 1.1771309)
4: (-14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1844063, 1.1844063)
5: (-7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2325692, 1.2325692)
6: (-11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1429083, 1.1429081)
7: (-6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9562483, 0.9562483)
8: (-4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1519935, 1.1519938)
9: (-5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9297078, 0.9297078)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.05 + 34.19 = 58.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.7649598, upper bound: 0.7649598

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5817
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 943

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5817

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649591, upper bound: 0.7649602
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649589, upper bound: 0.7649600
time: 3.58 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.40 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.40
Output dim: 3, lower bound: -0.7649591, upper bound: 0.7649602
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.40
Output dim: 3, lower bound: -0.7649589, upper bound: 0.7649600

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3245473, 1.3262224
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9597919, 0.9533896
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.2097063, 1.2077155
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1778650, 1.1768434
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1835604, 1.1865501
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2316775, 1.2348332
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1448545, 1.1421382
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9565415, 0.9561322
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1578312, 1.1496935
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9286792, 0.9323132

Time for backsubstitution: 22.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7623928, upper bound: 0.7649601
time: 3.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649590, upper bound: 0.7623939
time: 3.92 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3250217, 1.3245478
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9533894, 0.9551978
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.2077155, 1.2082758
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1768436, 1.1771309
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1844063, 1.1835604
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2325692, 1.2316775
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1421385, 1.1429081
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9561324, 0.9562483
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1496935, 1.1519938
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9297078, 0.9286792

Time for backsubstitution: 22.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 943

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5857

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7611690, upper bound: 0.7649597
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649592, upper bound: 0.7611692
time: 4.61 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.55 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.55
Output dim: 3, lower bound: -0.7623928, upper bound: 0.7649601
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.55
Output dim: 3, lower bound: -0.7649590, upper bound: 0.7623939
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.55
Output dim: 3, lower bound: -0.7611690, upper bound: 0.7649597
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.55
Output dim: 3, lower bound: -0.7649592, upper bound: 0.7611692

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3245473, 1.3262219
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9597752, 0.9533763
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.2097018, 1.2077122
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1778679, 1.1768467
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1835580, 1.1865463
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2316718, 1.2348261
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1448531, 1.1421363
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9565346, 0.9561229
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1578345, 1.1496956
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9286771, 0.9323111

Time for backsubstitution: 23.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4665

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7623812, upper bound: 0.7644019
time: 4.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7618336, upper bound: 0.7649471
time: 4.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3245473, 1.3262224
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9597785, 0.9533730
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.2097027, 1.2077112
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1778688, 1.1768463
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1835570, 1.1865473
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2316699, 1.2348275
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1448526, 1.1421368
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9565322, 0.9561250
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1578336, 1.1496966
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9286771, 0.9323113

Time for backsubstitution: 23.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 536

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7510621, upper bound: 0.7623815
time: 3.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649479, upper bound: 0.7484910
time: 4.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3134851, 1.3091660
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9343479, 0.9409149
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1895466, 1.1840496
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1642265, 1.1676700
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1828852, 1.1824203
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2273397, 1.2277565
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1377482, 1.1396158
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9337254, 0.9394450
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1407375, 1.1455641
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9228330, 0.9195116

Time for backsubstitution: 22.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 943

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7605684, upper bound: 0.7649558
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7605682, upper bound: 0.7605802
time: 4.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3096395, 1.3130112
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9391067, 0.9361563
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1834888, 1.1901073
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1673827, 1.1645136
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1832657, 1.1820402
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2286477, 1.2264481
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1388459, 1.1385181
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9393291, 0.9338415
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1432648, 1.1430371
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9205403, 0.9218040

Time for backsubstitution: 22.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7623929, upper bound: 0.7611704
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649591, upper bound: 0.7585956
time: 3.77 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.04 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.04
Output dim: 3, lower bound: -0.7623812, upper bound: 0.7644019
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.04
Output dim: 3, lower bound: -0.7618336, upper bound: 0.7649471
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.04
Output dim: 3, lower bound: -0.7510621, upper bound: 0.7623815
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.04
Output dim: 3, lower bound: -0.7649479, upper bound: 0.7484910
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.04
Output dim: 3, lower bound: -0.7605684, upper bound: 0.7649558
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 30.04
Output dim: 3, lower bound: -0.7605682, upper bound: 0.7605802
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.04
Output dim: 3, lower bound: -0.7623929, upper bound: 0.7611704
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.04
Output dim: 3, lower bound: -0.7649591, upper bound: 0.7585956

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3239980, 1.3306990
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9599030, 0.9533601
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.2134352, 1.2072556
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1820703, 1.1763303
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1866589, 1.1861668
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2316036, 1.2353764
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1444149, 1.1457088
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9597981, 0.9557240
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1577334, 1.1505258
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9320416, 0.9318995

Time for backsubstitution: 22.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 943

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 442

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7612421, upper bound: 0.7643853
time: 5.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7623663, upper bound: 0.7632659
time: 4.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3245473, 1.3256731
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9597590, 0.9533763
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.2092452, 1.2077122
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1773515, 1.1768467
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1831784, 1.1865463
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2316718, 1.2347579
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1448531, 1.1416979
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9561360, 0.9561229
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1578345, 1.1495948
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9282660, 0.9323111

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 442

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7606945, upper bound: 0.7649322
time: 5.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7618188, upper bound: 0.7638113
time: 7.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.2585731, 1.2382927
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9369457, 0.9370911
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1887858, 1.1795261
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1325712, 1.1428494
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1599698, 1.1726952
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.1607857, 1.1816792
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.0809655, 1.0569980
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9531617, 0.9548364
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.0591002, 1.0174615
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9034176, 0.9141986

Time for backsubstitution: 22.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 4665

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7502056, upper bound: 0.7618247
time: 6.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7502056, upper bound: 0.7482914
time: 6.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.2366176, 1.2602482
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9434965, 0.9305401
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1815178, 1.1867938
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1438718, 1.1315489
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1697049, 1.1629601
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.1785216, 1.1639433
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.0597138, 1.0782497
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9552436, 0.9527545
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.0255980, 1.0509634
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9105635, 0.9070525

Time for backsubstitution: 22.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 63

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649479, upper bound: 0.7474280
time: 4.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7638443, upper bound: 0.7484920
time: 3.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3097997, 1.3037181
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9306777, 0.9391468
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1837676, 1.1763067
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1619346, 1.1670287
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1844244, 1.1834989
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2245131, 1.2256346
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1376202, 1.1395197
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9329491, 0.9399455
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1407368, 1.1457627
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9205790, 0.9165077

Time for backsubstitution: 22.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4665

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7605570, upper bound: 0.7643962
time: 4.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7600103, upper bound: 0.7649429
time: 4.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3096390, 1.3130112
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9390905, 0.9361434
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1834850, 1.1901045
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1673844, 1.1645162
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1832623, 1.1820364
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2286434, 1.2264423
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1388445, 1.1385157
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9393229, 0.9338336
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1432691, 1.1430407
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9205382, 0.9218018

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 536

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7484910, upper bound: 0.7611582
time: 4.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7623818, upper bound: 0.7472702
time: 3.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3096385, 1.3130116
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9390938, 0.9361401
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1834860, 1.1901035
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1673849, 1.1645157
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1832614, 1.1820374
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2286420, 1.2264438
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1388435, 1.1385162
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9393210, 0.9338357
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1432681, 1.1430416
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9205382, 0.9218023

Time for backsubstitution: 21.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 943

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7556606, upper bound: 0.7580426
time: 4.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7643978, upper bound: 0.7493017
time: 4.52 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.45
Output dim: 3, lower bound: -0.7612421, upper bound: 0.7643853
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.45
Output dim: 3, lower bound: -0.7623663, upper bound: 0.7632659
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.45
Output dim: 3, lower bound: -0.7606945, upper bound: 0.7649322
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.45
Output dim: 3, lower bound: -0.7618188, upper bound: 0.7638113
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.45
Output dim: 3, lower bound: -0.7502056, upper bound: 0.7618247
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.45
Output dim: 3, lower bound: -0.7502056, upper bound: 0.7482914
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.45
Output dim: 3, lower bound: -0.7649479, upper bound: 0.7474280
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.45
Output dim: 3, lower bound: -0.7638443, upper bound: 0.7484920
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.45
Output dim: 3, lower bound: -0.7605570, upper bound: 0.7643962
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.45
Output dim: 3, lower bound: -0.7600103, upper bound: 0.7649429
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.45
Output dim: 3, lower bound: -0.7484910, upper bound: 0.7611582
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.45
Output dim: 3, lower bound: -0.7623818, upper bound: 0.7472702
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.45
Output dim: 3, lower bound: -0.7556606, upper bound: 0.7580426
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.45
Output dim: 3, lower bound: -0.7643978, upper bound: 0.7493017

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3226657, 1.3368940
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9610107, 0.9531207
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.2132716, 1.2080147
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1808619, 1.1819546
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1889973, 1.1856627
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2427626, 1.2329788
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1447480, 1.1456370
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9591792, 0.9586008
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1648898, 1.1489844
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9314508, 0.9346416

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 63

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7612421, upper bound: 0.7632804
time: 4.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7601206, upper bound: 0.7643852
time: 4.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3239980, 1.3293667
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9596641, 0.9533601
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.2134352, 1.2070920
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1820703, 1.1751223
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1861548, 1.1861668
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2292061, 1.2353764
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1443431, 1.1457088
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9597981, 0.9551055
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1561918, 1.1505258
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9320416, 0.9313092

Time for backsubstitution: 21.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7530659, upper bound: 0.7626958
time: 4.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7618079, upper bound: 0.7539586
time: 4.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3232145, 1.3318682
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9608667, 0.9531374
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.2090816, 1.2084713
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1761432, 1.1824710
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1855164, 1.1860423
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2428308, 1.2323604
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1451862, 1.1416259
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9555171, 0.9589996
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1649909, 1.1480534
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9276752, 0.9350531

Time for backsubstitution: 21.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5857

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7568960, upper bound: 0.7649323
time: 4.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7606948, upper bound: 0.7611424
time: 4.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3245473, 1.3243408
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9595196, 0.9533763
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.2092452, 1.2075486
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1773515, 1.1756384
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1826739, 1.1865463
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2292738, 1.2347579
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1447818, 1.1416979
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9561360, 0.9555044
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1562924, 1.1495948
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9282660, 0.9317205

Time for backsubstitution: 21.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 943

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7525212, upper bound: 0.7632441
time: 4.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7612596, upper bound: 0.7545035
time: 4.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.2582803, 1.2376823
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9369433, 0.9371829
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1887786, 1.1797166
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1308885, 1.1420445
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1597962, 1.1723557
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.1607714, 1.1822674
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.0805781, 1.0561953
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9531608, 0.9549174
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.0601304, 1.0172231
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9034033, 0.9146283

Time for backsubstitution: 21.37 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.24 + 560.69 = 618.93 seconds
