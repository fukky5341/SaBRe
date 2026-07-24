## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.11490763


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2869463, 0.2869463)
1: (-1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1361805, 0.1361805)
2: (7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3099370, 0.3099370)
3: (-3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1867815, 0.1867815)
4: (-10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1977763, 0.1977763)
5: (-8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3067017, 0.3067017)
6: (-7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2110730, 0.2110729)
7: (-5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.2082946, 0.2082946)
8: (-0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1505284, 0.1505284)
9: (-11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1758047, 0.1758049)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 20.99 + 32.71 = 53.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.1209554, upper bound: 0.1209554

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 2494

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1173858, upper bound: 0.1176223
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1176224, upper bound: 0.1173857
time: 2.84 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.01 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.01
Output dim: 2, lower bound: -0.1173858, upper bound: 0.1176223
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.01
Output dim: 2, lower bound: -0.1176224, upper bound: 0.1173857

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2865620, 0.2866445
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1358911, 0.1358726
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3099518, 0.3099318
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1852646, 0.1861974
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1978083, 0.1936731
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3065047, 0.3067367
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2104808, 0.2108207
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.2072630, 0.2036507
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1483088, 0.1498152
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1757562, 0.1750607

Time for backsubstitution: 7.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1110

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1162519, upper bound: 0.1169357
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1166992, upper bound: 0.1164885
time: 3.07 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2866445, 0.2865620
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1358726, 0.1358911
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3099318, 0.3099513
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1861974, 0.1852646
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1936731, 0.1978083
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3067367, 0.3065050
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2108208, 0.2104810
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.2036505, 0.2072632
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1498152, 0.1483088
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1750610, 0.1757563

Time for backsubstitution: 8.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1110

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1164885, upper bound: 0.1166991
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1169354, upper bound: 0.1162516
time: 3.09 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 14.58 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.58
Output dim: 2, lower bound: -0.1162519, upper bound: 0.1169357
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.58
Output dim: 2, lower bound: -0.1166992, upper bound: 0.1164885
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.58
Output dim: 2, lower bound: -0.1164885, upper bound: 0.1166991
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.58
Output dim: 2, lower bound: -0.1169354, upper bound: 0.1162516

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2756946, 0.2756457
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1380008, 0.1380829
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3173304, 0.3165312
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1819906, 0.1830212
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1893015, 0.1826925
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2993119, 0.3001974
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2019960, 0.2047024
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1995831, 0.1937821
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1473187, 0.1488327
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1784813, 0.1763134

Time for backsubstitution: 7.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 2322

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1157073, upper bound: 0.1166164
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1159328, upper bound: 0.1163913
time: 2.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2755635, 0.2757769
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1381013, 0.1379822
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3165512, 0.3173103
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1820884, 0.1829234
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1868274, 0.1851666
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2999654, 0.2995439
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2043628, 0.2023357
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1973946, 0.1959708
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1473262, 0.1488252
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1770091, 0.1777856

Time for backsubstitution: 8.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 2322

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1161545, upper bound: 0.1161692
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1163800, upper bound: 0.1159441
time: 2.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2757771, 0.2755632
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1379822, 0.1381013
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3173103, 0.3165512
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1829234, 0.1820884
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1851666, 0.1868274
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2995439, 0.2999654
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2023357, 0.2043628
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1959705, 0.1973946
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1488252, 0.1473262
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1777856, 0.1770091

Time for backsubstitution: 7.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 2322

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1159441, upper bound: 0.1163800
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1161694, upper bound: 0.1161547
time: 2.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2756455, 0.2756946
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1380829, 0.1380008
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3165312, 0.3173304
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1830212, 0.1819906
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1826925, 0.1893015
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3001974, 0.2993119
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2047025, 0.2019960
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1937821, 0.1995833
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1488327, 0.1473187
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1763134, 0.1784812

Time for backsubstitution: 8.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2322

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1163913, upper bound: 0.1159328
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1166166, upper bound: 0.1157075
time: 2.82 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 14.11 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.11
Output dim: 2, lower bound: -0.1157073, upper bound: 0.1166164
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.11
Output dim: 2, lower bound: -0.1159328, upper bound: 0.1163913
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.11
Output dim: 2, lower bound: -0.1161545, upper bound: 0.1161692
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.11
Output dim: 2, lower bound: -0.1163800, upper bound: 0.1159441
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.11
Output dim: 2, lower bound: -0.1159441, upper bound: 0.1163800
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.11
Output dim: 2, lower bound: -0.1161694, upper bound: 0.1161547
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.11
Output dim: 2, lower bound: -0.1163913, upper bound: 0.1159328
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.11
Output dim: 2, lower bound: -0.1166166, upper bound: 0.1157075

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2756462, 0.2751577
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1380004, 0.1383144
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3173470, 0.3165231
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1817052, 0.1828220
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1893001, 0.1830945
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2991421, 0.2998998
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2019945, 0.2045982
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1992526, 0.1937301
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1471113, 0.1484816
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1787989, 0.1763115

Time for backsubstitution: 8.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1132630, upper bound: 0.1148364
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1132988, upper bound: 0.1139465
time: 2.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2752066, 0.2755976
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1382322, 0.1380824
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3173223, 0.3165483
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1817914, 0.1827358
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1897037, 0.1826911
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2990143, 0.3000276
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2018918, 0.2047009
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1995311, 0.1934516
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1469678, 0.1486253
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1784792, 0.1766312

Time for backsubstitution: 7.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1132627, upper bound: 0.1139825
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1141526, upper bound: 0.1139467
time: 2.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2755151, 0.2752888
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1381010, 0.1382138
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3165684, 0.3173022
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1818030, 0.1827244
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1868260, 0.1855688
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2997956, 0.2992463
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2043613, 0.2022314
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1970642, 0.1959186
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1471188, 0.1484741
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1773267, 0.1777837

Time for backsubstitution: 8.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1137102, upper bound: 0.1143892
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1137460, upper bound: 0.1134993
time: 2.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2750750, 0.2757287
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1383328, 0.1379819
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3165431, 0.3173275
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1818892, 0.1826380
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1872294, 0.1851652
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2996678, 0.2993741
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2042586, 0.2023342
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1973426, 0.1956401
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1469752, 0.1486177
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1770070, 0.1781034

Time for backsubstitution: 8.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1137099, upper bound: 0.1135353
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1145998, upper bound: 0.1134995
time: 3.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2757287, 0.2750752
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1379819, 0.1383328
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3173270, 0.3165431
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1826380, 0.1818892
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1851652, 0.1872294
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2993741, 0.2996678
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2023340, 0.2042586
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1956401, 0.1973426
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1486177, 0.1469752
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1781034, 0.1770070

Time for backsubstitution: 8.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 214

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1134995, upper bound: 0.1145998
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1135353, upper bound: 0.1137099
time: 2.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2752886, 0.2755151
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1382138, 0.1381010
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3173022, 0.3165684
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1827244, 0.1818030
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1855688, 0.1868260
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2992463, 0.2997956
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2022315, 0.2043612
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1959186, 0.1970642
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1484741, 0.1471188
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1777837, 0.1773267

Time for backsubstitution: 7.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 214

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1134993, upper bound: 0.1137459
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1143892, upper bound: 0.1137101
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2755976, 0.2752066
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1380824, 0.1382322
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3165483, 0.3173223
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1827358, 0.1817914
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1826911, 0.1897037
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3000276, 0.2990143
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2047011, 0.2018918
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1934516, 0.1995311
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1486253, 0.1469678
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1766312, 0.1784792

Time for backsubstitution: 8.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 214

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1139467, upper bound: 0.1141526
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1139825, upper bound: 0.1132627
time: 2.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2751579, 0.2756464
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1383144, 0.1380004
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3165231, 0.3173470
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1828220, 0.1817052
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1830945, 0.1893001
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2998998, 0.2991421
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2045983, 0.2019945
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1937301, 0.1992526
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1484816, 0.1471113
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1763115, 0.1787989

Time for backsubstitution: 7.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 214

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1139465, upper bound: 0.1132987
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1148364, upper bound: 0.1132629
time: 2.86 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 13.71 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1132630, upper bound: 0.1148364
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1132988, upper bound: 0.1139465
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1132627, upper bound: 0.1139825
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1141526, upper bound: 0.1139467
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1137102, upper bound: 0.1143892
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1137460, upper bound: 0.1134993
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1137099, upper bound: 0.1135353
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1145998, upper bound: 0.1134995
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1134995, upper bound: 0.1145998
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1135353, upper bound: 0.1137099
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1134993, upper bound: 0.1137459
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1143892, upper bound: 0.1137101
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1139467, upper bound: 0.1141526
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1139825, upper bound: 0.1132627
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1139465, upper bound: 0.1132987
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 13.71
Output dim: 2, lower bound: -0.1148364, upper bound: 0.1132629

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 53.70 + 200.91 = 254.61 seconds
