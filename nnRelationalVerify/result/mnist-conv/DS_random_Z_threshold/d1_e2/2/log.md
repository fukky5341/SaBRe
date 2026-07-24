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
execution time: IAR + RelationalAnalysis = 22.15 + 34.01 = 56.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.1209554, upper bound: 0.1209554

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 1262

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1110

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1198393, upper bound: 0.1203006
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1203006, upper bound: 0.1198393
time: 3.14 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.24 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.24
Output dim: 2, lower bound: -0.1198393, upper bound: 0.1203006
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.24
Output dim: 2, lower bound: -0.1203006, upper bound: 0.1198393

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2760789, 0.2759476
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1382904, 0.1383909
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3173146, 0.3165359
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1835076, 0.1836053
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1892700, 0.1867959
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2995095, 0.3001630
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2025880, 0.2049546
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.2006149, 0.1984262
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1495382, 0.1495457
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1785295, 0.1770574

Time for backsubstitution: 8.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2466

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1189668, upper bound: 0.1195091
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1190620, upper bound: 0.1194140
time: 3.12 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2759478, 0.2760789
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1383909, 0.1382904
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3165359, 0.3173146
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1836053, 0.1835076
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1867959, 0.1892700
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3001630, 0.2995098
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2049547, 0.2025878
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1984262, 0.2006149
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1495457, 0.1495382
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1770573, 0.1785296

Time for backsubstitution: 8.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 60

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1198217, upper bound: 0.1192345
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1197086, upper bound: 0.1193475
time: 2.86 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 14.31 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.31
Output dim: 2, lower bound: -0.1189668, upper bound: 0.1195091
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.31
Output dim: 2, lower bound: -0.1190620, upper bound: 0.1194140
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.31
Output dim: 2, lower bound: -0.1198217, upper bound: 0.1192345
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.31
Output dim: 2, lower bound: -0.1197086, upper bound: 0.1193475

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2754071, 0.2755785
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1381528, 0.1381993
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3173199, 0.3164692
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1819384, 0.1824039
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1865296, 0.1850104
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2994382, 0.3000031
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2022022, 0.2047687
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1997492, 0.1981232
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1495432, 0.1495173
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1780763, 0.1762071

Time for backsubstitution: 8.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 2322

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 410

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1183415, upper bound: 0.1192212
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1186789, upper bound: 0.1188838
time: 2.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2760789, 0.2752755
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1382904, 0.1382533
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3172483, 0.3165359
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1835076, 0.1820362
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1874845, 0.1867959
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2993495, 0.3001630
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2024018, 0.2049546
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.2006149, 0.1975608
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1495098, 0.1495457
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1785295, 0.1766042

Time for backsubstitution: 8.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1262

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1151176, upper bound: 0.1154813
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1151293, upper bound: 0.1154697
time: 2.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2761776, 0.2763710
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1386884, 0.1384524
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3131337, 0.3137898
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1830810, 0.1830274
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1868989, 0.1893554
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3010700, 0.2994957
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2049804, 0.2025871
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1940737, 0.1967676
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1495328, 0.1495186
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1770182, 0.1784953

Time for backsubstitution: 8.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 2322

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2466

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1189658, upper bound: 0.1184955
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1190608, upper bound: 0.1184002
time: 2.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2762396, 0.2763088
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1385530, 0.1385880
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3130107, 0.3139129
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1831250, 0.1829833
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1868813, 0.1893730
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3001492, 0.3004165
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2049539, 0.2026137
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1945791, 0.1962621
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1495261, 0.1495253
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1770232, 0.1784904

Time for backsubstitution: 8.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 214

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1170004, upper bound: 0.1175093
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1178169, upper bound: 0.1167434
time: 3.25 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 15.22 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 2, lower bound: -0.1183415, upper bound: 0.1192212
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 2, lower bound: -0.1186789, upper bound: 0.1188838
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 2, lower bound: -0.1151176, upper bound: 0.1154813
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 2, lower bound: -0.1151293, upper bound: 0.1154697
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 2, lower bound: -0.1189658, upper bound: 0.1184955
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 2, lower bound: -0.1190608, upper bound: 0.1184002
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 2, lower bound: -0.1170004, upper bound: 0.1175093
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 2, lower bound: -0.1178169, upper bound: 0.1167434

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2581015, 0.2582560
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1371179, 0.1371107
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3168926, 0.3160815
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1797138, 0.1803861
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1869330, 0.1846762
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2841516, 0.2844293
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2016408, 0.2041692
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1973898, 0.1958394
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1482679, 0.1466613
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1760173, 0.1747928

Time for backsubstitution: 8.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2322

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1178027, upper bound: 0.1189026
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1180280, upper bound: 0.1186772
time: 3.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2580841, 0.2582734
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1370641, 0.1371645
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3169317, 0.3160424
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1799207, 0.1801792
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1861954, 0.1854138
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2838645, 0.2847161
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2016027, 0.2042072
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1974657, 0.1957636
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1466870, 0.1482421
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1766620, 0.1741482

Time for backsubstitution: 7.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 2494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 753

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1142861, upper bound: 0.1182359
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1180247, upper bound: 0.1147195
time: 3.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2755816, 0.2777112
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1383336, 0.1382065
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3171120, 0.3168540
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1844031, 0.1785785
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1870162, 0.1869180
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2993109, 0.2991395
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2017874, 0.2052561
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.2007456, 0.1973927
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1492161, 0.1513853
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1802282, 0.1757721

Time for backsubstitution: 8.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2928

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1149284, upper bound: 0.1154541
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1150920, upper bound: 0.1153985
time: 2.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2760789, 0.2747781
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1382436, 0.1382533
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3172483, 0.3163996
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1800498, 0.1820362
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1874845, 0.1863275
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2983255, 0.3001630
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2024018, 0.2043401
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.2004466, 0.1975608
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1495098, 0.1492519
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1776977, 0.1766042

Time for backsubstitution: 8.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2322

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1145847, upper bound: 0.1151504
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1148102, upper bound: 0.1149252
time: 2.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2755058, 0.2760022
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1385509, 0.1382608
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3131390, 0.3137231
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1815118, 0.1818259
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1841581, 0.1875699
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3009984, 0.2993355
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2045949, 0.2024010
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1932087, 0.1964650
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1495376, 0.1494901
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1765654, 0.1776456

Time for backsubstitution: 8.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1262

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1150215, upper bound: 0.1145629
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1150332, upper bound: 0.1145512
time: 3.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2761776, 0.2756991
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1386884, 0.1383148
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3130674, 0.3137898
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1830810, 0.1814581
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1851132, 0.1893554
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3009098, 0.2994957
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2047944, 0.2025871
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1940737, 0.1959026
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1495043, 0.1495186
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1770182, 0.1780427

Time for backsubstitution: 9.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 186

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1189441, upper bound: 0.1182077
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1189378, upper bound: 0.1169186
time: 3.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2727871, 0.2696414
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1372392, 0.1373109
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.2987080, 0.3006268
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1803049, 0.1805300
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1887870, 0.1919487
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2956555, 0.2963398
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.1811033, 0.1801796
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1884775, 0.1905918
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1487899, 0.1486837
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1703827, 0.1725483

Time for backsubstitution: 8.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1262

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1130580, upper bound: 0.1135980
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1130697, upper bound: 0.1135863
time: 2.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2695723, 0.2728558
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1372759, 0.1372741
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.2998843, 0.2996101
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1806717, 0.1801583
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1894569, 0.1912785
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2960725, 0.2953899
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.1825198, 0.1787632
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1886373, 0.1901608
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1486845, 0.1487890
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1710701, 0.1718500

Time for backsubstitution: 8.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 2466

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 186

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1177052, upper bound: 0.1165684
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1176987, upper bound: 0.1152792
time: 2.94 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 14.64 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1178027, upper bound: 0.1189026
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1180280, upper bound: 0.1186772
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1142861, upper bound: 0.1182359
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1180247, upper bound: 0.1147195
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1149284, upper bound: 0.1154541
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1150920, upper bound: 0.1153985
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1145847, upper bound: 0.1151504
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1148102, upper bound: 0.1149252
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1150215, upper bound: 0.1145629
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1150332, upper bound: 0.1145512
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1189441, upper bound: 0.1182077
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1189378, upper bound: 0.1169186
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1130580, upper bound: 0.1135980
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1130697, upper bound: 0.1135863
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1177052, upper bound: 0.1165684
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 2, lower bound: -0.1176987, upper bound: 0.1152792

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2580540, 0.2577686
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1371173, 0.1373419
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3169093, 0.3160729
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1794283, 0.1801869
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1869314, 0.1850781
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2839828, 0.2841327
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2016392, 0.2040646
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1970596, 0.1957874
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1480608, 0.1463106
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1763353, 0.1747910

Time for backsubstitution: 8.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2928

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1177133, upper bound: 0.1188769
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1177770, upper bound: 0.1188131
time: 2.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2576141, 0.2582083
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1373491, 0.1371100
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3168840, 0.3160977
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1795146, 0.1801007
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1873348, 0.1846747
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2838550, 0.2842607
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2015364, 0.2041674
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1973381, 0.1955090
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1479173, 0.1464542
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1760159, 0.1751107

Time for backsubstitution: 8.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 158

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1108003, upper bound: 0.1106469
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1108345, upper bound: 0.1106275
time: 2.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2573335, 0.2564080
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1181380, 0.1210372
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3188448, 0.3185658
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1797324, 0.1800129
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1687715, 0.1648908
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2836797, 0.2834842
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.1849736, 0.1855687
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1961296, 0.1951721
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1373047, 0.1373477
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1617999, 0.1619172

Time for backsubstitution: 8.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2494

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1106503, upper bound: 0.1148366
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1108869, upper bound: 0.1146000
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2562189, 0.2576056
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1209369, 0.1183102
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3194551, 0.3179550
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1797544, 0.1799994
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1656724, 0.1679896
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2826326, 0.2845314
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.1829642, 0.1875781
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1968739, 0.1944826
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1359007, 0.1388596
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1644306, 0.1592861

Time for backsubstitution: 8.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 186

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1165432, upper bound: 0.1145964
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1178323, upper bound: 0.1146029
time: 2.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2780633, 0.2808876
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1386155, 0.1350738
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3223877, 0.3233240
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1871567, 0.1817812
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1887772, 0.1886935
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3022549, 0.3020346
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2038186, 0.2074519
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.2030468, 0.1998112
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1468309, 0.1492751
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1844459, 0.1790419

Time for backsubstitution: 8.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 60

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1144801, upper bound: 0.1148877
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1143619, upper bound: 0.1150059
time: 3.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2787585, 0.2801924
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1352006, 0.1384884
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3232384, 0.3221309
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1874189, 0.1813318
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1887918, 0.1886528
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3022058, 0.3020835
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2039833, 0.2072871
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.2031641, 0.1996939
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1471053, 0.1490006
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1834974, 0.1795001

Time for backsubstitution: 7.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2322

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1145476, upper bound: 0.1150794
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1147729, upper bound: 0.1148540
time: 2.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2760305, 0.2742901
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1382431, 0.1384848
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3172650, 0.3163915
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1797643, 0.1818370
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1874833, 0.1867294
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2981558, 0.2998655
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2024003, 0.2042357
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.2001164, 0.1975088
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1493027, 0.1489010
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1780152, 0.1766021

Time for backsubstitution: 8.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 410

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1139652, upper bound: 0.1148631
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1142974, upper bound: 0.1145308
time: 3.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2755909, 0.2747300
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1384749, 0.1382530
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3172398, 0.3164167
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1798506, 0.1817508
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1878867, 0.1863260
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2980280, 0.2999935
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2022976, 0.2043384
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.2003946, 0.1972306
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1491592, 0.1490446
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1776953, 0.1769218

Time for backsubstitution: 8.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 753

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1088521, upper bound: 0.1143331
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1142190, upper bound: 0.1105793
time: 2.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2750080, 0.2784371
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1385944, 0.1382143
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3130026, 0.3140416
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1824074, 0.1783682
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1836898, 0.1876924
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3009598, 0.2983115
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2039800, 0.2027022
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1933393, 0.1962967
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1492435, 0.1513293
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1782639, 0.1768134

Time for backsubstitution: 7.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 214

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1123428, upper bound: 0.1126162
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1131125, upper bound: 0.1117967
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2755058, 0.2755044
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1385044, 0.1382608
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3131390, 0.3135872
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1780541, 0.1818259
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1841581, 0.1871016
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2999747, 0.2993355
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2045949, 0.2017862
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1930404, 0.1964650
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1495376, 0.1491959
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1757333, 0.1776456

Time for backsubstitution: 8.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 410

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 214

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1123545, upper bound: 0.1126045
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1131242, upper bound: 0.1117850
time: 2.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2693059, 0.2674675
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1282964, 0.1286318
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3128166, 0.3135996
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1810644, 0.1789298
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1772408, 0.1841320
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2998593, 0.2978294
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.1962934, 0.1925020
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1858702, 0.1887081
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1452036, 0.1452487
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1760170, 0.1771753

Time for backsubstitution: 8.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 410

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 753

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1145222, upper bound: 0.1175369
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1182810, upper bound: 0.1136224
time: 3.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2679470, 0.2691400
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1289105, 0.1279229
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3128777, 0.3134952
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1805527, 0.1794415
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1801291, 0.1814831
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2992442, 0.2983055
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.1947093, 0.1941818
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1869872, 0.1876991
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1452342, 0.1452180
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1761508, 0.1770415

Time for backsubstitution: 8.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1262

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 158

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1114964, upper bound: 0.1056158
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1115305, upper bound: 0.1054518
time: 2.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2627020, 0.2646258
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1268833, 0.1275905
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.2996321, 0.2994189
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1786544, 0.1776295
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1815839, 0.1860542
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2950218, 0.2937241
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.1740190, 0.1686784
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1804338, 0.1829662
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1443840, 0.1445191
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1700687, 0.1709825

Time for backsubstitution: 8.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2466

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 410

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1171882, upper bound: 0.1162670
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1174006, upper bound: 0.1158601
time: 3.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2613425, 0.2662981
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1274974, 0.1268815
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.2996931, 0.2993145
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1781429, 0.1781411
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1844721, 0.1834054
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2944067, 0.2942004
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.1724350, 0.1703583
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1815505, 0.1819572
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1444147, 0.1444885
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1702027, 0.1708487

Time for backsubstitution: 8.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2928

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1176046, upper bound: 0.1152534
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1176731, upper bound: 0.1151898
time: 3.93 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 15.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1177133, upper bound: 0.1188769
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1177770, upper bound: 0.1188131
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1108003, upper bound: 0.1106469
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1108345, upper bound: 0.1106275
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1106503, upper bound: 0.1148366
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1108869, upper bound: 0.1146000
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1165432, upper bound: 0.1145964
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1178323, upper bound: 0.1146029
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1144801, upper bound: 0.1148877
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1143619, upper bound: 0.1150059
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1145476, upper bound: 0.1150794
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1147729, upper bound: 0.1148540
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1139652, upper bound: 0.1148631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1142974, upper bound: 0.1145308
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1088521, upper bound: 0.1143331
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1142190, upper bound: 0.1105793
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1123428, upper bound: 0.1126162
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1131125, upper bound: 0.1117967
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1123545, upper bound: 0.1126045
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1131242, upper bound: 0.1117850
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1145222, upper bound: 0.1175369
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1182810, upper bound: 0.1136224
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1114964, upper bound: 0.1056158
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1115305, upper bound: 0.1054518
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1171882, upper bound: 0.1162670
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1174006, upper bound: 0.1158601
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1176046, upper bound: 0.1152534
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.53
Output dim: 2, lower bound: -0.1176731, upper bound: 0.1151898

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2604978, 0.2609074
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1373682, 0.1341782
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3221755, 0.3221893
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1822520, 0.1834600
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1887279, 0.1868889
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2869508, 0.2870517
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2033700, 0.2059604
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1992993, 0.1981442
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1456758, 0.1441998
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1795716, 0.1775687

Time for backsubstitution: 8.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 186

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1162315, upper bound: 0.1187536
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1175206, upper bound: 0.1187601
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2611928, 0.2602122
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1339533, 0.1375930
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3230262, 0.3213387
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1827015, 0.1830106
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1887422, 0.1868746
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2869017, 0.2871008
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2035347, 0.2057955
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1994166, 0.1980274
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1459502, 0.1439254
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1791131, 0.1780272

Time for backsubstitution: 7.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2494

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1141385, upper bound: 0.1154703
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1142504, upper bound: 0.1149557
time: 2.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2496605, 0.2493749
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1105444, 0.1085320
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3191614, 0.3177652
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1777375, 0.1774708
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1577992, 0.1630045
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2814429, 0.2828655
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.1745591, 0.1774931
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1886702, 0.1873958
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1316003, 0.1345899
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1634300, 0.1584189

Time for backsubstitution: 8.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2322

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1160611, upper bound: 0.1127138
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1162872, upper bound: 0.1141007
time: 2.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2479882, 0.2507341
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1112535, 0.1079178
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3192654, 0.3177042
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1772259, 0.1779824
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1604481, 0.1601164
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2809668, 0.2834806
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.1728792, 0.1790770
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1896794, 0.1862788
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1316310, 0.1345593
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1635635, 0.1582851

Time for backsubstitution: 8.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2322

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1173503, upper bound: 0.1127203
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1175764, upper bound: 0.1141072
time: 2.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2783549, 0.2811174
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1387779, 0.1353714
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3188629, 0.3199220
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1866765, 0.1812572
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1888628, 0.1887970
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3022416, 0.3029411
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2038174, 0.2074777
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1992004, 0.1954594
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1468108, 0.1492612
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1844122, 0.1790035

Time for backsubstitution: 8.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 753

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1097767, upper bound: 0.1143426
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1136910, upper bound: 0.1105837
time: 3.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2787104, 0.2797046
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1352003, 0.1387199
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3232565, 0.3221233
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1871334, 0.1811326
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1887906, 0.1890552
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3020368, 0.3017867
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2039816, 0.2071826
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.2028341, 0.1996422
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1468977, 0.1486493
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1838152, 0.1794982

Time for backsubstitution: 8.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 2494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 60

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1141463, upper bound: 0.1145453
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1139471, upper bound: 0.1146697
time: 3.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2702820, 0.2672465
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1112624, 0.1143311
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3146591, 0.3160613
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1808976, 0.1787848
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1637548, 0.1675993
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2969966, 0.2939215
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.1798059, 0.1740052
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1858230, 0.1894047
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1376216, 0.1362627
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1632333, 0.1670115

Time for backsubstitution: 9.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 214
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 158

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1072620, upper bound: 0.1065342
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1072799, upper bound: 0.1063844
time: 3.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2690847, 0.2684441
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1139958, 0.1116041
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3152781, 0.3154504
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1809196, 0.1787629
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1607085, 0.1706982
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2959495, 0.2949686
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.1777965, 0.1760145
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1865671, 0.1886604
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1362212, 0.1376666
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1658642, 0.1643915

Time for backsubstitution: 8.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2322

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1178514, upper bound: 0.1120557
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1180690, upper bound: 0.1131684
time: 3.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2465250, 0.2482724
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1260900, 0.1267433
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.2992048, 0.2990308
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1757826, 0.1749653
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1807802, 0.1845131
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2789710, 0.2773881
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.1743439, 0.1689653
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1778953, 0.1803861
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1431277, 0.1416770
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1687899, 0.1703475

Time for backsubstitution: 9.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2322
type: DSZ, layer: 3, pos: 2466
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 753
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 2494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2322

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1161482, upper bound: 0.1160036
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1170440, upper bound: 0.1160039
time: 3.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2465074, 0.2484491
1: -1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1260362, 0.1267972
2: 7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.2992435, 0.2989912
3: -3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1759902, 0.1745993
4: -10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1800427, 0.1852505
5: -8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.2786841, 0.2776732
6: -7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.1743058, 0.1690032
7: -5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.1778536, 0.1803105
8: -0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1415420, 0.1432579
9: -11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1694338, 0.1697028

Time for backsubstitution: 8.68 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.16 + 547.62 = 603.78 seconds
