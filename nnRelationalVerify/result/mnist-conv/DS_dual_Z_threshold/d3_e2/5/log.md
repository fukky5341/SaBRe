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
execution time: IAR + RelationalAnalysis = 23.67 + 34.67 = 58.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.7649598, upper bound: 0.7649598

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 5817
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 5857

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7611699, upper bound: 0.7649604
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649601, upper bound: 0.7611712
time: 3.89 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.38 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.38
Output dim: 3, lower bound: -0.7611699, upper bound: 0.7649604
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.38
Output dim: 3, lower bound: -0.7649601, upper bound: 0.7611712

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3134851, 1.3096395
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9361563, 0.9409149
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1901071, 1.1840496
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1645136, 1.1676700
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1828852, 1.1832657
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2273397, 1.2286477
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1385183, 1.1396158
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9338412, 0.9394450
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1430376, 1.1455641
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9228330, 0.9205403

Time for backsubstitution: 21.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5817
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 5817

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7611692, upper bound: 0.7649593
time: 5.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7611690, upper bound: 0.7649597
time: 4.03 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3096395, 1.3134851
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9409146, 0.9361563
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1840498, 1.1901073
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1676698, 1.1645136
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1832657, 1.1828852
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2286477, 1.2273397
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1396160, 1.1385181
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9394450, 0.9338415
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1455638, 1.1430371
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9205403, 0.9228327

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5817
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 5817

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649594, upper bound: 0.7611703
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649592, upper bound: 0.7611692
time: 4.83 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.74 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.74
Output dim: 3, lower bound: -0.7611692, upper bound: 0.7649593
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.74
Output dim: 3, lower bound: -0.7611690, upper bound: 0.7649597
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.74
Output dim: 3, lower bound: -0.7649594, upper bound: 0.7611703
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.74
Output dim: 3, lower bound: -0.7649592, upper bound: 0.7611692

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3130112, 1.3108406
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9407504, 0.9391065
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1915374, 1.1834891
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1652484, 1.1673830
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1820402, 1.1854100
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2264481, 1.2309122
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1404648, 1.1388462
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9341350, 0.9393289
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1488752, 1.1432645
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9218040, 0.9231458

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 4665

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7611577, upper bound: 0.7644011
time: 4.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7606110, upper bound: 0.7649490
time: 4.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2

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

Time for backsubstitution: 22.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4665

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7611575, upper bound: 0.7644013
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7606108, upper bound: 0.7649480
time: 5.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3091660, 1.3146863
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9455087, 0.9343481
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1854796, 1.1895466
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1684046, 1.1642265
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1824203, 1.1850295
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2277565, 1.2296047
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1415629, 1.1377485
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9397383, 0.9337254
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1514020, 1.1407378
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9195118, 0.9254382

Time for backsubstitution: 22.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 4665

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649479, upper bound: 0.7606107
time: 4.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7644013, upper bound: 0.7611574
time: 4.69 seconds

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

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4665
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 4665

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649477, upper bound: 0.7606110
time: 4.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7644011, upper bound: 0.7611577
time: 4.77 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 32.07 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.07
Output dim: 3, lower bound: -0.7611577, upper bound: 0.7644011
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.07
Output dim: 3, lower bound: -0.7606110, upper bound: 0.7649490
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.07
Output dim: 3, lower bound: -0.7611575, upper bound: 0.7644013
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.07
Output dim: 3, lower bound: -0.7606108, upper bound: 0.7649480
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.07
Output dim: 3, lower bound: -0.7649479, upper bound: 0.7606107
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.07
Output dim: 3, lower bound: -0.7644013, upper bound: 0.7611574
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.07
Output dim: 3, lower bound: -0.7649477, upper bound: 0.7606110
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.07
Output dim: 3, lower bound: -0.7644011, upper bound: 0.7611577

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3124623, 1.3153176
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9408784, 0.9390905
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1952705, 1.1830323
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1694512, 1.1668670
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1851411, 1.1850295
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2263803, 1.2314620
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1400266, 1.1424184
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9373970, 0.9389296
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1487737, 1.1440940
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9251680, 0.9227340

Time for backsubstitution: 22.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 442

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7600215, upper bound: 0.7643859
time: 4.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7611426, upper bound: 0.7632663
time: 4.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3130112, 1.3102918
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9407344, 0.9391065
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1910806, 1.1834891
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1647325, 1.1673830
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1816602, 1.1854100
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2264481, 1.2308440
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1404648, 1.1384077
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9337354, 0.9393289
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1488752, 1.1431632
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9213924, 0.9231458

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 442

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7594740, upper bound: 0.7649325
time: 4.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7605959, upper bound: 0.7638126
time: 4.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3129358, 1.3136430
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9344759, 0.9408989
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1932797, 1.1835923
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1684294, 1.1671541
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1859860, 1.1820402
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2272720, 1.2283063
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1373100, 1.1431880
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9369874, 0.9390457
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1406364, 1.1463935
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9261971, 0.9191000

Time for backsubstitution: 22.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 442

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7600213, upper bound: 0.7643872
time: 4.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7611424, upper bound: 0.7632652
time: 5.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3134851, 1.3086171
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9343319, 0.9409149
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1890898, 1.1840496
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1637106, 1.1676700
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1825051, 1.1824203
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2273397, 1.2276874
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1377482, 1.1391773
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9333258, 0.9394450
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1407375, 1.1454628
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9224215, 0.9195116

Time for backsubstitution: 22.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 442

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7594738, upper bound: 0.7649327
time: 4.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7605957, upper bound: 0.7638119
time: 5.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3086171, 1.3191633
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9456367, 0.9343319
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1892128, 1.1890898
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1726074, 1.1637106
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1855211, 1.1846495
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2276878, 1.2301545
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1411242, 1.1413209
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9430003, 0.9333258
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1513004, 1.1415672
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9228754, 0.9250267

Time for backsubstitution: 22.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 442

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7638118, upper bound: 0.7605956
time: 5.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649327, upper bound: 0.7594738
time: 5.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3091660, 1.3141375
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9454927, 0.9343481
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1850228, 1.1895466
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1678886, 1.1642265
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1820402, 1.1850295
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2277565, 1.2295356
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1415629, 1.1373100
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9393392, 0.9337254
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1514020, 1.1406364
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9190998, 0.9254382

Time for backsubstitution: 23.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 442

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7632652, upper bound: 0.7611423
time: 5.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7643859, upper bound: 0.7600213
time: 4.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3090901, 1.3174882
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9392343, 0.9361403
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1872220, 1.1896501
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1715856, 1.1639979
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1863661, 1.1816602
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2285800, 1.2269988
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1384077, 1.1420906
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9425912, 0.9334421
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1431632, 1.1438668
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9239044, 0.9213924

Time for backsubstitution: 22.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 442

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7638116, upper bound: 0.7605960
time: 5.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649325, upper bound: 0.7594740
time: 5.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3096395, 1.3124623
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9390903, 0.9361563
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1830325, 1.1901073
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1668668, 1.1645136
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1828852, 1.1820402
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2286477, 1.2263799
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1388459, 1.1380796
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9389296, 0.9338415
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1432648, 1.1429358
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9201288, 0.9218040

Time for backsubstitution: 22.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 442

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7632650, upper bound: 0.7611426
time: 4.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7643857, upper bound: 0.7600214
time: 4.70 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 32.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7600215, upper bound: 0.7643859
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7611426, upper bound: 0.7632663
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7594740, upper bound: 0.7649325
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7605959, upper bound: 0.7638126
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7600213, upper bound: 0.7643872
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7611424, upper bound: 0.7632652
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7594738, upper bound: 0.7649327
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7605957, upper bound: 0.7638119
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7638118, upper bound: 0.7605956
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7649327, upper bound: 0.7594738
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7632652, upper bound: 0.7611423
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7643859, upper bound: 0.7600213
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7638116, upper bound: 0.7605960
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7649325, upper bound: 0.7594740
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7632650, upper bound: 0.7611426
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.70
Output dim: 3, lower bound: -0.7643857, upper bound: 0.7600214

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3111296, 1.3215127
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9419861, 0.9388514
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1951075, 1.1837916
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1682429, 1.1724916
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1874795, 1.1845260
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2375388, 1.2290645
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1403599, 1.1423469
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9367790, 0.9418066
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1559308, 1.1425529
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9245772, 0.9254758

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 943

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7594212, upper bound: 0.7643821
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7594210, upper bound: 0.7600073
time: 4.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3124623, 1.3139849
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9406395, 0.9390905
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1952705, 1.1828690
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1694512, 1.1656590
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1846371, 1.1850295
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2239823, 1.2314620
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1399546, 1.1424184
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9373970, 0.9383111
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1472323, 1.1440940
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9251680, 0.9221432

Time for backsubstitution: 22.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 943

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7605412, upper bound: 0.7632614
time: 4.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7605410, upper bound: 0.7588861
time: 5.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3116784, 1.3164868
1: -4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9418416, 0.9388676
2: -5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.1909175, 1.1842487
3: 5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1635242, 1.1730072
4: -14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1839991, 1.1849060
5: -7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2376075, 1.2284455
6: -11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1407981, 1.1383359
7: -6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9331169, 0.9422059
8: -4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1560318, 1.1416218
9: -5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9208016, 0.9258876

Time for backsubstitution: 22.16 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.34 + 545.15 = 603.49 seconds
