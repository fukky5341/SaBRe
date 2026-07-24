## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.13366404


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2675802, 0.2675802)
1: (2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2295630, 0.2295631)
2: (-4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2331604, 0.2331605)
3: (-14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4674888, 0.4674888)
4: (-3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2475597, 0.2475598)
5: (-8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3414524, 0.3414524)
6: (-4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2934346, 0.2934346)
7: (-8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3316460, 0.3316460)
8: (-1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.3006594, 0.3006594)
9: (-7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2711308, 0.2711308)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.19 + 32.90 = 55.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.1452866, upper bound: 0.1452870

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 4598

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1451890, upper bound: 0.1452854
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1452850, upper bound: 0.1451894
time: 2.95 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.13 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.13
Output dim: 1, lower bound: -0.1451890, upper bound: 0.1452854
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.13
Output dim: 1, lower bound: -0.1452850, upper bound: 0.1451894

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2674878, 0.2675886
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2292247, 0.2295976
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2329627, 0.2331822
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4675524, 0.4668515
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2476076, 0.2470753
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3410981, 0.3414826
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2931640, 0.2934581
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3317630, 0.3304993
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.3006511, 0.3006600
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2711654, 0.2707733

Time for backsubstitution: 20.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417623, upper bound: 0.1417796
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417712, upper bound: 0.1417296
time: 2.87 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2675802, 0.2674878
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2295630, 0.2292247
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2331604, 0.2329627
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4668515, 0.4674888
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2470753, 0.2475598
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3414524, 0.3410981
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2934346, 0.2931639
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3304994, 0.3316460
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.3006594, 0.3006511
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2707733, 0.2711308

Time for backsubstitution: 20.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417296, upper bound: 0.1417711
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417797, upper bound: 0.1417623
time: 2.91 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 26.67 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 26.67
Output dim: 1, lower bound: -0.1417623, upper bound: 0.1417796
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 26.67
Output dim: 1, lower bound: -0.1417712, upper bound: 0.1417296
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 26.67
Output dim: 1, lower bound: -0.1417296, upper bound: 0.1417711
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 26.67
Output dim: 1, lower bound: -0.1417797, upper bound: 0.1417623

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2444130, 0.2435996
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2184923, 0.2189374
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2318674, 0.2320364
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4669621, 0.4661803
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2409601, 0.2419537
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3396575, 0.3402319
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2832348, 0.2817976
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3275828, 0.3286769
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2887414, 0.2891185
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2660911, 0.2654008

Time for backsubstitution: 20.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 409

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1412306, upper bound: 0.1411128
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1410196, upper bound: 0.1412702
time: 2.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2434987, 0.2440988
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2185645, 0.2186267
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2318169, 0.2320603
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4668810, 0.4662440
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2422978, 0.2404279
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3398104, 0.3400421
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2815034, 0.2831892
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3298166, 0.3263187
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2888572, 0.2887502
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2657928, 0.2655920

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 409

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1412428, upper bound: 0.1409753
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1410506, upper bound: 0.1411963
time: 3.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2440988, 0.2434988
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2186144, 0.2185645
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2320411, 0.2318170
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4662440, 0.4668171
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2404280, 0.2422670
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3400123, 0.3398104
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2831893, 0.2815034
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3263187, 0.3297110
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2887495, 0.2888572
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2655920, 0.2657586

Time for backsubstitution: 22.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 409

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1411964, upper bound: 0.1410501
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1409753, upper bound: 0.1412427
time: 3.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2435911, 0.2443843
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2189030, 0.2184923
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2320147, 0.2318674
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4661803, 0.4668970
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2419537, 0.2409123
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3401990, 0.3396575
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2817740, 0.2832308
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3286769, 0.3274655
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2890964, 0.2887414
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2654008, 0.2660475

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 409

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1412703, upper bound: 0.1410197
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1411125, upper bound: 0.1412305
time: 2.89 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 27.73 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.73
Output dim: 1, lower bound: -0.1412306, upper bound: 0.1411128
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.73
Output dim: 1, lower bound: -0.1410196, upper bound: 0.1412702
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.73
Output dim: 1, lower bound: -0.1412428, upper bound: 0.1409753
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.73
Output dim: 1, lower bound: -0.1410506, upper bound: 0.1411963
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.73
Output dim: 1, lower bound: -0.1411964, upper bound: 0.1410501
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.73
Output dim: 1, lower bound: -0.1409753, upper bound: 0.1412427
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.73
Output dim: 1, lower bound: -0.1412703, upper bound: 0.1410197
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.73
Output dim: 1, lower bound: -0.1411125, upper bound: 0.1412305

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2440637, 0.2496480
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2278492, 0.2281153
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2253238, 0.2258164
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4597557, 0.4589775
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2419692, 0.2416205
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3371472, 0.3374233
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2858317, 0.2870945
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3099577, 0.3101254
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2964993, 0.2964894
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2566698, 0.2547624

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1387843, upper bound: 0.1395041
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1396250, upper bound: 0.1386618
time: 3.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2491859, 0.2441646
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2277176, 0.2282221
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2254435, 0.2255432
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4596782, 0.4589000
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2421528, 0.2413170
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3370390, 0.3374467
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2868001, 0.2859998
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3110192, 0.3086939
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2964137, 0.2965081
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2551544, 0.2560322

Time for backsubstitution: 21.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1385674, upper bound: 0.1396666
time: 3.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1394081, upper bound: 0.1388248
time: 3.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2440637, 0.2496480
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2278492, 0.2281153
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2253238, 0.2258164
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4597557, 0.4589775
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2419692, 0.2416205
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3371472, 0.3374233
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2858317, 0.2870945
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3099577, 0.3101254
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2964993, 0.2964894
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2566698, 0.2547624

Time for backsubstitution: 20.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1387961, upper bound: 0.1393650
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1396367, upper bound: 0.1385227
time: 3.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2491859, 0.2441646
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2277176, 0.2282221
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2254435, 0.2255432
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4596782, 0.4589000
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2421528, 0.2413170
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3370390, 0.3374467
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2868001, 0.2859998
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3110192, 0.3086939
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2964137, 0.2965081
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2551544, 0.2560322

Time for backsubstitution: 20.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1385973, upper bound: 0.1395906
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1394391, upper bound: 0.1387500
time: 3.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2441561, 0.2491860
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2281873, 0.2277178
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2255217, 0.2254435
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4589000, 0.4596138
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2413170, 0.2421046
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3374238, 0.3370390
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2859867, 0.2868001
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3086936, 0.3109360
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2965071, 0.2964135
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2560321, 0.2551199

Time for backsubstitution: 20.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1387502, upper bound: 0.1394389
time: 4.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1395913, upper bound: 0.1385967
time: 3.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2496091, 0.2440638
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2280784, 0.2278492
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2257820, 0.2253237
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4589775, 0.4596775
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2416204, 0.2419107
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3373933, 0.3371472
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2870710, 0.2858316
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3101251, 0.3098406
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2964828, 0.2964993
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2547623, 0.2566133

Time for backsubstitution: 20.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1385233, upper bound: 0.1396373
time: 3.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1393647, upper bound: 0.1387963
time: 3.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2441561, 0.2491860
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2281873, 0.2277178
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2255217, 0.2254435
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4589000, 0.4596138
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2413170, 0.2421046
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3374238, 0.3370390
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2859867, 0.2868001
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3086936, 0.3109360
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2965071, 0.2964135
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2560321, 0.2551199

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1388253, upper bound: 0.1394087
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1396660, upper bound: 0.1385670
time: 3.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2496091, 0.2440638
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2280784, 0.2278492
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2257820, 0.2253237
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4589775, 0.4596775
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2416204, 0.2419107
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3373933, 0.3371472
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2870710, 0.2858316
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3101251, 0.3098406
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2964828, 0.2964993
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2547623, 0.2566133

Time for backsubstitution: 20.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1386623, upper bound: 0.1396253
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1395034, upper bound: 0.1387837
time: 3.32 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1387843, upper bound: 0.1395041
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1396250, upper bound: 0.1386618
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1385674, upper bound: 0.1396666
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1394081, upper bound: 0.1388248
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1387961, upper bound: 0.1393650
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1396367, upper bound: 0.1385227
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1385973, upper bound: 0.1395906
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1394391, upper bound: 0.1387500
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1387502, upper bound: 0.1394389
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1395913, upper bound: 0.1385967
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1385233, upper bound: 0.1396373
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1393647, upper bound: 0.1387963
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1388253, upper bound: 0.1394087
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1396660, upper bound: 0.1385670
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1386623, upper bound: 0.1396253
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 1, lower bound: -0.1395034, upper bound: 0.1387837

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2619582, 0.2632080
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2278270, 0.2285903
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2321554, 0.2314749
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4671695, 0.4659657
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2458301, 0.2458382
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3296301, 0.3347402
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2929409, 0.2934108
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3310356, 0.3293970
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2968297, 0.2954767
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2619830, 0.2587791

Time for backsubstitution: 21.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Candidate
type: DSZ, layer: 3, pos: 409

### Candidate
type: DSZ, layer: 3, pos: 1851

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1382444, upper bound: 0.1390582
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1383392, upper bound: 0.1389626
time: 3.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2631072, 0.2620591
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2282175, 0.2281997
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2312554, 0.2323749
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4666665, 0.4664688
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2463703, 0.2452977
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3343565, 0.3300145
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2931166, 0.2932351
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3306611, 0.3297718
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2954676, 0.2968386
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2591714, 0.2615910

Time for backsubstitution: 21.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Candidate
type: DSZ, layer: 3, pos: 409

### Candidate
type: DSZ, layer: 3, pos: 1851

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1390847, upper bound: 0.1382169
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1391797, upper bound: 0.1381228
time: 3.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2619582, 0.2632080
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2278270, 0.2285903
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2321554, 0.2314749
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4671695, 0.4659657
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2458301, 0.2458382
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3296301, 0.3347402
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2929409, 0.2934108
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3310356, 0.3293970
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2968297, 0.2954767
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2619830, 0.2587791

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Candidate
type: DSZ, layer: 3, pos: 409

### Candidate
type: DSZ, layer: 3, pos: 1851

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1380278, upper bound: 0.1392208
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1381224, upper bound: 0.1391243
time: 4.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2631072, 0.2620591
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2282175, 0.2281997
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2312554, 0.2323749
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4666665, 0.4664688
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2463703, 0.2452977
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3343565, 0.3300145
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2931166, 0.2932351
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3306611, 0.3297718
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2954676, 0.2968386
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2591714, 0.2615910

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Candidate
type: DSZ, layer: 3, pos: 409

### Candidate
type: DSZ, layer: 3, pos: 1851

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1388675, upper bound: 0.1383805
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1389632, upper bound: 0.1382848
time: 3.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2619582, 0.2632080
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2278270, 0.2285903
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2321554, 0.2314749
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4671695, 0.4659657
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2458301, 0.2458382
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3296301, 0.3347402
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2929409, 0.2934108
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3310356, 0.3293970
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2968297, 0.2954767
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2619830, 0.2587791

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Candidate
type: DSZ, layer: 3, pos: 409

### Candidate
type: DSZ, layer: 3, pos: 1851

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1382561, upper bound: 0.1389184
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1383508, upper bound: 0.1388225
time: 4.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2631072, 0.2620591
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2282175, 0.2281997
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2312554, 0.2323749
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4666665, 0.4664688
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2463703, 0.2452977
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3343565, 0.3300145
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2931166, 0.2932351
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3306611, 0.3297718
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2954676, 0.2968386
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2591714, 0.2615910

Time for backsubstitution: 22.09 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.08 + 566.47 = 621.55 seconds
