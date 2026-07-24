## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.061818120000000004


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2605984, 0.2605989)
1: (-3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1932678, 0.1932678)
2: (-8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2409225, 0.2409225)
3: (-1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2066152, 0.2066151)
4: (-4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1671135, 0.1671135)
5: (-6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1912547, 0.1912547)
6: (-15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2285557, 0.2285556)
7: (4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1057246, 0.1057247)
8: (-4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2120063, 0.2120063)
9: (-0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1830323, 0.1830324)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.53 + 32.76 = 54.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0686868, upper bound: 0.0686868

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 312

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670258, upper bound: 0.0670431
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670431, upper bound: 0.0670258
time: 3.11 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.07 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.07
Output dim: 7, lower bound: -0.0670258, upper bound: 0.0670431
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.07
Output dim: 7, lower bound: -0.0670431, upper bound: 0.0670258

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2598908, 0.2599375
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1930890, 0.1926243
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2400539, 0.2401588
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2064419, 0.2062680
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1665804, 0.1663382
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1914903, 0.1907774
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2279310, 0.2283787
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1044594, 0.1050189
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2106251, 0.2105651
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1831317, 0.1825047

Time for backsubstitution: 7.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 1829

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0660713, upper bound: 0.0665325
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0665153, upper bound: 0.0660886
time: 2.89 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2605984, 0.2598908
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1926243, 0.1932678
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2409225, 0.2400539
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2066152, 0.2064419
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1671135, 0.1665804
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1907774, 0.1912547
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2285557, 0.2279309
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1050189, 0.1057247
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2120063, 0.2106251
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1825047, 0.1830324

Time for backsubstitution: 7.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 1829

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0660886, upper bound: 0.0665152
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0665325, upper bound: 0.0660714
time: 2.68 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 12.79 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 12.79
Output dim: 7, lower bound: -0.0660713, upper bound: 0.0665325
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 12.79
Output dim: 7, lower bound: -0.0665153, upper bound: 0.0660886
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 12.79
Output dim: 7, lower bound: -0.0660886, upper bound: 0.0665152
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 12.79
Output dim: 7, lower bound: -0.0665325, upper bound: 0.0660714

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2565234, 0.2573552
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1918974, 0.1891563
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2379434, 0.2377563
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1773015, 0.1734843
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1626122, 0.1626460
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1778259, 0.1781456
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2164514, 0.2185433
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1015366, 0.1024214
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2002867, 0.2005205
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1639687, 0.1667397

Time for backsubstitution: 7.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 1852

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0646780, upper bound: 0.0662622
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0658010, upper bound: 0.0651390
time: 2.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2573085, 0.2565699
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1896210, 0.1914327
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2376513, 0.2380483
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1736580, 0.1771276
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1628882, 0.1623700
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1788585, 0.1771131
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2180958, 0.2168992
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1018620, 0.1020960
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2005806, 0.2002268
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1673670, 0.1633415

Time for backsubstitution: 8.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 1852

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651219, upper bound: 0.0658183
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0662450, upper bound: 0.0646952
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2572310, 0.2573085
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1914327, 0.1897998
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2388120, 0.2376513
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1774746, 0.1736581
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1631449, 0.1628882
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1771131, 0.1786230
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2170763, 0.2180958
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1020959, 0.1031271
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2016680, 0.2005806
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1633415, 0.1672676

Time for backsubstitution: 8.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 1852

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0646950, upper bound: 0.0662449
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0658183, upper bound: 0.0651218
time: 2.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2580161, 0.2565234
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1891563, 0.1920760
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2385201, 0.2379434
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1738313, 0.1773014
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1634209, 0.1626122
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1781456, 0.1775904
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2187204, 0.2164514
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1024213, 0.1028017
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2019618, 0.2002867
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1667397, 0.1638693

Time for backsubstitution: 8.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 1852

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651389, upper bound: 0.0658010
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0662619, upper bound: 0.0646776
time: 2.73 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 13.70 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 7, lower bound: -0.0646780, upper bound: 0.0662622
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 7, lower bound: -0.0658010, upper bound: 0.0651390
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 7, lower bound: -0.0651219, upper bound: 0.0658183
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 7, lower bound: -0.0662450, upper bound: 0.0646952
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 7, lower bound: -0.0646950, upper bound: 0.0662449
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 7, lower bound: -0.0658183, upper bound: 0.0651218
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 7, lower bound: -0.0651389, upper bound: 0.0658010
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 7, lower bound: -0.0662619, upper bound: 0.0646776

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2575767, 0.2583411
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1924138, 0.1891484
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2344069, 0.2344583
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1776451, 0.1737671
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1616756, 0.1621169
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1776776, 0.1778570
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2165506, 0.2186371
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0993716, 0.1007030
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2012343, 0.2015367
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1622462, 0.1654999

Time for backsubstitution: 7.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 311

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0626248, upper bound: 0.0632967
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0617687, upper bound: 0.0641914
time: 2.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2575092, 0.2584085
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1918895, 0.1896729
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2346454, 0.2342197
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1775842, 0.1738279
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1620831, 0.1617094
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1775373, 0.1779974
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2165453, 0.2186424
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0998181, 0.1002563
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2013028, 0.2014682
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1627288, 0.1650172

Time for backsubstitution: 7.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 311

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0637479, upper bound: 0.0621736
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0628918, upper bound: 0.0630683
time: 2.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2583618, 0.2575557
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1901374, 0.1914248
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2341149, 0.2347502
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1740018, 0.1774104
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1619516, 0.1618409
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1787103, 0.1768244
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2181947, 0.2169927
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0996970, 0.1003776
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2015282, 0.2012428
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1656444, 0.1621016

Time for backsubstitution: 7.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 311

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0630687, upper bound: 0.0628917
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0621740, upper bound: 0.0637475
time: 2.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2582943, 0.2576234
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1896131, 0.1919491
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2343533, 0.2345117
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1739409, 0.1774713
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1623592, 0.1614335
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1785699, 0.1769647
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2181895, 0.2169982
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1001436, 0.0999309
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2015966, 0.2011744
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1661271, 0.1616189

Time for backsubstitution: 7.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 311

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0641919, upper bound: 0.0617686
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0632970, upper bound: 0.0626244
time: 2.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2582843, 0.2582943
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1919491, 0.1897917
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2352757, 0.2343534
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1778183, 0.1739409
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1622084, 0.1623591
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1769647, 0.1783344
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2171752, 0.2181894
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0999309, 0.1014087
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2026156, 0.2015966
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1616189, 0.1660279

Time for backsubstitution: 7.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 311

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0626248, upper bound: 0.0632967
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0617687, upper bound: 0.0641914
time: 2.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2582169, 0.2583618
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1914248, 0.1903160
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2355142, 0.2341148
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1777574, 0.1740018
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1626160, 0.1619516
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1768244, 0.1784748
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2171700, 0.2181948
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1003776, 0.1009621
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2026840, 0.2015282
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1621016, 0.1655452

Time for backsubstitution: 8.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 311

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0637479, upper bound: 0.0621736
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0628918, upper bound: 0.0630683
time: 2.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2590694, 0.2575092
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1896727, 0.1920681
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2349837, 0.2346455
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1741749, 0.1775843
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1624843, 0.1620831
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1779973, 0.1773018
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2188194, 0.2165452
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1002563, 0.1010833
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2029094, 0.2013028
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1650172, 0.1626296

Time for backsubstitution: 7.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 311

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0630687, upper bound: 0.0628917
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0621740, upper bound: 0.0637475
time: 2.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2590020, 0.2575767
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1891484, 0.1925924
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2352221, 0.2344068
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1741141, 0.1776451
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1628919, 0.1616756
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1778570, 0.1774421
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2188141, 0.2165505
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1007030, 0.1006367
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2029779, 0.2012343
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1654999, 0.1621470

Time for backsubstitution: 7.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 311

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0641919, upper bound: 0.0617686
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0632970, upper bound: 0.0626244
time: 2.74 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 13.18 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0626248, upper bound: 0.0632967
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0617687, upper bound: 0.0641914
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0637479, upper bound: 0.0621736
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0628918, upper bound: 0.0630683
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0630687, upper bound: 0.0628917
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0621740, upper bound: 0.0637475
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0641919, upper bound: 0.0617686
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0632970, upper bound: 0.0626244
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0626248, upper bound: 0.0632967
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0617687, upper bound: 0.0641914
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0637479, upper bound: 0.0621736
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0628918, upper bound: 0.0630683
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0630687, upper bound: 0.0628917
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0621740, upper bound: 0.0637475
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0641919, upper bound: 0.0617686
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 7, lower bound: -0.0632970, upper bound: 0.0626244

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2568212, 0.2577438
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1921139, 0.1888433
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2340393, 0.2341909
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1762593, 0.1724237
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1594811, 0.1602535
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1726702, 0.1736341
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2154214, 0.2177849
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0991681, 0.1000009
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2009374, 0.2010456
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1551129, 0.1621910

Time for backsubstitution: 7.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0615521, upper bound: 0.0618186
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0615038, upper bound: 0.0621740
time: 2.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2569795, 0.2575955
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1921272, 0.1888487
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2341394, 0.2340908
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1763530, 0.1723812
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1598122, 0.1600682
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1734548, 0.1729732
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2158158, 0.2175081
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0986694, 0.1005087
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2007434, 0.2012547
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1591301, 0.1583666

Time for backsubstitution: 7.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0606964, upper bound: 0.0627134
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0606478, upper bound: 0.0630688
time: 2.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2567537, 0.2578113
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1915898, 0.1893675
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2342777, 0.2339525
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1761984, 0.1724846
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1598886, 0.1598459
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1725298, 0.1737745
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2154162, 0.2177901
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0996148, 0.0995542
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2010058, 0.2009772
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1555954, 0.1617085

Time for backsubstitution: 8.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0626253, upper bound: 0.0610527
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0622698, upper bound: 0.0611011
time: 2.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2569120, 0.2576630
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1916027, 0.1893730
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2343781, 0.2338521
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1762922, 0.1724421
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1602196, 0.1596607
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1733145, 0.1731136
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2158103, 0.2175133
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0991160, 0.1000621
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2008119, 0.2011863
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1596128, 0.1578839

Time for backsubstitution: 7.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0617693, upper bound: 0.0619474
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0614139, upper bound: 0.0619957
time: 2.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2576065, 0.2569585
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1898377, 0.1910827
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2337472, 0.2344830
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1726159, 0.1760678
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1597571, 0.1599774
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1737027, 0.1726015
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2170656, 0.2161405
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0994936, 0.0996754
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2012311, 0.2007519
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1585110, 0.1589596

Time for backsubstitution: 7.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619962, upper bound: 0.0614137
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619477, upper bound: 0.0617690
time: 2.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2577646, 0.2568104
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1898875, 0.1911249
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2338476, 0.2343826
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1727089, 0.1760246
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1600881, 0.1597922
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1744874, 0.1719407
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2174599, 0.2158637
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0989948, 0.1001834
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2010372, 0.2009609
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1623616, 0.1549683

Time for backsubstitution: 7.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0611012, upper bound: 0.0622695
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0610528, upper bound: 0.0626248
time: 2.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2575388, 0.2570262
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1893134, 0.1916070
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2339857, 0.2342443
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1725550, 0.1761287
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1601645, 0.1595700
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1735624, 0.1727418
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2170603, 0.2161460
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0999402, 0.0992287
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2012995, 0.2006834
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1589937, 0.1584769

Time for backsubstitution: 7.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0630692, upper bound: 0.0606477
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0627137, upper bound: 0.0606960
time: 2.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2576971, 0.2568779
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1893632, 0.1916492
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2340860, 0.2341442
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1726480, 0.1760854
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1604957, 0.1593848
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1743470, 0.1720811
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2174547, 0.2158689
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0994415, 0.0997367
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2011056, 0.2008924
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1628442, 0.1544856

Time for backsubstitution: 7.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0621742, upper bound: 0.0615036
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0618187, upper bound: 0.0615518
time: 2.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2575288, 0.2576971
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1916494, 0.1894865
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2349081, 0.2340860
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1764452, 0.1726480
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1600139, 0.1604956
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1720811, 0.1741850
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2160487, 0.2174547
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0997368, 0.1008012
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2023187, 0.2011056
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1544856, 0.1627191

Time for backsubstitution: 7.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0615521, upper bound: 0.0618186
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0615038, upper bound: 0.0621740
time: 3.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2576871, 0.2575388
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1916070, 0.1894917
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2350082, 0.2339859
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1765389, 0.1725550
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1603450, 0.1601645
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1727419, 0.1735241
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2164428, 0.2170603
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0992288, 0.1013091
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2021247, 0.2012995
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1584771, 0.1588945

Time for backsubstitution: 8.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0606964, upper bound: 0.0627134
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0606478, upper bound: 0.0630688
time: 2.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2574613, 0.2577646
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1911249, 0.1900108
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2351465, 0.2338476
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1763843, 0.1727089
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1604214, 0.1600882
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1719407, 0.1743254
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2160432, 0.2174599
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1001833, 0.1003545
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2023871, 0.2010372
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1549683, 0.1622365

Time for backsubstitution: 7.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0626253, upper bound: 0.0610527
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0622698, upper bound: 0.0611011
time: 3.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2576196, 0.2576065
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1910827, 0.1900163
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2352469, 0.2337472
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1764781, 0.1726159
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1607525, 0.1597570
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1726015, 0.1736645
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2164376, 0.2170656
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0996755, 0.1008624
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2021931, 0.2012311
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1589596, 0.1584119

Time for backsubstitution: 7.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0617693, upper bound: 0.0619474
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0614139, upper bound: 0.0619957
time: 3.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2583141, 0.2569120
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1893730, 0.1917260
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2346160, 0.2343781
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1728019, 0.1762922
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1602898, 0.1602197
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1731136, 0.1731524
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2176931, 0.2158103
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1000622, 0.1004757
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2026124, 0.2008119
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1578839, 0.1594876

Time for backsubstitution: 7.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619962, upper bound: 0.0614137
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619477, upper bound: 0.0617690
time: 2.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2584722, 0.2567537
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1893675, 0.1917682
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2347164, 0.2342777
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1728948, 0.1761984
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1606210, 0.1598886
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1737745, 0.1724916
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2180872, 0.2154162
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0995542, 0.1009836
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2024184, 0.2010058
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1617085, 0.1554962

Time for backsubstitution: 8.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0611012, upper bound: 0.0622695
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0610528, upper bound: 0.0626248
time: 3.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2582467, 0.2569795
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1888485, 0.1922503
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2348545, 0.2341394
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1727409, 0.1763530
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1606974, 0.1598122
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1729733, 0.1732928
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2176876, 0.2158155
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1005087, 0.1000291
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2026808, 0.2007434
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1583666, 0.1590050

Time for backsubstitution: 7.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0630692, upper bound: 0.0606477
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0627137, upper bound: 0.0606960
time: 2.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2584047, 0.2568212
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1888433, 0.1922925
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2349548, 0.2340393
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1728339, 0.1762593
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1610284, 0.1594811
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1736342, 0.1726320
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2180820, 0.2154214
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1000009, 0.1005370
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2024869, 0.2009374
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1621910, 0.1550137

Time for backsubstitution: 7.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0621742, upper bound: 0.0615036
time: 2.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0618187, upper bound: 0.0615518
time: 2.80 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 13.14 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0615521, upper bound: 0.0618186
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0615038, upper bound: 0.0621740
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0606964, upper bound: 0.0627134
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0606478, upper bound: 0.0630688
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0626253, upper bound: 0.0610527
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0622698, upper bound: 0.0611011
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0617693, upper bound: 0.0619474
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0614139, upper bound: 0.0619957
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0619962, upper bound: 0.0614137
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0619477, upper bound: 0.0617690
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0611012, upper bound: 0.0622695
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0610528, upper bound: 0.0626248
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0630692, upper bound: 0.0606477
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0627137, upper bound: 0.0606960
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0621742, upper bound: 0.0615036
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0618187, upper bound: 0.0615518
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0615521, upper bound: 0.0618186
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0615038, upper bound: 0.0621740
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0606964, upper bound: 0.0627134
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0606478, upper bound: 0.0630688
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0626253, upper bound: 0.0610527
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0622698, upper bound: 0.0611011
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0617693, upper bound: 0.0619474
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0614139, upper bound: 0.0619957
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0619962, upper bound: 0.0614137
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0619477, upper bound: 0.0617690
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0611012, upper bound: 0.0622695
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0610528, upper bound: 0.0626248
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0630692, upper bound: 0.0606477
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0627137, upper bound: 0.0606960
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0621742, upper bound: 0.0615036
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.14
Output dim: 7, lower bound: -0.0618187, upper bound: 0.0615518

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2477319, 0.2497416
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1914475, 0.1880108
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2325654, 0.2329443
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1620122, 0.1585597
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1550504, 0.1551774
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1726568, 0.1736321
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2151620, 0.2174695
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0941772, 0.0945636
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1899481, 0.1908365
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1460863, 0.1516134

Time for backsubstitution: 8.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 557

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0613998, upper bound: 0.0580436
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0573187, upper bound: 0.0616447
time: 3.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2488194, 0.2486546
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1912813, 0.1881772
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2328475, 0.2327170
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1623951, 0.1581527
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1544052, 0.1558228
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1726682, 0.1736207
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2151058, 0.2175255
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0937405, 0.0950099
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1909246, 0.1900564
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1445351, 0.1531653

Time for backsubstitution: 8.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 557

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0613515, upper bound: 0.0583989
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0572704, upper bound: 0.0620000
time: 3.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2478902, 0.2495933
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1914606, 0.1880161
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2326658, 0.2328439
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1621061, 0.1585170
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1553814, 0.1549921
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1734414, 0.1729712
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2155564, 0.2171926
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0936784, 0.0950715
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1897542, 0.1910456
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1501036, 0.1477888

Time for backsubstitution: 8.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 557

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0605219, upper bound: 0.0584801
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0568820, upper bound: 0.0625612
time: 2.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2489777, 0.2485063
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1912947, 0.1881827
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2329478, 0.2326169
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1624890, 0.1581103
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1547363, 0.1556374
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1734527, 0.1729599
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2155001, 0.2172487
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0932417, 0.0955178
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1907306, 0.1902655
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1485524, 0.1493407

Time for backsubstitution: 8.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 557

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0604738, upper bound: 0.0588355
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0568336, upper bound: 0.0629166
time: 3.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2476645, 0.2498093
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1909237, 0.1885351
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2328038, 0.2327607
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1619275, 0.1586205
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1554579, 0.1547701
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1725165, 0.1737725
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2151568, 0.2174747
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0946239, 0.0941266
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1900165, 0.1909646
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1465697, 0.1511308

Time for backsubstitution: 7.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 557

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0624727, upper bound: 0.0572776
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0583917, upper bound: 0.0608787
time: 3.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2487514, 0.2487221
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1907573, 0.1887010
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2330310, 0.2324786
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1623343, 0.1582376
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1548125, 0.1554152
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1725278, 0.1737611
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2151005, 0.2175307
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0941775, 0.0945632
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1907966, 0.1899880
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1450177, 0.1526819

Time for backsubstitution: 8.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 557

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0621172, upper bound: 0.0573259
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0580362, upper bound: 0.0609270
time: 2.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2478228, 0.2496612
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1909368, 0.1885403
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2329042, 0.2326603
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1620212, 0.1585779
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1557889, 0.1545848
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1733010, 0.1731116
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2155511, 0.2171979
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0941250, 0.0946345
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1898227, 0.1911734
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1505870, 0.1473061

Time for backsubstitution: 7.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 557

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0615950, upper bound: 0.0577142
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0579550, upper bound: 0.0617953
time: 2.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2489097, 0.2485738
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1907701, 0.1887065
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2331314, 0.2323782
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1624280, 0.1581950
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1551435, 0.1552299
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1733124, 0.1731002
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2154949, 0.2172539
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0936787, 0.0950711
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1906028, 0.1901971
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1490350, 0.1488574

Time for backsubstitution: 7.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 557

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0612397, upper bound: 0.0577626
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0575996, upper bound: 0.0618435
time: 3.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2485173, 0.2489562
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1891711, 0.1902503
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2322733, 0.2332363
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1583689, 0.1622037
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1553262, 0.1549014
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1736894, 0.1725994
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2168062, 0.2158251
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0945026, 0.0942381
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1902418, 0.1905428
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1494845, 0.1483819

Time for backsubstitution: 8.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 557

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0618437, upper bound: 0.0575996
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0577627, upper bound: 0.0612397
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2496045, 0.2478695
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1890051, 0.1904167
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2325554, 0.2330091
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1587518, 0.1617969
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1546811, 0.1555467
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1737007, 0.1725881
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2167501, 0.2158811
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0940660, 0.0946844
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1912184, 0.1897627
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1479334, 0.1499339

Time for backsubstitution: 8.07 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 54.28 + 552.63 = 606.91 seconds
