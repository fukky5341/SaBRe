## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.13544686250000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1734597, 0.1734598)
1: (-16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2724416, 0.2724416)
2: (-4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1778619, 0.1778619)
3: (-10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1877030, 0.1877030)
4: (-10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2270476, 0.2270477)
5: (-2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1730332, 0.1730332)
6: (-2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1486965, 0.1486965)
7: (-7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2213484, 0.2213485)
8: (-0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1635133, 0.1635132)
9: (-4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1734947, 0.1734947)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.83 + 33.91 = 55.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.1361274, upper bound: 0.1361275

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 919

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 439

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1347769, upper bound: 0.1361212
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361210, upper bound: 0.1347771
time: 3.03 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.29 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.29
Output dim: 0, lower bound: -0.1347769, upper bound: 0.1361212
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.29
Output dim: 0, lower bound: -0.1361210, upper bound: 0.1347771

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1724665, 0.1738473
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2688544, 0.2738420
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1791002, 0.1746882
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1879499, 0.1870706
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2265489, 0.2272418
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1715487, 0.1736130
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1482834, 0.1488567
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2213770, 0.2212762
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1638627, 0.1626173
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1716869, 0.1741998

Time for backsubstitution: 19.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 919

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1346626, upper bound: 0.1361211
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1347768, upper bound: 0.1360069
time: 3.27 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1734597, 0.1724664
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2724416, 0.2688544
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1746885, 0.1778619
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1870706, 0.1877030
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2270476, 0.2265488
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1730332, 0.1715487
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1486965, 0.1482834
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2212762, 0.2213485
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1626174, 0.1635132
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1734947, 0.1716869

Time for backsubstitution: 20.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 919

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1360067, upper bound: 0.1347769
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361209, upper bound: 0.1346628
time: 3.00 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 26.25 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 26.25
Output dim: 0, lower bound: -0.1346626, upper bound: 0.1361211
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 26.25
Output dim: 0, lower bound: -0.1347768, upper bound: 0.1360069
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 26.25
Output dim: 0, lower bound: -0.1360067, upper bound: 0.1347769
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 26.25
Output dim: 0, lower bound: -0.1361209, upper bound: 0.1346628

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1723234, 0.1738405
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2682273, 0.2738106
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1790797, 0.1742793
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1874343, 0.1870453
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2264009, 0.2272344
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1708843, 0.1735806
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1482749, 0.1486806
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2209934, 0.2212558
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1634624, 0.1625985
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1716783, 0.1740131

Time for backsubstitution: 22.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 919

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 919

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1345168, upper bound: 0.1353807
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1339224, upper bound: 0.1359751
time: 3.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1724598, 0.1737044
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2688229, 0.2732148
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1786915, 0.1746676
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1879245, 0.1865550
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2265415, 0.2270937
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1715163, 0.1729486
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1481075, 0.1488480
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2213565, 0.2208927
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1638439, 0.1622173
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1715002, 0.1741914

Time for backsubstitution: 20.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 919

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 919

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1346309, upper bound: 0.1352667
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1340364, upper bound: 0.1358609
time: 3.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1733169, 0.1724596
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2718148, 0.2688229
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1746675, 0.1774530
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1865550, 0.1876777
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2268993, 0.2265415
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1723689, 0.1715163
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1486880, 0.1481075
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2208925, 0.2213281
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1622174, 0.1634943
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1734862, 0.1715002

Time for backsubstitution: 20.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 919

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 919

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1358608, upper bound: 0.1340366
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1352665, upper bound: 0.1346311
time: 3.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1734530, 0.1723235
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2724106, 0.2682272
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1742793, 0.1778413
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1870453, 0.1871874
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2270399, 0.2264009
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1730009, 0.1708843
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1485206, 0.1482748
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2212557, 0.2209649
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1625984, 0.1631131
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1733080, 0.1716784

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 919

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 919

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1359750, upper bound: 0.1339226
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1353806, upper bound: 0.1345169
time: 3.14 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.09 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 28.09
Output dim: 0, lower bound: -0.1345168, upper bound: 0.1353807
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.09
Output dim: 0, lower bound: -0.1339224, upper bound: 0.1359751
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 28.09
Output dim: 0, lower bound: -0.1346309, upper bound: 0.1352667
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.09
Output dim: 0, lower bound: -0.1340364, upper bound: 0.1358609
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.09
Output dim: 0, lower bound: -0.1358608, upper bound: 0.1340366
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 28.09
Output dim: 0, lower bound: -0.1352665, upper bound: 0.1346311
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.09
Output dim: 0, lower bound: -0.1359750, upper bound: 0.1339226
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 28.09
Output dim: 0, lower bound: -0.1353806, upper bound: 0.1345169

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1722782, 0.1738405
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2682273, 0.2735116
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1787362, 0.1742793
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1874343, 0.1870453
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2264009, 0.2268293
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1708843, 0.1733099
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1480519, 0.1486806
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2209934, 0.2209384
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1634624, 0.1624125
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1716783, 0.1739918

Time for backsubstitution: 21.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 79
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2573
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 229
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 909

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 2133

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1175151, upper bound: 0.1305959
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1285426, upper bound: 0.1195677
time: 3.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1724143, 0.1737044
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2688229, 0.2729160
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1783478, 0.1746676
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1879245, 0.1865550
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2265415, 0.2266886
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1715163, 0.1726779
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1478845, 0.1488480
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2213565, 0.2205753
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1638439, 0.1620313
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1715002, 0.1741700

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 79
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2573
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 229
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 909

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 2133

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1176291, upper bound: 0.1304818
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1286566, upper bound: 0.1194537
time: 3.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1734080, 0.1724143
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2715158, 0.2690182
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1748581, 0.1771096
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1865550, 0.1876895
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2264946, 0.2268684
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1720983, 0.1716232
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1488149, 0.1478845
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2205752, 0.2214501
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1620314, 0.1635494
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1734648, 0.1716061

Time for backsubstitution: 22.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 79
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2573
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 229
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 909

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2133

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1194535, upper bound: 0.1286568
time: 4.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1304816, upper bound: 0.1176293
time: 3.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1735444, 0.1722782
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2721117, 0.2684226
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1744702, 0.1774979
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1870453, 0.1871992
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2266352, 0.2267277
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1727303, 0.1709913
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1486475, 0.1480519
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2209386, 0.2210867
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1624124, 0.1631682
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1732866, 0.1717843

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 79
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2573
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 229
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 909

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 2133

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1195676, upper bound: 0.1285427
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1305957, upper bound: 0.1175153
time: 3.07 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.10 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.10
Output dim: 0, lower bound: -0.1175151, upper bound: 0.1305959
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.10
Output dim: 0, lower bound: -0.1285426, upper bound: 0.1195677
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.10
Output dim: 0, lower bound: -0.1176291, upper bound: 0.1304818
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.10
Output dim: 0, lower bound: -0.1286566, upper bound: 0.1194537
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.10
Output dim: 0, lower bound: -0.1194535, upper bound: 0.1286568
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.10
Output dim: 0, lower bound: -0.1304816, upper bound: 0.1176293
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.10
Output dim: 0, lower bound: -0.1195676, upper bound: 0.1285427
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.10
Output dim: 0, lower bound: -0.1305957, upper bound: 0.1175153

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 55.74 + 286.44 = 342.17 seconds
