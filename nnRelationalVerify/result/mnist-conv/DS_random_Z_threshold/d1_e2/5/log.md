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
execution time: IAR + RelationalAnalysis = 22.92 + 33.28 = 56.20 seconds
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

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 439

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1347769, upper bound: 0.1361212
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361210, upper bound: 0.1347771
time: 2.99 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.04 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.04
Output dim: 0, lower bound: -0.1347769, upper bound: 0.1361212
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.04
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

Time for backsubstitution: 21.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 4656

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 919

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1346311, upper bound: 0.1353809
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1340366, upper bound: 0.1359753
time: 2.96 seconds

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

Time for backsubstitution: 22.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 919

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1360067, upper bound: 0.1347769
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361209, upper bound: 0.1346628
time: 2.96 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.88 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 28.88
Output dim: 0, lower bound: -0.1346311, upper bound: 0.1353809
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.88
Output dim: 0, lower bound: -0.1340366, upper bound: 0.1359753
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.88
Output dim: 0, lower bound: -0.1360067, upper bound: 0.1347769
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.88
Output dim: 0, lower bound: -0.1361209, upper bound: 0.1346628

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1724210, 0.1738473
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2688544, 0.2735434
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1787571, 0.1746882
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1879499, 0.1870707
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2265489, 0.2268369
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1715487, 0.1733427
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1480607, 0.1488567
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2213770, 0.2209586
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1638627, 0.1624312
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1716869, 0.1741785

Time for backsubstitution: 22.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1339224, upper bound: 0.1359751
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1340364, upper bound: 0.1358609
time: 3.02 seconds

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

Time for backsubstitution: 23.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 919

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 919

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1358608, upper bound: 0.1340366
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1352665, upper bound: 0.1346311
time: 3.05 seconds

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

Time for backsubstitution: 22.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 919

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 919

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1359750, upper bound: 0.1339226
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1353806, upper bound: 0.1345169
time: 3.00 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.39 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.39
Output dim: 0, lower bound: -0.1339224, upper bound: 0.1359751
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.39
Output dim: 0, lower bound: -0.1340364, upper bound: 0.1358609
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.39
Output dim: 0, lower bound: -0.1358608, upper bound: 0.1340366
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 28.39
Output dim: 0, lower bound: -0.1352665, upper bound: 0.1346311
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.39
Output dim: 0, lower bound: -0.1359750, upper bound: 0.1339226
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 28.39
Output dim: 0, lower bound: -0.1353806, upper bound: 0.1345169

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 909
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 79
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 229
type: DSZ, layer: 3, pos: 2573
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 1783

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 909

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1335366, upper bound: 0.1358763
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1338237, upper bound: 0.1355894
time: 3.19 seconds

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

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 229
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 2573
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 79
type: DSZ, layer: 3, pos: 909
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1783

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1329018, upper bound: 0.1357222
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1339049, upper bound: 0.1348261
time: 3.02 seconds

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

Time for backsubstitution: 22.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 229
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2573
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 909
type: DSZ, layer: 3, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 914

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1354338, upper bound: 0.1340059
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1358301, upper bound: 0.1332564
time: 3.10 seconds

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

Time for backsubstitution: 22.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2573
type: DSZ, layer: 3, pos: 909
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 79
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 229
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 2909

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2573

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1335298, upper bound: 0.1320498
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1340969, upper bound: 0.1314820
time: 3.13 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.71 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.71
Output dim: 0, lower bound: -0.1335366, upper bound: 0.1358763
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.71
Output dim: 0, lower bound: -0.1338237, upper bound: 0.1355894
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.71
Output dim: 0, lower bound: -0.1329018, upper bound: 0.1357222
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.71
Output dim: 0, lower bound: -0.1339049, upper bound: 0.1348261
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.71
Output dim: 0, lower bound: -0.1354338, upper bound: 0.1340059
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.71
Output dim: 0, lower bound: -0.1358301, upper bound: 0.1332564
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.71
Output dim: 0, lower bound: -0.1335298, upper bound: 0.1320498
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.71
Output dim: 0, lower bound: -0.1340969, upper bound: 0.1314820

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1722747, 0.1738403
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2682271, 0.2735087
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1787322, 0.1742795
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1874340, 0.1870423
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2264037, 0.2268243
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1708833, 0.1733086
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1480498, 0.1486806
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2209924, 0.2209370
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1634619, 0.1624124
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1716779, 0.1739900

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 229
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 2573
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 79
type: DSZ, layer: 3, pos: 2865

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 914

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1331129, upper bound: 0.1358457
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1335059, upper bound: 0.1350935
time: 3.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1722782, 0.1738372
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2682242, 0.2735116
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1787362, 0.1742793
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1874313, 0.1870453
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2263957, 0.2268293
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1708843, 0.1733094
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1480519, 0.1486787
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2209934, 0.2209376
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1634624, 0.1624125
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1716783, 0.1739913

Time for backsubstitution: 22.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 229
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 2573
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 79
type: DSZ, layer: 3, pos: 2865

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1327712, upper bound: 0.1354272
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1336837, upper bound: 0.1348236
time: 3.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1716689, 0.1731493
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2689010, 0.2730217
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1784123, 0.1748356
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1877989, 0.1864284
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2257302, 0.2256031
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1711920, 0.1723468
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1477515, 0.1487126
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2213421, 0.2205590
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1638135, 0.1620040
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1706235, 0.1729781

Time for backsubstitution: 22.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 909
type: DSZ, layer: 3, pos: 79
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2573

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1391

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1324948, upper bound: 0.1325430
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1297220, upper bound: 0.1353212
time: 3.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1732939, 0.1722870
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2699548, 0.2675259
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1697618, 0.1719804
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1853470, 0.1865982
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2264684, 0.2267261
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1672575, 0.1670565
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1485400, 0.1476915
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2123336, 0.2110244
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1488383, 0.1522030
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1731753, 0.1712348

Time for backsubstitution: 22.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 79
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 229
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 2573
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 909
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 2572

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1783

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1352797, upper bound: 0.1331629
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1357368, upper bound: 0.1327059
time: 3.16 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.92 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.92
Output dim: 0, lower bound: -0.1331129, upper bound: 0.1358457
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.92
Output dim: 0, lower bound: -0.1335059, upper bound: 0.1350935
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.92
Output dim: 0, lower bound: -0.1327712, upper bound: 0.1354272
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.92
Output dim: 0, lower bound: -0.1336837, upper bound: 0.1348236
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.92
Output dim: 0, lower bound: -0.1324948, upper bound: 0.1325430
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.92
Output dim: 0, lower bound: -0.1297220, upper bound: 0.1353212
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.92
Output dim: 0, lower bound: -0.1352797, upper bound: 0.1331629
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.92
Output dim: 0, lower bound: -0.1357368, upper bound: 0.1327059

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1721555, 0.1737268
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2667364, 0.2719938
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1736070, 0.1693099
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1863881, 0.1858373
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2262814, 0.2268033
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1663197, 0.1685719
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1478679, 0.1484057
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2105725, 0.2129118
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1523747, 0.1492195
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1713074, 0.1737114

Time for backsubstitution: 22.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1783
type: DSZ, layer: 3, pos: 229
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 2573
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 79
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 909
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 2572

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1783

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1325622, upper bound: 0.1357523
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1330194, upper bound: 0.1352953
time: 3.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1732315, 0.1722130
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2713777, 0.2688103
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1747849, 0.1770533
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1862720, 0.1874464
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2264774, 0.2268147
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1718738, 0.1714263
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1487113, 0.1478484
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2205653, 0.2214451
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1620358, 0.1635463
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1734493, 0.1715776

Time for backsubstitution: 22.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 79
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 229
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 909
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 2573
type: DSZ, layer: 3, pos: 2320

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1391

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1353364, upper bound: 0.1296772
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1325488, upper bound: 0.1323100
time: 3.10 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 29.05 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 29.05
Output dim: 0, lower bound: -0.1325622, upper bound: 0.1357523
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 29.05
Output dim: 0, lower bound: -0.1330194, upper bound: 0.1352953
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 29.05
Output dim: 0, lower bound: -0.1353364, upper bound: 0.1296772
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 29.05
Output dim: 0, lower bound: -0.1325488, upper bound: 0.1323100

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 9.8851662, 10.3085003, 9.8851662, 10.3085003, -0.1720768, 0.1736641
1: -16.8272705, -16.1140614, -16.8272705, -16.1140614, -0.2680193, 0.2733736
2: -4.3379030, -3.8939281, -4.3379030, -3.8939281, -0.1786799, 0.1742059
3: -10.4666405, -9.9474611, -10.4666405, -9.9474611, -0.1871911, 0.1867623
4: -10.2033596, -9.5819178, -10.2033596, -9.5819178, -0.2263471, 0.2268124
5: -2.6065121, -2.1945009, -2.6065121, -2.1945009, -0.1706872, 0.1730857
6: -2.0223312, -1.6953590, -2.0223312, -1.6953590, -0.1480157, 0.1485771
7: -7.7237086, -7.1458330, -7.7237086, -7.1458330, -0.2209883, 0.2209287
8: -0.9936709, -0.5890150, -0.9936709, -0.5890150, -0.1634592, 0.1624171
9: -4.8844619, -4.4187503, -4.8844619, -4.4187503, -0.1716499, 0.1739764

Time for backsubstitution: 22.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1391
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 79
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1830
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 229
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 1711
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 909
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2573

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1391

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1321664, upper bound: 0.1325646
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1295346, upper bound: 0.1353520
time: 3.23 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 29.28 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 29.28
Output dim: 0, lower bound: -0.1321664, upper bound: 0.1325646
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 29.28
Output dim: 0, lower bound: -0.1295346, upper bound: 0.1353520

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 56.20 + 467.02 = 523.23 seconds
