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
execution time: IAR + RelationalAnalysis = 22.87 + 33.52 = 56.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.1361274, upper bound: 0.1361275

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 439
type: B, layer: 1, pos: 439
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 919

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 439

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1347766, upper bound: 0.1361211
time: 2.93 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361202, upper bound: 0.1361212
time: 3.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.26 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.26
Output dim: 0, lower bound: -0.1347766, upper bound: 0.1361211
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.26
Output dim: 0, lower bound: -0.1361202, upper bound: 0.1361212

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 9.8866301, 10.3084927, 9.8851662, 10.3085003, -0.1717806, 0.1734595
1: -16.8232193, -16.1140671, -16.8272705, -16.1140614, -0.2683408, 0.2723644
2: -4.3379030, -3.8971777, -4.3379030, -3.8939281, -0.1778219, 0.1746024
3: -10.4664259, -9.9480820, -10.4666405, -9.9474611, -0.1871421, 0.1870700
4: -10.2023754, -9.5819178, -10.2033596, -9.5819178, -0.2259687, 0.2270229
5: -2.6050520, -2.1945989, -2.6065121, -2.1945009, -0.1715459, 0.1727026
6: -2.0218835, -1.6954622, -2.0223312, -1.6953590, -0.1482478, 0.1484858
7: -7.7236605, -7.1459236, -7.7237086, -7.1458330, -0.2212650, 0.2212594
8: -0.9936690, -0.5899405, -0.9936709, -0.5890150, -0.1635114, 0.1625874
9: -4.8826561, -4.4188504, -4.8844619, -4.4187503, -0.1716870, 0.1731349

Time for backsubstitution: 21.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 439
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 919

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 439

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1347765, upper bound: 0.1347766
time: 3.13 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1347765, upper bound: 0.1361211
time: 3.15 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 9.8845901, 10.3131332, 9.8851681, 10.3085003, -0.1739470, 0.1765223
1: -16.8276463, -16.0978241, -16.8272648, -16.1140614, -0.2741199, 0.2781390
2: -4.3522062, -3.8938241, -4.3379030, -3.8939345, -0.1809360, 0.1791936
3: -10.4690781, -9.9474525, -10.4666405, -9.9474611, -0.1896414, 0.1880084
4: -10.2038593, -9.5797558, -10.2033577, -9.5819178, -0.2274739, 0.2291837
5: -2.6065664, -2.1881423, -2.6065097, -2.1945009, -0.1738591, 0.1780593
6: -2.0223296, -1.6931822, -2.0223296, -1.6953595, -0.1488562, 0.1506590
7: -7.7240286, -7.1458330, -7.7237091, -7.1458340, -0.2216157, 0.2213815
8: -0.9981155, -0.5890112, -0.9936719, -0.5890164, -0.1667747, 0.1638697
9: -4.8844562, -4.4106555, -4.8844590, -4.4187512, -0.1742209, 0.1757801

Time for backsubstitution: 20.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 439
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 919

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 439

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361210, upper bound: 0.1347766
time: 3.52 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361210, upper bound: 0.1361212
time: 3.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.01 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 28.01
Output dim: 0, lower bound: -0.1347765, upper bound: 0.1347766
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.01
Output dim: 0, lower bound: -0.1347765, upper bound: 0.1361211
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.01
Output dim: 0, lower bound: -0.1361210, upper bound: 0.1347766
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.01
Output dim: 0, lower bound: -0.1361210, upper bound: 0.1361212

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 9.8866301, 10.3084927, 9.8845901, 10.3131332, -0.1748347, 0.1735560
1: -16.8232193, -16.1140671, -16.8276463, -16.0978241, -0.2740307, 0.2726331
2: -4.3379030, -3.8971777, -4.3522062, -3.8938241, -0.1779034, 0.1776735
3: -10.4664259, -9.9480820, -10.4690781, -9.9474525, -0.1871420, 0.1890088
4: -10.2023754, -9.5819178, -10.2038593, -9.5797558, -0.2281060, 0.2272531
5: -2.6050520, -2.1945989, -2.6065664, -2.1881423, -0.1765711, 0.1727226
6: -2.0218835, -1.6954622, -2.0223296, -1.6931822, -0.1502109, 0.1484848
7: -7.7236605, -7.1459236, -7.7240286, -7.1458330, -0.2212695, 0.2215266
8: -0.9936690, -0.5899405, -0.9981155, -0.5890112, -0.1635158, 0.1658479
9: -4.8826561, -4.4188504, -4.8844562, -4.4106555, -0.1739712, 0.1731311

Time for backsubstitution: 20.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 919

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1346623, upper bound: 0.1361210
time: 3.18 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1347764, upper bound: 0.1361210
time: 3.04 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 9.8845901, 10.3131332, 9.8866301, 10.3084927, -0.1735560, 0.1748346
1: -16.8276463, -16.0978241, -16.8232193, -16.1140671, -0.2726332, 0.2740307
2: -4.3522062, -3.8938241, -4.3379030, -3.8971777, -0.1776735, 0.1779033
3: -10.4690781, -9.9474525, -10.4664259, -9.9480820, -0.1890088, 0.1871420
4: -10.2038593, -9.5797558, -10.2023754, -9.5819178, -0.2272531, 0.2281060
5: -2.6065664, -2.1881423, -2.6050520, -2.1945989, -0.1727226, 0.1765711
6: -2.0223296, -1.6931822, -2.0218835, -1.6954622, -0.1484848, 0.1502109
7: -7.7240286, -7.1458330, -7.7236605, -7.1459236, -0.2215266, 0.2212696
8: -0.9981155, -0.5890112, -0.9936690, -0.5899405, -0.1658479, 0.1635157
9: -4.8844562, -4.4106555, -4.8826561, -4.4188504, -0.1731311, 0.1739712

Time for backsubstitution: 20.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 919

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361201, upper bound: 0.1346623
time: 3.12 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361201, upper bound: 0.1347765
time: 3.27 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 9.8845901, 10.3131332, 9.8845901, 10.3131332, -0.1740576, 0.1740575
1: -16.8276463, -16.0978241, -16.8276463, -16.0978241, -0.2743678, 0.2743678
2: -4.3522062, -3.8938241, -4.3522062, -3.8938241, -0.1791979, 0.1791979
3: -10.4690781, -9.9474525, -10.4690781, -9.9474525, -0.1880956, 0.1880955
4: -10.2038593, -9.5797558, -10.2038593, -9.5797558, -0.2274739, 0.2274737
5: -2.6065664, -2.1881423, -2.6065664, -2.1881423, -0.1739713, 0.1739713
6: -2.0223296, -1.6931822, -2.0223296, -1.6931822, -0.1492341, 0.1492341
7: -7.7240286, -7.1458330, -7.7240286, -7.1458330, -0.2214365, 0.2214365
8: -0.9981155, -0.5890112, -0.9981155, -0.5890112, -0.1643060, 0.1643060
9: -4.8844562, -4.4106555, -4.8844562, -4.4106555, -0.1743119, 0.1743119

Time for backsubstitution: 20.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 919

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1360067, upper bound: 0.1347770
time: 3.04 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361209, upper bound: 0.1347770
time: 3.04 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 26.95 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.95
Output dim: 0, lower bound: -0.1346623, upper bound: 0.1361210
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.95
Output dim: 0, lower bound: -0.1347764, upper bound: 0.1361210
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 26.95
Output dim: 0, lower bound: -0.1361201, upper bound: 0.1346623
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 26.95
Output dim: 0, lower bound: -0.1361201, upper bound: 0.1347765
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.95
Output dim: 0, lower bound: -0.1360067, upper bound: 0.1347770
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.95
Output dim: 0, lower bound: -0.1361209, upper bound: 0.1347770

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 9.8867626, 10.3084917, 9.8845968, 10.3131332, -0.1746907, 0.1735464
1: -16.8225574, -16.1140671, -16.8276253, -16.0978241, -0.2733657, 0.2726104
2: -4.3378911, -3.8974471, -4.3522043, -3.8938336, -0.1777523, 0.1772563
3: -10.4659634, -9.9480867, -10.4690609, -9.9474525, -0.1866237, 0.1889439
4: -10.2022600, -9.5819168, -10.2038584, -9.5797558, -0.2279551, 0.2272239
5: -2.6042910, -2.1946039, -2.6065402, -2.1881433, -0.1757972, 0.1726952
6: -2.0218835, -1.6956456, -2.0223296, -1.6931891, -0.1502049, 0.1483033
7: -7.7230577, -7.1459236, -7.7240086, -7.1458330, -0.2206625, 0.2215061
8: -0.9935389, -0.5899734, -0.9981127, -0.5890121, -0.1631064, 0.1655581
9: -4.8826561, -4.4190273, -4.8844566, -4.4106627, -0.1739532, 0.1729431

Time for backsubstitution: 21.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 4656

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 919

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1346615, upper bound: 0.1353799
time: 3.95 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1345165, upper bound: 0.1359740
time: 4.43 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 9.8865957, 10.3088903, 9.8845901, 10.3131332, -0.1748960, 0.1739440
1: -16.8233147, -16.1121750, -16.8276443, -16.0978241, -0.2740736, 0.2745242
2: -4.3388824, -3.8971739, -4.3522058, -3.8938248, -0.1786333, 0.1776652
3: -10.4664555, -9.9467010, -10.4690752, -9.9474525, -0.1873316, 0.1903157
4: -10.2024193, -9.5815201, -10.2038622, -9.5797558, -0.2282579, 0.2276083
5: -2.6051927, -2.1923990, -2.6065643, -2.1881440, -0.1765445, 0.1749382
6: -2.0224047, -1.6954460, -2.0223296, -1.6931828, -0.1507329, 0.1485057
7: -7.7237139, -7.1442108, -7.7240267, -7.1458330, -0.2212673, 0.2232405
8: -0.9936676, -0.5893126, -0.9981170, -0.5890112, -0.1642585, 0.1656296
9: -4.8831630, -4.4188371, -4.8844576, -4.4106579, -0.1739651, 0.1732022

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 4656

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 919

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1347756, upper bound: 0.1353798
time: 4.06 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1346306, upper bound: 0.1359740
time: 5.55 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: 9.8845968, 10.3131332, 9.8867626, 10.3084917, -0.1735464, 0.1746907
1: -16.8276253, -16.0978241, -16.8225574, -16.1140671, -0.2726103, 0.2733657
2: -4.3522043, -3.8938336, -4.3378911, -3.8974471, -0.1772563, 0.1777521
3: -10.4690609, -9.9474525, -10.4659634, -9.9480867, -0.1889439, 0.1866238
4: -10.2038584, -9.5797558, -10.2022600, -9.5819168, -0.2272239, 0.2279552
5: -2.6065402, -2.1881433, -2.6042910, -2.1946039, -0.1726950, 0.1757972
6: -2.0223296, -1.6931891, -2.0218835, -1.6956456, -0.1483033, 0.1502049
7: -7.7240086, -7.1458330, -7.7230577, -7.1459236, -0.2215061, 0.2206625
8: -0.9981127, -0.5890121, -0.9935389, -0.5899734, -0.1655581, 0.1631066
9: -4.8844566, -4.4106627, -4.8826561, -4.4190273, -0.1729431, 0.1739533

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 4656

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 919

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1353798, upper bound: 0.1346617
time: 3.35 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1359740, upper bound: 0.1345166
time: 3.32 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: 9.8845901, 10.3131332, 9.8865957, 10.3088903, -0.1739442, 0.1748960
1: -16.8276443, -16.0978241, -16.8233147, -16.1121750, -0.2745242, 0.2740737
2: -4.3522058, -3.8938248, -4.3388824, -3.8971739, -0.1776652, 0.1786336
3: -10.4690752, -9.9474525, -10.4664555, -9.9467010, -0.1903156, 0.1873316
4: -10.2038622, -9.5797558, -10.2024193, -9.5815201, -0.2276083, 0.2282579
5: -2.6065643, -2.1881440, -2.6051927, -2.1923990, -0.1749382, 0.1765445
6: -2.0223296, -1.6931828, -2.0224047, -1.6954460, -0.1485057, 0.1507328
7: -7.7240267, -7.1458330, -7.7237139, -7.1442108, -0.2232406, 0.2212672
8: -0.9981170, -0.5890112, -0.9936676, -0.5893126, -0.1656296, 0.1642585
9: -4.8844576, -4.4106579, -4.8831630, -4.4188371, -0.1732022, 0.1739651

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 4656

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 919

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1353797, upper bound: 0.1347758
time: 3.41 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1359740, upper bound: 0.1346307
time: 3.30 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 9.8847237, 10.3131332, 9.8845968, 10.3131332, -0.1739138, 0.1740477
1: -16.8269806, -16.0978241, -16.8276253, -16.0978241, -0.2737048, 0.2743456
2: -4.3521938, -3.8940923, -4.3522043, -3.8938336, -0.1790466, 0.1787838
3: -10.4686146, -9.9474592, -10.4690609, -9.9474525, -0.1875774, 0.1880305
4: -10.2037468, -9.5797558, -10.2038584, -9.5797558, -0.2273232, 0.2274446
5: -2.6058052, -2.1881454, -2.6065402, -2.1881433, -0.1731987, 0.1739438
6: -2.0223296, -1.6933668, -2.0223296, -1.6931891, -0.1492279, 0.1490523
7: -7.7234259, -7.1458330, -7.7240086, -7.1458330, -0.2208295, 0.2214161
8: -0.9979882, -0.5890450, -0.9981127, -0.5890121, -0.1638970, 0.1640216
9: -4.8844562, -4.4108310, -4.8844566, -4.4106627, -0.1742961, 0.1741242

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 4656

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 919

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1352665, upper bound: 0.1347762
time: 3.20 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1358607, upper bound: 0.1346311
time: 3.06 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 9.8845539, 10.3135300, 9.8845901, 10.3131332, -0.1741633, 0.1744454
1: -16.8277397, -16.0959358, -16.8276443, -16.0978241, -0.2744045, 0.2762591
2: -4.3531857, -3.8938181, -4.3522058, -3.8938248, -0.1799277, 0.1795810
3: -10.4691057, -9.9460716, -10.4690752, -9.9474525, -0.1882854, 0.1894023
4: -10.2039042, -9.5793552, -10.2038622, -9.5797558, -0.2276261, 0.2278289
5: -2.6067071, -2.1859429, -2.6065643, -2.1881440, -0.1739420, 0.1761869
6: -2.0228508, -1.6931676, -2.0223296, -1.6931828, -0.1497562, 0.1492553
7: -7.7240829, -7.1441221, -7.7240267, -7.1458330, -0.2214346, 0.2231508
8: -0.9981155, -0.5883837, -0.9981170, -0.5890112, -0.1650488, 0.1646022
9: -4.8849640, -4.4106426, -4.8844576, -4.4106579, -0.1748089, 0.1743830

Time for backsubstitution: 21.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 4656

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 919

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1353797, upper bound: 0.1347762
time: 3.30 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1359740, upper bound: 0.1346311
time: 3.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.14 seconds
NS_A1_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 28.14
Output dim: 0, lower bound: -0.1346615, upper bound: 0.1353799
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 28.14
Output dim: 0, lower bound: -0.1345165, upper bound: 0.1359740
NS_A1_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 28.14
Output dim: 0, lower bound: -0.1347756, upper bound: 0.1353798
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 28.14
Output dim: 0, lower bound: -0.1346306, upper bound: 0.1359740
NS_A2_B1_B1_B1, status: Status.VERIFIED, split count: 4, time: 28.14
Output dim: 0, lower bound: -0.1353798, upper bound: 0.1346617
NS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 28.14
Output dim: 0, lower bound: -0.1359740, upper bound: 0.1345166
NS_A2_B1_B2_B1, status: Status.VERIFIED, split count: 4, time: 28.14
Output dim: 0, lower bound: -0.1353797, upper bound: 0.1347758
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 28.14
Output dim: 0, lower bound: -0.1359740, upper bound: 0.1346307
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 28.14
Output dim: 0, lower bound: -0.1352665, upper bound: 0.1347762
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.14
Output dim: 0, lower bound: -0.1358607, upper bound: 0.1346311
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 28.14
Output dim: 0, lower bound: -0.1353797, upper bound: 0.1347762
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.14
Output dim: 0, lower bound: -0.1359740, upper bound: 0.1346311

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: 9.8862791, 10.3084927, 9.8845959, 10.3131332, -0.1747442, 0.1736495
1: -16.8225861, -16.1124649, -16.8276215, -16.0978241, -0.2734861, 0.2742105
2: -4.3396182, -3.8974459, -4.3522043, -3.8938341, -0.1794505, 0.1773071
3: -10.4659853, -9.9480515, -10.4690590, -9.9474525, -0.1866961, 0.1889588
4: -10.2022848, -9.5795574, -10.2038565, -9.5797567, -0.2283567, 0.2295396
5: -2.6042871, -2.1933615, -2.6065388, -2.1881430, -0.1758851, 0.1739453
6: -2.0230141, -1.6956381, -2.0223296, -1.6931896, -0.1512448, 0.1484562
7: -7.7230577, -7.1445084, -7.7240086, -7.1458330, -0.2207965, 0.2229164
8: -0.9935379, -0.5891895, -0.9981112, -0.5890126, -0.1631613, 0.1655594
9: -4.8826537, -4.4185982, -4.8844552, -4.4106627, -0.1739827, 0.1733853

Time for backsubstitution: 21.33 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2133
type: A, layer: 3, pos: 2133
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2572
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2573
type: A, layer: 3, pos: 2573
type: A, layer: 3, pos: 2865
type: B, layer: 3, pos: 2865
type: B, layer: 3, pos: 1391
type: A, layer: 3, pos: 1391
type: B, layer: 3, pos: 1830
type: A, layer: 3, pos: 1830
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 2909
type: A, layer: 3, pos: 2909
type: A, layer: 3, pos: 229
type: B, layer: 3, pos: 229
type: B, layer: 3, pos: 1711
type: A, layer: 3, pos: 1711
type: B, layer: 3, pos: 79
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 1783
type: B, layer: 3, pos: 1783
type: B, layer: 3, pos: 305
type: A, layer: 3, pos: 305
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 909
type: B, layer: 3, pos: 909

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 2133

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1291372, upper bound: 0.1195668
time: 8.22 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1291372, upper bound: 0.1305943
time: 4.70 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: 9.8861094, 10.3088894, 9.8845901, 10.3131351, -0.1749496, 0.1740470
1: -16.8233414, -16.1105766, -16.8276405, -16.0978241, -0.2741945, 0.2761230
2: -4.3406105, -3.8971725, -4.3522067, -3.8938265, -0.1803305, 0.1777967
3: -10.4664774, -9.9466648, -10.4690752, -9.9474535, -0.1874040, 0.1903297
4: -10.2024441, -9.5791578, -10.2038584, -9.5797558, -0.2286595, 0.2299232
5: -2.6051893, -2.1911569, -2.6065636, -2.1881430, -0.1766326, 0.1761880
6: -2.0235355, -1.6954401, -2.0223296, -1.6931834, -0.1512529, 0.1486586
7: -7.7237158, -7.1427951, -7.7240248, -7.1458330, -0.2214018, 0.2234456
8: -0.9936657, -0.5885291, -0.9981160, -0.5890107, -0.1643136, 0.1656308
9: -4.8831625, -4.4184098, -4.8844571, -4.4106560, -0.1739946, 0.1736445

Time for backsubstitution: 21.66 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2133
type: A, layer: 3, pos: 2133
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2572
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2573
type: A, layer: 3, pos: 2573
type: A, layer: 3, pos: 2865
type: B, layer: 3, pos: 2865
type: B, layer: 3, pos: 1391
type: A, layer: 3, pos: 1391
type: B, layer: 3, pos: 1830
type: A, layer: 3, pos: 1830
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 2909
type: A, layer: 3, pos: 2909
type: A, layer: 3, pos: 229
type: B, layer: 3, pos: 229
type: B, layer: 3, pos: 1711
type: A, layer: 3, pos: 1711
type: B, layer: 3, pos: 79
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 1783
type: B, layer: 3, pos: 1783
type: B, layer: 3, pos: 305
type: A, layer: 3, pos: 305
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 909
type: B, layer: 3, pos: 909

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 2133

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1292513, upper bound: 0.1195667
time: 3.81 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1292513, upper bound: 0.1305945
time: 3.31 seconds

## BFS NS instance: NS_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: 9.8845959, 10.3131332, 9.8862791, 10.3084927, -0.1736495, 0.1747442
1: -16.8276215, -16.0978241, -16.8225861, -16.1124649, -0.2742105, 0.2734863
2: -4.3522043, -3.8938341, -4.3396182, -3.8974459, -0.1773071, 0.1794504
3: -10.4690590, -9.9474525, -10.4659853, -9.9480515, -0.1889588, 0.1866961
4: -10.2038565, -9.5797567, -10.2022848, -9.5795574, -0.2295396, 0.2283567
5: -2.6065388, -2.1881430, -2.6042871, -2.1933615, -0.1739454, 0.1758851
6: -2.0223296, -1.6931896, -2.0230141, -1.6956381, -0.1484562, 0.1512448
7: -7.7240086, -7.1458330, -7.7230577, -7.1445084, -0.2229165, 0.2207965
8: -0.9981112, -0.5890126, -0.9935379, -0.5891895, -0.1655594, 0.1631612
9: -4.8844552, -4.4106627, -4.8826537, -4.4185982, -0.1733853, 0.1739827

Time for backsubstitution: 21.51 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2133
type: B, layer: 3, pos: 2133
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2572
type: A, layer: 3, pos: 2573
type: B, layer: 3, pos: 2573
type: B, layer: 3, pos: 2865
type: A, layer: 3, pos: 2865
type: A, layer: 3, pos: 1391
type: B, layer: 3, pos: 1391
type: A, layer: 3, pos: 1830
type: B, layer: 3, pos: 1830
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 2909
type: B, layer: 3, pos: 2909
type: B, layer: 3, pos: 229
type: A, layer: 3, pos: 229
type: A, layer: 3, pos: 1711
type: B, layer: 3, pos: 1711
type: A, layer: 3, pos: 79
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 1783
type: A, layer: 3, pos: 1783
type: A, layer: 3, pos: 305
type: B, layer: 3, pos: 305
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 909
type: A, layer: 3, pos: 909

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2133

## Relational analysis of NS_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1195666, upper bound: 0.1291373
time: 3.24 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1305943, upper bound: 0.1291373
time: 3.47 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: 9.8845901, 10.3131351, 9.8861094, 10.3088894, -0.1740469, 0.1749495
1: -16.8276405, -16.0978241, -16.8233414, -16.1105766, -0.2761232, 0.2741945
2: -4.3522067, -3.8938265, -4.3406105, -3.8971725, -0.1777967, 0.1803305
3: -10.4690752, -9.9474535, -10.4664774, -9.9466648, -0.1903296, 0.1874040
4: -10.2038584, -9.5797558, -10.2024441, -9.5791578, -0.2299232, 0.2286596
5: -2.6065636, -2.1881430, -2.6051893, -2.1911569, -0.1761880, 0.1766326
6: -2.0223296, -1.6931834, -2.0235355, -1.6954401, -0.1486586, 0.1512529
7: -7.7240248, -7.1458330, -7.7237158, -7.1427951, -0.2234456, 0.2214017
8: -0.9981160, -0.5890107, -0.9936657, -0.5885291, -0.1656308, 0.1643134
9: -4.8844571, -4.4106560, -4.8831625, -4.4184098, -0.1736445, 0.1739946

Time for backsubstitution: 21.50 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2133
type: B, layer: 3, pos: 2133
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2572
type: A, layer: 3, pos: 2573
type: B, layer: 3, pos: 2573
type: B, layer: 3, pos: 2865
type: A, layer: 3, pos: 2865
type: A, layer: 3, pos: 1391
type: B, layer: 3, pos: 1391
type: A, layer: 3, pos: 1830
type: B, layer: 3, pos: 1830
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 2909
type: B, layer: 3, pos: 2909
type: B, layer: 3, pos: 229
type: A, layer: 3, pos: 229
type: A, layer: 3, pos: 1711
type: B, layer: 3, pos: 1711
type: A, layer: 3, pos: 79
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 1783
type: A, layer: 3, pos: 1783
type: A, layer: 3, pos: 305
type: B, layer: 3, pos: 305
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 909
type: A, layer: 3, pos: 909

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 2133

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1195666, upper bound: 0.1292515
time: 3.25 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1305943, upper bound: 0.1292515
time: 3.12 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 9.8847237, 10.3131332, 9.8841105, 10.3131332, -0.1740170, 0.1745571
1: -16.8269806, -16.0978241, -16.8276539, -16.0962276, -0.2753049, 0.2745984
2: -4.3521938, -3.8940930, -4.3539310, -3.8938317, -0.1792582, 0.1804816
3: -10.4686146, -9.9474592, -10.4690838, -9.9474163, -0.1875921, 0.1881033
4: -10.2037420, -9.5797558, -10.2038813, -9.5773945, -0.2296383, 0.2278448
5: -2.6058044, -2.1881454, -2.6065378, -2.1868999, -0.1744494, 0.1740561
6: -2.0223296, -1.6933663, -2.0234590, -1.6931829, -0.1493807, 0.1501767
7: -7.7234282, -7.1458330, -7.7240095, -7.1444173, -0.2222401, 0.2215502
8: -0.9979877, -0.5890455, -0.9981103, -0.5882277, -0.1646807, 0.1640764
9: -4.8844557, -4.4108310, -4.8844571, -4.4102340, -0.1747384, 0.1742448

Time for backsubstitution: 21.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 919

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1358599, upper bound: 0.1345170
time: 3.17 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1358599, upper bound: 0.1346311
time: 3.12 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 9.8845558, 10.3135319, 9.8841076, 10.3131323, -0.1742663, 0.1749545
1: -16.8277416, -16.0959358, -16.8276730, -16.0962276, -0.2760044, 0.2765121
2: -4.3531866, -3.8938198, -4.3539314, -3.8938241, -0.1801395, 0.1810647
3: -10.4691048, -9.9460716, -10.4690971, -9.9474163, -0.1883004, 0.1894749
4: -10.2038994, -9.5793562, -10.2038860, -9.5773945, -0.2299414, 0.2282289
5: -2.6067052, -2.1859410, -2.6065605, -2.1869016, -0.1751924, 0.1762991
6: -2.0228508, -1.6931690, -2.0234590, -1.6931760, -0.1499089, 0.1503795
7: -7.7240791, -7.1441221, -7.7240276, -7.1444173, -0.2228452, 0.2232852
8: -0.9981165, -0.5883851, -0.9981155, -0.5882273, -0.1658328, 0.1646570
9: -4.8849635, -4.4106426, -4.8844552, -4.4102283, -0.1752512, 0.1745040

Time for backsubstitution: 21.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 4656

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 919

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1359741, upper bound: 0.1340366
time: 3.29 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1359741, upper bound: 0.1346311
time: 3.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 28.20 seconds
NS_A1_B2_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 28.20
Output dim: 0, lower bound: -0.1291372, upper bound: 0.1195668
NS_A1_B2_A1_A2_B2, status: Status.VERIFIED, split count: 5, time: 28.20
Output dim: 0, lower bound: -0.1291372, upper bound: 0.1305943
NS_A1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 28.20
Output dim: 0, lower bound: -0.1292513, upper bound: 0.1195667
NS_A1_B2_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 28.20
Output dim: 0, lower bound: -0.1292513, upper bound: 0.1305945
NS_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 28.20
Output dim: 0, lower bound: -0.1195666, upper bound: 0.1291373
NS_A2_B1_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 28.20
Output dim: 0, lower bound: -0.1305943, upper bound: 0.1291373
NS_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 28.20
Output dim: 0, lower bound: -0.1195666, upper bound: 0.1292515
NS_A2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 28.20
Output dim: 0, lower bound: -0.1305943, upper bound: 0.1292515
NS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 28.20
Output dim: 0, lower bound: -0.1358599, upper bound: 0.1345170
NS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 28.20
Output dim: 0, lower bound: -0.1358599, upper bound: 0.1346311
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.20
Output dim: 0, lower bound: -0.1359741, upper bound: 0.1340366
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.20
Output dim: 0, lower bound: -0.1359741, upper bound: 0.1346311

## BFS NS instance: NS_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 9.8847237, 10.3131332, 9.8842392, 10.3131323, -0.1740122, 0.1744182
1: -16.8269806, -16.0978241, -16.8270149, -16.0962276, -0.2753049, 0.2739574
2: -4.3521938, -3.8940930, -4.3539209, -3.8940909, -0.1788623, 0.1803491
3: -10.4686146, -9.9474592, -10.4686356, -9.9474220, -0.1875463, 0.1876042
4: -10.2037420, -9.5797558, -10.2037725, -9.5773964, -0.2296154, 0.2277001
5: -2.6058044, -2.1881454, -2.6058006, -2.1869035, -0.1744484, 0.1733099
6: -2.0223296, -1.6933663, -2.0234590, -1.6933593, -0.1492052, 0.1501767
7: -7.7234282, -7.1458330, -7.7234278, -7.1444173, -0.2222401, 0.2209632
8: -0.9979877, -0.5890455, -0.9979873, -0.5882607, -0.1644188, 0.1636901
9: -4.8844557, -4.4108310, -4.8844543, -4.4104033, -0.1745571, 0.1742356

Time for backsubstitution: 21.55 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2133
type: A, layer: 3, pos: 2133
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2572
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2573
type: A, layer: 3, pos: 2573
type: A, layer: 3, pos: 2865
type: B, layer: 3, pos: 2865
type: A, layer: 3, pos: 1391
type: B, layer: 3, pos: 1391
type: A, layer: 3, pos: 1830
type: B, layer: 3, pos: 1830
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 2909
type: A, layer: 3, pos: 2909
type: B, layer: 3, pos: 229
type: A, layer: 3, pos: 229
type: B, layer: 3, pos: 1711
type: A, layer: 3, pos: 1711
type: B, layer: 3, pos: 79
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 1783
type: B, layer: 3, pos: 1783
type: B, layer: 3, pos: 305
type: A, layer: 3, pos: 305
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 909
type: B, layer: 3, pos: 909

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 2133

## Relational analysis of NS_A2_B2_A1_B2_B1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1304807, upper bound: 0.1181096
time: 3.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_B2

### Relational analysis result of NS_A2_B2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1304815, upper bound: 0.1291373
time: 3.22 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 9.8847237, 10.3131332, 9.8840694, 10.3135319, -0.1744052, 0.1746125
1: -16.8269806, -16.0978241, -16.8277721, -16.0943356, -0.2771972, 0.2746841
2: -4.3521938, -3.8940930, -4.3549132, -3.8938179, -0.1791419, 0.1805503
3: -10.4686146, -9.9474592, -10.4691257, -9.9460325, -0.1889009, 0.1881170
4: -10.2037420, -9.5797558, -10.2039280, -9.5769987, -0.2299945, 0.2278723
5: -2.6058044, -2.1881454, -2.6067028, -2.1847003, -0.1766666, 0.1740817
6: -2.0223296, -1.6933663, -2.0239801, -1.6931609, -0.1494156, 0.1506984
7: -7.7234282, -7.1458330, -7.7240834, -7.1427045, -0.2229805, 0.2215861
8: -0.9979877, -0.5890455, -0.9981151, -0.5875993, -0.1649868, 0.1638179
9: -4.8844557, -4.4108310, -4.8849635, -4.4102144, -0.1747634, 0.1747422

Time for backsubstitution: 21.52 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.39 + 545.69 = 602.09 seconds
