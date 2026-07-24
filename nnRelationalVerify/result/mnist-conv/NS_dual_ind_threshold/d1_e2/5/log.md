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
execution time: IAR + RelationalAnalysis = 22.85 + 32.84 = 55.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.1361274, upper bound: 0.1361275

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 439
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 919

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 439

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1347766, upper bound: 0.1361211
time: 2.97 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361202, upper bound: 0.1361212
time: 3.17 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.29 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.29
Output dim: 0, lower bound: -0.1347766, upper bound: 0.1361211
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.29
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

Time for backsubstitution: 20.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 439
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 919

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 439

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1347765, upper bound: 0.1347766
time: 2.95 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1347765, upper bound: 0.1361211
time: 2.94 seconds

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

Time for backsubstitution: 20.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 439
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 919

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 439

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361210, upper bound: 0.1347766
time: 3.18 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361210, upper bound: 0.1361212
time: 3.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 26.99 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 26.99
Output dim: 0, lower bound: -0.1347765, upper bound: 0.1347766
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.99
Output dim: 0, lower bound: -0.1347765, upper bound: 0.1361211
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.99
Output dim: 0, lower bound: -0.1361210, upper bound: 0.1347766
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.99
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

Time for backsubstitution: 19.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 919

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1346623, upper bound: 0.1361210
time: 3.11 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1347764, upper bound: 0.1361210
time: 3.01 seconds

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

Time for backsubstitution: 20.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 919

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1360059, upper bound: 0.1347765
time: 3.04 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361200, upper bound: 0.1347765
time: 3.01 seconds

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
type: A, layer: 1, pos: 919

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1360067, upper bound: 0.1347770
time: 3.01 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1361209, upper bound: 0.1347770
time: 2.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 26.83 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.83
Output dim: 0, lower bound: -0.1346623, upper bound: 0.1361210
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.83
Output dim: 0, lower bound: -0.1347764, upper bound: 0.1361210
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.83
Output dim: 0, lower bound: -0.1360059, upper bound: 0.1347765
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.83
Output dim: 0, lower bound: -0.1361200, upper bound: 0.1347765
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.83
Output dim: 0, lower bound: -0.1360067, upper bound: 0.1347770
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.83
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

Time for backsubstitution: 20.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 919

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1346623, upper bound: 0.1360060
time: 3.11 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1346623, upper bound: 0.1361202
time: 3.14 seconds

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

Time for backsubstitution: 20.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 4656

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 919

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1340361, upper bound: 0.1361194
time: 3.18 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1346304, upper bound: 0.1359743
time: 3.14 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 9.8847237, 10.3131332, 9.8866348, 10.3084927, -0.1734126, 0.1748231
1: -16.8269806, -16.0978241, -16.8231964, -16.1140671, -0.2719699, 0.2740004
2: -4.3521938, -3.8940923, -4.3379021, -3.8971872, -0.1775171, 0.1774890
3: -10.4686146, -9.9474592, -10.4664116, -9.9480839, -0.1884904, 0.1870770
4: -10.2037468, -9.5797558, -10.2023726, -9.5819159, -0.2271026, 0.2280769
5: -2.6058052, -2.1881454, -2.6050262, -2.1946001, -0.1719505, 0.1765351
6: -2.0223296, -1.6933668, -2.0218835, -1.6954695, -0.1484787, 0.1500288
7: -7.7234259, -7.1458330, -7.7236404, -7.1459236, -0.2209194, 0.2212491
8: -0.9979882, -0.5890450, -0.9936643, -0.5899420, -0.1654345, 0.1632313
9: -4.8844562, -4.4108310, -4.8826571, -4.4188571, -0.1731153, 0.1737828

Time for backsubstitution: 20.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 919

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1360059, upper bound: 0.1346624
time: 3.02 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1360059, upper bound: 0.1347766
time: 3.13 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 9.8845539, 10.3135300, 9.8866291, 10.3084927, -0.1736620, 0.1748322
1: -16.8277397, -16.0959358, -16.8232193, -16.1140671, -0.2726699, 0.2740304
2: -4.3531857, -3.8938181, -4.3379030, -3.8971801, -0.1775622, 0.1782866
3: -10.4691057, -9.9460716, -10.4664240, -9.9480820, -0.1891961, 0.1884488
4: -10.2039042, -9.5793552, -10.2023783, -9.5819159, -0.2274055, 0.2284611
5: -2.6067071, -2.1859429, -2.6050501, -2.1946008, -0.1726935, 0.1765985
6: -2.0228508, -1.6931676, -2.0218835, -1.6954635, -0.1490068, 0.1502309
7: -7.7240829, -7.1441221, -7.7236614, -7.1459236, -0.2215240, 0.2229838
8: -0.9981155, -0.5883837, -0.9936676, -0.5899405, -0.1658297, 0.1638117
9: -4.8849640, -4.4106426, -4.8826561, -4.4188509, -0.1736279, 0.1739990

Time for backsubstitution: 20.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 4656

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 919

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1353797, upper bound: 0.1347758
time: 3.19 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1359740, upper bound: 0.1346307
time: 3.08 seconds

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

Time for backsubstitution: 20.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 919

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1360059, upper bound: 0.1346627
time: 3.13 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1360059, upper bound: 0.1347769
time: 3.22 seconds

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

Time for backsubstitution: 20.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 4656

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 919

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1353797, upper bound: 0.1347762
time: 3.24 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1359740, upper bound: 0.1346311
time: 3.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 27.14 seconds
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 0, lower bound: -0.1346623, upper bound: 0.1360060
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 0, lower bound: -0.1346623, upper bound: 0.1361202
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 0, lower bound: -0.1340361, upper bound: 0.1361194
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 0, lower bound: -0.1346304, upper bound: 0.1359743
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 0, lower bound: -0.1360059, upper bound: 0.1346624
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 0, lower bound: -0.1360059, upper bound: 0.1347766
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 27.14
Output dim: 0, lower bound: -0.1353797, upper bound: 0.1347758
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 0, lower bound: -0.1359740, upper bound: 0.1346307
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 0, lower bound: -0.1360059, upper bound: 0.1346627
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 0, lower bound: -0.1360059, upper bound: 0.1347769
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 27.14
Output dim: 0, lower bound: -0.1353797, upper bound: 0.1347762
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 0, lower bound: -0.1359740, upper bound: 0.1346311

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 9.8867626, 10.3084917, 9.8847237, 10.3131332, -0.1746860, 0.1734078
1: -16.8225574, -16.1140671, -16.8269806, -16.0978241, -0.2733657, 0.2719698
2: -4.3378911, -3.8974471, -4.3521938, -3.8940923, -0.1773562, 0.1771246
3: -10.4659634, -9.9480867, -10.4686146, -9.9474592, -0.1865779, 0.1884445
4: -10.2022600, -9.5819168, -10.2037468, -9.5797558, -0.2279319, 0.2270793
5: -2.6042910, -2.1946039, -2.6058052, -2.1881454, -0.1757960, 0.1719493
6: -2.0218835, -1.6956456, -2.0223296, -1.6933668, -0.1500288, 0.1483033
7: -7.7230577, -7.1459236, -7.7234259, -7.1458330, -0.2206625, 0.2209193
8: -0.9935389, -0.5899734, -0.9979882, -0.5890450, -0.1628447, 0.1651747
9: -4.8826561, -4.4190273, -4.8844562, -4.4108310, -0.1737737, 0.1729337

Time for backsubstitution: 19.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 919

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 919

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1346615, upper bound: 0.1352661
time: 3.25 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1345165, upper bound: 0.1358601
time: 3.69 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 9.8867626, 10.3084917, 9.8845539, 10.3135300, -0.1746883, 0.1736021
1: -16.8225574, -16.1140671, -16.8277397, -16.0959358, -0.2733657, 0.2726955
2: -4.3378911, -3.8974471, -4.3531857, -3.8938181, -0.1776357, 0.1771512
3: -10.4659634, -9.9480867, -10.4691057, -9.9460716, -0.1879334, 0.1889551
4: -10.2022600, -9.5819168, -10.2039042, -9.5793552, -0.2283114, 0.2272513
5: -2.6042910, -2.1946039, -2.6067071, -2.1859429, -0.1758247, 0.1727203
6: -2.0218835, -1.6956456, -2.0228508, -1.6931676, -0.1502382, 0.1488257
7: -7.7230577, -7.1459236, -7.7240829, -7.1441221, -0.2223783, 0.2215408
8: -0.9935389, -0.5899734, -0.9981155, -0.5883837, -0.1634123, 0.1652988
9: -4.8826561, -4.4190273, -4.8849640, -4.4106426, -0.1739743, 0.1734406

Time for backsubstitution: 24.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 919

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 919

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1346615, upper bound: 0.1353799
time: 3.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1345165, upper bound: 0.1359740
time: 4.35 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 9.8865957, 10.3088903, 9.8845940, 10.3130322, -0.1747901, 0.1739438
1: -16.8233147, -16.1121750, -16.8272667, -16.0978241, -0.2740736, 0.2742212
2: -4.3388824, -3.8971739, -4.3522043, -3.8942344, -0.1782894, 0.1776637
3: -10.4664555, -9.9467010, -10.4690704, -9.9474535, -0.1873311, 0.1903152
4: -10.2024193, -9.5815201, -10.2033024, -9.5797558, -0.2282549, 0.2271997
5: -2.6051927, -2.1923990, -2.6062729, -2.1881452, -0.1765438, 0.1746681
6: -2.0224047, -1.6954460, -2.0223296, -1.6934505, -0.1505083, 0.1485053
7: -7.7237139, -7.1442108, -7.7236891, -7.1458340, -0.2212666, 0.2229228
8: -0.9936676, -0.5893126, -0.9979296, -0.5890112, -0.1642586, 0.1654425
9: -4.8831630, -4.4188371, -4.8843622, -4.4106598, -0.1739640, 0.1731806

Time for backsubstitution: 21.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 919

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 919

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1340361, upper bound: 0.1353798
time: 5.32 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1340361, upper bound: 0.1359743
time: 3.08 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 9.8865967, 10.3088894, 9.8841076, 10.3131323, -0.1749149, 0.1744531
1: -16.8233147, -16.1121750, -16.8276730, -16.0962276, -0.2740738, 0.2747769
2: -4.3388824, -3.8971753, -4.3539314, -3.8938241, -0.1788452, 0.1776688
3: -10.4664545, -9.9467010, -10.4690971, -9.9474163, -0.1873463, 0.1903867
4: -10.2024155, -9.5815191, -10.2038860, -9.5773945, -0.2297380, 0.2280085
5: -2.6051905, -2.1923990, -2.6065605, -2.1869016, -0.1765642, 0.1750506
6: -2.0224047, -1.6954463, -2.0234590, -1.6931760, -0.1508847, 0.1496300
7: -7.7237144, -7.1442108, -7.7240276, -7.1444173, -0.2226775, 0.2233744
8: -0.9936666, -0.5893126, -0.9981155, -0.5882273, -0.1650424, 0.1656854
9: -4.8831615, -4.4188371, -4.8844552, -4.4102283, -0.1739943, 0.1733231

Time for backsubstitution: 21.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 919

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 919

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1346305, upper bound: 0.1353799
time: 3.19 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1346305, upper bound: 0.1359743
time: 3.14 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 9.8847237, 10.3131332, 9.8867626, 10.3084917, -0.1734078, 0.1746860
1: -16.8269806, -16.0978241, -16.8225574, -16.1140671, -0.2719699, 0.2733657
2: -4.3521938, -3.8940923, -4.3378911, -3.8974471, -0.1771246, 0.1773565
3: -10.4686146, -9.9474592, -10.4659634, -9.9480867, -0.1884445, 0.1865779
4: -10.2037468, -9.5797558, -10.2022600, -9.5819168, -0.2270794, 0.2279319
5: -2.6058052, -2.1881454, -2.6042910, -2.1946039, -0.1719493, 0.1757960
6: -2.0223296, -1.6933668, -2.0218835, -1.6956456, -0.1483033, 0.1500288
7: -7.7234259, -7.1458330, -7.7230577, -7.1459236, -0.2209194, 0.2206625
8: -0.9979882, -0.5890450, -0.9935389, -0.5899734, -0.1651747, 0.1628448
9: -4.8844562, -4.4108310, -4.8826561, -4.4190273, -0.1729337, 0.1737737

Time for backsubstitution: 21.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 919

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 919

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1360051, upper bound: 0.1339225
time: 3.29 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1358600, upper bound: 0.1345166
time: 3.17 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 9.8847237, 10.3131332, 9.8865957, 10.3088903, -0.1738008, 0.1748754
1: -16.8269806, -16.0978241, -16.8233147, -16.1121750, -0.2738631, 0.2740725
2: -4.3521938, -3.8940923, -4.3388824, -3.8971739, -0.1773939, 0.1782248
3: -10.4686146, -9.9474592, -10.4664555, -9.9467010, -0.1897998, 0.1870896
4: -10.2037468, -9.5797558, -10.2024193, -9.5815201, -0.2274590, 0.2281027
5: -2.6058052, -2.1881454, -2.6051927, -2.1923990, -0.1741679, 0.1765392
6: -2.0223296, -1.6933668, -2.0224047, -1.6954460, -0.1485132, 0.1505513
7: -7.7234259, -7.1458330, -7.7237139, -7.1442108, -0.2226350, 0.2212843
8: -0.9979882, -0.5890450, -0.9936676, -0.5893126, -0.1652281, 0.1629728
9: -4.8844562, -4.4108310, -4.8831630, -4.4188371, -0.1731395, 0.1737770

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 919

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 919

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1360051, upper bound: 0.1340363
time: 3.32 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1358600, upper bound: 0.1346306
time: 3.45 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 9.8845558, 10.3135319, 9.8861446, 10.3084927, -0.1737649, 0.1748855
1: -16.8277416, -16.0959358, -16.8232498, -16.1124649, -0.2742696, 0.2741853
2: -4.3531866, -3.8938198, -4.3396297, -3.8971784, -0.1776937, 0.1799846
3: -10.4691048, -9.9460716, -10.4664469, -9.9480457, -0.1892111, 0.1885212
4: -10.2038994, -9.5793562, -10.2024002, -9.5795574, -0.2297210, 0.2288625
5: -2.6067052, -2.1859410, -2.6050475, -2.1933584, -0.1739439, 0.1766862
6: -2.0228508, -1.6931690, -2.0230141, -1.6954565, -0.1491600, 0.1512735
7: -7.7240791, -7.1441221, -7.7236624, -7.1445084, -0.2229341, 0.2231178
8: -0.9981165, -0.5883851, -0.9936671, -0.5891562, -0.1658309, 0.1638665
9: -4.8849635, -4.4106426, -4.8826556, -4.4184246, -0.1740701, 0.1740283

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 919

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 919

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1359741, upper bound: 0.1340363
time: 3.11 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1359741, upper bound: 0.1346306
time: 3.07 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 9.8847237, 10.3131332, 9.8847237, 10.3131332, -0.1739091, 0.1739092
1: -16.8269806, -16.0978241, -16.8269806, -16.0978241, -0.2737048, 0.2737046
2: -4.3521938, -3.8940923, -4.3521938, -3.8940923, -0.1786511, 0.1786510
3: -10.4686146, -9.9474592, -10.4686146, -9.9474592, -0.1875316, 0.1875316
4: -10.2037468, -9.5797558, -10.2037468, -9.5797558, -0.2272999, 0.2272999
5: -2.6058052, -2.1881454, -2.6058052, -2.1881454, -0.1731976, 0.1731977
6: -2.0223296, -1.6933668, -2.0223296, -1.6933668, -0.1490523, 0.1490523
7: -7.7234259, -7.1458330, -7.7234259, -7.1458330, -0.2208295, 0.2208295
8: -0.9979882, -0.5890450, -0.9979882, -0.5890450, -0.1636353, 0.1636353
9: -4.8844562, -4.4108310, -4.8844562, -4.4108310, -0.1741147, 0.1741148

Time for backsubstitution: 21.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 919

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 919

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1360059, upper bound: 0.1339228
time: 3.14 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1358600, upper bound: 0.1345169
time: 3.04 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 9.8847237, 10.3131332, 9.8845539, 10.3135300, -0.1743021, 0.1741034
1: -16.8269806, -16.0978241, -16.8277397, -16.0959358, -0.2755979, 0.2744303
2: -4.3521938, -3.8940923, -4.3531857, -3.8938181, -0.1789299, 0.1795191
3: -10.4686146, -9.9474592, -10.4691057, -9.9460716, -0.1888869, 0.1880441
4: -10.2037468, -9.5797558, -10.2039042, -9.5793552, -0.2276795, 0.2274716
5: -2.6058052, -2.1881454, -2.6067071, -2.1859429, -0.1754165, 0.1739687
6: -2.0223296, -1.6933668, -2.0228508, -1.6931676, -0.1492627, 0.1495749
7: -7.7234259, -7.1458330, -7.7240829, -7.1441221, -0.2225454, 0.2214515
8: -0.9979882, -0.5890450, -0.9981155, -0.5883837, -0.1642028, 0.1637632
9: -4.8844562, -4.4108310, -4.8849640, -4.4106426, -0.1743211, 0.1746216

Time for backsubstitution: 21.60 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.70 + 554.67 = 610.37 seconds
