## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.08844139


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2214638, 0.2214637)
1: (-5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1752607, 0.1752607)
2: (-12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1957591, 0.1957589)
3: (-9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1939447, 0.1939447)
4: (6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1413702, 0.1413702)
5: (-5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1783422, 0.1783422)
6: (-13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2266999, 0.2266999)
7: (-5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1561695, 0.1561695)
8: (-2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1177218, 0.1177218)
9: (-4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1155073, 0.1155073)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.92 + 33.14 = 55.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0930962, upper bound: 0.0930960

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 442

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 6181

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930957, upper bound: 0.0927537
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0927539, upper bound: 0.0930955
time: 2.60 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.46 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.46
Output dim: 4, lower bound: -0.0930957, upper bound: 0.0927537
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.46
Output dim: 4, lower bound: -0.0927539, upper bound: 0.0930955

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2182952, 0.2198743
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1751571, 0.1750530
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1940973, 0.1949251
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1894673, 0.1917075
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1409330, 0.1404932
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1744561, 0.1763985
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2216078, 0.2241534
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1559438, 0.1557176
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1172589, 0.1168054
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1142201, 0.1148612

Time for backsubstitution: 21.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 442

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 442

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0905168, upper bound: 0.0927477
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930895, upper bound: 0.0901752
time: 2.78 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2198743, 0.2182953
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1750530, 0.1751571
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1949251, 0.1940973
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1917075, 0.1894672
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1404932, 0.1409330
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1763985, 0.1744561
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2241534, 0.2216079
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1557176, 0.1559438
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1168054, 0.1172589
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1148612, 0.1142201

Time for backsubstitution: 21.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 442

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0901750, upper bound: 0.0930896
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0927476, upper bound: 0.0905170
time: 2.65 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 26.78 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 26.78
Output dim: 4, lower bound: -0.0905168, upper bound: 0.0927477
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 26.78
Output dim: 4, lower bound: -0.0930895, upper bound: 0.0901752
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 26.78
Output dim: 4, lower bound: -0.0901750, upper bound: 0.0930896
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 26.78
Output dim: 4, lower bound: -0.0927476, upper bound: 0.0905170

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2183436, 0.2196809
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1747754, 0.1751486
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1947759, 0.1922359
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1898662, 0.1901262
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1383983, 0.1411318
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1734231, 0.1766582
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2198324, 0.2246011
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1542571, 0.1561413
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1168492, 0.1169078
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1150285, 0.1116416

Time for backsubstitution: 21.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0903270, upper bound: 0.0913307
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0887147, upper bound: 0.0925621
time: 2.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2181021, 0.2198743
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1751571, 0.1746712
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1914080, 0.1949251
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1878859, 0.1917075
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1409330, 0.1379585
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1744561, 0.1753656
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2216078, 0.2223780
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1559438, 0.1540308
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1172589, 0.1163958
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1110006, 0.1148612

Time for backsubstitution: 21.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0928997, upper bound: 0.0887582
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0912872, upper bound: 0.0899894
time: 2.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2199227, 0.2181020
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1746713, 0.1752527
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1956037, 0.1914082
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1921064, 0.1878859
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1379585, 0.1415716
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1753656, 0.1747158
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2223780, 0.2220556
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1540308, 0.1563675
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1163958, 0.1173612
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1156695, 0.1110006

Time for backsubstitution: 21.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0899894, upper bound: 0.0912872
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0887581, upper bound: 0.0928997
time: 2.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2196809, 0.2182953
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1750530, 0.1747752
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1922358, 0.1940973
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1901261, 0.1894672
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1404932, 0.1383983
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1763985, 0.1734231
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2241534, 0.2198324
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1557176, 0.1542571
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1168054, 0.1168492
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1116416, 0.1142201

Time for backsubstitution: 21.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0925620, upper bound: 0.0887147
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0913307, upper bound: 0.0903269
time: 2.68 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 27.51 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.51
Output dim: 4, lower bound: -0.0903270, upper bound: 0.0913307
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.51
Output dim: 4, lower bound: -0.0887147, upper bound: 0.0925621
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.51
Output dim: 4, lower bound: -0.0928997, upper bound: 0.0887582
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.51
Output dim: 4, lower bound: -0.0912872, upper bound: 0.0899894
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.51
Output dim: 4, lower bound: -0.0899894, upper bound: 0.0912872
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.51
Output dim: 4, lower bound: -0.0887581, upper bound: 0.0928997
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.51
Output dim: 4, lower bound: -0.0925620, upper bound: 0.0887147
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.51
Output dim: 4, lower bound: -0.0913307, upper bound: 0.0903269

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.1859527, 0.1907800
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1438354, 0.1455301
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1925797, 0.1901686
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1856833, 0.1869231
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1370634, 0.1396394
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1386563, 0.1417958
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2074838, 0.2134604
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1530943, 0.1551542
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1169419, 0.1170318
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1071543, 0.1045392

Time for backsubstitution: 20.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0888067, upper bound: 0.0895129
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0888043, upper bound: 0.0897169
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.1890702, 0.1872900
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1448230, 0.1442088
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1927085, 0.1899552
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1865053, 0.1859434
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1368849, 0.1397969
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1381893, 0.1418914
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2085257, 0.2122525
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1532700, 0.1549149
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1169698, 0.1170005
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1077922, 0.1037675

Time for backsubstitution: 20.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0870870, upper bound: 0.0910279
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0868832, upper bound: 0.0910303
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.1857110, 0.1909735
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1442170, 0.1450527
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1892121, 0.1928579
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1837029, 0.1885045
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1395983, 0.1364661
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1396893, 0.1405031
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2092590, 0.2112373
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1547810, 0.1530437
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1173516, 0.1165198
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1031265, 0.1077586

Time for backsubstitution: 21.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0913791, upper bound: 0.0869405
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0913767, upper bound: 0.0871445
time: 2.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.1888286, 0.1874834
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1452044, 0.1437314
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1893408, 0.1926444
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1845251, 0.1875246
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1394199, 0.1366236
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1392223, 0.1405988
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2103009, 0.2100294
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1549566, 0.1528044
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1173794, 0.1164885
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1037644, 0.1069870

Time for backsubstitution: 20.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0896594, upper bound: 0.0884555
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0894555, upper bound: 0.0884578
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.1875316, 0.1888286
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1437315, 0.1453002
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1933229, 0.1893408
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1879234, 0.1845251
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1366236, 0.1400582
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1405987, 0.1394819
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2100294, 0.2107489
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1528044, 0.1553805
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1164885, 0.1174818
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1077954, 0.1037643

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0884583, upper bound: 0.0894550
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0884559, upper bound: 0.0896593
time: 2.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.1910217, 0.1857110
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1450528, 0.1443129
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1935363, 0.1892121
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1889033, 0.1837029
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1364661, 0.1402366
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1405031, 0.1399489
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2112373, 0.2097070
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1530437, 0.1552049
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1165198, 0.1174539
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1085671, 0.1031264

Time for backsubstitution: 21.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0871447, upper bound: 0.0913763
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0869411, upper bound: 0.0913787
time: 2.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.1872900, 0.1890221
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1441128, 0.1448228
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1899552, 0.1920300
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1859432, 0.1861064
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1391585, 0.1368849
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1416318, 0.1381893
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2118046, 0.2085257
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1544911, 0.1532700
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1168981, 0.1169698
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1037675, 0.1069838

Time for backsubstitution: 21.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0910307, upper bound: 0.0868826
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0910283, upper bound: 0.0870868
time: 2.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.1907800, 0.1859044
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1454343, 0.1438354
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1901686, 0.1919012
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1869231, 0.1852844
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1390010, 0.1370634
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1415361, 0.1386563
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2130126, 0.2074838
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1547304, 0.1530943
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1169295, 0.1169419
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1045392, 0.1063459

Time for backsubstitution: 21.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0897170, upper bound: 0.0888038
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0895134, upper bound: 0.0888062
time: 2.93 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 27.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0888067, upper bound: 0.0895129
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0888043, upper bound: 0.0897169
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0870870, upper bound: 0.0910279
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0868832, upper bound: 0.0910303
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0913791, upper bound: 0.0869405
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0913767, upper bound: 0.0871445
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0896594, upper bound: 0.0884555
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0894555, upper bound: 0.0884578
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0884583, upper bound: 0.0894550
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0884559, upper bound: 0.0896593
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0871447, upper bound: 0.0913763
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0869411, upper bound: 0.0913787
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0910307, upper bound: 0.0868826
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0910283, upper bound: 0.0870868
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0897170, upper bound: 0.0888038
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 4, lower bound: -0.0895134, upper bound: 0.0888062

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.1875767, 0.1937692
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1642709, 0.1645920
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1938611, 0.1913216
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1623815, 0.1623054
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1296097, 0.1323403
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1716077, 0.1744044
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.1908581, 0.1985437
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1201903, 0.1180497
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1083335, 0.1078652
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1131334, 0.1092515

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Candidate
type: DSZ, layer: 3, pos: 332

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0884850, upper bound: 0.0884538
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0877469, upper bound: 0.0891917
time: 2.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.1924318, 0.1889139
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1642187, 0.1646441
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1938616, 0.1913214
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1620452, 0.1626415
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1296068, 0.1323432
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1711693, 0.1748428
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.1937759, 0.1956269
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1161655, 0.1220747
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1078067, 0.1083921
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1126386, 0.1097466

Time for backsubstitution: 21.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Candidate
type: DSZ, layer: 3, pos: 332

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0884827, upper bound: 0.0886578
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0877445, upper bound: 0.0893958
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.1875767, 0.1937692
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1642709, 0.1645920
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1938611, 0.1913216
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1623815, 0.1623054
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1296097, 0.1323403
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1716077, 0.1744044
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.1908581, 0.1985437
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1201903, 0.1180497
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1083335, 0.1078652
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1131334, 0.1092515

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Candidate
type: DSZ, layer: 3, pos: 332

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0867657, upper bound: 0.0899687
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0860276, upper bound: 0.0907067
time: 2.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.1924318, 0.1889139
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1642187, 0.1646441
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1938616, 0.1913214
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1620452, 0.1626415
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1296068, 0.1323432
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1711693, 0.1748428
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.1937759, 0.1956269
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1161655, 0.1220747
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1078067, 0.1083921
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1126386, 0.1097466

Time for backsubstitution: 21.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Candidate
type: DSZ, layer: 3, pos: 332

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0865615, upper bound: 0.0899711
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0858234, upper bound: 0.0907091
time: 2.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.1873350, 0.1939626
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1646525, 0.1641146
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1904937, 0.1940109
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1604012, 0.1638868
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1321446, 0.1291671
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1726407, 0.1731118
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.1926336, 0.1963205
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1218771, 0.1159392
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1087431, 0.1073532
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1091055, 0.1124709

Time for backsubstitution: 21.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Candidate
type: DSZ, layer: 3, pos: 332

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0910574, upper bound: 0.0858813
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0903193, upper bound: 0.0866193
time: 2.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.1921903, 0.1891073
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1646004, 0.1641668
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1904937, 0.1940106
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1600651, 0.1642230
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1321418, 0.1291699
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1722022, 0.1735502
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.1955514, 0.1934037
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1178521, 0.1199641
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1082163, 0.1078801
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1086107, 0.1129660

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Candidate
type: DSZ, layer: 3, pos: 332

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0910551, upper bound: 0.0860853
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0903169, upper bound: 0.0868234
time: 2.79 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 27.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.30
Output dim: 4, lower bound: -0.0884850, upper bound: 0.0884538
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.30
Output dim: 4, lower bound: -0.0877469, upper bound: 0.0891917
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.30
Output dim: 4, lower bound: -0.0884827, upper bound: 0.0886578
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.30
Output dim: 4, lower bound: -0.0877445, upper bound: 0.0893958
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.30
Output dim: 4, lower bound: -0.0867657, upper bound: 0.0899687
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.30
Output dim: 4, lower bound: -0.0860276, upper bound: 0.0907067
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.30
Output dim: 4, lower bound: -0.0865615, upper bound: 0.0899711
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.30
Output dim: 4, lower bound: -0.0858234, upper bound: 0.0907091
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.30
Output dim: 4, lower bound: -0.0910574, upper bound: 0.0858813
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.30
Output dim: 4, lower bound: -0.0903193, upper bound: 0.0866193
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.30
Output dim: 4, lower bound: -0.0910551, upper bound: 0.0860853
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.30
Output dim: 4, lower bound: -0.0903169, upper bound: 0.0868234
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.30
Output dim: 4, lower bound: -0.0896594, upper bound: 0.0884555
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.30
Output dim: 4, lower bound: -0.0894555, upper bound: 0.0884578
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.30
Output dim: 4, lower bound: -0.0884583, upper bound: 0.0894550
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.30
Output dim: 4, lower bound: -0.0884559, upper bound: 0.0896593
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.30
Output dim: 4, lower bound: -0.0871447, upper bound: 0.0913763
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.30
Output dim: 4, lower bound: -0.0869411, upper bound: 0.0913787
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.30
Output dim: 4, lower bound: -0.0910307, upper bound: 0.0868826
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.30
Output dim: 4, lower bound: -0.0910283, upper bound: 0.0870868
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.30
Output dim: 4, lower bound: -0.0897170, upper bound: 0.0888038
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.30
Output dim: 4, lower bound: -0.0895134, upper bound: 0.0888062

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.06 + 548.67 = 603.73 seconds
