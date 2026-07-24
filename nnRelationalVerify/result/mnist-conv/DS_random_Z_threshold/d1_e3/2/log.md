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
execution time: IAR + RelationalAnalysis = 24.54 + 32.57 = 57.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0930962, upper bound: 0.0930960

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6181

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930957, upper bound: 0.0927537
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0927539, upper bound: 0.0930955
time: 2.57 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.18 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.18
Output dim: 4, lower bound: -0.0930957, upper bound: 0.0927537
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.18
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

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 442

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0905168, upper bound: 0.0927477
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930895, upper bound: 0.0901752
time: 2.70 seconds

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

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 442

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0901750, upper bound: 0.0930896
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0927476, upper bound: 0.0905170
time: 2.49 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.08 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.08
Output dim: 4, lower bound: -0.0905168, upper bound: 0.0927477
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.08
Output dim: 4, lower bound: -0.0930895, upper bound: 0.0901752
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.08
Output dim: 4, lower bound: -0.0901750, upper bound: 0.0930896
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.08
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

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0903270, upper bound: 0.0913307
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0887147, upper bound: 0.0925621
time: 2.61 seconds

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

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3125

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0928421, upper bound: 0.0865164
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0894358, upper bound: 0.0899278
time: 2.70 seconds

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

Time for backsubstitution: 23.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 332

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 157

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0901211, upper bound: 0.0904636
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0901211, upper bound: 0.0922416
time: 2.53 seconds

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

Time for backsubstitution: 22.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 1249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 157

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0918999, upper bound: 0.0904630
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0901217, upper bound: 0.0904630
time: 2.65 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 27.85 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.85
Output dim: 4, lower bound: -0.0903270, upper bound: 0.0913307
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.85
Output dim: 4, lower bound: -0.0887147, upper bound: 0.0925621
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.85
Output dim: 4, lower bound: -0.0928421, upper bound: 0.0865164
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.85
Output dim: 4, lower bound: -0.0894358, upper bound: 0.0899278
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.85
Output dim: 4, lower bound: -0.0901211, upper bound: 0.0904636
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.85
Output dim: 4, lower bound: -0.0901211, upper bound: 0.0922416
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.85
Output dim: 4, lower bound: -0.0918999, upper bound: 0.0904630
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.85
Output dim: 4, lower bound: -0.0901217, upper bound: 0.0904630

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

Time for backsubstitution: 23.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 2334

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3125

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0900797, upper bound: 0.0876814
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0866789, upper bound: 0.0910834
time: 2.61 seconds

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

Time for backsubstitution: 23.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1732

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2334

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0854760, upper bound: 0.0899583
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0863311, upper bound: 0.0890529
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2024918, 0.2044235
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1737335, 0.1727339
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1831254, 0.1865042
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1880189, 0.1918609
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1351488, 0.1311574
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1752365, 0.1763815
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2122715, 0.2140312
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1535054, 0.1511980
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1169277, 0.1162981
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1000322, 0.1046365

Time for backsubstitution: 23.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0926524, upper bound: 0.0851089
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0910398, upper bound: 0.0863405
time: 2.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2026515, 0.2042639
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1732196, 0.1732479
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1829870, 0.1866425
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1880392, 0.1918408
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1341320, 0.1321743
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1754720, 0.1761459
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2132610, 0.2130415
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1531110, 0.1515925
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1171612, 0.1160647
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1007761, 0.1038927

Time for backsubstitution: 23.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 332

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0878672, upper bound: 0.0883675
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0878648, upper bound: 0.0883699
time: 2.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2195446, 0.2179391
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1742331, 0.1748128
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1956384, 0.1914433
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1913326, 0.1871167
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1379117, 0.1415248
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1746397, 0.1731026
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2222813, 0.2218364
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1540405, 0.1561998
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1162067, 0.1168638
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1153675, 0.1107005

Time for backsubstitution: 23.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3125

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0898738, upper bound: 0.0868084
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0864620, upper bound: 0.0902162
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2197597, 0.2180552
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1742313, 0.1751026
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1956398, 0.1914430
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1918771, 0.1871121
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1379118, 0.1416098
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1751162, 0.1739899
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2223462, 0.2219590
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1540732, 0.1563772
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1163480, 0.1171722
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1155910, 0.1106985

Time for backsubstitution: 23.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 1446

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0899354, upper bound: 0.0904392
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0887042, upper bound: 0.0920518
time: 2.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2195182, 0.2181326
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1750543, 0.1743354
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1922708, 0.1941866
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1893525, 0.1895491
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1406856, 0.1383516
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1756728, 0.1726940
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2240568, 0.2197363
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1557269, 0.1542662
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1166163, 0.1166601
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1113396, 0.1143135

Time for backsubstitution: 22.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 332

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0915129, upper bound: 0.0893343
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0907711, upper bound: 0.0900762
time: 2.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2195182, 0.2179602
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1746128, 0.1743371
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1922712, 0.1941323
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1893570, 0.1886928
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1404464, 0.1383516
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1749632, 0.1726973
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2239590, 0.2197359
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1555853, 0.1542667
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1163695, 0.1166602
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1113416, 0.1139179

Time for backsubstitution: 22.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0885630, upper bound: 0.0889092
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0885607, upper bound: 0.0889115
time: 2.66 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 27.92 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0900797, upper bound: 0.0876814
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0866789, upper bound: 0.0910834
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0854760, upper bound: 0.0899583
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0863311, upper bound: 0.0890529
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0926524, upper bound: 0.0851089
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0910398, upper bound: 0.0863405
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0878672, upper bound: 0.0883675
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0878648, upper bound: 0.0883699
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0898738, upper bound: 0.0868084
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0864620, upper bound: 0.0902162
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0899354, upper bound: 0.0904392
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0887042, upper bound: 0.0920518
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0915129, upper bound: 0.0893343
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0907711, upper bound: 0.0900762
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0885630, upper bound: 0.0889092
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.92
Output dim: 4, lower bound: -0.0885607, upper bound: 0.0889115

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2027333, 0.2042303
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1733520, 0.1732113
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1864932, 0.1838148
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1899992, 0.1902796
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1326140, 0.1343306
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1742035, 0.1776741
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2104961, 0.2162544
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1518188, 0.1533085
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1165181, 0.1168101
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1040601, 0.1014171

Time for backsubstitution: 22.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0885593, upper bound: 0.0858386
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0885569, upper bound: 0.0860426
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2028928, 0.2040706
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1728380, 0.1737252
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1863546, 0.1839532
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1900195, 0.1902593
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1315972, 0.1353475
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1744390, 0.1774386
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2114856, 0.2152648
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1514243, 0.1537030
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1167516, 0.1165767
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1048039, 0.1006733

Time for backsubstitution: 21.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 157

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0866248, upper bound: 0.0884575
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0866248, upper bound: 0.0902356
time: 2.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2180243, 0.2195122
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1737429, 0.1721194
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1973400, 0.1911292
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1880659, 0.1871392
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1357500, 0.1404703
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1720650, 0.1749082
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2195864, 0.2244377
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1517200, 0.1585355
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1152611, 0.1151644
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1155298, 0.1106533

Time for backsubstitution: 22.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1732

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0838813, upper bound: 0.0878667
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0827741, upper bound: 0.0883664
time: 3.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.6748533, -9.1102934, -9.6748533, -9.1102934, -0.2181749, 0.2196809
1: -5.5039563, -5.0927405, -5.5039563, -5.0927405, -0.1747754, 0.1741161
2: -12.0584517, -11.6201696, -12.0584517, -11.6201696, -0.1936691, 0.1922359
3: -9.5751858, -9.1182966, -9.5751858, -9.1182966, -0.1868790, 0.1901262
4: 6.2165461, 6.6384802, 6.2165461, 6.6384802, -0.1383983, 0.1384835
5: -5.0103426, -4.5674219, -5.0103426, -4.5674219, -0.1716731, 0.1766582
6: -13.4792080, -12.8041668, -13.4792080, -12.8041668, -0.2196690, 0.2246011
7: -5.3893228, -5.0399065, -5.3893228, -5.0399065, -0.1542571, 0.1536043
8: -2.6510296, -2.2606416, -2.6510296, -2.2606416, -0.1168492, 0.1153197
9: -4.7955704, -4.4272785, -4.7955704, -4.4272785, -0.1140401, 0.1116416

Time for backsubstitution: 22.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3125

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0860607, upper bound: 0.0855491
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0825782, upper bound: 0.0888057
time: 2.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 22.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 3125

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1732

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0918611, upper bound: 0.0759508
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0887962, upper bound: 0.0844905
time: 2.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 22.82 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.10 + 561.01 = 618.12 seconds
