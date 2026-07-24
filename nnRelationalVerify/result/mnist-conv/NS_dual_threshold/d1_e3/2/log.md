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
execution time: IAR + RelationalAnalysis = 21.92 + 32.99 = 54.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0930962, upper bound: 0.0930960

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 6181
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0904660, upper bound: 0.0930898
time: 2.57 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930887, upper bound: 0.0930898
time: 2.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.34 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.34
Output dim: 4, lower bound: -0.0904660, upper bound: 0.0930898
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.34
Output dim: 4, lower bound: -0.0930887, upper bound: 0.0930898

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -9.6748505, -9.1103783, -9.6748533, -9.1102934, -0.2213029, 0.2211058
1: -5.5031271, -5.0927458, -5.5039563, -5.0927405, -0.1744366, 0.1752604
2: -12.0584450, -11.6228561, -12.0584517, -11.6201696, -0.1958029, 0.1930672
3: -9.5751677, -9.1206570, -9.5751858, -9.1182966, -0.1939824, 0.1915797
4: 6.2191954, 6.6384497, 6.2165461, 6.6384802, -0.1387174, 0.1414500
5: -5.0098014, -4.5674295, -5.0103426, -4.5674219, -0.1765760, 0.1776059
6: -13.4776249, -12.8041773, -13.4792080, -12.8041668, -0.2248255, 0.2264916
7: -5.3876553, -5.0399065, -5.3893228, -5.0399065, -0.1544117, 0.1561172
8: -2.6507521, -2.2606459, -2.6510296, -2.2606416, -0.1170939, 0.1175106
9: -4.7955642, -4.4308338, -4.7955704, -4.4272785, -0.1155573, 0.1119463

Time for backsubstitution: 20.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 6181

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0904656, upper bound: 0.0904654
time: 2.64 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0904656, upper bound: 0.0930898
time: 2.67 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -9.6752787, -9.1102848, -9.6748533, -9.1102934, -0.2217503, 0.2224698
1: -5.5041428, -5.0876541, -5.5039554, -5.0927410, -0.1756908, 0.1770518
2: -12.0761404, -11.6200542, -12.0584526, -11.6201725, -0.1985145, 0.1966299
3: -9.5903502, -9.1182699, -9.5751858, -9.1182976, -0.1970994, 0.1943611
4: 6.2165346, 6.6560507, 6.2165489, 6.6384811, -0.1420158, 0.1430969
5: -5.0104346, -4.5653958, -5.0103416, -4.5674219, -0.1810486, 0.1796333
6: -13.4793997, -12.7933769, -13.4792080, -12.8041658, -0.2287551, 0.2285974
7: -5.3895330, -5.0292664, -5.3893194, -5.0399065, -0.1571157, 0.1604970
8: -2.6510916, -2.2592545, -2.6510286, -2.2606421, -0.1190742, 0.1189499
9: -4.8186359, -4.4272509, -4.7955704, -4.4272823, -0.1174232, 0.1163523

Time for backsubstitution: 20.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 6181

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930899, upper bound: 0.0904656
time: 2.55 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930899, upper bound: 0.0930899
time: 2.51 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 25.41 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 4, lower bound: -0.0904656, upper bound: 0.0904654
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 4, lower bound: -0.0904656, upper bound: 0.0930898
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 4, lower bound: -0.0930899, upper bound: 0.0904656
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 4, lower bound: -0.0930899, upper bound: 0.0930899

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -9.6748505, -9.1103783, -9.6748505, -9.1103783, -0.2211032, 0.2211033
1: -5.5031271, -5.0927458, -5.5031271, -5.0927458, -0.1744322, 0.1744322
2: -12.0584450, -11.6228561, -12.0584450, -11.6228561, -0.1930567, 0.1930568
3: -9.5751677, -9.1206570, -9.5751677, -9.1206570, -0.1915463, 0.1915464
4: 6.2191954, 6.6384497, 6.2191954, 6.6384497, -0.1386744, 0.1386744
5: -5.0098014, -4.5674295, -5.0098014, -4.5674295, -0.1765455, 0.1765455
6: -13.4776249, -12.8041773, -13.4776249, -12.8041773, -0.2247152, 0.2247152
7: -5.3876553, -5.0399065, -5.3876553, -5.0399065, -0.1544117, 0.1544116
8: -2.6507521, -2.2606459, -2.6507521, -2.2606459, -0.1170915, 0.1170915
9: -4.7955642, -4.4308338, -4.7955642, -4.4308338, -0.1119424, 0.1119424

Time for backsubstitution: 20.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6181
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0904032, upper bound: 0.0899984
time: 2.58 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0904654, upper bound: 0.0904650
time: 2.67 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -9.6748505, -9.1103783, -9.6752815, -9.1102877, -0.2211976, 0.2215505
1: -5.5031271, -5.0927458, -5.5041437, -5.0882068, -0.1761538, 0.1755905
2: -12.0584450, -11.6228561, -12.0760717, -11.6200533, -0.1959044, 0.1958128
3: -9.5751677, -9.1206570, -9.5892754, -9.1182699, -0.1939254, 0.1945857
4: 6.2191954, 6.6384497, 6.2165365, 6.6557522, -0.1404013, 0.1413289
5: -5.0098014, -4.5674295, -5.0104322, -4.5653963, -0.1785729, 0.1772525
6: -13.4776249, -12.8041773, -13.4794016, -12.7933769, -0.2268186, 0.2264944
7: -5.3876553, -5.0399065, -5.3895330, -5.0292664, -0.1587915, 0.1561989
8: -2.6507521, -2.2606459, -2.6510906, -2.2592564, -0.1185309, 0.1173786
9: -4.7955642, -4.4308338, -4.8181400, -4.4272513, -0.1155366, 0.1137957

Time for backsubstitution: 21.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6181
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0904032, upper bound: 0.0926228
time: 2.62 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0904654, upper bound: 0.0930894
time: 2.72 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -9.6752815, -9.1102877, -9.6748505, -9.1103783, -0.2215505, 0.2211976
1: -5.5041437, -5.0882068, -5.5031271, -5.0927458, -0.1755905, 0.1761538
2: -12.0760717, -11.6200533, -12.0584450, -11.6228561, -0.1958127, 0.1959046
3: -9.5892754, -9.1182699, -9.5751677, -9.1206570, -0.1945857, 0.1939253
4: 6.2165365, 6.6557522, 6.2191954, 6.6384497, -0.1413288, 0.1404013
5: -5.0104322, -4.5653963, -5.0098014, -4.5674295, -0.1772525, 0.1785729
6: -13.4794016, -12.7933769, -13.4776249, -12.8041773, -0.2264944, 0.2268186
7: -5.3895330, -5.0292664, -5.3876553, -5.0399065, -0.1561989, 0.1587915
8: -2.6510906, -2.2592564, -2.6507521, -2.2606459, -0.1173786, 0.1185309
9: -4.8181400, -4.4272513, -4.7955642, -4.4308338, -0.1137956, 0.1155366

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 6181

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0926214, upper bound: 0.0904026
time: 2.67 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930883, upper bound: 0.0904651
time: 2.74 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -9.6752787, -9.1102858, -9.6752787, -9.1102858, -0.2228498, 0.2228497
1: -5.5041437, -5.0872302, -5.5041437, -5.0872302, -0.1774575, 0.1775541
2: -12.0761919, -11.6200542, -12.0761919, -11.6200542, -0.1968087, 0.1968087
3: -9.5911713, -9.1182699, -9.5911713, -9.1182699, -0.1976322, 0.1972311
4: 6.2165365, 6.6562824, 6.2165365, 6.6562824, -0.1431357, 0.1432229
5: -5.0104356, -4.5653954, -5.0104356, -4.5653954, -0.1809145, 0.1811765
6: -13.4794025, -12.7933769, -13.4794025, -12.7933769, -0.2289841, 0.2289841
7: -5.3895330, -5.0292664, -5.3895330, -5.0292664, -0.1572518, 0.1572518
8: -2.6510944, -2.2592535, -2.6510944, -2.2592535, -0.1194394, 0.1194394
9: -4.8190160, -4.4272509, -4.8190160, -4.4272509, -0.1183292, 0.1175128

Time for backsubstitution: 21.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6181
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930273, upper bound: 0.0900501
time: 2.59 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930895, upper bound: 0.0905168
time: 2.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 26.91 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.91
Output dim: 4, lower bound: -0.0904032, upper bound: 0.0899984
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.91
Output dim: 4, lower bound: -0.0904654, upper bound: 0.0904650
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.91
Output dim: 4, lower bound: -0.0904032, upper bound: 0.0926228
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.91
Output dim: 4, lower bound: -0.0904654, upper bound: 0.0930894
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 26.91
Output dim: 4, lower bound: -0.0926214, upper bound: 0.0904026
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 26.91
Output dim: 4, lower bound: -0.0930883, upper bound: 0.0904651
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.91
Output dim: 4, lower bound: -0.0930273, upper bound: 0.0900501
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.91
Output dim: 4, lower bound: -0.0930895, upper bound: 0.0905168

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.6704969, -9.1105423, -9.6731062, -9.1104431, -0.2166955, 0.2192982
1: -5.5023479, -5.0927782, -5.5028095, -5.0927596, -0.1730232, 0.1736662
2: -12.0564871, -11.6236944, -12.0576611, -11.6231890, -0.1907024, 0.1914778
3: -9.5704002, -9.1213493, -9.5732641, -9.1209297, -0.1863244, 0.1885689
4: 6.2198257, 6.6375618, 6.2194433, 6.6380939, -0.1370877, 0.1369679
5: -5.0052133, -4.5678730, -5.0079656, -4.5676036, -0.1715111, 0.1740173
6: -13.4717121, -12.8042698, -13.4752636, -12.8042135, -0.2186879, 0.2222645
7: -5.3870010, -5.0404577, -5.3873987, -5.0401287, -0.1531503, 0.1533071
8: -2.6504021, -2.2616959, -2.6506147, -2.2610703, -0.1158623, 0.1156528
9: -4.7939901, -4.4312997, -4.7949347, -4.4310179, -0.1101412, 0.1107829

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 1249

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 3, pos: 1732

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0896684, upper bound: 0.0856662
time: 3.20 seconds

## Relational analysis of NS_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0896686, upper bound: 0.0892636
time: 3.14 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -9.6752930, -9.1024904, -9.6748447, -9.1103792, -0.2198918, 0.2228975
1: -5.5039687, -5.0919685, -5.5031252, -5.0927467, -0.1745888, 0.1755270
2: -12.0589066, -11.6185818, -12.0584421, -11.6228580, -0.1922623, 0.1961304
3: -9.5755281, -9.1114531, -9.5751629, -9.1206589, -0.1896695, 0.1943163
4: 6.2176719, 6.6385345, 6.2191968, 6.6384487, -0.1398432, 0.1386470
5: -5.0105934, -4.5585051, -5.0097980, -4.5674281, -0.1748998, 0.1788321
6: -13.4780674, -12.7932911, -13.4776192, -12.8041782, -0.2224940, 0.2266887
7: -5.3885770, -5.0397730, -5.3876543, -5.0399075, -0.1550596, 0.1544462
8: -2.6524973, -2.2604699, -2.6507535, -2.2606487, -0.1187185, 0.1167740
9: -4.7957730, -4.4276962, -4.7955604, -4.4308333, -0.1113700, 0.1138289

Time for backsubstitution: 21.70 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249

Time for candidate selection: 0.43 seconds

### Candidate
type: A, layer: 3, pos: 1732

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0897269, upper bound: 0.0861332
time: 3.24 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0897274, upper bound: 0.0897266
time: 3.10 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.6704969, -9.1105423, -9.6735363, -9.1103516, -0.2167898, 0.2197452
1: -5.5023479, -5.0927782, -5.5038261, -5.0882187, -0.1746589, 0.1748247
2: -12.0564871, -11.6236944, -12.0752888, -11.6203823, -0.1935503, 0.1940384
3: -9.5704002, -9.1213493, -9.5873699, -9.1185398, -0.1887035, 0.1911551
4: 6.2198257, 6.6375618, 6.2167845, 6.6553946, -0.1387245, 0.1396220
5: -5.0052133, -4.5678730, -5.0085974, -4.5655713, -0.1735383, 0.1747248
6: -13.4717121, -12.8042698, -13.4770355, -12.7934113, -0.2207825, 0.2240441
7: -5.3870010, -5.0404577, -5.3892756, -5.0294886, -0.1574879, 0.1550949
8: -2.6504021, -2.2616959, -2.6509538, -2.2596788, -0.1173016, 0.1159399
9: -4.7939901, -4.4312997, -4.8175101, -4.4274349, -0.1137356, 0.1124570

Time for backsubstitution: 21.79 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 157

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 1732

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0860710, upper bound: 0.0918240
time: 2.86 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0896684, upper bound: 0.0918243
time: 3.00 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -9.6752930, -9.1024904, -9.6752729, -9.1102867, -0.2199855, 0.2231231
1: -5.5039687, -5.0919685, -5.5041413, -5.0882063, -0.1762668, 0.1766853
2: -12.0589066, -11.6185818, -12.0760708, -11.6200542, -0.1951098, 0.1961858
3: -9.5755281, -9.1114531, -9.5892696, -9.1182709, -0.1920483, 0.1946998
4: 6.2176719, 6.6385345, 6.2165356, 6.6557493, -0.1402574, 0.1413013
5: -5.0105934, -4.5585051, -5.0104289, -4.5653968, -0.1769272, 0.1795390
6: -13.4780674, -12.7932911, -13.4793921, -12.7933769, -0.2246003, 0.2284747
7: -5.3885770, -5.0397730, -5.3895321, -5.0292673, -0.1588186, 0.1562338
8: -2.6524973, -2.2604699, -2.6510911, -2.2592568, -0.1196773, 0.1170611
9: -4.7957730, -4.4276962, -4.8181372, -4.4272518, -0.1149642, 0.1139264

Time for backsubstitution: 21.77 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 157

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 1732

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0861332, upper bound: 0.0922866
time: 3.05 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0897272, upper bound: 0.0922869
time: 3.06 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -9.6735363, -9.1103516, -9.6704969, -9.1105423, -0.2197453, 0.2167898
1: -5.5038261, -5.0882187, -5.5023479, -5.0927782, -0.1748247, 0.1746589
2: -12.0752888, -11.6203823, -12.0564871, -11.6236944, -0.1940384, 0.1935505
3: -9.5873699, -9.1185398, -9.5704002, -9.1213493, -0.1911550, 0.1887035
4: 6.2167845, 6.6553946, 6.2198257, 6.6375618, -0.1396220, 0.1387245
5: -5.0085974, -4.5655713, -5.0052133, -4.5678730, -0.1747248, 0.1735383
6: -13.4770355, -12.7934113, -13.4717121, -12.8042698, -0.2240441, 0.2207824
7: -5.3892756, -5.0294886, -5.3870010, -5.0404577, -0.1550949, 0.1574879
8: -2.6509538, -2.2596788, -2.6504021, -2.2616959, -0.1159399, 0.1173016
9: -4.8175101, -4.4274349, -4.7939901, -4.4312997, -0.1124570, 0.1137356

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 157

Time for candidate selection: 0.46 seconds

### Candidate
type: A, layer: 3, pos: 1732

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0918240, upper bound: 0.0860711
time: 3.32 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0918242, upper bound: 0.0896681
time: 2.98 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -9.6752729, -9.1102867, -9.6752930, -9.1024904, -0.2231231, 0.2199856
1: -5.5041413, -5.0882063, -5.5039687, -5.0919685, -0.1766853, 0.1762668
2: -12.0760708, -11.6200542, -12.0589066, -11.6185818, -0.1961858, 0.1951100
3: -9.5892696, -9.1182709, -9.5755281, -9.1114531, -0.1946998, 0.1920483
4: 6.2165356, 6.6557493, 6.2176719, 6.6385345, -0.1413013, 0.1402574
5: -5.0104289, -4.5653968, -5.0105934, -4.5585051, -0.1795388, 0.1769272
6: -13.4793921, -12.7933769, -13.4780674, -12.7932911, -0.2284747, 0.2246004
7: -5.3895321, -5.0292673, -5.3885770, -5.0397730, -0.1562338, 0.1588186
8: -2.6510911, -2.2592568, -2.6524973, -2.2604699, -0.1170611, 0.1196773
9: -4.8181372, -4.4272518, -4.7957730, -4.4276962, -0.1139264, 0.1149642

Time for backsubstitution: 21.73 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 1249
type: A, layer: 3, pos: 157

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 1732

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0922865, upper bound: 0.0861332
time: 3.33 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0922868, upper bound: 0.0897265
time: 3.02 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -9.6709261, -9.1104507, -9.6735373, -9.1103497, -0.2184424, 0.2210451
1: -5.5033646, -5.0872612, -5.5038257, -5.0872436, -0.1759627, 0.1767037
2: -12.0742378, -11.6208887, -12.0754099, -11.6203823, -0.1944546, 0.1952306
3: -9.5864029, -9.1189604, -9.5892668, -9.1185398, -0.1923995, 0.1938004
4: 6.2171659, 6.6553936, 6.2167845, 6.6559258, -0.1414588, 0.1415160
5: -5.0058479, -4.5658412, -5.0085993, -4.5655713, -0.1758548, 0.1782050
6: -13.4734879, -12.7934675, -13.4770374, -12.7934113, -0.2229563, 0.2264548
7: -5.3888783, -5.0298176, -5.3892751, -5.0294886, -0.1559907, 0.1561474
8: -2.6507406, -2.2603040, -2.6509557, -2.2596793, -0.1182104, 0.1180011
9: -4.8174434, -4.4277167, -4.8183866, -4.4274359, -0.1165171, 0.1161744

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2559
type: B, layer: 3, pos: 2559
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 1249
type: A, layer: 3, pos: 157
type: B, layer: 3, pos: 157

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 2559

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2559

### Candidate
type: A, layer: 3, pos: 1732

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0922285, upper bound: 0.0857173
time: 3.06 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0922285, upper bound: 0.0893162
time: 2.85 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -9.6757221, -9.1023960, -9.6752720, -9.1102848, -0.2216382, 0.2235410
1: -5.5049868, -5.0864506, -5.5041428, -5.0872297, -0.1775711, 0.1775602
2: -12.0766544, -11.6157770, -12.0761909, -11.6200542, -0.1960166, 0.1990589
3: -9.5915337, -9.1090641, -9.5911655, -9.1182709, -0.1957370, 0.1973450
4: 6.2150121, 6.6563668, 6.2165356, 6.6562815, -0.1429919, 0.1431952
5: -5.0112286, -4.5564713, -5.0104313, -4.5653963, -0.1792756, 0.1812961
6: -13.4798441, -12.7824917, -13.4793930, -12.7933769, -0.2267666, 0.2294942
7: -5.3904548, -5.0291328, -5.3895335, -5.0292673, -0.1579009, 0.1572864
8: -2.6528373, -2.2590766, -2.6510925, -2.2592564, -0.1203879, 0.1191224
9: -4.8192253, -4.4241157, -4.8190126, -4.4272513, -0.1177573, 0.1176441

Time for backsubstitution: 21.63 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2559
type: B, layer: 3, pos: 2559
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 157
type: B, layer: 3, pos: 157

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 2559

## Relational analysis of NS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2559

### Candidate
type: A, layer: 3, pos: 1732

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0922867, upper bound: 0.0861841
time: 3.02 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0922870, upper bound: 0.0897788
time: 2.83 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 33.60 seconds
NS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0896684, upper bound: 0.0856662
NS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0896686, upper bound: 0.0892636
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0897269, upper bound: 0.0861332
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0897274, upper bound: 0.0897266
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0860710, upper bound: 0.0918240
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0896684, upper bound: 0.0918243
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0861332, upper bound: 0.0922866
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0897272, upper bound: 0.0922869
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0918240, upper bound: 0.0860711
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0918242, upper bound: 0.0896681
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0922865, upper bound: 0.0861332
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0922868, upper bound: 0.0897265
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0922285, upper bound: 0.0857173
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0922285, upper bound: 0.0893162
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0922867, upper bound: 0.0861841
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 33.60
Output dim: 4, lower bound: -0.0922870, upper bound: 0.0897788

## BFS NS instance: NS_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -9.6704960, -9.1107521, -9.6730919, -9.1105423, -0.2165798, 0.2190326
1: -5.5008802, -5.0922298, -5.5021176, -5.0927601, -0.1723861, 0.1751425
2: -12.0549793, -11.6239557, -12.0569515, -11.6231947, -0.1895730, 0.1912529
3: -9.5666142, -9.1193399, -9.5714836, -9.1209307, -0.1840377, 0.1914699
4: 6.2206173, 6.6296349, 6.2194438, 6.6343670, -0.1307092, 0.1286601
5: -5.0050869, -4.5691061, -5.0079584, -4.5681839, -0.1706387, 0.1726928
6: -13.4782982, -12.8072491, -13.4752493, -12.8056135, -0.2153403, 0.2147138
7: -5.3839579, -5.0377140, -5.3859682, -5.0401287, -0.1489778, 0.1522375
8: -2.6491442, -2.2609448, -2.6500244, -2.2610760, -0.1135891, 0.1140965
9: -4.7926307, -4.4289436, -4.7942948, -4.4310193, -0.1081906, 0.1111030

Time for backsubstitution: 22.28 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 1249

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A1_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0865168, upper bound: 0.0825074
time: 2.87 seconds

## Relational analysis of NS_A1_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0880494, upper bound: 0.0840558
time: 3.11 seconds

## BFS NS instance: NS_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -9.6704836, -9.1106291, -9.6731005, -9.1104717, -0.2166522, 0.2190386
1: -5.5017633, -5.0927782, -5.5026083, -5.0927601, -0.1742446, 0.1733268
2: -12.0562744, -11.6237001, -12.0575876, -11.6231899, -0.1901873, 0.1913260
3: -9.5686302, -9.1213493, -9.5726538, -9.1209288, -0.1895264, 0.1877904
4: 6.2198257, 6.6363444, 6.2194438, 6.6376438, -0.1367944, 0.1283072
5: -5.0052061, -4.5680885, -5.0079627, -4.5676775, -0.1714332, 0.1726658
6: -13.4717016, -12.8080139, -13.4752560, -12.8055000, -0.2182028, 0.2138870
7: -5.3850546, -5.0404577, -5.3867283, -5.0401287, -0.1496475, 0.1529336
8: -2.6497717, -2.2617011, -2.6503973, -2.2610736, -0.1131773, 0.1156113
9: -4.7925897, -4.4313030, -4.7944522, -4.4310188, -0.1091896, 0.1104683

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 1249

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0865167, upper bound: 0.0861078
time: 2.68 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0881025, upper bound: 0.0876974
time: 2.97 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -9.6752892, -9.1026993, -9.6748314, -9.1104765, -0.2197757, 0.2226315
1: -5.5024986, -5.0914230, -5.5024309, -5.0927472, -0.1739509, 0.1770031
2: -12.0573969, -11.6188412, -12.0577316, -11.6228657, -0.1911329, 0.1958882
3: -9.5717430, -9.1094456, -9.5733833, -9.1206589, -0.1873710, 0.1972828
4: 6.2184629, 6.6306081, 6.2191954, 6.6347218, -0.1334651, 0.1303017
5: -5.0104671, -4.5597391, -5.0097923, -4.5680084, -0.1740276, 0.1775066
6: -13.4846525, -12.7962723, -13.4776058, -12.8055763, -0.2191370, 0.2191483
7: -5.3855343, -5.0370283, -5.3862247, -5.0399075, -0.1508716, 0.1533579
8: -2.6512384, -2.2597213, -2.6501622, -2.2606554, -0.1164421, 0.1152189
9: -4.7944131, -4.4253392, -4.7949228, -4.4308362, -0.1094005, 0.1141266

Time for backsubstitution: 21.62 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2334

## Relational analysis of NS_A1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0863334, upper bound: 0.0835413
time: 2.94 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0867806, upper bound: 0.0831756
time: 3.08 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -9.6752787, -9.1025753, -9.6748371, -9.1104069, -0.2198476, 0.2226393
1: -5.5033836, -5.0919690, -5.5029240, -5.0927463, -0.1758095, 0.1751878
2: -12.0586929, -11.6185875, -12.0583668, -11.6228609, -0.1917474, 0.1959825
3: -9.5737572, -9.1114521, -9.5745525, -9.1206608, -0.1928598, 0.1935376
4: 6.2176719, 6.6373444, 6.2191954, 6.6380110, -0.1395503, 0.1299488
5: -5.0105858, -4.5587196, -5.0097961, -4.5675039, -0.1748215, 0.1774851
6: -13.4780560, -12.7970343, -13.4776154, -12.8054619, -0.2220079, 0.2183678
7: -5.3866305, -5.0397730, -5.3869858, -5.0399075, -0.1515357, 0.1540728
8: -2.6518664, -2.2604752, -2.6505356, -2.2606521, -0.1160304, 0.1167321
9: -4.7943721, -4.4276991, -4.7950797, -4.4308348, -0.1104044, 0.1135153

Time for backsubstitution: 21.55 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2334

## Relational analysis of NS_A1_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0863367, upper bound: 0.0871471
time: 2.96 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0867825, upper bound: 0.0867816
time: 3.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -9.6704845, -9.1106405, -9.6735106, -9.1105747, -0.2165135, 0.2196027
1: -5.5016565, -5.0927782, -5.5022759, -5.0878630, -0.1761265, 0.1740699
2: -12.0557814, -11.6237011, -12.0738468, -11.6206226, -0.1933489, 0.1928419
3: -9.5686235, -9.1213484, -9.5837469, -9.1164522, -0.1916896, 0.1887707
4: 6.2198257, 6.6338434, 6.2174473, 6.6478100, -0.1302910, 0.1333802
5: -5.0052066, -4.5684519, -5.0084915, -4.5667524, -0.1722708, 0.1738724
6: -13.4717007, -12.8056679, -13.4837427, -12.7962656, -0.2130758, 0.2208401
7: -5.3855739, -5.0404577, -5.3859730, -5.0272617, -0.1562382, 0.1507385
8: -2.6498117, -2.2617025, -2.6496086, -2.2591143, -0.1155613, 0.1136309
9: -4.7933531, -4.4313035, -4.8162093, -4.4250288, -0.1141134, 0.1103965

Time for backsubstitution: 21.66 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 157

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0829157, upper bound: 0.0886675
time: 2.77 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0844602, upper bound: 0.0902014
time: 2.97 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.6704922, -9.1105709, -9.6735210, -9.1104488, -0.2165244, 0.2196987
1: -5.5021591, -5.0927777, -5.5031872, -5.0882187, -0.1743419, 0.1759528
2: -12.0564194, -11.6236963, -12.0749989, -11.6203928, -0.1934047, 0.1934607
3: -9.5698299, -9.1213474, -9.5852203, -9.1185417, -0.1879739, 0.1942949
4: 6.2198257, 6.6371183, 6.2167835, 6.6536980, -0.1299244, 0.1393477
5: -5.0052099, -4.5679426, -5.0085897, -4.5658555, -0.1721219, 0.1746476
6: -13.4717102, -12.8054762, -13.4770222, -12.7978191, -0.2122825, 0.2235842
7: -5.3863745, -5.0404577, -5.3871498, -5.0294886, -0.1571399, 0.1514955
8: -2.6501989, -2.2616968, -2.6502628, -2.2596860, -0.1172589, 0.1132494
9: -4.7935395, -4.4313011, -4.8158531, -4.4274373, -0.1134396, 0.1113662

Time for backsubstitution: 21.82 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 54.91 + 562.55 = 617.46 seconds
