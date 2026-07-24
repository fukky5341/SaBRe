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
execution time: IAR + RelationalAnalysis = 22.92 + 32.53 = 55.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0930962, upper bound: 0.0930960

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 6181

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0904660, upper bound: 0.0930898
time: 2.60 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930887, upper bound: 0.0930898
time: 2.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.36 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.36
Output dim: 4, lower bound: -0.0904660, upper bound: 0.0930898
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.36
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

Time for backsubstitution: 21.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0904656, upper bound: 0.0904654
time: 2.83 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0904656, upper bound: 0.0930898
time: 2.82 seconds

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

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930899, upper bound: 0.0904656
time: 2.75 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930899, upper bound: 0.0930899
time: 2.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 26.66 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.66
Output dim: 4, lower bound: -0.0904656, upper bound: 0.0904654
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.66
Output dim: 4, lower bound: -0.0904656, upper bound: 0.0930898
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.66
Output dim: 4, lower bound: -0.0930899, upper bound: 0.0904656
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.66
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

Time for backsubstitution: 21.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6181

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0904032, upper bound: 0.0899984
time: 2.63 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0904654, upper bound: 0.0904650
time: 2.66 seconds

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

Time for backsubstitution: 21.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0904032, upper bound: 0.0926228
time: 2.59 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0904654, upper bound: 0.0930894
time: 2.68 seconds

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

Time for backsubstitution: 21.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6181

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930259, upper bound: 0.0899981
time: 2.71 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930882, upper bound: 0.0904650
time: 2.77 seconds

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

Time for backsubstitution: 21.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930273, upper bound: 0.0900501
time: 2.60 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0930895, upper bound: 0.0905168
time: 2.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 26.86 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.86
Output dim: 4, lower bound: -0.0904032, upper bound: 0.0899984
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.86
Output dim: 4, lower bound: -0.0904654, upper bound: 0.0904650
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.86
Output dim: 4, lower bound: -0.0904032, upper bound: 0.0926228
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.86
Output dim: 4, lower bound: -0.0904654, upper bound: 0.0930894
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.86
Output dim: 4, lower bound: -0.0930259, upper bound: 0.0899981
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.86
Output dim: 4, lower bound: -0.0930882, upper bound: 0.0904650
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.86
Output dim: 4, lower bound: -0.0930273, upper bound: 0.0900501
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.86
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

Time for backsubstitution: 21.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0899987
time: 2.68 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0899987
time: 2.69 seconds

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

Time for backsubstitution: 21.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0904032
time: 2.61 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0904655
time: 2.65 seconds

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

Time for backsubstitution: 21.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0926215
time: 2.66 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0926214
time: 2.63 seconds

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

Time for backsubstitution: 21.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0930259
time: 2.63 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0930882
time: 2.63 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -9.6709251, -9.1104517, -9.6731062, -9.1104431, -0.2171421, 0.2193929
1: -5.5033646, -5.0882373, -5.5028095, -5.0927596, -0.1741824, 0.1753036
2: -12.0741177, -11.6208887, -12.0576611, -11.6231890, -0.1934547, 0.1943270
3: -9.5845070, -9.1189604, -9.5732641, -9.1209297, -0.1893530, 0.1909480
4: 6.2171655, 6.6548615, 6.2194433, 6.6380939, -0.1397419, 0.1386699
5: -5.0058451, -4.5658393, -5.0079656, -4.5676036, -0.1722189, 0.1757830
6: -13.4734879, -12.7934675, -13.4752636, -12.8042135, -0.2204677, 0.2238153
7: -5.3888783, -5.0298176, -5.3873987, -5.0401287, -0.1549377, 0.1576734
8: -2.6507387, -2.2603035, -2.6506147, -2.2610703, -0.1161496, 0.1170918
9: -4.8165669, -4.4277172, -4.7949347, -4.4310179, -0.1119835, 0.1143774

Time for backsubstitution: 21.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0926214, upper bound: 0.0899987
time: 2.67 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0926214, upper bound: 0.0899987
time: 2.68 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -9.6757212, -9.1023979, -9.6748447, -9.1103792, -0.2203391, 0.2229915
1: -5.5049877, -5.0874271, -5.5031252, -5.0927467, -0.1757478, 0.1761601
2: -12.0765333, -11.6157761, -12.0584421, -11.6228580, -0.1950202, 0.1989826
3: -9.5896358, -9.1090631, -9.5751629, -9.1206589, -0.1926901, 0.1967015
4: 6.2150126, 6.6558352, 6.2191968, 6.6384487, -0.1424979, 0.1400615
5: -5.0112267, -4.5564718, -5.0097980, -4.5674281, -0.1756034, 0.1788743
6: -13.4798431, -12.7824917, -13.4776192, -12.8041782, -0.2242769, 0.2268550
7: -5.3904533, -5.0291328, -5.3876543, -5.0399075, -0.1568471, 0.1586720
8: -2.6528354, -2.2590766, -2.6507535, -2.2606487, -0.1190056, 0.1182138
9: -4.8183494, -4.4241147, -4.7955604, -4.4308333, -0.1132239, 0.1174294

Time for backsubstitution: 21.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0926214, upper bound: 0.0904030
time: 2.65 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0926214, upper bound: 0.0904655
time: 2.79 seconds

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

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0926214, upper bound: 0.0900503
time: 2.85 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0926214, upper bound: 0.0900503
time: 2.87 seconds

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

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0926227, upper bound: 0.0904545
time: 2.67 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0926227, upper bound: 0.0904548
time: 2.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 27.63 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0899987
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0899987
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0904032
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0904655
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0926215
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0926214
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0930259
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0899987, upper bound: 0.0930882
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0926214, upper bound: 0.0899987
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0926214, upper bound: 0.0899987
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0926214, upper bound: 0.0904030
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0926214, upper bound: 0.0904655
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0926214, upper bound: 0.0900503
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0926214, upper bound: 0.0900503
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0926227, upper bound: 0.0904545
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 4, lower bound: -0.0926227, upper bound: 0.0904548

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -9.6704969, -9.1105423, -9.6704969, -9.1105423, -0.2166635, 0.2166634
1: -5.5023479, -5.0927782, -5.5023479, -5.0927782, -0.1728798, 0.1728798
2: -12.0564871, -11.6236944, -12.0564871, -11.6236944, -0.1902437, 0.1902437
3: -9.5704002, -9.1213493, -9.5704002, -9.1213493, -0.1856878, 0.1856878
4: 6.2198257, 6.6375618, 6.2198257, 6.6375618, -0.1363208, 0.1363209
5: -5.0052133, -4.5678730, -5.0052133, -4.5678730, -0.1711388, 0.1711388
6: -13.4717121, -12.8042698, -13.4717121, -12.8042698, -0.2186568, 0.2186568
7: -5.3870010, -5.0404577, -5.3870010, -5.0404577, -0.1527251, 0.1527251
8: -2.6504021, -2.2616959, -2.6504021, -2.2616959, -0.1151858, 0.1151858
9: -4.7939901, -4.4312997, -4.7939901, -4.4312997, -0.1098247, 0.1098247

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1249

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 1732

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0892638, upper bound: 0.0856663
time: 3.54 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0892638, upper bound: 0.0892634
time: 3.04 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -9.6704969, -9.1105423, -9.6752930, -9.1024904, -0.2184968, 0.2212102
1: -5.5023479, -5.0927782, -5.5039687, -5.0919685, -0.1736944, 0.1743499
2: -12.0564871, -11.6236944, -12.0589066, -11.6185818, -0.1940746, 0.1923385
3: -9.5704002, -9.1213493, -9.5755281, -9.1114531, -0.1895121, 0.1908051
4: 6.2198257, 6.6375618, 6.2176719, 6.6385345, -0.1375376, 0.1385697
5: -5.0052133, -4.5678730, -5.0105934, -4.5585051, -0.1740162, 0.1762183
6: -13.4717121, -12.8042698, -13.4780674, -12.7932911, -0.2206739, 0.2241879
7: -5.3870010, -5.0404577, -5.3885770, -5.0397730, -0.1534631, 0.1543533
8: -2.6504021, -2.2616959, -2.6524973, -2.2604699, -0.1164562, 0.1175936
9: -4.7939901, -4.4312997, -4.7957730, -4.4276962, -0.1122327, 0.1114923

Time for backsubstitution: 21.55 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1249

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 1732

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0892638, upper bound: 0.0856662
time: 3.09 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0892638, upper bound: 0.0892633
time: 3.41 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -9.6752930, -9.1024904, -9.6704969, -9.1105423, -0.2212102, 0.2184968
1: -5.5039687, -5.0919685, -5.5023479, -5.0927782, -0.1743499, 0.1736944
2: -12.0589066, -11.6185818, -12.0564871, -11.6236944, -0.1923385, 0.1940746
3: -9.5755281, -9.1114531, -9.5704002, -9.1213493, -0.1908051, 0.1895121
4: 6.2176719, 6.6385345, 6.2198257, 6.6375618, -0.1385697, 0.1375376
5: -5.0105934, -4.5585051, -5.0052133, -4.5678730, -0.1762183, 0.1740162
6: -13.4780674, -12.7932911, -13.4717121, -12.8042698, -0.2241880, 0.2206738
7: -5.3885770, -5.0397730, -5.3870010, -5.0404577, -0.1543533, 0.1534631
8: -2.6524973, -2.2604699, -2.6504021, -2.2616959, -0.1175936, 0.1164562
9: -4.7957730, -4.4276962, -4.7939901, -4.4312997, -0.1114923, 0.1122327

Time for backsubstitution: 21.54 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1249

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 1732

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0892640, upper bound: 0.0860709
time: 2.99 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0892638, upper bound: 0.0896679
time: 3.04 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.6752930, -9.1024904, -9.6752930, -9.1024904, -0.2214706, 0.2214705
1: -5.5039687, -5.0919685, -5.5039687, -5.0919685, -0.1758935, 0.1758935
2: -12.0589066, -11.6185818, -12.0589066, -11.6185818, -0.1931987, 0.1931987
3: -9.5755281, -9.1114531, -9.5755281, -9.1114531, -0.1898080, 0.1898079
4: 6.2176719, 6.6385345, 6.2176719, 6.6385345, -0.1387285, 0.1387285
5: -5.0105934, -4.5585051, -5.0105934, -4.5585051, -0.1759901, 0.1759901
6: -13.4780674, -12.7932911, -13.4780674, -12.7932911, -0.2231930, 0.2231930
7: -5.3885770, -5.0397730, -5.3885770, -5.0397730, -0.1546379, 0.1546379
8: -2.6524973, -2.2604699, -2.6524973, -2.2604699, -0.1168093, 0.1168093
9: -4.7957730, -4.4276962, -4.7957730, -4.4276962, -0.1119784, 0.1119784

Time for backsubstitution: 21.53 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1249

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 1732

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0892638, upper bound: 0.0860710
time: 3.09 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0892638, upper bound: 0.0896678
time: 3.41 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -9.6704969, -9.1105423, -9.6709251, -9.1104517, -0.2167574, 0.2171099
1: -5.5023479, -5.0927782, -5.5033646, -5.0882373, -0.1745349, 0.1740389
2: -12.0564871, -11.6236944, -12.0741177, -11.6208887, -0.1930931, 0.1929926
3: -9.5704002, -9.1213493, -9.5845070, -9.1189604, -0.1880670, 0.1887168
4: 6.2198257, 6.6375618, 6.2171655, 6.6548615, -0.1380439, 0.1389751
5: -5.0052133, -4.5678730, -5.0058451, -4.5658393, -0.1731660, 0.1718465
6: -13.4717121, -12.8042698, -13.4734879, -12.7934675, -0.2207506, 0.2204367
7: -5.3870010, -5.0404577, -5.3888783, -5.0298176, -0.1571069, 0.1545124
8: -2.6504021, -2.2616959, -2.6507387, -2.2603035, -0.1166248, 0.1154729
9: -4.7939901, -4.4312997, -4.8165669, -4.4277172, -0.1134194, 0.1116525

Time for backsubstitution: 21.55 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1249

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 1732

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0892638, upper bound: 0.0887674
time: 3.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0892638, upper bound: 0.0918239
time: 3.06 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.6704969, -9.1105423, -9.6757212, -9.1023979, -0.2185907, 0.2214353
1: -5.5023479, -5.0927782, -5.5049877, -5.0874271, -0.1746788, 0.1755090
2: -12.0564871, -11.6236944, -12.0765333, -11.6157761, -0.1969266, 0.1941429
3: -9.5704002, -9.1213493, -9.5896358, -9.1090631, -0.1918973, 0.1916229
4: 6.2198257, 6.6375618, 6.2150126, 6.6558352, -0.1389047, 0.1412244
5: -5.0052133, -4.5678730, -5.0112267, -4.5564718, -0.1740584, 0.1769223
6: -13.4717121, -12.8042698, -13.4798431, -12.7824917, -0.2208401, 0.2259774
7: -5.3870010, -5.0404577, -5.3904533, -5.0291328, -0.1576583, 0.1561410
8: -2.6504021, -2.2616959, -2.6528354, -2.2590766, -0.1178960, 0.1178808
9: -4.7939901, -4.4312997, -4.8183494, -4.4241147, -0.1158332, 0.1125514

Time for backsubstitution: 21.46 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.45 + 546.89 = 602.34 seconds
