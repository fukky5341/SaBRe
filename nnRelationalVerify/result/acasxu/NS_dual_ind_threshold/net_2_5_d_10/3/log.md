## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.917404983


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732)
1: (-0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920)
2: (-0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294)
3: (-0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840)
4: (-0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.23 + 0.92 = 3.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.9266717, upper bound: 0.9266717

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257123, upper bound: 0.9224874
time: 0.33 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257123, upper bound: 0.9261087
time: 0.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.80 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -0.9257123, upper bound: 0.9224874
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -0.9257123, upper bound: 0.9261087

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.1124772, 1.0333321, -0.1100792, 1.0819939, -1.1944711, 1.1434114
1: -0.1168870, 0.1800460, -0.1110148, 0.1496772, -0.2665642, 0.2910608
2: -0.0336507, 0.2684456, -0.0314484, 0.2418811, -0.2755317, 0.2998939
3: -0.1114517, 0.1417743, -0.1107760, 0.1208080, -0.2322597, 0.2525503
4: -0.0372688, 0.2626405, -0.0367316, 0.2377871, -0.2750560, 0.2993721

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9222105, upper bound: 0.9222105
time: 0.30 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9222105, upper bound: 0.9224874
time: 0.29 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0692852, 0.5910217, -0.1100792, 1.0819939, -1.1512791, 0.7011009
1: -0.0682940, 0.1111395, -0.1110148, 0.1496772, -0.2179712, 0.2221543
2: -0.0113181, 0.1750363, -0.0314484, 0.2418811, -0.2531992, 0.2064846
3: -0.0773237, 0.0747272, -0.1107760, 0.1208080, -0.1981317, 0.1855032
4: -0.0151952, 0.1743963, -0.0367316, 0.2377871, -0.2529824, 0.2111280

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9224874, upper bound: 0.9257123
time: 0.35 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9224874, upper bound: 0.9261088
time: 0.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.84 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 0, lower bound: -0.9222105, upper bound: 0.9222105
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 0, lower bound: -0.9222105, upper bound: 0.9224874
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 0, lower bound: -0.9224874, upper bound: 0.9257123
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 0, lower bound: -0.9224874, upper bound: 0.9261088

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.1124772, 1.0333321, -0.1124772, 1.0333321, -1.1458093, 1.1458093
1: -0.1168870, 0.1800460, -0.1168870, 0.1800460, -0.2969330, 0.2969330
2: -0.0336507, 0.2684456, -0.0336507, 0.2684456, -0.3020962, 0.3020962
3: -0.1114517, 0.1417743, -0.1114517, 0.1417743, -0.2532260, 0.2532260
4: -0.0372688, 0.2626405, -0.0372688, 0.2626405, -0.2999094, 0.2999094

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9201087, upper bound: 0.9201275
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9212320, upper bound: 0.9212320
time: 0.33 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.1124772, 1.0333321, -0.0692852, 0.5910217, -0.7034988, 1.1026173
1: -0.1168870, 0.1800460, -0.0682940, 0.1111395, -0.2280265, 0.2483400
2: -0.0336507, 0.2684456, -0.0113181, 0.1750363, -0.2086869, 0.2797637
3: -0.1114517, 0.1417743, -0.0773237, 0.0747272, -0.1861789, 0.2190980
4: -0.0372688, 0.2626405, -0.0151952, 0.1743963, -0.2116652, 0.2778357

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9201087, upper bound: 0.9204079
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9212320, upper bound: 0.9215859
time: 0.37 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0692852, 0.5910217, -0.1124772, 1.0333321, -1.1026173, 0.7034988
1: -0.0682940, 0.1111395, -0.1168870, 0.1800460, -0.2483400, 0.2280265
2: -0.0113181, 0.1750363, -0.0336507, 0.2684456, -0.2797637, 0.2086869
3: -0.0773237, 0.0747272, -0.1114517, 0.1417743, -0.2190980, 0.1861789
4: -0.0151952, 0.1743963, -0.0372688, 0.2626405, -0.2778357, 0.2116652

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9204284, upper bound: 0.9221685
time: 0.38 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9215859, upper bound: 0.9256088
time: 0.37 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0692852, 0.5910217, -0.0692852, 0.5910217, -0.6603069, 0.6603069
1: -0.0682940, 0.1111395, -0.0682940, 0.1111395, -0.1794335, 0.1794335
2: -0.0113181, 0.1750363, -0.0113181, 0.1750363, -0.1863543, 0.1863543
3: -0.0773237, 0.0747272, -0.0773237, 0.0747272, -0.1520509, 0.1520509
4: -0.0151952, 0.1743963, -0.0151952, 0.1743963, -0.1895916, 0.1895916

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9204284, upper bound: 0.9223308
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9215859, upper bound: 0.9260747
time: 0.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.21 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.9201087, upper bound: 0.9201275
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.9212320, upper bound: 0.9212320
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.9201087, upper bound: 0.9204079
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.9212320, upper bound: 0.9215859
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.9204284, upper bound: 0.9221685
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.9215859, upper bound: 0.9256088
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.9204284, upper bound: 0.9223308
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.9215859, upper bound: 0.9260747

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0706510, 0.7790555, -0.1124772, 1.0333321, -1.1039830, 0.8915327
1: -0.0764569, 0.0875176, -0.1168870, 0.1800460, -0.2565029, 0.2044046
2: -0.0152188, 0.1739050, -0.0336507, 0.2684456, -0.2836644, 0.2075557
3: -0.0846703, 0.0690454, -0.1114517, 0.1417743, -0.2264446, 0.1804971
4: -0.0195002, 0.1793149, -0.0372688, 0.2626405, -0.2821407, 0.2165837

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9190042, upper bound: 0.9190042
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9190042, upper bound: 0.9201275
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1071744, 1.0085871, -0.1124772, 1.0333321, -1.1405065, 1.1210643
1: -0.1098092, 0.1602736, -0.1168870, 0.1800460, -0.2898552, 0.2771606
2: -0.0307444, 0.2449260, -0.0336507, 0.2684456, -0.2991900, 0.2785767
3: -0.1075596, 0.1269811, -0.1114517, 0.1417743, -0.2493339, 0.2384328
4: -0.0345333, 0.2402283, -0.0372688, 0.2626405, -0.2971738, 0.2774971

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9201275, upper bound: 0.9201087
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9201275, upper bound: 0.9212320
time: 0.37 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0706510, 0.7790555, -0.0692852, 0.5910217, -0.6616726, 0.8483407
1: -0.0764569, 0.0875176, -0.0682940, 0.1111395, -0.1875964, 0.1558116
2: -0.0152188, 0.1739050, -0.0113181, 0.1750363, -0.1902551, 0.1852231
3: -0.0846703, 0.0690454, -0.0773237, 0.0747272, -0.1593975, 0.1463691
4: -0.0195002, 0.1793149, -0.0151952, 0.1743963, -0.1938965, 0.1945101

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9210452, upper bound: 0.9193240
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9210452, upper bound: 0.9204079
time: 0.37 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1071744, 1.0085871, -0.0692852, 0.5910217, -0.6981961, 1.0778723
1: -0.1098092, 0.1602736, -0.0682940, 0.1111395, -0.2209487, 0.2285676
2: -0.0307444, 0.2449260, -0.0113181, 0.1750363, -0.2057807, 0.2562441
3: -0.1075596, 0.1269811, -0.0773237, 0.0747272, -0.1822868, 0.2043047
4: -0.0345333, 0.2402283, -0.0151952, 0.1743963, -0.2089296, 0.2554235

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9221685, upper bound: 0.9204285
time: 0.38 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9221685, upper bound: 0.9215859
time: 0.40 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0285890, 0.3429806, -0.1124772, 1.0333321, -1.0619211, 0.4554578
1: -0.0353233, 0.0591630, -0.1168870, 0.1800460, -0.2153693, 0.1760500
2: 0.0049988, 0.1127314, -0.0336507, 0.2684456, -0.2634467, 0.1463821
3: -0.0519922, 0.0294693, -0.1114517, 0.1417743, -0.1937665, 0.1409210
4: 0.0014093, 0.1187848, -0.0372688, 0.2626405, -0.2612312, 0.1560536

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9193240, upper bound: 0.9210452
time: 0.36 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9193240, upper bound: 0.9221685
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0624029, 0.5478467, -0.1124772, 1.0333321, -1.0957350, 0.6603239
1: -0.0596335, 0.0941162, -0.1168870, 0.1800460, -0.2396795, 0.2110032
2: -0.0075298, 0.1537760, -0.0336507, 0.2684456, -0.2759754, 0.1874267
3: -0.0723288, 0.0593504, -0.1114517, 0.1417743, -0.2141031, 0.1708021
4: -0.0114261, 0.1549907, -0.0372688, 0.2626405, -0.2740666, 0.1922595

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9204079, upper bound: 0.9241987
time: 0.35 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9204079, upper bound: 0.9256088
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0285890, 0.3429806, -0.0692852, 0.5910217, -0.6196107, 0.4122658
1: -0.0353233, 0.0591630, -0.0682940, 0.1111395, -0.1464628, 0.1274570
2: 0.0049988, 0.1127314, -0.0113181, 0.1750363, -0.1700374, 0.1240495
3: -0.0519922, 0.0294693, -0.0773237, 0.0747272, -0.1267194, 0.1067930
4: 0.0014093, 0.1187848, -0.0151952, 0.1743963, -0.1729870, 0.1339800

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9213649, upper bound: 0.9213594
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9213649, upper bound: 0.9223308
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0624029, 0.5478467, -0.0692852, 0.5910217, -0.6534245, 0.6171319
1: -0.0596335, 0.0941162, -0.0682940, 0.1111395, -0.1707730, 0.1624102
2: -0.0075298, 0.1537760, -0.0113181, 0.1750363, -0.1825660, 0.1650941
3: -0.0723288, 0.0593504, -0.0773237, 0.0747272, -0.1470560, 0.1366741
4: -0.0114261, 0.1549907, -0.0151952, 0.1743963, -0.1858224, 0.1701859

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9225148, upper bound: 0.9247697
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9225148, upper bound: 0.9260746
time: 0.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.25 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9190042, upper bound: 0.9190042
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9190042, upper bound: 0.9201275
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9201275, upper bound: 0.9201087
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9201275, upper bound: 0.9212320
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9210452, upper bound: 0.9193240
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9210452, upper bound: 0.9204079
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9221685, upper bound: 0.9204285
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9221685, upper bound: 0.9215859
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9193240, upper bound: 0.9210452
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9193240, upper bound: 0.9221685
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9204079, upper bound: 0.9241987
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9204079, upper bound: 0.9256088
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9213649, upper bound: 0.9213594
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9213649, upper bound: 0.9223308
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9225148, upper bound: 0.9247697
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -0.9225148, upper bound: 0.9260746

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0706510, 0.7790555, -0.0706510, 0.7790555, -0.8497065, 0.8497065
1: -0.0764569, 0.0875176, -0.0764569, 0.0875176, -0.1639745, 0.1639745
2: -0.0152188, 0.1739050, -0.0152188, 0.1739050, -0.1891238, 0.1891238
3: -0.0846703, 0.0690454, -0.0846703, 0.0690454, -0.1537157, 0.1537157
4: -0.0195002, 0.1793149, -0.0195002, 0.1793149, -0.1988150, 0.1988150

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9160194, upper bound: 0.9081060
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9185889, upper bound: 0.9185888
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0706510, 0.7790555, -0.1071744, 1.0085871, -1.0792381, 0.8862299
1: -0.0764569, 0.0875176, -0.1098092, 0.1602736, -0.2367306, 0.1973268
2: -0.0152188, 0.1739050, -0.0307444, 0.2449260, -0.2601449, 0.2046494
3: -0.0846703, 0.0690454, -0.1075596, 0.1269811, -0.2116514, 0.1766050
4: -0.0195002, 0.1793149, -0.0345333, 0.2402283, -0.2597285, 0.2138481

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9160194, upper bound: 0.9092647
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9185889, upper bound: 0.9197476
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1071744, 1.0085871, -0.0706510, 0.7790555, -0.8862299, 1.0792381
1: -0.1098092, 0.1602736, -0.0764569, 0.0875176, -0.1973268, 0.2367306
2: -0.0307444, 0.2449260, -0.0152188, 0.1739050, -0.2046494, 0.2601449
3: -0.1075596, 0.1269811, -0.0846703, 0.0690454, -0.1766050, 0.2116514
4: -0.0345333, 0.2402283, -0.0195002, 0.1793149, -0.2138481, 0.2597285

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9180434, upper bound: 0.9136910
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9197476, upper bound: 0.9196936
time: 0.39 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1071744, 1.0085871, -0.1071744, 1.0085871, -1.1157615, 1.1157615
1: -0.1098092, 0.1602736, -0.1098092, 0.1602736, -0.2700829, 0.2700829
2: -0.0307444, 0.2449260, -0.0307444, 0.2449260, -0.2756705, 0.2756705
3: -0.1075596, 0.1269811, -0.1075596, 0.1269811, -0.2345407, 0.2345407
4: -0.0345333, 0.2402283, -0.0345333, 0.2402283, -0.2747616, 0.2747616

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9180435, upper bound: 0.9147996
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9197476, upper bound: 0.9205644
time: 0.39 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0706510, 0.7790555, -0.0285890, 0.3429806, -0.4136316, 0.8076445
1: -0.0764569, 0.0875176, -0.0353233, 0.0591630, -0.1356199, 0.1228409
2: -0.0152188, 0.1739050, 0.0049988, 0.1127314, -0.1279503, 0.1689062
3: -0.0846703, 0.0690454, -0.0519922, 0.0294693, -0.1141396, 0.1210375
4: -0.0195002, 0.1793149, 0.0014093, 0.1187848, -0.1382849, 0.1779056

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9181964, upper bound: 0.9084434
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9207659, upper bound: 0.9189263
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0706510, 0.7790555, -0.0624029, 0.5478467, -0.6184977, 0.8414584
1: -0.0764569, 0.0875176, -0.0596335, 0.0941162, -0.1705731, 0.1471511
2: -0.0152188, 0.1739050, -0.0075298, 0.1537760, -0.1689948, 0.1814348
3: -0.0846703, 0.0690454, -0.0723288, 0.0593504, -0.1440207, 0.1413742
4: -0.0195002, 0.1793149, -0.0114261, 0.1549907, -0.1744908, 0.1907410

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9181964, upper bound: 0.9096099
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9207659, upper bound: 0.9200374
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1071744, 1.0085871, -0.0285890, 0.3429806, -0.4501550, 1.0371761
1: -0.1098092, 0.1602736, -0.0353233, 0.0591630, -0.1689722, 0.1955969
2: -0.0307444, 0.2449260, 0.0049988, 0.1127314, -0.1434759, 0.2399272
3: -0.1075596, 0.1269811, -0.0519922, 0.0294693, -0.1370289, 0.1789732
4: -0.0345333, 0.2402283, 0.0014093, 0.1187848, -0.1533180, 0.2388190

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9202205, upper bound: 0.9140285
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9219247, upper bound: 0.9200311
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1071744, 1.0085871, -0.0624029, 0.5478467, -0.6550211, 1.0709900
1: -0.1098092, 0.1602736, -0.0596335, 0.0941162, -0.2039254, 0.2199071
2: -0.0307444, 0.2449260, -0.0075298, 0.1537760, -0.1845205, 0.2524558
3: -0.1075596, 0.1269811, -0.0723288, 0.0593504, -0.1669101, 0.1993098
4: -0.0345333, 0.2402283, -0.0114261, 0.1549907, -0.1895239, 0.2516544

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9202205, upper bound: 0.9151834
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9219247, upper bound: 0.9209989
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0285890, 0.3429806, -0.0706510, 0.7790555, -0.8076445, 0.4136316
1: -0.0353233, 0.0591630, -0.0764569, 0.0875176, -0.1228409, 0.1356199
2: 0.0049988, 0.1127314, -0.0152188, 0.1739050, -0.1689062, 0.1279503
3: -0.0519922, 0.0294693, -0.0846703, 0.0690454, -0.1210375, 0.1141396
4: 0.0014093, 0.1187848, -0.0195002, 0.1793149, -0.1779056, 0.1382849

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0285890, 0.3429806, -0.1071744, 1.0085871, -1.0371761, 0.4501550
1: -0.0353233, 0.0591630, -0.1098092, 0.1602736, -0.1955969, 0.1689722
2: 0.0049988, 0.1127314, -0.0307444, 0.2449260, -0.2399272, 0.1434759
3: -0.0519922, 0.0294693, -0.1075596, 0.1269811, -0.1789732, 0.1370289
4: 0.0014093, 0.1187848, -0.0345333, 0.2402283, -0.2388190, 0.1533180

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0624029, 0.5478467, -0.0706510, 0.7790555, -0.8414584, 0.6184977
1: -0.0596335, 0.0941162, -0.0764569, 0.0875176, -0.1471511, 0.1705731
2: -0.0075298, 0.1537760, -0.0152188, 0.1739050, -0.1814348, 0.1689948
3: -0.0723288, 0.0593504, -0.0846703, 0.0690454, -0.1413742, 0.1440207
4: -0.0114261, 0.1549907, -0.0195002, 0.1793149, -0.1907410, 0.1744908

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0624029, 0.5478467, -0.1071744, 1.0085871, -1.0709900, 0.6550211
1: -0.0596335, 0.0941162, -0.1098092, 0.1602736, -0.2199071, 0.2039254
2: -0.0075298, 0.1537760, -0.0307444, 0.2449260, -0.2524558, 0.1845205
3: -0.0723288, 0.0593504, -0.1075596, 0.1269811, -0.1993098, 0.1669101
4: -0.0114261, 0.1549907, -0.0345333, 0.2402283, -0.2516544, 0.1895239

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0285890, 0.3429806, -0.0285890, 0.3429806, -0.3715696, 0.3715696
1: -0.0353233, 0.0591630, -0.0353233, 0.0591630, -0.0944863, 0.0944863
2: 0.0049988, 0.1127314, 0.0049988, 0.1127314, -0.1077326, 0.1077326
3: -0.0519922, 0.0294693, -0.0519922, 0.0294693, -0.0814614, 0.0814614
4: 0.0014093, 0.1187848, 0.0014093, 0.1187848, -0.1173755, 0.1173755

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0285890, 0.3429806, -0.0624029, 0.5478467, -0.5764357, 0.4053835
1: -0.0353233, 0.0591630, -0.0596335, 0.0941162, -0.1294395, 0.1187965
2: 0.0049988, 0.1127314, -0.0075298, 0.1537760, -0.1487772, 0.1202612
3: -0.0519922, 0.0294693, -0.0723288, 0.0593504, -0.1113426, 0.1017981
4: 0.0014093, 0.1187848, -0.0114261, 0.1549907, -0.1535814, 0.1302109

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0624029, 0.5478467, -0.0285890, 0.3429806, -0.4053835, 0.5764357
1: -0.0596335, 0.0941162, -0.0353233, 0.0591630, -0.1187965, 0.1294395
2: -0.0075298, 0.1537760, 0.0049988, 0.1127314, -0.1202612, 0.1487772
3: -0.0723288, 0.0593504, -0.0519922, 0.0294693, -0.1017981, 0.1113426
4: -0.0114261, 0.1549907, 0.0014093, 0.1187848, -0.1302109, 0.1535814

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0624029, 0.5478467, -0.0624029, 0.5478467, -0.6102496, 0.6102496
1: -0.0596335, 0.0941162, -0.0596335, 0.0941162, -0.1537497, 0.1537497
2: -0.0075298, 0.1537760, -0.0075298, 0.1537760, -0.1613058, 0.1613058
3: -0.0723288, 0.0593504, -0.0723288, 0.0593504, -0.1316792, 0.1316792
4: -0.0114261, 0.1549907, -0.0114261, 0.1549907, -0.1664168, 0.1664168

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.15 + 101.66 = 104.81 seconds
