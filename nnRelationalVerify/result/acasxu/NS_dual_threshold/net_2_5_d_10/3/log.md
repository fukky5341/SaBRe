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
execution time: IAR + RelationalAnalysis = 2.09 + 0.92 = 3.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.9266717, upper bound: 0.9266717

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257123, upper bound: 0.9224874
time: 0.32 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257123, upper bound: 0.9261087
time: 0.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.79 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.9257123, upper bound: 0.9224874
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.9257123, upper bound: 0.9261087

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.1124772, 1.0333321, -0.1100792, 1.0819939, -1.1944711, 1.1434114
1: -0.1168870, 0.1800460, -0.1110148, 0.1496772, -0.2665642, 0.2910608
2: -0.0336507, 0.2684456, -0.0314484, 0.2418811, -0.2755317, 0.2998939
3: -0.1114517, 0.1417743, -0.1107760, 0.1208080, -0.2322597, 0.2525503
4: -0.0372688, 0.2626405, -0.0367316, 0.2377871, -0.2750560, 0.2993721

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9222105, upper bound: 0.9222105
time: 0.28 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9222105, upper bound: 0.9224874
time: 0.28 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0692852, 0.5910217, -0.1100792, 1.0819939, -1.1512791, 0.7011009
1: -0.0682940, 0.1111395, -0.1110148, 0.1496772, -0.2179712, 0.2221543
2: -0.0113181, 0.1750363, -0.0314484, 0.2418811, -0.2531992, 0.2064846
3: -0.0773237, 0.0747272, -0.1107760, 0.1208080, -0.1981317, 0.1855032
4: -0.0151952, 0.1743963, -0.0367316, 0.2377871, -0.2529824, 0.2111280

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

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
time: 0.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.74 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 0, lower bound: -0.9222105, upper bound: 0.9222105
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 0, lower bound: -0.9222105, upper bound: 0.9224874
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 0, lower bound: -0.9224874, upper bound: 0.9257123
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 0, lower bound: -0.9224874, upper bound: 0.9261088

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.1124772, 1.0333321, -0.1124772, 1.0333321, -1.1458093, 1.1458093
1: -0.1168870, 0.1800460, -0.1168870, 0.1800460, -0.2969330, 0.2969330
2: -0.0336507, 0.2684456, -0.0336507, 0.2684456, -0.3020962, 0.3020962
3: -0.1114517, 0.1417743, -0.1114517, 0.1417743, -0.2532260, 0.2532260
4: -0.0372688, 0.2626405, -0.0372688, 0.2626405, -0.2999094, 0.2999094

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9205978, upper bound: 0.9173841
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9222105, upper bound: 0.9222105
time: 0.30 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.1124772, 1.0333321, -0.0692852, 0.5910217, -0.7034988, 1.1026173
1: -0.1168870, 0.1800460, -0.0682940, 0.1111395, -0.2280265, 0.2483400
2: -0.0336507, 0.2684456, -0.0113181, 0.1750363, -0.2086869, 0.2797637
3: -0.1114517, 0.1417743, -0.0773237, 0.0747272, -0.1861789, 0.2190980
4: -0.0372688, 0.2626405, -0.0151952, 0.1743963, -0.2116652, 0.2778357

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9201275, upper bound: 0.9204285
time: 0.28 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9212320, upper bound: 0.9215859
time: 0.31 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0692852, 0.5910217, -0.1124772, 1.0333321, -1.1026173, 0.7034988
1: -0.0682940, 0.1111395, -0.1168870, 0.1800460, -0.2483400, 0.2280265
2: -0.0113181, 0.1750363, -0.0336507, 0.2684456, -0.2797637, 0.2086869
3: -0.0773237, 0.0747272, -0.1114517, 0.1417743, -0.2190980, 0.1861789
4: -0.0151952, 0.1743963, -0.0372688, 0.2626405, -0.2778357, 0.2116652

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9204284, upper bound: 0.9221685
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9215859, upper bound: 0.9256088
time: 0.30 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0692852, 0.5910217, -0.0692852, 0.5910217, -0.6603069, 0.6603069
1: -0.0682940, 0.1111395, -0.0682940, 0.1111395, -0.1794335, 0.1794335
2: -0.0113181, 0.1750363, -0.0113181, 0.1750363, -0.1863543, 0.1863543
3: -0.0773237, 0.0747272, -0.0773237, 0.0747272, -0.1520509, 0.1520509
4: -0.0151952, 0.1743963, -0.0151952, 0.1743963, -0.1895916, 0.1895916

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9204284, upper bound: 0.9223308
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9215859, upper bound: 0.9260747
time: 0.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.13 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.9205978, upper bound: 0.9173841
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.9222105, upper bound: 0.9222105
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.9201275, upper bound: 0.9204285
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.9212320, upper bound: 0.9215859
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.9204284, upper bound: 0.9221685
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.9215859, upper bound: 0.9256088
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.9204284, upper bound: 0.9223308
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.9215859, upper bound: 0.9260747

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1437851, 1.0390234, -0.1124772, 1.0333321, -1.1771172, 1.1515006
1: -0.1356380, 0.2822680, -0.1168870, 0.1800460, -0.3156840, 0.3991550
2: -0.0403091, 0.3860257, -0.0336507, 0.2684456, -0.3087547, 0.4196763
3: -0.1215828, 0.1990471, -0.1114517, 0.1417743, -0.2633571, 0.3104988
4: -0.0424740, 0.3740170, -0.0372688, 0.2626405, -0.3051145, 0.4112858

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9180700, upper bound: 0.9153696
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9191744, upper bound: 0.9164928
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0946860, 0.8907769, -0.1124772, 1.0333321, -1.1280181, 1.0032541
1: -0.1009413, 0.1579757, -0.1168870, 0.1800460, -0.2809873, 0.2748627
2: -0.0243350, 0.2426133, -0.0336507, 0.2684456, -0.2927805, 0.2762640
3: -0.0963039, 0.1191721, -0.1114517, 0.1417743, -0.2380783, 0.2306238
4: -0.0269555, 0.2384191, -0.0372688, 0.2626405, -0.2895960, 0.2756879

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9173841, upper bound: 0.9205978
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9173841, upper bound: 0.9222105
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.1124772, 1.0333321, -0.0285890, 0.3429806, -0.4554578, 1.0619211
1: -0.1168870, 0.1800460, -0.0353233, 0.0591630, -0.1760500, 0.2153693
2: -0.0336507, 0.2684456, 0.0049988, 0.1127314, -0.1463821, 0.2634467
3: -0.1114517, 0.1417743, -0.0519922, 0.0294693, -0.1409210, 0.1937665
4: -0.0372688, 0.2626405, 0.0014093, 0.1187848, -0.1560536, 0.2612312

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9202205, upper bound: 0.9140286
time: 0.29 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9219247, upper bound: 0.9200311
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.1124772, 1.0333321, -0.0624029, 0.5478467, -0.6603239, 1.0957350
1: -0.1168870, 0.1800460, -0.0596335, 0.0941162, -0.2110032, 0.2396795
2: -0.0336507, 0.2684456, -0.0075298, 0.1537760, -0.1874267, 0.2759754
3: -0.1114517, 0.1417743, -0.0723288, 0.0593504, -0.1708021, 0.2141031
4: -0.0372688, 0.2626405, -0.0114261, 0.1549907, -0.1922595, 0.2740666

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9236329, upper bound: 0.9152404
time: 0.41 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9252479, upper bound: 0.9212079
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0285890, 0.3429806, -0.1124772, 1.0333321, -1.0619211, 0.4554578
1: -0.0353233, 0.0591630, -0.1168870, 0.1800460, -0.2153693, 0.1760500
2: 0.0049988, 0.1127314, -0.0336507, 0.2684456, -0.2634467, 0.1463821
3: -0.0519922, 0.0294693, -0.1114517, 0.1417743, -0.1937665, 0.1409210
4: 0.0014093, 0.1187848, -0.0372688, 0.2626405, -0.2612312, 0.1560536

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9140286, upper bound: 0.9202205
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9200311, upper bound: 0.9219247
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0624029, 0.5478467, -0.1124772, 1.0333321, -1.0957350, 0.6603239
1: -0.0596335, 0.0941162, -0.1168870, 0.1800460, -0.2396795, 0.2110032
2: -0.0075298, 0.1537760, -0.0336507, 0.2684456, -0.2759754, 0.1874267
3: -0.0723288, 0.0593504, -0.1114517, 0.1417743, -0.2141031, 0.1708021
4: -0.0114261, 0.1549907, -0.0372688, 0.2626405, -0.2740666, 0.1922595

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9152404, upper bound: 0.9236329
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9212079, upper bound: 0.9252479
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0285890, 0.3429806, -0.0692852, 0.5910217, -0.6196107, 0.4122658
1: -0.0353233, 0.0591630, -0.0682940, 0.1111395, -0.1464628, 0.1274570
2: 0.0049988, 0.1127314, -0.0113181, 0.1750363, -0.1700374, 0.1240495
3: -0.0519922, 0.0294693, -0.0773237, 0.0747272, -0.1267194, 0.1067930
4: 0.0014093, 0.1187848, -0.0151952, 0.1743963, -0.1729870, 0.1339800

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9208249, upper bound: 0.9223250
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9213649, upper bound: 0.9213594
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9213649, upper bound: 0.9223308
time: 0.29 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0624029, 0.5478467, -0.0692852, 0.5910217, -0.6534245, 0.6171319
1: -0.0596335, 0.0941162, -0.0682940, 0.1111395, -0.1707730, 0.1624102
2: -0.0075298, 0.1537760, -0.0113181, 0.1750363, -0.1825660, 0.1650941
3: -0.0723288, 0.0593504, -0.0773237, 0.0747272, -0.1470560, 0.1366741
4: -0.0114261, 0.1549907, -0.0151952, 0.1743963, -0.1858224, 0.1701859

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9225148, upper bound: 0.9247697
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9225148, upper bound: 0.9260746
time: 0.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.17 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9180700, upper bound: 0.9153696
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9191744, upper bound: 0.9164928
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9173841, upper bound: 0.9205978
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9173841, upper bound: 0.9222105
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9202205, upper bound: 0.9140286
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9219247, upper bound: 0.9200311
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9236329, upper bound: 0.9152404
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9252479, upper bound: 0.9212079
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9140286, upper bound: 0.9202205
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9200311, upper bound: 0.9219247
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9152404, upper bound: 0.9236329
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9212079, upper bound: 0.9252479
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9213649, upper bound: 0.9213594
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9213649, upper bound: 0.9223308
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9225148, upper bound: 0.9247697
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.9225148, upper bound: 0.9260746

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1437851, 1.0390234, -0.0706510, 0.7790555, -0.9228407, 1.1096743
1: -0.1356380, 0.2822680, -0.0764569, 0.0875176, -0.2231556, 0.3587250
2: -0.0403091, 0.3860257, -0.0152188, 0.1739050, -0.2142141, 0.4012445
3: -0.1215828, 0.1990471, -0.0846703, 0.0690454, -0.1906281, 0.2837173
4: -0.0424740, 0.3740170, -0.0195002, 0.1793149, -0.2217889, 0.3935172

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9172425, upper bound: 0.9118512
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176178, upper bound: 0.9143400
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1437851, 1.0390234, -0.1071744, 1.0085871, -1.1523722, 1.1461978
1: -0.1356380, 0.2822680, -0.1098092, 0.1602736, -0.2959117, 0.3920773
2: -0.0403091, 0.3860257, -0.0307444, 0.2449260, -0.2852352, 0.4167701
3: -0.1215828, 0.1990471, -0.1075596, 0.1269811, -0.2485638, 0.3066067
4: -0.0424740, 0.3740170, -0.0345333, 0.2402283, -0.2827023, 0.4085503

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9144353, upper bound: 0.9144353
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9144353, upper bound: 0.9164928
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0946860, 0.8907769, -0.1437851, 1.0390234, -1.1337094, 1.0345621
1: -0.1009413, 0.1579757, -0.1356380, 0.2822680, -0.3832094, 0.2936137
2: -0.0243350, 0.2426133, -0.0403091, 0.3860257, -0.4103606, 0.2829224
3: -0.0963039, 0.1191721, -0.1215828, 0.1990471, -0.2953510, 0.2407549
4: -0.0269555, 0.2384191, -0.0424740, 0.3740170, -0.4009725, 0.2808931

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9140216, upper bound: 0.9196576
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9167717, upper bound: 0.9201382
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0946860, 0.8907769, -0.0946860, 0.8907769, -0.9854630, 0.9854630
1: -0.1009413, 0.1579757, -0.1009413, 0.1579757, -0.2589170, 0.2589170
2: -0.0243350, 0.2426133, -0.0243350, 0.2426133, -0.2669483, 0.2669483
3: -0.0963039, 0.1191721, -0.0963039, 0.1191721, -0.2154761, 0.2154761
4: -0.0269555, 0.2384191, -0.0269555, 0.2384191, -0.2653746, 0.2653746

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9153696, upper bound: 0.9200216
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9164928, upper bound: 0.9210123
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0715488, 0.8423631, -0.0285890, 0.3429806, -0.4145295, 0.8709521
1: -0.0761303, 0.0800685, -0.0353233, 0.0591630, -0.1352933, 0.1153918
2: -0.0131412, 0.1637267, 0.0049988, 0.1127314, -0.1258727, 0.1587279
3: -0.0824926, 0.0633265, -0.0519922, 0.0294693, -0.1119619, 0.1153187
4: -0.0185535, 0.1671162, 0.0014093, 0.1187848, -0.1373382, 0.1657070

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9181964, upper bound: 0.9084435
time: 0.31 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9181964, upper bound: 0.9140286
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1116339, 1.0280960, -0.0285890, 0.3429806, -0.4546145, 1.0566850
1: -0.1160147, 0.1785062, -0.0353233, 0.0591630, -0.1751777, 0.2138294
2: -0.0332533, 0.2668678, 0.0049988, 0.1127314, -0.1459847, 0.2618689
3: -0.1109314, 0.1404098, -0.0519922, 0.0294693, -0.1404007, 0.1924020
4: -0.0368760, 0.2612246, 0.0014093, 0.1187848, -0.1556608, 0.2598153

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9207659, upper bound: 0.9189263
time: 0.32 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9207659, upper bound: 0.9200311
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0715488, 0.8423631, -0.0624029, 0.5478467, -0.6193956, 0.9047660
1: -0.0761303, 0.0800685, -0.0596335, 0.0941162, -0.1702465, 0.1397020
2: -0.0131412, 0.1637267, -0.0075298, 0.1537760, -0.1669173, 0.1712565
3: -0.0824926, 0.0633265, -0.0723288, 0.0593504, -0.1418430, 0.1356553
4: -0.0185535, 0.1671162, -0.0114261, 0.1549907, -0.1735441, 0.1785424

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9181964, upper bound: 0.9096099
time: 0.32 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9181964, upper bound: 0.9151834
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1116339, 1.0280960, -0.0624029, 0.5478467, -0.6594806, 1.0904988
1: -0.1160147, 0.1785062, -0.0596335, 0.0941162, -0.2101309, 0.2381397
2: -0.0332533, 0.2668678, -0.0075298, 0.1537760, -0.1870293, 0.2743976
3: -0.1109314, 0.1404098, -0.0723288, 0.0593504, -0.1702818, 0.2127386
4: -0.0368760, 0.2612246, -0.0114261, 0.1549907, -0.1918667, 0.2726507

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9207659, upper bound: 0.9200373
time: 0.31 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9207659, upper bound: 0.9209989
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0285890, 0.3429806, -0.0715488, 0.8423631, -0.8709521, 0.4145295
1: -0.0353233, 0.0591630, -0.0761303, 0.0800685, -0.1153918, 0.1352933
2: 0.0049988, 0.1127314, -0.0131412, 0.1637267, -0.1587279, 0.1258727
3: -0.0519922, 0.0294693, -0.0824926, 0.0633265, -0.1153187, 0.1119619
4: 0.0014093, 0.1187848, -0.0185535, 0.1671162, -0.1657070, 0.1373382

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9084435, upper bound: 0.9181964
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9084435, upper bound: 0.9202205
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0285890, 0.3429806, -0.1116339, 1.0280960, -1.0566850, 0.4546145
1: -0.0353233, 0.0591630, -0.1160147, 0.1785062, -0.2138294, 0.1751777
2: 0.0049988, 0.1127314, -0.0332533, 0.2668678, -0.2618689, 0.1459847
3: -0.0519922, 0.0294693, -0.1109314, 0.1404098, -0.1924020, 0.1404007
4: 0.0014093, 0.1187848, -0.0368760, 0.2612246, -0.2598153, 0.1556608

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9189263, upper bound: 0.9207659
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9189263, upper bound: 0.9219247
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0624029, 0.5478467, -0.0715488, 0.8423631, -0.9047660, 0.6193956
1: -0.0596335, 0.0941162, -0.0761303, 0.0800685, -0.1397020, 0.1702465
2: -0.0075298, 0.1537760, -0.0131412, 0.1637267, -0.1712565, 0.1669173
3: -0.0723288, 0.0593504, -0.0824926, 0.0633265, -0.1356553, 0.1418430
4: -0.0114261, 0.1549907, -0.0185535, 0.1671162, -0.1785424, 0.1735441

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9096099, upper bound: 0.9214790
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9096099, upper bound: 0.9230404
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0624029, 0.5478467, -0.1116339, 1.0280960, -1.0904988, 0.6594806
1: -0.0596335, 0.0941162, -0.1160147, 0.1785062, -0.2381397, 0.2101309
2: -0.0075298, 0.1537760, -0.0332533, 0.2668678, -0.2743976, 0.1870293
3: -0.0723288, 0.0593504, -0.1109314, 0.1404098, -0.2127386, 0.1702818
4: -0.0114261, 0.1549907, -0.0368760, 0.2612246, -0.2726507, 0.1918667

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9200374, upper bound: 0.9238312
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9200374, upper bound: 0.9247364
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0285890, 0.3429806, -0.0285890, 0.3429806, -0.3715696, 0.3715696
1: -0.0353233, 0.0591630, -0.0353233, 0.0591630, -0.0944863, 0.0944863
2: 0.0049988, 0.1127314, 0.0049988, 0.1127314, -0.1077326, 0.1077326
3: -0.0519922, 0.0294693, -0.0519922, 0.0294693, -0.0814614, 0.0814614
4: 0.0014093, 0.1187848, 0.0014093, 0.1187848, -0.1173755, 0.1173755

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0285890, 0.3429806, -0.0624029, 0.5478467, -0.5764357, 0.4053835
1: -0.0353233, 0.0591630, -0.0596335, 0.0941162, -0.1294395, 0.1187965
2: 0.0049988, 0.1127314, -0.0075298, 0.1537760, -0.1487772, 0.1202612
3: -0.0519922, 0.0294693, -0.0723288, 0.0593504, -0.1113426, 0.1017981
4: 0.0014093, 0.1187848, -0.0114261, 0.1549907, -0.1535814, 0.1302109

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0624029, 0.5478467, -0.0285890, 0.3429806, -0.4053835, 0.5764357
1: -0.0596335, 0.0941162, -0.0353233, 0.0591630, -0.1187965, 0.1294395
2: -0.0075298, 0.1537760, 0.0049988, 0.1127314, -0.1202612, 0.1487772
3: -0.0723288, 0.0593504, -0.0519922, 0.0294693, -0.1017981, 0.1113426
4: -0.0114261, 0.1549907, 0.0014093, 0.1187848, -0.1302109, 0.1535814

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9225148, upper bound: 0.9245903
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0624029, 0.5478467, -0.0624029, 0.5478467, -0.6102496, 0.6102496
1: -0.0596335, 0.0941162, -0.0596335, 0.0941162, -0.1537497, 0.1537497
2: -0.0075298, 0.1537760, -0.0075298, 0.1537760, -0.1613058, 0.1613058
3: -0.0723288, 0.0593504, -0.0723288, 0.0593504, -0.1316792, 0.1316792
4: -0.0114261, 0.1549907, -0.0114261, 0.1549907, -0.1664168, 0.1664168

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.01 + 87.02 = 90.02 seconds
