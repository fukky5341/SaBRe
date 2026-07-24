## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.12280842500000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0722671, 0.2333107, 0.0722671, 0.2333107, -0.1610436, 0.1610436)
1: (0.0389306, 0.2428598, 0.0389306, 0.2428598, -0.2039292, 0.2039292)
2: (0.0560051, 0.2667054, 0.0560051, 0.2667054, -0.2107002, 0.2107002)
3: (-0.0283638, 0.2345866, -0.0283638, 0.2345866, -0.2629504, 0.2629504)
4: (-0.0069483, 0.2760361, -0.0069483, 0.2760361, -0.2829843, 0.2829843)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.66 + 0.75 = 2.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.1444805, upper bound: 0.1444805

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1439579, upper bound: 0.1368941
time: 0.22 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1433590, upper bound: 0.1433589
time: 0.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.59 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.1439579, upper bound: 0.1368941
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.1433590, upper bound: 0.1433589

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0820065, 0.2198348, 0.0722671, 0.2333107, -0.1513042, 0.1475676
1: 0.0510250, 0.2293532, 0.0389306, 0.2428598, -0.1918348, 0.1904226
2: 0.0681462, 0.2446522, 0.0560051, 0.2667054, -0.1985592, 0.1886471
3: -0.0115750, 0.2246487, -0.0283638, 0.2345866, -0.2461616, 0.2530125
4: 0.0048897, 0.2434022, -0.0069483, 0.2760361, -0.2711464, 0.2503505

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1368941, upper bound: 0.1368941
time: 0.24 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1368941, upper bound: 0.1368941
time: 0.24 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0732364, 0.2302559, 0.0722671, 0.2333107, -0.1600743, 0.1579888
1: 0.0410464, 0.2404269, 0.0389306, 0.2428598, -0.2018134, 0.2014963
2: 0.0570673, 0.2617090, 0.0560051, 0.2667054, -0.2096381, 0.2057039
3: -0.0252035, 0.2326369, -0.0283638, 0.2345866, -0.2597901, 0.2610008
4: -0.0057263, 0.2690515, -0.0069483, 0.2760361, -0.2817624, 0.2759998

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1368941, upper bound: 0.1433590
time: 0.24 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1368941, upper bound: 0.1433590
time: 0.24 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.17 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -0.1368941, upper bound: 0.1368941
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -0.1368941, upper bound: 0.1368941
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -0.1368941, upper bound: 0.1433590
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -0.1368941, upper bound: 0.1433590

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0820065, 0.2198348, 0.0820065, 0.2198348, -0.1378283, 0.1378283
1: 0.0510250, 0.2293532, 0.0510250, 0.2293532, -0.1783282, 0.1783282
2: 0.0681462, 0.2446522, 0.0681462, 0.2446522, -0.1765060, 0.1765060
3: -0.0115750, 0.2246487, -0.0115750, 0.2246487, -0.2362237, 0.2362237
4: 0.0048897, 0.2434022, 0.0048897, 0.2434022, -0.2385125, 0.2385125

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1355605, upper bound: 0.1338131
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1334853, upper bound: 0.1329660
time: 0.23 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.0820065, 0.2198348, 0.0732364, 0.2302559, -0.1482494, 0.1465983
1: 0.0510250, 0.2293532, 0.0410464, 0.2404269, -0.1894019, 0.1883067
2: 0.0681462, 0.2446522, 0.0570673, 0.2617090, -0.1935628, 0.1875849
3: -0.0115750, 0.2246487, -0.0252035, 0.2326369, -0.2442120, 0.2498522
4: 0.0048897, 0.2434022, -0.0057263, 0.2690515, -0.2641618, 0.2491286

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1355605, upper bound: 0.1338131
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1334853, upper bound: 0.1329660
time: 0.24 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0732364, 0.2302559, 0.0820065, 0.2198348, -0.1465983, 0.1482494
1: 0.0410464, 0.2404269, 0.0510250, 0.2293532, -0.1883067, 0.1894019
2: 0.0570673, 0.2617090, 0.0681462, 0.2446522, -0.1875849, 0.1935628
3: -0.0252035, 0.2326369, -0.0115750, 0.2246487, -0.2498522, 0.2442120
4: -0.0057263, 0.2690515, 0.0048897, 0.2434022, -0.2491286, 0.2641618

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1349615, upper bound: 0.1404063
time: 0.25 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273
time: 0.24 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0732364, 0.2302559, 0.0732364, 0.2302559, -0.1570195, 0.1570195
1: 0.0410464, 0.2404269, 0.0410464, 0.2404269, -0.1993805, 0.1993805
2: 0.0570673, 0.2617090, 0.0570673, 0.2617090, -0.2046417, 0.2046417
3: -0.0252035, 0.2326369, -0.0252035, 0.2326369, -0.2578405, 0.2578405
4: -0.0057263, 0.2690515, -0.0057263, 0.2690515, -0.2747778, 0.2747778

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1349615, upper bound: 0.1404064
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273
time: 0.24 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.18 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -0.1355605, upper bound: 0.1338131
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -0.1334853, upper bound: 0.1329660
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -0.1355605, upper bound: 0.1338131
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -0.1334853, upper bound: 0.1329660
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -0.1349615, upper bound: 0.1404063
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -0.1349615, upper bound: 0.1404064
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0918941, 0.2120161, 0.0820065, 0.2198348, -0.1279406, 0.1300096
1: 0.0578728, 0.2103683, 0.0510250, 0.2293532, -0.1714804, 0.1593433
2: 0.0779552, 0.2351160, 0.0681462, 0.2446522, -0.1666970, 0.1669698
3: -0.0050839, 0.1987448, -0.0115750, 0.2246487, -0.2297325, 0.2103199
4: 0.0195638, 0.2308713, 0.0048897, 0.2434022, -0.2238384, 0.2259816

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1339596, upper bound: 0.1339596
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1339596, upper bound: 0.1339596
time: 0.24 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0767667, 0.2262760, 0.0820065, 0.2198348, -0.1430681, 0.1442695
1: 0.0229453, 0.2282764, 0.0510250, 0.2293532, -0.2064079, 0.1772514
2: 0.0670445, 0.2697541, 0.0681462, 0.2446522, -0.1776077, 0.2016079
3: -0.0565433, 0.2209895, -0.0115750, 0.2246487, -0.2811919, 0.2325645
4: 0.0042171, 0.2863551, 0.0048897, 0.2434022, -0.2391852, 0.2814654

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1339596, upper bound: 0.1339596
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1339596, upper bound: 0.1339596
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0918941, 0.2120161, 0.0732364, 0.2302559, -0.1383618, 0.1387797
1: 0.0578728, 0.2103683, 0.0410464, 0.2404269, -0.1825541, 0.1693218
2: 0.0779552, 0.2351160, 0.0570673, 0.2617090, -0.1837538, 0.1780487
3: -0.0050839, 0.1987448, -0.0252035, 0.2326369, -0.2377208, 0.2239484
4: 0.0195638, 0.2308713, -0.0057263, 0.2690515, -0.2494877, 0.2365976

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1340925, upper bound: 0.1329660
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1340925, upper bound: 0.1329660
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0767667, 0.2262760, 0.0732364, 0.2302559, -0.1534892, 0.1530395
1: 0.0229453, 0.2282764, 0.0410464, 0.2404269, -0.2174816, 0.1872300
2: 0.0670445, 0.2697541, 0.0570673, 0.2617090, -0.1946644, 0.2126868
3: -0.0565433, 0.2209895, -0.0252035, 0.2326369, -0.2891802, 0.2461930
4: 0.0042171, 0.2863551, -0.0057263, 0.2690515, -0.2648344, 0.2920814

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1340925, upper bound: 0.1329660
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1340925, upper bound: 0.1329660
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0819002, 0.2224207, 0.0820065, 0.2198348, -0.1379345, 0.1404142
1: 0.0490453, 0.2219181, 0.0510250, 0.2293532, -0.1803079, 0.1708931
2: 0.0662161, 0.2516250, 0.0681462, 0.2446522, -0.1784362, 0.1834788
3: -0.0185044, 0.2083519, -0.0115750, 0.2246487, -0.2431531, 0.2199270
4: 0.0070564, 0.2554384, 0.0048897, 0.2434022, -0.2363459, 0.2505487

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1329660, upper bound: 0.1340925
time: 0.25 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1329660, upper bound: 0.1340925
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0699754, 0.2389362, 0.0820065, 0.2198348, -0.1498594, 0.1569297
1: 0.0170647, 0.2386436, 0.0510250, 0.2293532, -0.2122885, 0.1876186
2: 0.0581372, 0.2883278, 0.0681462, 0.2446522, -0.1865150, 0.2201816
3: -0.0680611, 0.2284164, -0.0115750, 0.2246487, -0.2927097, 0.2399914
4: -0.0050042, 0.3116492, 0.0048897, 0.2434022, -0.2484064, 0.3067595

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1329660, upper bound: 0.1340925
time: 0.25 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1329660, upper bound: 0.1340925
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0819002, 0.2224207, 0.0732364, 0.2302559, -0.1483557, 0.1491843
1: 0.0490453, 0.2219181, 0.0410464, 0.2404269, -0.1913816, 0.1808717
2: 0.0662161, 0.2516250, 0.0570673, 0.2617090, -0.1954929, 0.1945577
3: -0.0185044, 0.2083519, -0.0252035, 0.2326369, -0.2511414, 0.2335554
4: 0.0070564, 0.2554384, -0.0057263, 0.2690515, -0.2619951, 0.2611647

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0699754, 0.2389362, 0.0732364, 0.2302559, -0.1602805, 0.1656998
1: 0.0170647, 0.2386436, 0.0410464, 0.2404269, -0.2233622, 0.1975972
2: 0.0581372, 0.2883278, 0.0570673, 0.2617090, -0.2035718, 0.2312605
3: -0.0680611, 0.2284164, -0.0252035, 0.2326369, -0.3006980, 0.2536199
4: -0.0050042, 0.3116492, -0.0057263, 0.2690515, -0.2740557, 0.3173755

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273
time: 0.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.24 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1339596, upper bound: 0.1339596
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1339596, upper bound: 0.1339596
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1339596, upper bound: 0.1339596
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1339596, upper bound: 0.1339596
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1340925, upper bound: 0.1329660
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1340925, upper bound: 0.1329660
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1340925, upper bound: 0.1329660
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1340925, upper bound: 0.1329660
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1329660, upper bound: 0.1340925
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1329660, upper bound: 0.1340925
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1329660, upper bound: 0.1340925
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1329660, upper bound: 0.1340925
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -0.1324916, upper bound: 0.1332273

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0918941, 0.2120161, 0.0918941, 0.2120161, -0.1201220, 0.1201220
1: 0.0578728, 0.2103683, 0.0578728, 0.2103683, -0.1524955, 0.1524955
2: 0.0779552, 0.2351160, 0.0779552, 0.2351160, -0.1571608, 0.1571608
3: -0.0050839, 0.1987448, -0.0050839, 0.1987448, -0.2038287, 0.2038287
4: 0.0195638, 0.2308713, 0.0195638, 0.2308713, -0.2113075, 0.2113075

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0918941, 0.2120161, 0.0767667, 0.2262760, -0.1343818, 0.1352495
1: 0.0578728, 0.2103683, 0.0229453, 0.2282764, -0.1704037, 0.1874230
2: 0.0779552, 0.2351160, 0.0670445, 0.2697541, -0.1917989, 0.1680715
3: -0.0050839, 0.1987448, -0.0565433, 0.2209895, -0.2260733, 0.2552881
4: 0.0195638, 0.2308713, 0.0042171, 0.2863551, -0.2667913, 0.2266542

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0767667, 0.2262760, 0.0918941, 0.2120161, -0.1352495, 0.1343818
1: 0.0229453, 0.2282764, 0.0578728, 0.2103683, -0.1874230, 0.1704037
2: 0.0670445, 0.2697541, 0.0779552, 0.2351160, -0.1680715, 0.1917989
3: -0.0565433, 0.2209895, -0.0050839, 0.1987448, -0.2552881, 0.2260733
4: 0.0042171, 0.2863551, 0.0195638, 0.2308713, -0.2266542, 0.2667913

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0767667, 0.2262760, 0.0767667, 0.2262760, -0.1495093, 0.1495093
1: 0.0229453, 0.2282764, 0.0229453, 0.2282764, -0.2053312, 0.2053312
2: 0.0670445, 0.2697541, 0.0670445, 0.2697541, -0.2027096, 0.2027096
3: -0.0565433, 0.2209895, -0.0565433, 0.2209895, -0.2775328, 0.2775328
4: 0.0042171, 0.2863551, 0.0042171, 0.2863551, -0.2821380, 0.2821380

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0918941, 0.2120161, 0.0819002, 0.2224207, -0.1305265, 0.1301159
1: 0.0578728, 0.2103683, 0.0490453, 0.2219181, -0.1640453, 0.1613230
2: 0.0779552, 0.2351160, 0.0662161, 0.2516250, -0.1736698, 0.1689000
3: -0.0050839, 0.1987448, -0.0185044, 0.2083519, -0.2134358, 0.2172493
4: 0.0195638, 0.2308713, 0.0070564, 0.2554384, -0.2358746, 0.2238149

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1340765, upper bound: 0.1298637
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0918941, 0.2120161, 0.0699754, 0.2389362, -0.1470420, 0.1420407
1: 0.0578728, 0.2103683, 0.0170647, 0.2386436, -0.1807709, 0.1933036
2: 0.0779552, 0.2351160, 0.0581372, 0.2883278, -0.2103726, 0.1769788
3: -0.0050839, 0.1987448, -0.0680611, 0.2284164, -0.2335002, 0.2668059
4: 0.0195638, 0.2308713, -0.0050042, 0.3116492, -0.2920854, 0.2358755

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1340765, upper bound: 0.1298637
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0767667, 0.2262760, 0.0819002, 0.2224207, -0.1456540, 0.1443757
1: 0.0229453, 0.2282764, 0.0490453, 0.2219181, -0.1989729, 0.1792312
2: 0.0670445, 0.2697541, 0.0662161, 0.2516250, -0.1845805, 0.2035381
3: -0.0565433, 0.2209895, -0.0185044, 0.2083519, -0.2648952, 0.2394939
4: 0.0042171, 0.2863551, 0.0070564, 0.2554384, -0.2512213, 0.2792987

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1319267, upper bound: 0.1293637
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0767667, 0.2262760, 0.0699754, 0.2389362, -0.1621695, 0.1563006
1: 0.0229453, 0.2282764, 0.0170647, 0.2386436, -0.2156984, 0.2112117
2: 0.0670445, 0.2697541, 0.0581372, 0.2883278, -0.2212833, 0.2116169
3: -0.0565433, 0.2209895, -0.0680611, 0.2284164, -0.2849597, 0.2890505
4: 0.0042171, 0.2863551, -0.0050042, 0.3116492, -0.3074321, 0.2913593

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1319267, upper bound: 0.1293637
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0819002, 0.2224207, 0.0918941, 0.2120161, -0.1301159, 0.1305265
1: 0.0490453, 0.2219181, 0.0578728, 0.2103683, -0.1613230, 0.1640453
2: 0.0662161, 0.2516250, 0.0779552, 0.2351160, -0.1689000, 0.1736698
3: -0.0185044, 0.2083519, -0.0050839, 0.1987448, -0.2172493, 0.2134358
4: 0.0070564, 0.2554384, 0.0195638, 0.2308713, -0.2238149, 0.2358746

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0819002, 0.2224207, 0.0767667, 0.2262760, -0.1443757, 0.1456540
1: 0.0490453, 0.2219181, 0.0229453, 0.2282764, -0.1792312, 0.1989729
2: 0.0662161, 0.2516250, 0.0670445, 0.2697541, -0.2035381, 0.1845805
3: -0.0185044, 0.2083519, -0.0565433, 0.2209895, -0.2394939, 0.2648952
4: 0.0070564, 0.2554384, 0.0042171, 0.2863551, -0.2792987, 0.2512213

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0699754, 0.2389362, 0.0918941, 0.2120161, -0.1420407, 0.1470420
1: 0.0170647, 0.2386436, 0.0578728, 0.2103683, -0.1933036, 0.1807709
2: 0.0581372, 0.2883278, 0.0779552, 0.2351160, -0.1769788, 0.2103726
3: -0.0680611, 0.2284164, -0.0050839, 0.1987448, -0.2668059, 0.2335002
4: -0.0050042, 0.3116492, 0.0195638, 0.2308713, -0.2358755, 0.2920854

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0699754, 0.2389362, 0.0767667, 0.2262760, -0.1563006, 0.1621695
1: 0.0170647, 0.2386436, 0.0229453, 0.2282764, -0.2112117, 0.2156984
2: 0.0581372, 0.2883278, 0.0670445, 0.2697541, -0.2116169, 0.2212833
3: -0.0680611, 0.2284164, -0.0565433, 0.2209895, -0.2890505, 0.2849597
4: -0.0050042, 0.3116492, 0.0042171, 0.2863551, -0.2913593, 0.3074321

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0819002, 0.2224207, 0.0819002, 0.2224207, -0.1405205, 0.1405205
1: 0.0490453, 0.2219181, 0.0490453, 0.2219181, -0.1728728, 0.1728728
2: 0.0662161, 0.2516250, 0.0662161, 0.2516250, -0.1854089, 0.1854089
3: -0.0185044, 0.2083519, -0.0185044, 0.2083519, -0.2268564, 0.2268564
4: 0.0070564, 0.2554384, 0.0070564, 0.2554384, -0.2483820, 0.2483820

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1289635, upper bound: 0.1314073
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1292403, upper bound: 0.1336889
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0819002, 0.2224207, 0.0699754, 0.2389362, -0.1570359, 0.1524453
1: 0.0490453, 0.2219181, 0.0170647, 0.2386436, -0.1895984, 0.2048534
2: 0.0662161, 0.2516250, 0.0581372, 0.2883278, -0.2221118, 0.1934878
3: -0.0185044, 0.2083519, -0.0680611, 0.2284164, -0.2469208, 0.2764130
4: 0.0070564, 0.2554384, -0.0050042, 0.3116492, -0.3045928, 0.2604426

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1289635, upper bound: 0.1314073
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1292403, upper bound: 0.1336889
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0699754, 0.2389362, 0.0819002, 0.2224207, -0.1524453, 0.1570359
1: 0.0170647, 0.2386436, 0.0490453, 0.2219181, -0.2048534, 0.1895984
2: 0.0581372, 0.2883278, 0.0662161, 0.2516250, -0.1934878, 0.2221118
3: -0.0680611, 0.2284164, -0.0185044, 0.2083519, -0.2764130, 0.2469208
4: -0.0050042, 0.3116492, 0.0070564, 0.2554384, -0.2604426, 0.3045928

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1271687, upper bound: 0.1283737
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0699754, 0.2389362, 0.0699754, 0.2389362, -0.1689608, 0.1689608
1: 0.0170647, 0.2386436, 0.0170647, 0.2386436, -0.2215789, 0.2215789
2: 0.0581372, 0.2883278, 0.0581372, 0.2883278, -0.2301906, 0.2301906
3: -0.0680611, 0.2284164, -0.0680611, 0.2284164, -0.2964774, 0.2964774
4: -0.0050042, 0.3116492, -0.0050042, 0.3116492, -0.3166534, 0.3166534

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1271687, upper bound: 0.1283737
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
time: 0.24 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.22 seconds
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.1289635, upper bound: 0.1314073
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.1292403, upper bound: 0.1336889
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.1289635, upper bound: 0.1314073
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.1292403, upper bound: 0.1336889
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.1271687, upper bound: 0.1283737
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.1271687, upper bound: 0.1283737
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0870389, 0.2136717, 0.0819002, 0.2224207, -0.1353818, 0.1317715
1: 0.0558013, 0.2120326, 0.0490453, 0.2219181, -0.1661168, 0.1629874
2: 0.0713646, 0.2391730, 0.0662161, 0.2516250, -0.1802604, 0.1729569
3: -0.0070687, 0.1977674, -0.0185044, 0.2083519, -0.2154206, 0.2162718
4: 0.0130469, 0.2389780, 0.0070564, 0.2554384, -0.2423915, 0.2319216

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1346739, upper bound: 0.1383603
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1346739, upper bound: 0.1383603
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0877597, 0.2153121, 0.0819002, 0.2224207, -0.1346610, 0.1334118
1: 0.0573157, 0.2101370, 0.0490453, 0.2219181, -0.1646024, 0.1610917
2: 0.0727918, 0.2408701, 0.0662161, 0.2516250, -0.1788332, 0.1746540
3: -0.0085227, 0.1973067, -0.0185044, 0.2083519, -0.2168747, 0.2158111
4: 0.0132263, 0.2405431, 0.0070564, 0.2554384, -0.2422121, 0.2334868

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1351825, upper bound: 0.1408198
time: 0.27 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1351825, upper bound: 0.1408198
time: 0.28 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0870389, 0.2136717, 0.0699754, 0.2389362, -0.1518973, 0.1436963
1: 0.0558013, 0.2120326, 0.0170647, 0.2386436, -0.1828423, 0.1949679
2: 0.0713646, 0.2391730, 0.0581372, 0.2883278, -0.2169632, 0.1810358
3: -0.0070687, 0.1977674, -0.0680611, 0.2284164, -0.2354851, 0.2658284
4: 0.0130469, 0.2389780, -0.0050042, 0.3116492, -0.2986023, 0.2439822

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1289635, upper bound: 0.1314073
time: 0.27 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1289635, upper bound: 0.1314073
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0877597, 0.2153121, 0.0699754, 0.2389362, -0.1511765, 0.1453367
1: 0.0573157, 0.2101370, 0.0170647, 0.2386436, -0.1813279, 0.1930723
2: 0.0727918, 0.2408701, 0.0581372, 0.2883278, -0.2155360, 0.1827329
3: -0.0085227, 0.1973067, -0.0680611, 0.2284164, -0.2369391, 0.2653677
4: 0.0132263, 0.2405431, -0.0050042, 0.3116492, -0.2984229, 0.2455473

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1292403, upper bound: 0.1336889
time: 0.27 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1292403, upper bound: 0.1336889
time: 0.28 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0735340, 0.2321481, 0.0819002, 0.2224207, -0.1488867, 0.1502478
1: 0.0225614, 0.2318966, 0.0490453, 0.2219181, -0.1993567, 0.1828514
2: 0.0618032, 0.2791588, 0.0662161, 0.2516250, -0.1898218, 0.2129427
3: -0.0596962, 0.2215191, -0.0185044, 0.2083519, -0.2680481, 0.2400236
4: -0.0004705, 0.2995532, 0.0070564, 0.2554384, -0.2559089, 0.2924968

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1312787, upper bound: 0.1317141
time: 0.28 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1312787, upper bound: 0.1317140
time: 0.26 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0759843, 0.2316126, 0.0819002, 0.2224207, -0.1464364, 0.1497123
1: 0.0241483, 0.2267795, 0.0490453, 0.2219181, -0.1977698, 0.1777343
2: 0.0650663, 0.2780107, 0.0662161, 0.2516250, -0.1865587, 0.2117946
3: -0.0581607, 0.2172649, -0.0185044, 0.2083519, -0.2665126, 0.2357693
4: 0.0015443, 0.2965019, 0.0070564, 0.2554384, -0.2538941, 0.2894455

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1312787, upper bound: 0.1317141
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1312787, upper bound: 0.1317141
time: 0.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0735340, 0.2321481, 0.0699754, 0.2389362, -0.1654022, 0.1621727
1: 0.0225614, 0.2318966, 0.0170647, 0.2386436, -0.2160822, 0.2148319
2: 0.0618032, 0.2791588, 0.0581372, 0.2883278, -0.2265247, 0.2210216
3: -0.0596962, 0.2215191, -0.0680611, 0.2284164, -0.2881126, 0.2895802
4: -0.0004705, 0.2995532, -0.0050042, 0.3116492, -0.3121198, 0.3045574

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
time: 0.27 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
time: 0.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0759843, 0.2316126, 0.0699754, 0.2389362, -0.1629519, 0.1616372
1: 0.0241483, 0.2267795, 0.0170647, 0.2386436, -0.2144953, 0.2097148
2: 0.0650663, 0.2780107, 0.0581372, 0.2883278, -0.2232615, 0.2198735
3: -0.0581607, 0.2172649, -0.0680611, 0.2284164, -0.2865771, 0.2853259
4: 0.0015443, 0.2965019, -0.0050042, 0.3116492, -0.3101049, 0.3015061

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
time: 0.26 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.25 seconds
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1346739, upper bound: 0.1383603
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1346739, upper bound: 0.1383603
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1351825, upper bound: 0.1408198
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1351825, upper bound: 0.1408198
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1289635, upper bound: 0.1314073
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1289635, upper bound: 0.1314073
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1292403, upper bound: 0.1336889
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1292403, upper bound: 0.1336889
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1312787, upper bound: 0.1317141
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1312787, upper bound: 0.1317140
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1312787, upper bound: 0.1317141
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1312787, upper bound: 0.1317141
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -0.1260151, upper bound: 0.1275428

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0870389, 0.2136717, 0.0870389, 0.2136717, -0.1266328, 0.1266328
1: 0.0558013, 0.2120326, 0.0558013, 0.2120326, -0.1562313, 0.1562313
2: 0.0713646, 0.2391730, 0.0713646, 0.2391730, -0.1678083, 0.1678083
3: -0.0070687, 0.1977674, -0.0070687, 0.1977674, -0.2048361, 0.2048361
4: 0.0130469, 0.2389780, 0.0130469, 0.2389780, -0.2259311, 0.2259311

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0870389, 0.2136717, 0.0877597, 0.2153121, -0.1282731, 0.1259120
1: 0.0558013, 0.2120326, 0.0573157, 0.2101370, -0.1543357, 0.1547169
2: 0.0713646, 0.2391730, 0.0727918, 0.2408701, -0.1695054, 0.1663812
3: -0.0070687, 0.1977674, -0.0085227, 0.1973067, -0.2043754, 0.2062901
4: 0.0130469, 0.2389780, 0.0132263, 0.2405431, -0.2274963, 0.2257517

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0877597, 0.2153121, 0.0870389, 0.2136717, -0.1259120, 0.1282731
1: 0.0573157, 0.2101370, 0.0558013, 0.2120326, -0.1547169, 0.1543357
2: 0.0727918, 0.2408701, 0.0713646, 0.2391730, -0.1663812, 0.1695054
3: -0.0085227, 0.1973067, -0.0070687, 0.1977674, -0.2062901, 0.2043754
4: 0.0132263, 0.2405431, 0.0130469, 0.2389780, -0.2257517, 0.2274963

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0877597, 0.2153121, 0.0877597, 0.2153121, -0.1275523, 0.1275523
1: 0.0573157, 0.2101370, 0.0573157, 0.2101370, -0.1528213, 0.1528213
2: 0.0727918, 0.2408701, 0.0727918, 0.2408701, -0.1680783, 0.1680783
3: -0.0085227, 0.1973067, -0.0085227, 0.1973067, -0.2058294, 0.2058294
4: 0.0132263, 0.2405431, 0.0132263, 0.2405431, -0.2273168, 0.2273168

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0870389, 0.2136717, 0.0735340, 0.2321481, -0.1451091, 0.1401377
1: 0.0558013, 0.2120326, 0.0225614, 0.2318966, -0.1760953, 0.1894712
2: 0.0713646, 0.2391730, 0.0618032, 0.2791588, -0.2077941, 0.1773698
3: -0.0070687, 0.1977674, -0.0596962, 0.2215191, -0.2285878, 0.2574636
4: 0.0130469, 0.2389780, -0.0004705, 0.2995532, -0.2865063, 0.2394485

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0870389, 0.2136717, 0.0759843, 0.2316126, -0.1445736, 0.1376874
1: 0.0558013, 0.2120326, 0.0241483, 0.2267795, -0.1709782, 0.1878843
2: 0.0713646, 0.2391730, 0.0650663, 0.2780107, -0.2066461, 0.1741067
3: -0.0070687, 0.1977674, -0.0581607, 0.2172649, -0.2243336, 0.2559281
4: 0.0130469, 0.2389780, 0.0015443, 0.2965019, -0.2834550, 0.2374337

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0877597, 0.2153121, 0.0735340, 0.2321481, -0.1443883, 0.1417781
1: 0.0573157, 0.2101370, 0.0225614, 0.2318966, -0.1745809, 0.1875756
2: 0.0727918, 0.2408701, 0.0618032, 0.2791588, -0.2063670, 0.1790669
3: -0.0085227, 0.1973067, -0.0596962, 0.2215191, -0.2300419, 0.2570029
4: 0.0132263, 0.2405431, -0.0004705, 0.2995532, -0.2863269, 0.2410137

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0877597, 0.2153121, 0.0759843, 0.2316126, -0.1438528, 0.1393278
1: 0.0573157, 0.2101370, 0.0241483, 0.2267795, -0.1694638, 0.1859887
2: 0.0727918, 0.2408701, 0.0650663, 0.2780107, -0.2052189, 0.1758038
3: -0.0085227, 0.1973067, -0.0581607, 0.2172649, -0.2257876, 0.2554674
4: 0.0132263, 0.2405431, 0.0015443, 0.2965019, -0.2832756, 0.2389988

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0735340, 0.2321481, 0.0870389, 0.2136717, -0.1401377, 0.1451091
1: 0.0225614, 0.2318966, 0.0558013, 0.2120326, -0.1894712, 0.1760953
2: 0.0618032, 0.2791588, 0.0713646, 0.2391730, -0.1773698, 0.2077941
3: -0.0596962, 0.2215191, -0.0070687, 0.1977674, -0.2574636, 0.2285878
4: -0.0004705, 0.2995532, 0.0130469, 0.2389780, -0.2394485, 0.2865063

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0735340, 0.2321481, 0.0877597, 0.2153121, -0.1417781, 0.1443883
1: 0.0225614, 0.2318966, 0.0573157, 0.2101370, -0.1875756, 0.1745809
2: 0.0618032, 0.2791588, 0.0727918, 0.2408701, -0.1790669, 0.2063670
3: -0.0596962, 0.2215191, -0.0085227, 0.1973067, -0.2570029, 0.2300419
4: -0.0004705, 0.2995532, 0.0132263, 0.2405431, -0.2410137, 0.2863269

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0759843, 0.2316126, 0.0870389, 0.2136717, -0.1376874, 0.1445736
1: 0.0241483, 0.2267795, 0.0558013, 0.2120326, -0.1878843, 0.1709782
2: 0.0650663, 0.2780107, 0.0713646, 0.2391730, -0.1741067, 0.2066461
3: -0.0581607, 0.2172649, -0.0070687, 0.1977674, -0.2559281, 0.2243336
4: 0.0015443, 0.2965019, 0.0130469, 0.2389780, -0.2374337, 0.2834550

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0759843, 0.2316126, 0.0877597, 0.2153121, -0.1393278, 0.1438528
1: 0.0241483, 0.2267795, 0.0573157, 0.2101370, -0.1859887, 0.1694638
2: 0.0650663, 0.2780107, 0.0727918, 0.2408701, -0.1758038, 0.2052189
3: -0.0581607, 0.2172649, -0.0085227, 0.1973067, -0.2554674, 0.2257876
4: 0.0015443, 0.2965019, 0.0132263, 0.2405431, -0.2389988, 0.2832756

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0735340, 0.2321481, 0.0735340, 0.2321481, -0.1586141, 0.1586141
1: 0.0225614, 0.2318966, 0.0225614, 0.2318966, -0.2093352, 0.2093352
2: 0.0618032, 0.2791588, 0.0618032, 0.2791588, -0.2173556, 0.2173556
3: -0.0596962, 0.2215191, -0.0596962, 0.2215191, -0.2812153, 0.2812153
4: -0.0004705, 0.2995532, -0.0004705, 0.2995532, -0.3000237, 0.3000237

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0735340, 0.2321481, 0.0759843, 0.2316126, -0.1580786, 0.1561638
1: 0.0225614, 0.2318966, 0.0241483, 0.2267795, -0.2042181, 0.2077483
2: 0.0618032, 0.2791588, 0.0650663, 0.2780107, -0.2162075, 0.2140925
3: -0.0596962, 0.2215191, -0.0581607, 0.2172649, -0.2769611, 0.2796798
4: -0.0004705, 0.2995532, 0.0015443, 0.2965019, -0.2969724, 0.2980089

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0759843, 0.2316126, 0.0735340, 0.2321481, -0.1561638, 0.1580786
1: 0.0241483, 0.2267795, 0.0225614, 0.2318966, -0.2077483, 0.2042181
2: 0.0650663, 0.2780107, 0.0618032, 0.2791588, -0.2140925, 0.2162075
3: -0.0581607, 0.2172649, -0.0596962, 0.2215191, -0.2796798, 0.2769611
4: 0.0015443, 0.2965019, -0.0004705, 0.2995532, -0.2980089, 0.2969724

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0759843, 0.2316126, 0.0759843, 0.2316126, -0.1556283, 0.1556283
1: 0.0241483, 0.2267795, 0.0241483, 0.2267795, -0.2026312, 0.2026312
2: 0.0650663, 0.2780107, 0.0650663, 0.2780107, -0.2129444, 0.2129444
3: -0.0581607, 0.2172649, -0.0581607, 0.2172649, -0.2754256, 0.2754256
4: 0.0015443, 0.2965019, 0.0015443, 0.2965019, -0.2949576, 0.2949576

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.41 + 117.47 = 119.88 seconds
