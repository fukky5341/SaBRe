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
execution time: IAR + RelationalAnalysis = 1.61 + 0.74 = 2.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.1444805, upper bound: 0.1444805

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

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
time: 0.22 seconds

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

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1373135, upper bound: 0.1337799
time: 0.22 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1343781, upper bound: 0.1321562
time: 0.22 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0732364, 0.2302559, 0.0722671, 0.2333107, -0.1600743, 0.1579888
1: 0.0410464, 0.2404269, 0.0389306, 0.2428598, -0.2018134, 0.2014963
2: 0.0570673, 0.2617090, 0.0560051, 0.2667054, -0.2096381, 0.2057039
3: -0.0252035, 0.2326369, -0.0283638, 0.2345866, -0.2597901, 0.2610008
4: -0.0057263, 0.2690515, -0.0069483, 0.2760361, -0.2817624, 0.2759998

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
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
time: 0.23 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.10 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.1373135, upper bound: 0.1337799
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.1343781, upper bound: 0.1321562
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.1368941, upper bound: 0.1433590
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.1368941, upper bound: 0.1433590

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: 0.0821925, 0.2194903, 0.0722671, 0.2333107, -0.1511182, 0.1472231
1: 0.0513112, 0.2289833, 0.0389306, 0.2428598, -0.1915486, 0.1900527
2: 0.0683440, 0.2441752, 0.0560051, 0.2667054, -0.1983614, 0.1881701
3: -0.0111772, 0.2242729, -0.0283638, 0.2345866, -0.2457638, 0.2526367
4: 0.0050781, 0.2427847, -0.0069483, 0.2760361, -0.2709580, 0.2497329

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1357112, upper bound: 0.1337799
time: 0.23 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1357112, upper bound: 0.1337799
time: 0.23 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: 0.0796905, 0.2222642, 0.0722671, 0.2333107, -0.1536202, 0.1499970
1: 0.0471406, 0.2326190, 0.0389306, 0.2428598, -0.1957192, 0.1936885
2: 0.0658738, 0.2469145, 0.0560051, 0.2667054, -0.2008316, 0.1909094
3: -0.0120858, 0.2275630, -0.0283638, 0.2345866, -0.2466724, 0.2559268
4: 0.0039079, 0.2468308, -0.0069483, 0.2760361, -0.2721282, 0.2537790

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1343781, upper bound: 0.1321562
time: 0.24 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1343781, upper bound: 0.1321562
time: 0.23 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0732364, 0.2302559, 0.0820065, 0.2198348, -0.1465983, 0.1482494
1: 0.0410464, 0.2404269, 0.0510250, 0.2293532, -0.1883067, 0.1894019
2: 0.0570673, 0.2617090, 0.0681462, 0.2446522, -0.1875849, 0.1935628
3: -0.0252035, 0.2326369, -0.0115750, 0.2246487, -0.2498522, 0.2442120
4: -0.0057263, 0.2690515, 0.0048897, 0.2434022, -0.2491286, 0.2641618

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1337799, upper bound: 0.1367145
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336611
time: 0.23 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0732364, 0.2302559, 0.0732364, 0.2302559, -0.1570195, 0.1570195
1: 0.0410464, 0.2404269, 0.0410464, 0.2404269, -0.1993805, 0.1993805
2: 0.0570673, 0.2617090, 0.0570673, 0.2617090, -0.2046417, 0.2046417
3: -0.0252035, 0.2326369, -0.0252035, 0.2326369, -0.2578405, 0.2578405
4: -0.0057263, 0.2690515, -0.0057263, 0.2690515, -0.2747778, 0.2747778

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1351122, upper bound: 0.1403007
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336612
time: 0.24 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.12 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.1357112, upper bound: 0.1337799
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.1357112, upper bound: 0.1337799
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.1343781, upper bound: 0.1321562
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.1343781, upper bound: 0.1321562
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.1337799, upper bound: 0.1367145
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336611
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.1351122, upper bound: 0.1403007
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336612

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0821925, 0.2194903, 0.0820065, 0.2198348, -0.1376422, 0.1374838
1: 0.0513112, 0.2289833, 0.0510250, 0.2293532, -0.1780420, 0.1779583
2: 0.0683440, 0.2441752, 0.0681462, 0.2446522, -0.1763082, 0.1760290
3: -0.0111772, 0.2242729, -0.0115750, 0.2246487, -0.2358258, 0.2358479
4: 0.0050781, 0.2427847, 0.0048897, 0.2434022, -0.2383241, 0.2378950

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1357112, upper bound: 0.1337799
time: 0.23 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1357112, upper bound: 0.1337799
time: 0.24 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0821925, 0.2194903, 0.0732364, 0.2302559, -0.1480634, 0.1462539
1: 0.0513112, 0.2289833, 0.0410464, 0.2404269, -0.1891157, 0.1879369
2: 0.0683440, 0.2441752, 0.0570673, 0.2617090, -0.1933650, 0.1871079
3: -0.0111772, 0.2242729, -0.0252035, 0.2326369, -0.2438141, 0.2494764
4: 0.0050781, 0.2427847, -0.0057263, 0.2690515, -0.2639734, 0.2485110

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1324356, upper bound: 0.1322566
time: 0.23 seconds

## Relational analysis of NS_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1357112, upper bound: 0.1337799
time: 0.23 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1357112, upper bound: 0.1337799
time: 0.24 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0796905, 0.2222642, 0.0724867, 0.2328915, -0.1532010, 0.1497775
1: 0.0471406, 0.2326190, 0.0392322, 0.2424081, -0.1952675, 0.1933869
2: 0.0658738, 0.2469145, 0.0562323, 0.2661542, -0.2002804, 0.1906822
3: -0.0120858, 0.2275630, -0.0279511, 0.2341732, -0.2462590, 0.2555141
4: 0.0039079, 0.2468308, -0.0066978, 0.2753263, -0.2714185, 0.2535285

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1327758, upper bound: 0.1321562
time: 0.24 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1327758, upper bound: 0.1321562
time: 0.24 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0796905, 0.2222642, 0.0698787, 0.2366408, -0.1569503, 0.1523854
1: 0.0471406, 0.2326190, 0.0349536, 0.2473645, -0.2002239, 0.1976655
2: 0.0658738, 0.2469145, 0.0532843, 0.2714010, -0.2055272, 0.1936302
3: -0.0120858, 0.2275630, -0.0310813, 0.2384260, -0.2505118, 0.2586443
4: 0.0039079, 0.2468308, -0.0087516, 0.2829595, -0.2790517, 0.2555824

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1327758, upper bound: 0.1321562
time: 0.23 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1327758, upper bound: 0.1321562
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0732364, 0.2302559, 0.0821925, 0.2194903, -0.1462539, 0.1480634
1: 0.0410464, 0.2404269, 0.0513112, 0.2289833, -0.1879369, 0.1891157
2: 0.0570673, 0.2617090, 0.0683440, 0.2441752, -0.1871079, 0.1933650
3: -0.0252035, 0.2326369, -0.0111772, 0.2242729, -0.2494764, 0.2438141
4: -0.0057263, 0.2690515, 0.0050781, 0.2427847, -0.2485110, 0.2639734

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1322566, upper bound: 0.1347425
time: 0.24 seconds

## Relational analysis of NS_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1321562, upper bound: 0.1343780
time: 0.24 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1321562, upper bound: 0.1343780
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0732364, 0.2302559, 0.0796905, 0.2222642, -0.1490277, 0.1505654
1: 0.0410464, 0.2404269, 0.0471406, 0.2326190, -0.1915726, 0.1932863
2: 0.0570673, 0.2617090, 0.0658738, 0.2469145, -0.1898472, 0.1958352
3: -0.0252035, 0.2326369, -0.0120858, 0.2275630, -0.2527665, 0.2447227
4: -0.0057263, 0.2690515, 0.0039079, 0.2468308, -0.2525571, 0.2651436

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1321562, upper bound: 0.1343780
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1321562, upper bound: 0.1343780
time: 0.22 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0734453, 0.2298511, 0.0732364, 0.2302559, -0.1568106, 0.1566147
1: 0.0413309, 0.2399949, 0.0410464, 0.2404269, -0.1990960, 0.1989485
2: 0.0572846, 0.2611772, 0.0570673, 0.2617090, -0.2044244, 0.2041099
3: -0.0248084, 0.2322409, -0.0252035, 0.2326369, -0.2574453, 0.2574444
4: -0.0054841, 0.2683696, -0.0057263, 0.2690515, -0.2745356, 0.2740959

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336612
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336612
time: 0.26 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0707583, 0.2337075, 0.0732364, 0.2302559, -0.1594976, 0.1604711
1: 0.0369808, 0.2450551, 0.0410464, 0.2404269, -0.2034461, 0.2040086
2: 0.0542761, 0.2666780, 0.0570673, 0.2617090, -0.2074329, 0.2096107
3: -0.0279449, 0.2365744, -0.0252035, 0.2326369, -0.2605818, 0.2617779
4: -0.0075533, 0.2763611, -0.0057263, 0.2690515, -0.2766048, 0.2820874

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336611
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336611
time: 0.25 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.18 seconds
NS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1357112, upper bound: 0.1337799
NS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1357112, upper bound: 0.1337799
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1357112, upper bound: 0.1337799
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1357112, upper bound: 0.1337799
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1327758, upper bound: 0.1321562
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1327758, upper bound: 0.1321562
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1327758, upper bound: 0.1321562
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1327758, upper bound: 0.1321562
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1321562, upper bound: 0.1343780
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1321562, upper bound: 0.1343780
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1321562, upper bound: 0.1343780
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1321562, upper bound: 0.1343780
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336612
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336612
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336611
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -0.1320588, upper bound: 0.1336611

## BFS NS instance: NS_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0821925, 0.2194903, 0.0821925, 0.2194903, -0.1372977, 0.1372977
1: 0.0513112, 0.2289833, 0.0513112, 0.2289833, -0.1776721, 0.1776721
2: 0.0683440, 0.2441752, 0.0683440, 0.2441752, -0.1758312, 0.1758312
3: -0.0111772, 0.2242729, -0.0111772, 0.2242729, -0.2354501, 0.2354501
4: 0.0050781, 0.2427847, 0.0050781, 0.2427847, -0.2377066, 0.2377066

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## BFS NS instance: NS_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0821925, 0.2194903, 0.0796905, 0.2222642, -0.1400716, 0.1397998
1: 0.0513112, 0.2289833, 0.0471406, 0.2326190, -0.1813078, 0.1818427
2: 0.0683440, 0.2441752, 0.0658738, 0.2469145, -0.1785705, 0.1783014
3: -0.0111772, 0.2242729, -0.0120858, 0.2275630, -0.2387402, 0.2363587
4: 0.0050781, 0.2427847, 0.0039079, 0.2468308, -0.2417527, 0.2388768

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0821925, 0.2194903, 0.0734453, 0.2298511, -0.1476586, 0.1460450
1: 0.0513112, 0.2289833, 0.0413309, 0.2399949, -0.1886837, 0.1876524
2: 0.0683440, 0.2441752, 0.0572846, 0.2611772, -0.1928332, 0.1868906
3: -0.0111772, 0.2242729, -0.0248084, 0.2322409, -0.2434180, 0.2490813
4: 0.0050781, 0.2427847, -0.0054841, 0.2683696, -0.2632914, 0.2482688

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0821925, 0.2194903, 0.0707583, 0.2337075, -0.1515150, 0.1487320
1: 0.0513112, 0.2289833, 0.0369808, 0.2450551, -0.1937439, 0.1920025
2: 0.0683440, 0.2441752, 0.0542761, 0.2666780, -0.1983340, 0.1898991
3: -0.0111772, 0.2242729, -0.0279449, 0.2365744, -0.2477515, 0.2522178
4: 0.0050781, 0.2427847, -0.0075533, 0.2763611, -0.2712830, 0.2503380

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0796905, 0.2222642, 0.0821925, 0.2194903, -0.1397998, 0.1400716
1: 0.0471406, 0.2326190, 0.0513112, 0.2289833, -0.1818427, 0.1813078
2: 0.0658738, 0.2469145, 0.0683440, 0.2441752, -0.1783014, 0.1785705
3: -0.0120858, 0.2275630, -0.0111772, 0.2242729, -0.2363587, 0.2387402
4: 0.0039079, 0.2468308, 0.0050781, 0.2427847, -0.2388768, 0.2417527

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1326131, upper bound: 0.1320859
time: 0.24 seconds

## Relational analysis of NS_A1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0796905, 0.2222642, 0.0734453, 0.2298511, -0.1501606, 0.1488189
1: 0.0471406, 0.2326190, 0.0413309, 0.2399949, -0.1928543, 0.1912881
2: 0.0658738, 0.2469145, 0.0572846, 0.2611772, -0.1953034, 0.1896299
3: -0.0120858, 0.2275630, -0.0248084, 0.2322409, -0.2443267, 0.2523714
4: 0.0039079, 0.2468308, -0.0054841, 0.2683696, -0.2644617, 0.2523149

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1326131, upper bound: 0.1320859
time: 0.24 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0796905, 0.2222642, 0.0791481, 0.2234208, -0.1437303, 0.1431160
1: 0.0471406, 0.2326190, 0.0458111, 0.2337078, -0.1865672, 0.1868080
2: 0.0658738, 0.2469145, 0.0653405, 0.2495278, -0.1836540, 0.1815740
3: -0.0120858, 0.2275630, -0.0146942, 0.2284608, -0.2405466, 0.2422572
4: 0.0039079, 0.2468308, 0.0034690, 0.2505844, -0.2466766, 0.2433618

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0796905, 0.2222642, 0.0707583, 0.2337075, -0.1540171, 0.1515059
1: 0.0471406, 0.2326190, 0.0369808, 0.2450551, -0.1979145, 0.1956382
2: 0.0658738, 0.2469145, 0.0542761, 0.2666780, -0.2008042, 0.1926384
3: -0.0120858, 0.2275630, -0.0279449, 0.2365744, -0.2486601, 0.2555079
4: 0.0039079, 0.2468308, -0.0075533, 0.2763611, -0.2724532, 0.2543840

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0734453, 0.2298511, 0.0821925, 0.2194903, -0.1460450, 0.1476586
1: 0.0413309, 0.2399949, 0.0513112, 0.2289833, -0.1876524, 0.1886837
2: 0.0572846, 0.2611772, 0.0683440, 0.2441752, -0.1868906, 0.1928332
3: -0.0248084, 0.2322409, -0.0111772, 0.2242729, -0.2490813, 0.2434180
4: -0.0054841, 0.2683696, 0.0050781, 0.2427847, -0.2482688, 0.2632914

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0707583, 0.2337075, 0.0821925, 0.2194903, -0.1487320, 0.1515150
1: 0.0369808, 0.2450551, 0.0513112, 0.2289833, -0.1920025, 0.1937439
2: 0.0542761, 0.2666780, 0.0683440, 0.2441752, -0.1898991, 0.1983340
3: -0.0279449, 0.2365744, -0.0111772, 0.2242729, -0.2522178, 0.2477515
4: -0.0075533, 0.2763611, 0.0050781, 0.2427847, -0.2503380, 0.2712830

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0734453, 0.2298511, 0.0796905, 0.2222642, -0.1488189, 0.1501606
1: 0.0413309, 0.2399949, 0.0471406, 0.2326190, -0.1912881, 0.1928543
2: 0.0572846, 0.2611772, 0.0658738, 0.2469145, -0.1896299, 0.1953034
3: -0.0248084, 0.2322409, -0.0120858, 0.2275630, -0.2523714, 0.2443267
4: -0.0054841, 0.2683696, 0.0039079, 0.2468308, -0.2523149, 0.2644617

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1302934, upper bound: 0.1342325
time: 0.25 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0707583, 0.2337075, 0.0796905, 0.2222642, -0.1515059, 0.1540171
1: 0.0369808, 0.2450551, 0.0471406, 0.2326190, -0.1956382, 0.1979145
2: 0.0542761, 0.2666780, 0.0658738, 0.2469145, -0.1926384, 0.2008042
3: -0.0279449, 0.2365744, -0.0120858, 0.2275630, -0.2555079, 0.2486601
4: -0.0075533, 0.2763611, 0.0039079, 0.2468308, -0.2543840, 0.2724532

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0734453, 0.2298511, 0.0734453, 0.2298511, -0.1564059, 0.1564059
1: 0.0413309, 0.2399949, 0.0413309, 0.2399949, -0.1986640, 0.1986640
2: 0.0572846, 0.2611772, 0.0572846, 0.2611772, -0.2038926, 0.2038926
3: -0.0248084, 0.2322409, -0.0248084, 0.2322409, -0.2570493, 0.2570493
4: -0.0054841, 0.2683696, -0.0054841, 0.2683696, -0.2738537, 0.2738537

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1337354, upper bound: 0.1359947
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1337475, upper bound: 0.1381466
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0734453, 0.2298511, 0.0707583, 0.2337075, -0.1602623, 0.1590928
1: 0.0413309, 0.2399949, 0.0369808, 0.2450551, -0.2037241, 0.2030141
2: 0.0572846, 0.2611772, 0.0542761, 0.2666780, -0.2093934, 0.2069011
3: -0.0248084, 0.2322409, -0.0279449, 0.2365744, -0.2613828, 0.2601857
4: -0.0054841, 0.2683696, -0.0075533, 0.2763611, -0.2818452, 0.2759228

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1338535, upper bound: 0.1398741
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1337475, upper bound: 0.1381466
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0707583, 0.2337075, 0.0734453, 0.2298511, -0.1590928, 0.1602623
1: 0.0369808, 0.2450551, 0.0413309, 0.2399949, -0.2030141, 0.2037241
2: 0.0542761, 0.2666780, 0.0572846, 0.2611772, -0.2069011, 0.2093934
3: -0.0279449, 0.2365744, -0.0248084, 0.2322409, -0.2601857, 0.2613828
4: -0.0075533, 0.2763611, -0.0054841, 0.2683696, -0.2759228, 0.2818452

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1307534
time: 0.27 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1316573
time: 0.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0707583, 0.2337075, 0.0707583, 0.2337075, -0.1629493, 0.1629493
1: 0.0369808, 0.2450551, 0.0369808, 0.2450551, -0.2080743, 0.2080743
2: 0.0542761, 0.2666780, 0.0542761, 0.2666780, -0.2124019, 0.2124019
3: -0.0279449, 0.2365744, -0.0279449, 0.2365744, -0.2645192, 0.2645192
4: -0.0075533, 0.2763611, -0.0075533, 0.2763611, -0.2839144, 0.2839144

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1307534
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1316573
time: 0.24 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.49 seconds
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -0.1337354, upper bound: 0.1359947
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -0.1337475, upper bound: 0.1381466
NS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -0.1338535, upper bound: 0.1398741
NS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -0.1337475, upper bound: 0.1381466
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1307534
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1316573
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1307534
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1316573

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0783825, 0.2205054, 0.0734453, 0.2298511, -0.1514686, 0.1470601
1: 0.0482821, 0.2302842, 0.0413309, 0.2399949, -0.1917129, 0.1889533
2: 0.0624542, 0.2478489, 0.0572846, 0.2611772, -0.1987230, 0.1905643
3: -0.0129582, 0.2227221, -0.0248084, 0.2322409, -0.2451991, 0.2475305
4: 0.0008159, 0.2506008, -0.0054841, 0.2683696, -0.2675536, 0.2560849

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1357363, upper bound: 0.1391876
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1357363, upper bound: 0.1391876
time: 0.26 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0788392, 0.2222926, 0.0734453, 0.2298511, -0.1510119, 0.1488473
1: 0.0492356, 0.2283794, 0.0413309, 0.2399949, -0.1907593, 0.1870484
2: 0.0635781, 0.2500039, 0.0572846, 0.2611772, -0.1975991, 0.1927193
3: -0.0150195, 0.2220079, -0.0248084, 0.2322409, -0.2472603, 0.2468163
4: 0.0004846, 0.2528384, -0.0054841, 0.2683696, -0.2678849, 0.2583226

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1359554, upper bound: 0.1413395
time: 0.27 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1359554, upper bound: 0.1413395
time: 0.26 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0734453, 0.2298511, 0.0751822, 0.2249900, -0.1515447, 0.1546690
1: 0.0413309, 0.2399949, 0.0431372, 0.2356555, -0.1943246, 0.1968578
2: 0.0572846, 0.2611772, 0.0588951, 0.2540951, -0.1968106, 0.2022821
3: -0.0248084, 0.2322409, -0.0163606, 0.2275606, -0.2523690, 0.2486015
4: -0.0054841, 0.2683696, -0.0018036, 0.2591435, -0.2646277, 0.2701732

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1337354, upper bound: 0.1359948
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1337354, upper bound: 0.1381466
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0734453, 0.2298511, 0.0759371, 0.2269447, -0.1534995, 0.1539141
1: 0.0413309, 0.2399949, 0.0437303, 0.2337886, -0.1924577, 0.1962647
2: 0.0572846, 0.2611772, 0.0602925, 0.2563978, -0.1991133, 0.2008846
3: -0.0248084, 0.2322409, -0.0192161, 0.2267310, -0.2515394, 0.2514569
4: -0.0054841, 0.2683696, -0.0018764, 0.2617289, -0.2672130, 0.2702460

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1337354, upper bound: 0.1359948
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1337354, upper bound: 0.1359948
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0751822, 0.2249900, 0.0734453, 0.2298511, -0.1546690, 0.1515447
1: 0.0431372, 0.2356555, 0.0413309, 0.2399949, -0.1968578, 0.1943246
2: 0.0588951, 0.2540951, 0.0572846, 0.2611772, -0.2022821, 0.1968106
3: -0.0163606, 0.2275606, -0.0248084, 0.2322409, -0.2486015, 0.2523690
4: -0.0018036, 0.2591435, -0.0054841, 0.2683696, -0.2701732, 0.2646277

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1326454, upper bound: 0.1339462
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1326454, upper bound: 0.1339462
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0759371, 0.2269447, 0.0734453, 0.2298511, -0.1539141, 0.1534995
1: 0.0437303, 0.2337886, 0.0413309, 0.2399949, -0.1962647, 0.1924577
2: 0.0602925, 0.2563978, 0.0572846, 0.2611772, -0.2008846, 0.1991133
3: -0.0192161, 0.2267310, -0.0248084, 0.2322409, -0.2514569, 0.2515394
4: -0.0018764, 0.2617289, -0.0054841, 0.2683696, -0.2702460, 0.2672130

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1328689, upper bound: 0.1348501
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1328689, upper bound: 0.1348501
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0751822, 0.2249900, 0.0707583, 0.2337075, -0.1585254, 0.1542317
1: 0.0431372, 0.2356555, 0.0369808, 0.2450551, -0.2019179, 0.1986747
2: 0.0588951, 0.2540951, 0.0542761, 0.2666780, -0.2077829, 0.1998190
3: -0.0163606, 0.2275606, -0.0279449, 0.2365744, -0.2529350, 0.2555055
4: -0.0018036, 0.2591435, -0.0075533, 0.2763611, -0.2781647, 0.2666968

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1307534
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1307534
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0759371, 0.2269447, 0.0707583, 0.2337075, -0.1577705, 0.1561865
1: 0.0437303, 0.2337886, 0.0369808, 0.2450551, -0.2013248, 0.1968078
2: 0.0602925, 0.2563978, 0.0542761, 0.2666780, -0.2063854, 0.2021217
3: -0.0192161, 0.2267310, -0.0279449, 0.2365744, -0.2557904, 0.2546759
4: -0.0018764, 0.2617289, -0.0075533, 0.2763611, -0.2782375, 0.2692822

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1316573
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1316573
time: 0.25 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.19 seconds
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1357363, upper bound: 0.1391876
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1357363, upper bound: 0.1391876
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1359554, upper bound: 0.1413395
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1359554, upper bound: 0.1413395
NS_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1337354, upper bound: 0.1359948
NS_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1337354, upper bound: 0.1381466
NS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1337354, upper bound: 0.1359948
NS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1337354, upper bound: 0.1359948
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1326454, upper bound: 0.1339462
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1326454, upper bound: 0.1339462
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1328689, upper bound: 0.1348501
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1328689, upper bound: 0.1348501
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1307534
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1307534
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1316573
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -0.1306871, upper bound: 0.1316573

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0783825, 0.2205054, 0.0783825, 0.2205054, -0.1421228, 0.1421228
1: 0.0482821, 0.2302842, 0.0482821, 0.2302842, -0.1820022, 0.1820022
2: 0.0624542, 0.2478489, 0.0624542, 0.2478489, -0.1853947, 0.1853947
3: -0.0129582, 0.2227221, -0.0129582, 0.2227221, -0.2356803, 0.2356803
4: 0.0008159, 0.2506008, 0.0008159, 0.2506008, -0.2497848, 0.2497848

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0783825, 0.2205054, 0.0788392, 0.2222926, -0.1439101, 0.1416661
1: 0.0482821, 0.2302842, 0.0492356, 0.2283794, -0.1800973, 0.1810486
2: 0.0624542, 0.2478489, 0.0635781, 0.2500039, -0.1875497, 0.1842708
3: -0.0129582, 0.2227221, -0.0150195, 0.2220079, -0.2349662, 0.2377415
4: 0.0008159, 0.2506008, 0.0004846, 0.2528384, -0.2520225, 0.2501161

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0788392, 0.2222926, 0.0783825, 0.2205054, -0.1416661, 0.1439101
1: 0.0492356, 0.2283794, 0.0482821, 0.2302842, -0.1810486, 0.1800973
2: 0.0635781, 0.2500039, 0.0624542, 0.2478489, -0.1842708, 0.1875497
3: -0.0150195, 0.2220079, -0.0129582, 0.2227221, -0.2377415, 0.2349662
4: 0.0004846, 0.2528384, 0.0008159, 0.2506008, -0.2501161, 0.2520225

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0788392, 0.2222926, 0.0788392, 0.2222926, -0.1434534, 0.1434534
1: 0.0492356, 0.2283794, 0.0492356, 0.2283794, -0.1791438, 0.1791438
2: 0.0635781, 0.2500039, 0.0635781, 0.2500039, -0.1864258, 0.1864258
3: -0.0150195, 0.2220079, -0.0150195, 0.2220079, -0.2370274, 0.2370274
4: 0.0004846, 0.2528384, 0.0004846, 0.2528384, -0.2523538, 0.2523538

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0783825, 0.2205054, 0.0751822, 0.2249900, -0.1466074, 0.1453232
1: 0.0482821, 0.2302842, 0.0431372, 0.2356555, -0.1873734, 0.1871470
2: 0.0624542, 0.2478489, 0.0588951, 0.2540951, -0.1916409, 0.1889538
3: -0.0129582, 0.2227221, -0.0163606, 0.2275606, -0.2405189, 0.2390827
4: 0.0008159, 0.2506008, -0.0018036, 0.2591435, -0.2583276, 0.2524044

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0788392, 0.2222926, 0.0751822, 0.2249900, -0.1461507, 0.1471104
1: 0.0492356, 0.2283794, 0.0431372, 0.2356555, -0.1864199, 0.1852422
2: 0.0635781, 0.2500039, 0.0588951, 0.2540951, -0.1905170, 0.1911088
3: -0.0150195, 0.2220079, -0.0163606, 0.2275606, -0.2425801, 0.2383686
4: 0.0004846, 0.2528384, -0.0018036, 0.2591435, -0.2586589, 0.2546421

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0783825, 0.2205054, 0.0759371, 0.2269447, -0.1485622, 0.1445683
1: 0.0482821, 0.2302842, 0.0437303, 0.2337886, -0.1855066, 0.1865540
2: 0.0624542, 0.2478489, 0.0602925, 0.2563978, -0.1939436, 0.1875564
3: -0.0129582, 0.2227221, -0.0192161, 0.2267310, -0.2396892, 0.2419381
4: 0.0008159, 0.2506008, -0.0018764, 0.2617289, -0.2609130, 0.2524772

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0788392, 0.2222926, 0.0759371, 0.2269447, -0.1481055, 0.1463555
1: 0.0492356, 0.2283794, 0.0437303, 0.2337886, -0.1845530, 0.1846491
2: 0.0635781, 0.2500039, 0.0602925, 0.2563978, -0.1928197, 0.1897113
3: -0.0150195, 0.2220079, -0.0192161, 0.2267310, -0.2417505, 0.2412240
4: 0.0004846, 0.2528384, -0.0018764, 0.2617289, -0.2612443, 0.2547148

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0751822, 0.2249900, 0.0783825, 0.2205054, -0.1453232, 0.1466074
1: 0.0431372, 0.2356555, 0.0482821, 0.2302842, -0.1871470, 0.1873734
2: 0.0588951, 0.2540951, 0.0624542, 0.2478489, -0.1889538, 0.1916409
3: -0.0163606, 0.2275606, -0.0129582, 0.2227221, -0.2390827, 0.2405189
4: -0.0018036, 0.2591435, 0.0008159, 0.2506008, -0.2524044, 0.2583276

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0751822, 0.2249900, 0.0788392, 0.2222926, -0.1471104, 0.1461507
1: 0.0431372, 0.2356555, 0.0492356, 0.2283794, -0.1852422, 0.1864199
2: 0.0588951, 0.2540951, 0.0635781, 0.2500039, -0.1911088, 0.1905170
3: -0.0163606, 0.2275606, -0.0150195, 0.2220079, -0.2383686, 0.2425801
4: -0.0018036, 0.2591435, 0.0004846, 0.2528384, -0.2546421, 0.2586589

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0759371, 0.2269447, 0.0783825, 0.2205054, -0.1445683, 0.1485622
1: 0.0437303, 0.2337886, 0.0482821, 0.2302842, -0.1865540, 0.1855066
2: 0.0602925, 0.2563978, 0.0624542, 0.2478489, -0.1875564, 0.1939436
3: -0.0192161, 0.2267310, -0.0129582, 0.2227221, -0.2419381, 0.2396892
4: -0.0018764, 0.2617289, 0.0008159, 0.2506008, -0.2524772, 0.2609130

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0759371, 0.2269447, 0.0788392, 0.2222926, -0.1463555, 0.1481055
1: 0.0437303, 0.2337886, 0.0492356, 0.2283794, -0.1846491, 0.1845530
2: 0.0602925, 0.2563978, 0.0635781, 0.2500039, -0.1897113, 0.1928197
3: -0.0192161, 0.2267310, -0.0150195, 0.2220079, -0.2412240, 0.2417505
4: -0.0018764, 0.2617289, 0.0004846, 0.2528384, -0.2547148, 0.2612443

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0751822, 0.2249900, 0.0751822, 0.2249900, -0.1498078, 0.1498078
1: 0.0431372, 0.2356555, 0.0431372, 0.2356555, -0.1925183, 0.1925183
2: 0.0588951, 0.2540951, 0.0588951, 0.2540951, -0.1952001, 0.1952001
3: -0.0163606, 0.2275606, -0.0163606, 0.2275606, -0.2439213, 0.2439213
4: -0.0018036, 0.2591435, -0.0018036, 0.2591435, -0.2609472, 0.2609472

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0751822, 0.2249900, 0.0759371, 0.2269447, -0.1517626, 0.1490529
1: 0.0431372, 0.2356555, 0.0437303, 0.2337886, -0.1906514, 0.1919252
2: 0.0588951, 0.2540951, 0.0602925, 0.2563978, -0.1975028, 0.1938026
3: -0.0163606, 0.2275606, -0.0192161, 0.2267310, -0.2430916, 0.2467767
4: -0.0018036, 0.2591435, -0.0018764, 0.2617289, -0.2635325, 0.2610199

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0759371, 0.2269447, 0.0751822, 0.2249900, -0.1490529, 0.1517626
1: 0.0437303, 0.2337886, 0.0431372, 0.2356555, -0.1919252, 0.1906514
2: 0.0602925, 0.2563978, 0.0588951, 0.2540951, -0.1938026, 0.1975028
3: -0.0192161, 0.2267310, -0.0163606, 0.2275606, -0.2467767, 0.2430916
4: -0.0018764, 0.2617289, -0.0018036, 0.2591435, -0.2610199, 0.2635325

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0759371, 0.2269447, 0.0759371, 0.2269447, -0.1510077, 0.1510077
1: 0.0437303, 0.2337886, 0.0437303, 0.2337886, -0.1900584, 0.1900584
2: 0.0602925, 0.2563978, 0.0602925, 0.2563978, -0.1961053, 0.1961053
3: -0.0192161, 0.2267310, -0.0192161, 0.2267310, -0.2459471, 0.2459471
4: -0.0018764, 0.2617289, -0.0018764, 0.2617289, -0.2636053, 0.2636053

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.35 + 110.89 = 113.24 seconds
