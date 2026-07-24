## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 110.29190320664999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-49.3401833, 71.9582520, -49.3401833, 71.9582520, -121.2984314, 121.2984314)
1: (-39.4408836, 66.3552780, -39.4408836, 66.3552780, -105.7961578, 105.7961578)
2: (-33.4991455, 70.2209854, -33.4991455, 70.2209854, -103.7201309, 103.7201309)
3: (-53.5254898, 67.6930313, -53.5254898, 67.6930313, -121.2185211, 121.2185211)
4: (-38.8239403, 75.6963577, -38.8239403, 75.6963577, -114.5202942, 114.5202942)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 2.20 = 3.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -110.3029335, upper bound: 110.3029335

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2817471, upper bound: 110.2977862
time: 1.02 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2766108, upper bound: 110.2766108
time: 0.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.00 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.00
Output dim: 3, lower bound: -110.2817471, upper bound: 110.2977862
NS_A2, status: Status.VERIFIED, split count: 1, time: 2.00
Output dim: 3, lower bound: -110.2766108, upper bound: 110.2766108

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -44.0261078, 64.2232971, -49.3401833, 71.9582520, -115.9843597, 113.5634766
1: -35.3125572, 59.3043060, -39.4408836, 66.3552780, -101.6678238, 98.7451935
2: -29.9509239, 62.9027901, -33.4991455, 70.2209854, -100.1719055, 96.4019241
3: -48.1699867, 60.4233284, -53.5254898, 67.6930313, -115.8630219, 113.9488220
4: -34.6387939, 67.9170990, -38.8239403, 75.6963577, -110.3351440, 106.7410355

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2795197, upper bound: 110.2960826
time: 0.88 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2799244, upper bound: 110.2966942
time: 0.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.11 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 3, lower bound: -110.2795197, upper bound: 110.2960826
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 3, lower bound: -110.2799244, upper bound: 110.2966942

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -26.7596970, 38.9945946, -49.3253250, 71.9359512, -98.6956329, 88.3198853
1: -21.9750061, 35.9142647, -39.4294357, 66.3344650, -88.3094482, 75.3437042
2: -18.6125469, 37.9555130, -33.4894142, 70.1986847, -88.8112259, 71.4449310
3: -30.1447124, 36.8029709, -53.5098991, 67.6721725, -97.8168793, 90.3128586
4: -21.6015415, 41.3991547, -38.8128777, 75.6727676, -97.2743073, 80.2120361

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2795197, upper bound: 110.2960826
time: 0.58 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2720348, upper bound: 110.2610935
time: 1.02 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -43.6388092, 63.6405106, -49.3401833, 71.9582520, -115.5970459, 112.9806976
1: -34.9972839, 58.7506714, -39.4408836, 66.3552780, -101.3525543, 98.1915436
2: -29.6778946, 62.3347054, -33.4991455, 70.2209854, -99.8988800, 95.8338470
3: -47.7600365, 59.8603249, -53.5254898, 67.6930313, -115.4530640, 113.3858109
4: -34.3257332, 67.3130035, -38.8239403, 75.6963577, -110.0220947, 106.1369324

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2795197, upper bound: 110.2966942
time: 0.63 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2795197, upper bound: 110.2966942
time: 0.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.00 seconds
NS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 3, lower bound: -110.2795197, upper bound: 110.2960826
NS_A1_A1_A2, status: Status.VERIFIED, split count: 3, time: 3.00
Output dim: 3, lower bound: -110.2720348, upper bound: 110.2610935
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 3, lower bound: -110.2795197, upper bound: 110.2966942
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 3, lower bound: -110.2795197, upper bound: 110.2966942

## BFS NS instance: NS_A1_A1_A1

### Backsubstitution after applying NS history:
0: -23.0428982, 33.3883057, -49.3253250, 71.9359512, -94.9788513, 82.7136078
1: -18.9467678, 30.6928692, -39.4294357, 66.3344650, -85.2812195, 70.1223068
2: -16.0146427, 32.4796982, -33.4894142, 70.1986847, -86.2133255, 65.9691086
3: -26.0264225, 31.4916210, -53.5098991, 67.6721725, -93.6985931, 85.0014954
4: -18.5842152, 35.5076752, -38.8128777, 75.6727676, -94.2569809, 74.3205490

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2768493, upper bound: 110.2960451
time: 0.81 seconds

## Relational analysis of NS_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2634205, upper bound: 110.2954215
time: 0.67 seconds

## Relational analysis of NS_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_A1_B1

### Relational analysis result of NS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2737501, upper bound: 110.2937702
time: 0.64 seconds

## Relational analysis of NS_A1_A1_A1_B2

### Relational analysis result of NS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2770403, upper bound: 110.2945450
time: 0.96 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -43.6388092, 63.6405106, -44.0261078, 64.2232971, -107.8620834, 107.6666183
1: -34.9972839, 58.7506714, -35.3125572, 59.3043060, -94.3015823, 94.0632172
2: -29.6778946, 62.3347054, -29.9509239, 62.9027901, -92.5806808, 92.2856293
3: -47.7600365, 59.8603249, -48.1699867, 60.4233284, -108.1833649, 108.0303116
4: -34.3257332, 67.3130035, -34.6387939, 67.9170990, -102.2428284, 101.9517899

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2798247, upper bound: 110.2913262
time: 1.01 seconds

## Relational analysis of NS_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2760084, upper bound: 110.2951128
time: 0.63 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2776117, upper bound: 110.2957570
time: 0.64 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -43.6388092, 63.6405106, -51.8509254, 75.2086945, -118.8474884, 115.4914398
1: -34.9972839, 58.7506714, -41.3074570, 69.3799744, -104.3772507, 100.0581207
2: -29.6778946, 62.3347054, -35.1515656, 73.4152451, -103.0931396, 97.4862671
3: -47.7600365, 59.8603249, -56.1897697, 70.8145828, -118.5746155, 116.0500870
4: -34.3257332, 67.3130035, -40.8006821, 79.2240067, -113.5497437, 108.1136856

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2768493, upper bound: 110.2966567
time: 0.93 seconds

## Relational analysis of NS_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2795197, upper bound: 110.2966942
time: 0.99 seconds

## Relational analysis of NS_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2720348, upper bound: 110.2803353
time: 0.83 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.37 seconds
NS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.37
Output dim: 3, lower bound: -110.2737501, upper bound: 110.2937702
NS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.37
Output dim: 3, lower bound: -110.2770403, upper bound: 110.2945450
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.37
Output dim: 3, lower bound: -110.2760084, upper bound: 110.2951128
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.37
Output dim: 3, lower bound: -110.2776117, upper bound: 110.2957570
NS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.37
Output dim: 3, lower bound: -110.2795197, upper bound: 110.2966942
NS_A1_A2_B2_A2, status: Status.VERIFIED, split count: 4, time: 5.37
Output dim: 3, lower bound: -110.2720348, upper bound: 110.2803353

## BFS NS instance: NS_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -23.0428982, 33.3883057, -42.3015327, 61.8034286, -84.8463287, 75.6898346
1: -18.9467678, 30.6928692, -33.8693771, 56.9463997, -75.8931656, 64.5622406
2: -16.0146427, 32.4796982, -28.7918110, 60.0724258, -76.0870514, 61.2715073
3: -26.0264225, 31.4916210, -45.7080841, 58.3093376, -84.3357620, 77.1997070
4: -18.5842152, 35.5076752, -33.5916443, 64.7364807, -83.3206940, 69.0993118

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2713212, upper bound: 110.2920846
time: 0.62 seconds

## Relational analysis of NS_A1_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2689211, upper bound: 110.2916016
time: 0.94 seconds

## BFS NS instance: NS_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -23.0428982, 33.3883057, -47.9857178, 69.9693680, -93.0122681, 81.3740158
1: -18.9467678, 30.6928692, -38.3581047, 64.5269318, -83.4736938, 69.0509720
2: -16.0146427, 32.4796982, -32.5711823, 68.3045120, -84.3191528, 65.0508804
3: -26.0264225, 31.4916210, -52.1085281, 65.8173065, -91.8437271, 83.6001434
4: -18.5842152, 35.5076752, -37.7180328, 73.6558914, -92.2401047, 73.2256927

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2693429, upper bound: 110.2944199
time: 1.09 seconds

## Relational analysis of NS_A1_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2770403, upper bound: 110.2945367
time: 0.75 seconds

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -43.6388092, 63.6405106, -37.0266342, 54.0763702, -97.7151642, 100.6671448
1: -34.9972839, 58.7506714, -29.7324085, 49.9074249, -84.9047012, 88.4830780
2: -29.6778946, 62.3347054, -25.2101345, 52.7719002, -82.4497986, 87.5448380
3: -47.7600365, 59.8603249, -40.2778206, 51.0218964, -98.7819366, 100.1381454
4: -34.3257332, 67.3130035, -29.3433304, 56.9319763, -91.2577057, 96.6563339

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2970429, upper bound: 110.3003719
time: 1.12 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2976144, upper bound: 110.3004009
time: 0.64 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -43.6388092, 63.6405106, -42.7648430, 62.3412361, -105.9800415, 106.4053497
1: -34.9972839, 58.7506714, -34.3122368, 57.5657349, -92.5630188, 93.0629120
2: -29.6778946, 62.3347054, -29.0913887, 61.0827332, -90.7606277, 91.4260941
3: -47.7600365, 59.8603249, -46.8368301, 58.6393738, -106.3994141, 106.6971588
4: -34.3257332, 67.3130035, -33.6217575, 65.9797211, -100.3054504, 100.9347610

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2933393, upper bound: 110.2998292
time: 0.76 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2991988, upper bound: 110.2991988
time: 1.02 seconds

## BFS NS instance: NS_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -37.0164680, 53.6788406, -51.8509254, 75.2086945, -112.2251587, 105.5297699
1: -29.7266006, 49.4665794, -41.3074570, 69.3799744, -99.1065674, 90.7740326
2: -25.1554070, 52.5385590, -35.1515656, 73.4152451, -98.5706482, 87.6901245
3: -40.6107368, 50.4548759, -56.1897697, 70.8145828, -111.4253235, 106.6446304
4: -29.0924835, 56.8481140, -40.8006821, 79.2240067, -108.3164749, 97.6487961

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_B2_A1_A1

### Relational analysis result of NS_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2637907, upper bound: 110.2960334
time: 0.89 seconds

## Relational analysis of NS_A1_A2_B2_A1_A2

### Relational analysis result of NS_A1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2636449, upper bound: 110.2859694
time: 0.63 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.35 seconds
NS_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 3, lower bound: -110.2713212, upper bound: 110.2920846
NS_A1_A1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 3, lower bound: -110.2689211, upper bound: 110.2916016
NS_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 3, lower bound: -110.2693429, upper bound: 110.2944199
NS_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 3, lower bound: -110.2770403, upper bound: 110.2945367
NS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 3, lower bound: -110.2970429, upper bound: 110.3003719
NS_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 3, lower bound: -110.2976144, upper bound: 110.3004009
NS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 3, lower bound: -110.2933393, upper bound: 110.2998292
NS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 3, lower bound: -110.2991988, upper bound: 110.2991988
NS_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 3, lower bound: -110.2637907, upper bound: 110.2960334
NS_A1_A2_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 3, lower bound: -110.2636449, upper bound: 110.2859694

## BFS NS instance: NS_A1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -23.0428982, 33.3883057, -40.0520096, 58.4614677, -81.5043640, 73.4403152
1: -18.9467678, 30.6928692, -32.0848503, 53.8710289, -72.8177948, 62.7777100
2: -16.0146427, 32.4796982, -27.2324753, 56.8728485, -72.8874893, 59.7121735
3: -26.0264225, 31.4916210, -43.3454857, 55.1305847, -81.1570053, 74.8370972
4: -18.5842152, 35.5076752, -31.7575531, 61.3126450, -79.8968430, 67.2652206

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A1_A1_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2713206, upper bound: 110.2920846
time: 0.59 seconds

## Relational analysis of NS_A1_A1_A1_B1_B1_B2

### Relational analysis result of NS_A1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2713206, upper bound: 110.2920846
time: 0.70 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -23.0428982, 33.3883057, -43.3520050, 62.9735718, -86.0164719, 76.7402954
1: -18.9467678, 30.6928692, -34.7014313, 58.0655556, -77.0123062, 65.3942871
2: -16.0146427, 32.4796982, -29.4227467, 61.5683556, -77.5830002, 61.9024353
3: -26.0264225, 31.4916210, -47.3250313, 59.2092018, -85.2356262, 78.8166504
4: -18.5842152, 35.5076752, -34.0199852, 66.5135040, -85.0977173, 69.5276413

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2686914, upper bound: 110.2928517
time: 0.92 seconds

## Relational analysis of NS_A1_A1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2693429, upper bound: 110.2944199
time: 0.63 seconds

## Relational analysis of NS_A1_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2693429, upper bound: 110.2944199
time: 0.87 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -23.0428982, 33.3883057, -47.0518723, 68.5811996, -91.6240997, 80.4401779
1: -18.9467678, 30.6928692, -37.6278496, 63.2247162, -82.1714706, 68.3207092
2: -16.0146427, 32.4796982, -31.9534950, 66.9125519, -82.9271927, 64.4331894
3: -26.0264225, 31.4916210, -51.1100693, 64.5105896, -90.5370102, 82.6016922
4: -18.5842152, 35.5076752, -37.0142365, 72.1744995, -90.7587128, 72.5219040

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2753471, upper bound: 110.2929322
time: 0.93 seconds

## Relational analysis of NS_A1_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2722443, upper bound: 110.2919046
time: 0.95 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -43.6388092, 63.6405106, -22.0261707, 32.0627861, -75.7015915, 85.6666794
1: -34.9972839, 58.7506714, -18.1256790, 29.4096718, -64.4069519, 76.8763275
2: -29.6778946, 62.3347054, -15.3146496, 30.9919376, -60.6698303, 77.6493530
3: -47.7600365, 59.8603249, -24.5969238, 30.2939625, -78.0540009, 84.4572449
4: -34.3257332, 67.3130035, -17.8736763, 33.7776642, -68.1033936, 85.1866760

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B1_B1_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2956280, upper bound: 110.2837304
time: 0.76 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2621744, upper bound: 110.2765630
time: 0.94 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -43.6388092, 63.6405106, -36.6411972, 53.5077209, -97.1464996, 100.2817001
1: -34.9972839, 58.7506714, -29.4214668, 49.3871002, -84.3843765, 88.1721191
2: -29.6778946, 62.3347054, -24.9394436, 52.2345467, -81.9124451, 87.2741470
3: -47.7600365, 59.8603249, -39.8699837, 50.4863091, -98.2463455, 99.7303085
4: -34.3257332, 67.3130035, -29.0274220, 56.3519859, -90.6777191, 96.3404160

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2963520, upper bound: 110.2995012
time: 0.84 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2957216, upper bound: 110.2988258
time: 0.91 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -43.6388092, 63.6405106, -40.5834656, 59.0667877, -102.7055893, 104.2239761
1: -34.9972839, 58.7506714, -32.5769310, 54.5345688, -89.5318527, 91.3275909
2: -29.6778946, 62.3347054, -27.6003761, 57.9234161, -87.6013107, 89.9350815
3: -47.7600365, 59.8603249, -44.5465240, 55.5283699, -103.2884064, 104.4068451
4: -34.3257332, 67.3130035, -31.8727283, 62.6195221, -96.9452515, 99.1857147

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B1_B2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2933393, upper bound: 110.2990614
time: 0.71 seconds

## Relational analysis of NS_A1_A2_B1_B2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2970577, upper bound: 110.2995286
time: 0.97 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -43.6388092, 63.6405106, -40.7060699, 59.4078522, -103.0466309, 104.3465805
1: -34.9972839, 58.7506714, -32.6908722, 54.8694611, -89.8667297, 91.4415436
2: -29.6778946, 62.3347054, -27.6843185, 58.2127571, -87.8906479, 90.0190277
3: -47.7600365, 59.8603249, -44.5952148, 55.9273682, -103.6874084, 104.4555359
4: -34.3257332, 67.3130035, -32.0053253, 62.8714714, -97.1972046, 99.3183289

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B1_B2_B2_B1

### Relational analysis result of NS_A1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2957216, upper bound: 110.2989154
time: 0.67 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2_B2

### Relational analysis result of NS_A1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2974797, upper bound: 110.2991932
time: 0.97 seconds

## BFS NS instance: NS_A1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -36.5485191, 52.9725723, -51.8509254, 75.2086945, -111.7572174, 104.8235016
1: -29.3577385, 48.8141594, -41.3074570, 69.3799744, -98.7377167, 90.1216125
2: -24.8364735, 51.8516884, -35.1515656, 73.4152451, -98.2517166, 87.0032501
3: -40.1164246, 49.7925301, -56.1897697, 70.8145828, -110.9310074, 105.9822998
4: -28.7198143, 56.1157532, -40.8006821, 79.2240067, -107.9438171, 96.9164352

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B2_A1_A1_B1

### Relational analysis result of NS_A1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2630734, upper bound: 110.2946406
time: 0.77 seconds

## Relational analysis of NS_A1_A2_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_A1_A1_B1

### Relational analysis result of NS_A1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2637853, upper bound: 110.2960233
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B2_A1_A1_B2

### Relational analysis result of NS_A1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2557575, upper bound: 110.2952017
time: 0.92 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.65 seconds
NS_A1_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2713206, upper bound: 110.2920846
NS_A1_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2713206, upper bound: 110.2920846
NS_A1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2693429, upper bound: 110.2944199
NS_A1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2693429, upper bound: 110.2944199
NS_A1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2753471, upper bound: 110.2929322
NS_A1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2722443, upper bound: 110.2919046
NS_A1_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2956280, upper bound: 110.2837304
NS_A1_A2_B1_B1_B1_B2, status: Status.VERIFIED, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2621744, upper bound: 110.2765630
NS_A1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2963520, upper bound: 110.2995012
NS_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2957216, upper bound: 110.2988258
NS_A1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2933393, upper bound: 110.2990614
NS_A1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2970577, upper bound: 110.2995286
NS_A1_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2957216, upper bound: 110.2989154
NS_A1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2974797, upper bound: 110.2991932
NS_A1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2637853, upper bound: 110.2960233
NS_A1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.65
Output dim: 3, lower bound: -110.2557575, upper bound: 110.2952017

## BFS NS instance: NS_A1_A1_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -23.0428982, 33.3883057, -34.9027100, 50.9208145, -73.9637146, 68.2909927
1: -18.9467678, 30.6928692, -28.0472393, 46.9997711, -65.9465408, 58.7401085
2: -16.0146427, 32.4796982, -23.7506275, 49.7510605, -65.7657013, 56.2303238
3: -26.0264225, 31.4916210, -38.0662193, 48.0333519, -74.0597763, 69.5578308
4: -18.5842152, 35.5076752, -27.5988464, 53.7108688, -72.2950745, 63.1065063

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_A1_B1_B1_B1_A1

### Relational analysis result of NS_A1_A1_A1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2712778, upper bound: 110.2913741
time: 0.87 seconds

## Relational analysis of NS_A1_A1_A1_B1_B1_B1_A2

### Relational analysis result of NS_A1_A1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2712778, upper bound: 110.2920846
time: 0.96 seconds

## BFS NS instance: NS_A1_A1_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -23.0428982, 33.3883057, -43.5814056, 63.2594337, -86.3023300, 76.9697113
1: -18.9467678, 30.6928692, -34.7996941, 58.2790146, -77.2257767, 65.4925537
2: -16.0146427, 32.4796982, -29.5976639, 61.5638428, -77.5784836, 62.0773544
3: -26.0264225, 31.4916210, -47.1538162, 59.6528549, -85.6792755, 78.6454163
4: -18.5842152, 35.5076752, -34.5439682, 66.4336700, -85.0178833, 70.0516357

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_A1_B1_B1_B2_A1

### Relational analysis result of NS_A1_A1_A1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2712778, upper bound: 110.2913741
time: 0.91 seconds

## Relational analysis of NS_A1_A1_A1_B1_B1_B2_A2

### Relational analysis result of NS_A1_A1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2712778, upper bound: 110.2920846
time: 0.83 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -23.0428982, 33.3883057, -38.2283974, 55.4927177, -78.5356140, 71.6166992
1: -18.9467678, 30.6928692, -30.7143955, 51.2659187, -70.2126770, 61.4072533
2: -16.0146427, 32.4796982, -25.9953079, 54.4987946, -70.5134354, 58.4750061
3: -26.0264225, 31.4916210, -42.1360931, 52.1705360, -78.1969528, 73.6277008
4: -18.5842152, 35.5076752, -29.9862823, 58.9917145, -77.5759277, 65.4939346

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_A1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -23.0428982, 33.3883057, -45.9603539, 66.3199005, -89.3628006, 79.3486557
1: -18.9467678, 30.6928692, -36.6502953, 61.1593056, -80.1060410, 67.3431625
2: -16.0146427, 32.4796982, -31.1476078, 64.8426666, -80.8573074, 63.6273041
3: -26.0264225, 31.4916210, -50.0867043, 62.4112663, -88.4376907, 81.5783234
4: -18.5842152, 35.5076752, -36.0966759, 70.1333847, -88.7175980, 71.6043396

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -23.0428982, 33.3883057, -44.8150711, 65.2394257, -88.2823257, 78.2033691
1: -18.9467678, 30.6928692, -35.8529053, 60.1202164, -79.0669708, 66.5457764
2: -16.0146427, 32.4796982, -30.4267750, 63.6847878, -79.6994324, 62.9064674
3: -26.0264225, 31.4916210, -48.7642784, 61.3322716, -87.3586884, 80.2558899
4: -18.5842152, 35.5076752, -35.2225876, 68.7372360, -87.3214493, 70.7302322

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2712783, upper bound: 110.2922220
time: 0.71 seconds

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2712783, upper bound: 110.2926265
time: 0.76 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -23.0428982, 33.3883057, -44.9724121, 65.6226120, -88.6655121, 78.3607178
1: -18.9467678, 30.6928692, -36.0078392, 60.4982147, -79.4449768, 66.7007065
2: -16.0146427, 32.4796982, -30.5463448, 64.0023270, -80.0169678, 63.0260429
3: -26.0264225, 31.4916210, -48.8436317, 61.7738724, -87.8002777, 80.3352509
4: -18.5842152, 35.5076752, -35.4072418, 69.0218887, -87.6060944, 70.9149017

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_A1_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2688782, upper bound: 110.2911944
time: 1.01 seconds

## Relational analysis of NS_A1_A1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_A1_A1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2688782, upper bound: 110.2918630
time: 0.60 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -43.6388092, 63.6405106, -17.8042068, 25.7366276, -69.3754272, 81.4447174
1: -34.9972839, 58.7506714, -14.7358599, 23.5445499, -58.5418320, 73.4865265
2: -29.6778946, 62.3347054, -12.3865213, 24.8870449, -54.5649376, 74.7212296
3: -47.7600365, 59.8603249, -20.0918941, 24.2903156, -72.0503540, 79.9522171
4: -34.3257332, 67.3130035, -14.4392452, 27.2208958, -61.5466309, 81.7522507

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2962482, upper bound: 110.2805221
time: 0.95 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2940668, upper bound: 110.2739373
time: 0.60 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2940668, upper bound: 110.2838859
time: 0.91 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -41.4839478, 60.4205627, -36.6411972, 53.5077209, -94.9916611, 97.0617523
1: -33.2954216, 55.7789612, -29.4214668, 49.3871002, -82.6825104, 85.2004242
2: -28.2168522, 59.2334900, -24.9394436, 52.2345467, -80.4513931, 84.1729279
3: -45.5037689, 56.8079185, -39.8699837, 50.4863091, -95.9900818, 96.6779022
4: -32.6043282, 64.0128632, -29.0274220, 56.3519859, -88.9563065, 93.0402832

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2940372, upper bound: 110.2812196
time: 1.02 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2962015, upper bound: 110.2902495
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2955547, upper bound: 110.2984463
time: 0.99 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2953678, upper bound: 110.2984419
time: 1.03 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -41.7486420, 60.9535446, -36.6411972, 53.5077209, -95.2563477, 97.5947266
1: -33.5105209, 56.2746353, -29.4214668, 49.3871002, -82.8976212, 85.6960983
2: -28.3866997, 59.6931763, -24.9394436, 52.2345467, -80.6212387, 84.6326141
3: -45.6848030, 57.3771210, -39.8699837, 50.4863091, -96.1711044, 97.2471008
4: -32.8503838, 64.4475250, -29.0274220, 56.3519859, -89.2023468, 93.4749374

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2947122, upper bound: 110.2977043
time: 0.74 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2949342, upper bound: 110.2978996
time: 0.84 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -43.6388092, 63.6405106, -24.1472416, 34.9470558, -78.5858383, 87.7877502
1: -34.9972839, 58.7506714, -19.8914948, 32.2284966, -67.2257843, 78.6421356
2: -29.6778946, 62.3347054, -16.8092842, 34.1107864, -63.7886810, 79.1439896
3: -47.7600365, 59.8603249, -27.4348984, 33.0227242, -80.7827606, 87.2952194
4: -34.3257332, 67.3130035, -19.4530334, 37.3334503, -71.6591797, 86.7660370

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B2_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2929662, upper bound: 110.2955842
time: 0.73 seconds

## Relational analysis of NS_A1_A2_B1_B2_B1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2929662, upper bound: 110.2990614
time: 0.67 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -43.6388092, 63.6405106, -40.2609520, 58.5737915, -102.2125854, 103.9014587
1: -34.9972839, 58.7506714, -32.3201103, 54.0791168, -89.0764008, 91.0707703
2: -29.6778946, 62.3347054, -27.3807240, 57.4484787, -87.1263733, 89.7154312
3: -47.7600365, 59.8603249, -44.2037888, 55.0638237, -102.8238602, 104.0641174
4: -34.3257332, 67.3130035, -31.6159039, 62.1161880, -96.4419250, 98.9289093

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B2_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2933393, upper bound: 110.2981972
time: 1.12 seconds

## Relational analysis of NS_A1_A2_B1_B2_B1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2970577, upper bound: 110.2995286
time: 0.99 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -43.6388092, 63.6405106, -24.5585728, 35.6476822, -79.2864914, 88.1990814
1: -34.9972839, 58.7506714, -20.1751938, 32.7953796, -67.7926636, 78.9258499
2: -29.6778946, 62.3347054, -17.0628605, 34.6541252, -64.3320160, 79.3975677
3: -47.7600365, 59.8603249, -27.6626434, 33.6470490, -81.4070816, 87.5229645
4: -34.3257332, 67.3130035, -19.8094215, 37.8483315, -72.1740646, 87.1224213

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2953485, upper bound: 110.2954382
time: 1.03 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2929662, upper bound: 110.2989154
time: 0.92 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -43.6388092, 63.6405106, -40.2895660, 58.7787056, -102.4175110, 103.9300613
1: -34.9972839, 58.7506714, -32.3486443, 54.2659302, -89.2631912, 91.0993118
2: -29.6778946, 62.3347054, -27.3884315, 57.5934296, -87.2713242, 89.7231369
3: -47.7600365, 59.8603249, -44.1501656, 55.3143311, -103.0743713, 104.0104828
4: -34.3257332, 67.3130035, -31.6658173, 62.2130051, -96.5387421, 98.9788208

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2966927, upper bound: 110.2980631
time: 1.08 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2969148, upper bound: 110.2982361
time: 0.91 seconds

## BFS NS instance: NS_A1_A2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -36.5485191, 52.9725723, -47.3718414, 68.4347076, -104.9832306, 100.3444138
1: -29.3577385, 48.8141594, -37.7667084, 63.1103287, -92.4680634, 86.5808640
2: -24.8364735, 51.8516884, -32.1066017, 66.8840408, -91.7205048, 83.9582901
3: -40.1164246, 49.7925301, -51.5684471, 64.4096680, -104.5260925, 101.3609772
4: -28.7198143, 56.1157532, -37.2258720, 72.3013840, -101.0211945, 93.3416290

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_B2_A1_A1_B1_B1

### Relational analysis result of NS_A1_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2634205, upper bound: 110.2960233
time: 0.98 seconds

## Relational analysis of NS_A1_A2_B2_A1_A1_B1_B2

### Relational analysis result of NS_A1_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2634205, upper bound: 110.2960233
time: 0.88 seconds

## BFS NS instance: NS_A1_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -36.5485191, 52.9725723, -50.9855347, 73.9237595, -110.4722748, 103.9581070
1: -29.3577385, 48.8141594, -40.6328430, 68.1736450, -97.5313797, 89.4469986
2: -24.8364735, 51.8516884, -34.5824242, 72.1236649, -96.9601364, 86.4341049
3: -40.1164246, 49.7925301, -55.2625122, 69.6043320, -109.7207565, 105.0550385
4: -28.7198143, 56.1157532, -40.1541786, 77.8493958, -106.5692062, 96.2699280

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2565747, upper bound: 110.2845760
time: 0.96 seconds

## Relational analysis of NS_A1_A2_B2_A1_A1_B2_A2

### Relational analysis result of NS_A1_A2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2565747, upper bound: 110.2952017
time: 0.80 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.69 seconds
NS_A1_A1_A1_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2712778, upper bound: 110.2913741
NS_A1_A1_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2712778, upper bound: 110.2920846
NS_A1_A1_A1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2712778, upper bound: 110.2913741
NS_A1_A1_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2712778, upper bound: 110.2920846
NS_A1_A1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2712783, upper bound: 110.2922220
NS_A1_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2712783, upper bound: 110.2926265
NS_A1_A1_A1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2688782, upper bound: 110.2911944
NS_A1_A1_A1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2688782, upper bound: 110.2918630
NS_A1_A2_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2940668, upper bound: 110.2739373
NS_A1_A2_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2940668, upper bound: 110.2838859
NS_A1_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2955547, upper bound: 110.2984463
NS_A1_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2953678, upper bound: 110.2984419
NS_A1_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2947122, upper bound: 110.2977043
NS_A1_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2949342, upper bound: 110.2978996
NS_A1_A2_B1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2929662, upper bound: 110.2955842
NS_A1_A2_B1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2929662, upper bound: 110.2990614
NS_A1_A2_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2933393, upper bound: 110.2981972
NS_A1_A2_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2970577, upper bound: 110.2995286
NS_A1_A2_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2953485, upper bound: 110.2954382
NS_A1_A2_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2929662, upper bound: 110.2989154
NS_A1_A2_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2966927, upper bound: 110.2980631
NS_A1_A2_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2969148, upper bound: 110.2982361
NS_A1_A2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2634205, upper bound: 110.2960233
NS_A1_A2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2634205, upper bound: 110.2960233
NS_A1_A2_B2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2565747, upper bound: 110.2845760
NS_A1_A2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 3, lower bound: -110.2565747, upper bound: 110.2952017

## BFS NS instance: NS_A1_A1_A1_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -22.1585312, 31.9563618, -34.9027100, 50.9208145, -73.0793304, 66.8590622
1: -18.2378006, 29.4273300, -28.0472393, 46.9997711, -65.2375717, 57.4745712
2: -15.3983307, 31.1384811, -23.7506275, 49.7510605, -65.1493759, 54.8891029
3: -25.0903549, 30.1918354, -38.0662193, 48.0333519, -73.1237030, 68.2580261
4: -17.8444347, 34.0910263, -27.5988464, 53.7108688, -71.5552826, 61.6898651

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_B1_B1_B1_A2_A1

### Relational analysis result of NS_A1_A1_A1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2711857, upper bound: 110.2924772
time: 0.99 seconds

## Relational analysis of NS_A1_A1_A1_B1_B1_B1_A2_A2

### Relational analysis result of NS_A1_A1_A1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2711857, upper bound: 110.2931887
time: 0.89 seconds

## BFS NS instance: NS_A1_A1_A1_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -22.1585312, 31.9563618, -43.5814056, 63.2594337, -85.4179688, 75.5377655
1: -18.2378006, 29.4273300, -34.7996941, 58.2790146, -76.5168152, 64.2270050
2: -15.3983307, 31.1384811, -29.5976639, 61.5638428, -76.9621582, 60.7361336
3: -25.0903549, 30.1918354, -47.1538162, 59.6528549, -84.7432098, 77.3456268
4: -17.8444347, 34.0910263, -34.5439682, 66.4336700, -84.2780991, 68.6349945

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_B1_B1_B2_A2_A1

### Relational analysis result of NS_A1_A1_A1_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2709650, upper bound: 110.2913731
time: 0.60 seconds

## Relational analysis of NS_A1_A1_A1_B1_B1_B2_A2_A2

### Relational analysis result of NS_A1_A1_A1_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2709650, upper bound: 110.2920846
time: 0.76 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -17.8042068, 25.7366276, -44.8150711, 65.2394257, -83.0436325, 70.5516968
1: -14.7358599, 23.5445499, -35.8529053, 60.1202164, -74.8560638, 59.3974457
2: -12.3865213, 24.8870449, -30.4267750, 63.6847878, -76.0713120, 55.3138199
3: -20.0918941, 24.2903156, -48.7642784, 61.3322716, -81.4241638, 73.0545959
4: -14.4392452, 27.2208958, -35.2225876, 68.7372360, -83.1764832, 62.4434738

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A1_B1

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2712778, upper bound: 110.2922220
time: 0.69 seconds

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A1_B2

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2712778, upper bound: 110.2922220
time: 0.92 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -22.1585312, 31.9563618, -44.8150711, 65.2394257, -87.3979568, 76.7714310
1: -18.2378006, 29.4273300, -35.8529053, 60.1202164, -78.3580017, 65.2802353
2: -15.3983307, 31.1384811, -30.4267750, 63.6847878, -79.0831070, 61.5652428
3: -25.0903549, 30.1918354, -48.7642784, 61.3322716, -86.4226227, 78.9560928
4: -17.8444347, 34.0910263, -35.2225876, 68.7372360, -86.5816574, 69.3136063

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A2_A1

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2709650, upper bound: 110.2919662
time: 0.68 seconds

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A2_A2

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2709650, upper bound: 110.2926265
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -36.6411972, 53.5077209, -17.8042068, 25.7366276, -62.3778191, 71.3119202
1: -29.4214668, 49.3871002, -14.7358599, 23.5445499, -52.9660187, 64.1229477
2: -24.9394436, 52.2345467, -12.3865213, 24.8870449, -49.8264885, 64.6210709
3: -39.8699837, 50.4863091, -20.0918941, 24.2903156, -64.1603012, 70.5782013
4: -29.0274220, 56.3519859, -14.4392452, 27.2208958, -56.2483177, 70.7912292

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2924782, upper bound: 110.2714990
time: 1.12 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2925611, upper bound: 110.2697493
time: 0.96 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -42.4109230, 61.8017998, -17.8042068, 25.7366276, -68.1475525, 79.6060028
1: -34.0227127, 57.0550194, -14.7358599, 23.5445499, -57.5672607, 71.7908783
2: -28.8415699, 60.5567589, -12.3865213, 24.8870449, -53.7286148, 72.9432831
3: -46.4586220, 58.1191177, -20.0918941, 24.2903156, -70.7489319, 78.2110138
4: -33.3352890, 65.4217911, -14.4392452, 27.2208958, -60.5561829, 79.8610382

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2940668, upper bound: 110.2838859
time: 1.21 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2940668, upper bound: 110.2838859
time: 0.91 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -39.3334885, 57.1704597, -36.6411972, 53.5077209, -92.8411942, 93.8116302
1: -31.5721359, 52.8182869, -29.4214668, 49.3871002, -80.9592285, 82.2397461
2: -26.7320309, 56.1400452, -24.9394436, 52.2345467, -78.9665756, 81.0794907
3: -43.2408867, 53.7558174, -39.8699837, 50.4863091, -93.7271729, 93.6258011
4: -30.8398476, 60.7224808, -29.0274220, 56.3519859, -87.1918335, 89.7499008

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2954587, upper bound: 110.2897041
time: 0.92 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2955519, upper bound: 110.2984463
time: 0.95 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -37.6876755, 54.9266701, -36.6411972, 53.5077209, -91.1953964, 91.5678635
1: -30.2497196, 50.6742554, -29.4214668, 49.3871002, -79.6367950, 80.0957108
2: -25.6112423, 53.7606354, -24.9394436, 52.2345467, -77.8457794, 78.7000809
3: -41.2169876, 51.7038002, -39.8699837, 50.4863091, -91.7032928, 91.5737839
4: -29.6541958, 58.0571556, -29.0274220, 56.3519859, -86.0061722, 87.0845718

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2939535, upper bound: 110.2923407
time: 1.12 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2939535, upper bound: 110.2984419
time: 0.95 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -39.4900780, 57.5602989, -36.6411972, 53.5077209, -92.9977798, 94.2014847
1: -31.6919422, 53.1494827, -29.4214668, 49.3871002, -81.0790253, 82.5709305
2: -26.8132915, 56.4491920, -24.9394436, 52.2345467, -79.0478363, 81.3886337
3: -43.3081322, 54.1574860, -39.8699837, 50.4863091, -93.7944412, 94.0274658
4: -30.9794979, 60.9937134, -29.0274220, 56.3519859, -87.3314819, 90.0211182

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_A1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2925208, upper bound: 110.2970823
time: 0.90 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_A1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2925208, upper bound: 110.2970937
time: 1.03 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -38.2779274, 55.9137802, -36.6411972, 53.5077209, -91.7856445, 92.5549469
1: -30.7478714, 51.6129456, -29.4214668, 49.3871002, -80.1349716, 81.0344086
2: -26.0317593, 54.6833229, -24.9394436, 52.2345467, -78.2662888, 79.6227570
3: -41.7778282, 52.7051926, -39.8699837, 50.4863091, -92.2641296, 92.5751801
4: -30.2181091, 59.0113335, -29.0274220, 56.3519859, -86.5700760, 88.0387573

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2925208, upper bound: 110.2972776
time: 0.92 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2924764, upper bound: 110.2973840
time: 0.68 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -36.6411972, 53.5077209, -24.1472416, 34.9470558, -71.5882339, 77.6549606
1: -29.4214668, 49.3871002, -19.8914948, 32.2284966, -61.6499634, 69.2785721
2: -24.9394436, 52.2345467, -16.8092842, 34.1107864, -59.0502319, 69.0438156
3: -39.8699837, 50.4863091, -27.4348984, 33.0227242, -72.8926926, 77.9212036
4: -29.0274220, 56.3519859, -19.4530334, 37.3334503, -66.3608704, 75.8050079

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B2_B1_B1_A1_A1

### Relational analysis result of NS_A1_A2_B1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2932019
time: 0.63 seconds

## Relational analysis of NS_A1_A2_B1_B2_B1_B1_A1_A2

### Relational analysis result of NS_A1_A2_B1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2955842
time: 0.93 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -42.4109230, 61.8017998, -24.1472416, 34.9470558, -77.3579712, 85.9490433
1: -34.0227127, 57.0550194, -19.8914948, 32.2284966, -66.2512054, 76.9465027
2: -28.8415699, 60.5567589, -16.8092842, 34.1107864, -62.9523544, 77.3660202
3: -46.4586220, 58.1191177, -27.4348984, 33.0227242, -79.4813156, 85.5540085
4: -33.3352890, 65.4217911, -19.4530334, 37.3334503, -70.6687393, 84.8748245

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B2_B1_B1_A2_A1

### Relational analysis result of NS_A1_A2_B1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2990614
time: 1.01 seconds

## Relational analysis of NS_A1_A2_B1_B2_B1_B1_A2_A2

### Relational analysis result of NS_A1_A2_B1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2990614
time: 0.88 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -41.4839478, 60.4205627, -40.2609520, 58.5737915, -100.0577393, 100.6815033
1: -33.2954216, 55.7789612, -32.3201103, 54.0791168, -87.3745422, 88.0990753
2: -28.2168522, 59.2334900, -27.3807240, 57.4484787, -85.6653214, 86.6142120
3: -45.5037689, 56.8079185, -44.2037888, 55.0638237, -100.5675964, 101.0117035
4: -32.6043282, 64.0128632, -31.6159039, 62.1161880, -94.7205124, 95.6287689

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B2_B1_B2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2923377, upper bound: 110.2932019
time: 0.93 seconds

## Relational analysis of NS_A1_A2_B1_B2_B1_B2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2923377, upper bound: 110.2995286
time: 1.05 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -41.7486420, 60.9535446, -40.2609520, 58.5737915, -100.3224258, 101.2144775
1: -33.5105209, 56.2746353, -32.3201103, 54.0791168, -87.5896378, 88.5947342
2: -28.3866997, 59.6931763, -27.3807240, 57.4484787, -85.8351669, 87.0738983
3: -45.6848030, 57.3771210, -44.2037888, 55.0638237, -100.7486267, 101.5809097
4: -32.8503838, 64.4475250, -31.6159039, 62.1161880, -94.9665527, 96.0634232

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B2_B1_B2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2923377, upper bound: 110.2955881
time: 0.88 seconds

## Relational analysis of NS_A1_A2_B1_B2_B1_B2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2923377, upper bound: 110.2995286
time: 0.60 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -36.6411972, 53.5077209, -24.5585728, 35.6476822, -72.2888794, 78.0662842
1: -29.4214668, 49.3871002, -20.1751938, 32.7953796, -62.2168465, 69.5622711
2: -24.9394436, 52.2345467, -17.0628605, 34.6541252, -59.5935669, 69.2974091
3: -39.8699837, 50.4863091, -27.6626434, 33.6470490, -73.5170288, 78.1489563
4: -29.0274220, 56.3519859, -19.8094215, 37.8483315, -66.8757477, 76.1613998

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B2_B2_B1_A1_A1

### Relational analysis result of NS_A1_A2_B1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2930589
time: 0.87 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2_B1_A1_A2

### Relational analysis result of NS_A1_A2_B1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2954411
time: 0.93 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -42.4109230, 61.8017998, -24.5585728, 35.6476822, -78.0586090, 86.3603745
1: -34.0227127, 57.0550194, -20.1751938, 32.7953796, -66.8180923, 77.2302094
2: -28.8415699, 60.5567589, -17.0628605, 34.6541252, -63.4956932, 77.6196213
3: -46.4586220, 58.1191177, -27.6626434, 33.6470490, -80.1056366, 85.7817612
4: -33.3352890, 65.4217911, -19.8094215, 37.8483315, -71.1836243, 85.2312164

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B2_B2_B1_A2_A1

### Relational analysis result of NS_A1_A2_B1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2989184
time: 1.04 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2_B1_A2_A2

### Relational analysis result of NS_A1_A2_B1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2989184
time: 0.99 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -41.4607048, 60.3708534, -40.2895660, 58.7787056, -100.2394104, 100.6604004
1: -33.2594566, 55.7631645, -32.3486443, 54.2659302, -87.5253754, 88.1118088
2: -28.1800098, 59.2180862, -27.3884315, 57.5934296, -85.7734375, 86.6065216
3: -45.4684715, 56.7737236, -44.1501656, 55.3143311, -100.7827988, 100.9238892
4: -32.5424652, 63.9942856, -31.6658173, 62.2130051, -94.7554550, 95.6601028

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2980363, upper bound: 110.2980630
time: 0.78 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2980363, upper bound: 110.2980630
time: 1.15 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -39.7737923, 58.0298805, -40.2895660, 58.7787056, -98.5524979, 98.3194275
1: -31.9083290, 53.5324593, -32.3486443, 54.2659302, -86.1742477, 85.8810959
2: -27.0376472, 56.7330818, -27.3884315, 57.5934296, -84.6310730, 84.1215134
3: -43.3893280, 54.6376343, -44.1501656, 55.3143311, -98.7036591, 98.7877808
4: -31.3577919, 61.2228966, -31.6658173, 62.2130051, -93.5708008, 92.8887177

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2982583, upper bound: 110.2982361
time: 0.99 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2982583, upper bound: 110.2982361
time: 0.70 seconds

## BFS NS instance: NS_A1_A2_B2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -36.5485191, 52.9725723, -46.8781281, 67.6992264, -104.2477417, 99.8507004
1: -29.3577385, 48.8141594, -37.3850899, 62.4281769, -91.7859192, 86.1992416
2: -24.8364735, 51.8516884, -31.7776775, 66.1662445, -91.0027084, 83.6293640
3: -40.1164246, 49.7925301, -51.0553131, 63.7152748, -103.8316956, 100.8478394
4: -28.7198143, 56.1157532, -36.8398056, 71.5378647, -100.2576752, 92.9555588

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_A1_A1_B1_B1_A1

### Relational analysis result of NS_A1_A2_B2_A1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2635683, upper bound: 110.2853976
time: 0.75 seconds

## Relational analysis of NS_A1_A2_B2_A1_A1_B1_B1_A2

### Relational analysis result of NS_A1_A2_B2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2635683, upper bound: 110.2960233
time: 0.91 seconds

## BFS NS instance: NS_A1_A2_B2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -36.5485191, 52.9725723, -47.5451851, 68.7545547, -105.3030701, 100.5177612
1: -29.3577385, 48.8141594, -37.9440422, 63.3302460, -92.6879883, 86.7582016
2: -24.8364735, 51.8516884, -32.2653580, 67.1647644, -92.0012360, 84.1170502
3: -40.1164246, 49.7925301, -51.7972107, 64.6090240, -104.7254486, 101.5897369
4: -28.7198143, 56.1157532, -37.4488029, 72.5966644, -101.3164673, 93.5645523

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_A1_A1_B1_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2635683, upper bound: 110.2853976
time: 0.94 seconds

## Relational analysis of NS_A1_A2_B2_A1_A1_B1_B2_A2

### Relational analysis result of NS_A1_A2_B2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2635683, upper bound: 110.2960233
time: 0.59 seconds

## BFS NS instance: NS_A1_A2_B2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -35.9560547, 52.0968666, -50.9855347, 73.9237595, -109.8798141, 103.0823898
1: -28.8900509, 47.9917946, -40.6328430, 68.1736450, -97.0636902, 88.6246338
2: -24.4437256, 50.9698906, -34.5824242, 72.1236649, -96.5673904, 85.5522995
3: -39.4713860, 48.9712791, -55.2625122, 69.6043320, -109.0757141, 104.2337952
4: -28.2736664, 55.1733856, -40.1541786, 77.8493958, -106.1230621, 95.3275528

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_B2_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2565747, upper bound: 110.2952017
time: 0.86 seconds

## Relational analysis of NS_A1_A2_B2_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2565747, upper bound: 110.2952017
time: 0.87 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.38 seconds
NS_A1_A1_A1_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2711857, upper bound: 110.2924772
NS_A1_A1_A1_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2711857, upper bound: 110.2931887
NS_A1_A1_A1_B1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2709650, upper bound: 110.2913731
NS_A1_A1_A1_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2709650, upper bound: 110.2920846
NS_A1_A1_A1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2712778, upper bound: 110.2922220
NS_A1_A1_A1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2712778, upper bound: 110.2922220
NS_A1_A1_A1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2709650, upper bound: 110.2919662
NS_A1_A1_A1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2709650, upper bound: 110.2926265
NS_A1_A2_B1_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2924782, upper bound: 110.2714990
NS_A1_A2_B1_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2925611, upper bound: 110.2697493
NS_A1_A2_B1_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2940668, upper bound: 110.2838859
NS_A1_A2_B1_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2940668, upper bound: 110.2838859
NS_A1_A2_B1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2954587, upper bound: 110.2897041
NS_A1_A2_B1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2955519, upper bound: 110.2984463
NS_A1_A2_B1_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2939535, upper bound: 110.2923407
NS_A1_A2_B1_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2939535, upper bound: 110.2984419
NS_A1_A2_B1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2925208, upper bound: 110.2970823
NS_A1_A2_B1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2925208, upper bound: 110.2970937
NS_A1_A2_B1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2925208, upper bound: 110.2972776
NS_A1_A2_B1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2924764, upper bound: 110.2973840
NS_A1_A2_B1_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2932019
NS_A1_A2_B1_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2955842
NS_A1_A2_B1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2990614
NS_A1_A2_B1_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2990614
NS_A1_A2_B1_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2923377, upper bound: 110.2932019
NS_A1_A2_B1_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2923377, upper bound: 110.2995286
NS_A1_A2_B1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2923377, upper bound: 110.2955881
NS_A1_A2_B1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2923377, upper bound: 110.2995286
NS_A1_A2_B1_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2930589
NS_A1_A2_B1_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2954411
NS_A1_A2_B1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2989184
NS_A1_A2_B1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2940981, upper bound: 110.2989184
NS_A1_A2_B1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2980363, upper bound: 110.2980630
NS_A1_A2_B1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2980363, upper bound: 110.2980630
NS_A1_A2_B1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2982583, upper bound: 110.2982361
NS_A1_A2_B1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2982583, upper bound: 110.2982361
NS_A1_A2_B2_A1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2635683, upper bound: 110.2853976
NS_A1_A2_B2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2635683, upper bound: 110.2960233
NS_A1_A2_B2_A1_A1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2635683, upper bound: 110.2853976
NS_A1_A2_B2_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2635683, upper bound: 110.2960233
NS_A1_A2_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2565747, upper bound: 110.2952017
NS_A1_A2_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.38
Output dim: 3, lower bound: -110.2565747, upper bound: 110.2952017

## BFS NS instance: NS_A1_A1_A1_B1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -20.4448090, 29.3368053, -34.9027100, 50.9208145, -71.3656235, 64.2394943
1: -16.8664875, 27.0127811, -28.0472393, 46.9997711, -63.8662567, 55.0600204
2: -14.2126341, 28.6432552, -23.7506275, 49.7510605, -63.9636841, 52.3938828
3: -23.3248024, 27.7187843, -38.0662193, 48.0333519, -71.3581543, 65.7849960
4: -16.4322357, 31.4498653, -27.5988464, 53.7108688, -70.1430664, 59.0487061

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A1_A1_B1_B1_B1_A2_A1_B1

### Relational analysis result of NS_A1_A1_A1_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2715419, upper bound: 110.2924772
time: 0.66 seconds

## Relational analysis of NS_A1_A1_A1_B1_B1_B1_A2_A1_B2

### Relational analysis result of NS_A1_A1_A1_B1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2715419, upper bound: 110.2924772
time: 1.03 seconds

## BFS NS instance: NS_A1_A1_A1_B1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -20.5661011, 29.5958290, -34.9027100, 50.9208145, -71.4869080, 64.4985275
1: -16.9293365, 27.1672230, -28.0472393, 46.9997711, -63.9291077, 55.2144623
2: -14.2718859, 28.7536163, -23.7506275, 49.7510605, -64.0229492, 52.5042419
3: -23.2549477, 27.9210701, -38.0662193, 48.0333519, -71.2882996, 65.9872818
4: -16.5585098, 31.5066795, -27.5988464, 53.7108688, -70.2693634, 59.1055222

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A1_A1_B1_B1_B1_A2_A2_B1

### Relational analysis result of NS_A1_A1_A1_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2715419, upper bound: 110.2931887
time: 0.97 seconds

## Relational analysis of NS_A1_A1_A1_B1_B1_B1_A2_A2_B2

### Relational analysis result of NS_A1_A1_A1_B1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2715419, upper bound: 110.2931887
time: 0.63 seconds

## BFS NS instance: NS_A1_A1_A1_B1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -20.5661011, 29.5958290, -43.5814056, 63.2594337, -83.8255310, 73.1772308
1: -16.9293365, 27.1672230, -34.7996941, 58.2790146, -75.2083359, 61.9669037
2: -14.2718859, 28.7536163, -29.5976639, 61.5638428, -75.8357315, 58.3512802
3: -23.2549477, 27.9210701, -47.1538162, 59.6528549, -82.9078064, 75.0748749
4: -16.5585098, 31.5066795, -34.5439682, 66.4336700, -82.9921799, 66.0506439

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A1_A1_B1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_A1_A1_B1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2713212, upper bound: 110.2920846
time: 0.69 seconds

## Relational analysis of NS_A1_A1_A1_B1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_A1_A1_B1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2713212, upper bound: 110.2920846
time: 1.03 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -17.8042068, 25.7366276, -39.6714973, 57.6982498, -75.5024490, 65.4081268
1: -14.7358599, 23.5445499, -31.8559265, 53.2457047, -67.9815598, 55.4004745
2: -12.3865213, 24.8870449, -26.9906673, 56.5492134, -68.9357376, 51.8777084
3: -20.0918941, 24.2903156, -43.5606842, 54.2363358, -74.3282166, 67.8509979
4: -14.4392452, 27.2208958, -31.1761322, 61.1561584, -75.5953979, 58.3970222

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A1_B1_A1

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2749909, upper bound: 110.2902427
time: 0.73 seconds

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A1_B1_A2

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2749909, upper bound: 110.2922220
time: 1.11 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -17.8042068, 25.7366276, -47.2119980, 68.2719955, -86.0762024, 72.9486008
1: -14.7358599, 23.5445499, -37.6303406, 62.9293442, -77.6652069, 61.1748886
2: -12.3865213, 24.8870449, -32.0000839, 66.6647949, -79.0513153, 56.8871269
3: -20.0918941, 24.2903156, -51.3128471, 64.2400665, -84.3319550, 75.6031647
4: -14.4392452, 27.2208958, -37.1154442, 72.0461273, -86.4853745, 64.3363419

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2749909, upper bound: 110.2902427
time: 1.01 seconds

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2749909, upper bound: 110.2922220
time: 0.68 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -20.4448090, 29.3368053, -44.8150711, 65.2394257, -85.6842346, 74.1518631
1: -16.8664875, 27.0127811, -35.8529053, 60.1202164, -76.9866867, 62.8656845
2: -14.2126341, 28.6432552, -30.4267750, 63.6847878, -77.8974228, 59.0700302
3: -23.3248024, 27.7187843, -48.7642784, 61.3322716, -84.6570663, 76.4830627
4: -16.4322357, 31.4498653, -35.2225876, 68.7372360, -85.1694412, 66.6724396

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2709650, upper bound: 110.2917181
time: 0.81 seconds

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2744382, upper bound: 110.2919662
time: 1.20 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -20.5661011, 29.5958290, -44.8150711, 65.2394257, -85.8055267, 74.4109039
1: -16.9293365, 27.1672230, -35.8529053, 60.1202164, -77.0495224, 63.0201263
2: -14.2718859, 28.7536163, -30.4267750, 63.6847878, -77.9566727, 59.1803894
3: -23.2549477, 27.9210701, -48.7642784, 61.3322716, -84.5872192, 76.6853409
4: -16.5585098, 31.5066795, -35.2225876, 68.7372360, -85.2957458, 66.7292557

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A2_A2_B1

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2744382, upper bound: 110.2924511
time: 1.10 seconds

## Relational analysis of NS_A1_A1_A1_B2_B2_B1_A2_A2_B2

### Relational analysis result of NS_A1_A1_A1_B2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2744382, upper bound: 110.2926265
time: 1.07 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -34.6021156, 50.4781685, -17.8042068, 25.7366276, -60.3387451, 68.2823792
1: -27.8080616, 46.5959473, -14.7358599, 23.5445499, -51.3526115, 61.3318062
2: -23.5447140, 49.3333931, -12.3865213, 24.8870449, -48.4317551, 61.7199135
3: -37.7599602, 47.6195831, -20.0918941, 24.2903156, -62.0502777, 67.7114792
4: -27.3529243, 53.2658768, -14.4392452, 27.2208958, -54.5738220, 67.7051239

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2924782, upper bound: 110.2714990
time: 0.60 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -34.8739777, 50.9410629, -17.8042068, 25.7366276, -60.6105995, 68.7452621
1: -28.0718346, 47.0488205, -14.7358599, 23.5445499, -51.6163826, 61.7846794
2: -23.7538452, 49.7960548, -12.3865213, 24.8870449, -48.6408920, 62.1825752
3: -38.0938187, 48.1170311, -20.0918941, 24.2903156, -62.3841286, 68.2089233
4: -27.6374874, 53.7609291, -14.4392452, 27.2208958, -54.8583832, 68.2001724

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2877017, upper bound: 110.2694359
time: 0.85 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B1_A1_A2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2877017, upper bound: 110.2697493
time: 1.03 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -35.8064842, 51.8580856, -17.8042068, 25.7366276, -61.5431023, 69.6622925
1: -28.7594032, 47.7943954, -14.7358599, 23.5445499, -52.3039513, 62.5302544
2: -24.3237953, 50.7885857, -12.3865213, 24.8870449, -49.2108383, 63.1751060
3: -39.3242073, 48.7321053, -20.0918941, 24.2903156, -63.6145134, 68.8239975
4: -28.1049366, 54.9842949, -14.4392452, 27.2208958, -55.3258324, 69.4235306

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_A2_B1_B1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -42.2494164, 61.2992172, -17.8042068, 25.7366276, -67.9860458, 79.1034241
1: -33.7633629, 56.5813026, -14.7358599, 23.5445499, -57.3079071, 71.3171616
2: -28.6620636, 60.1363792, -12.3865213, 24.8870449, -53.5491028, 72.5229034
3: -46.1217461, 57.6378708, -20.0918941, 24.2903156, -70.4120636, 77.7297668
4: -33.2315903, 64.9618378, -14.4392452, 27.2208958, -60.4524841, 79.4010849

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_A2_B1_B1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -35.1777611, 50.8454208, -36.6411972, 53.5077209, -88.6854706, 87.4866028
1: -28.2725925, 47.0015564, -29.4214668, 49.3871002, -77.6596909, 76.4230194
2: -23.8997879, 50.0527649, -24.9394436, 52.2345467, -76.1343384, 74.9922104
3: -38.9230156, 47.7884483, -39.8699837, 50.4863091, -89.4093246, 87.6584320
4: -27.5130901, 54.2688713, -29.0274220, 56.3519859, -83.8650513, 83.2962799

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A1_A1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2932674, upper bound: 110.2890821
time: 0.81 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A1_A1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2932674, upper bound: 110.2897041
time: 0.92 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -38.4492798, 55.8452606, -36.6411972, 53.5077209, -91.9570007, 92.4864502
1: -30.8718128, 51.5667152, -29.4214668, 49.3871002, -80.2589111, 80.9881821
2: -26.1397781, 54.8046417, -24.9394436, 52.2345467, -78.3743134, 79.7440872
3: -42.2790070, 52.5016708, -39.8699837, 50.4863091, -92.7653198, 92.3716583
4: -30.1650295, 59.2988129, -29.0274220, 56.3519859, -86.5170135, 88.3262329

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2933606, upper bound: 110.2978243
time: 0.90 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A1_A2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2933606, upper bound: 110.2984463
time: 0.64 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -32.5469093, 47.5639458, -36.6411972, 53.5077209, -86.0546188, 84.2051392
1: -26.2562714, 43.9286995, -29.4214668, 49.3871002, -75.6433716, 73.3501511
2: -22.1968575, 46.5146103, -24.9394436, 52.2345467, -74.4313965, 71.4540405
3: -35.6434555, 44.9257507, -39.8699837, 50.4863091, -86.1297607, 84.7957306
4: -25.8195095, 50.2267570, -29.0274220, 56.3519859, -82.1714935, 79.2541809

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A2_A1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -110.2917622, upper bound: 110.2917622
time: 0.70 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A2_A1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2917622, upper bound: 110.2923407
time: 0.98 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -36.4185638, 53.0245056, -36.6411972, 53.5077209, -89.9262772, 89.6657028
1: -29.2321167, 48.9271889, -29.4214668, 49.3871002, -78.6192017, 78.3486328
2: -24.7378082, 51.9307098, -24.9394436, 52.2345467, -76.9723511, 76.8701553
3: -39.8733788, 49.9112892, -39.8699837, 50.4863091, -90.3596878, 89.7812729
4: -28.6183739, 56.1097755, -29.0274220, 56.3519859, -84.9703522, 85.1371918

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A2_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2917622, upper bound: 110.2978200
time: 0.95 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A1_A2_A2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2917622, upper bound: 110.2984419
time: 1.06 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -39.4900780, 57.5602989, -34.6021156, 50.4781685, -89.9682388, 92.1624069
1: -31.6919422, 53.1494827, -27.8080616, 46.5959473, -78.2878723, 80.9575195
2: -26.8132915, 56.4491920, -23.5447140, 49.3333931, -76.1466827, 79.9939041
3: -43.3081322, 54.1574860, -37.7599602, 47.6195831, -90.9277191, 91.9174500
4: -30.9794979, 60.9937134, -27.3529243, 53.2658768, -84.2453766, 88.3466263

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

## BFS NS instance: NS_A1_A2_B1_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -39.4900780, 57.5602989, -34.8739777, 50.9410629, -90.4311218, 92.4342728
1: -31.6919422, 53.1494827, -28.0718346, 47.0488205, -78.7407608, 81.2212982
2: -26.8132915, 56.4491920, -23.7538452, 49.7960548, -76.6093445, 80.2030334
3: -43.3081322, 54.1574860, -38.0938187, 48.1170311, -91.4251633, 92.2512817
4: -30.9794979, 60.9937134, -27.6374874, 53.7609291, -84.7404251, 88.6311951

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_A2_B1_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -38.2779274, 55.9137802, -34.6021156, 50.4781685, -88.7560959, 90.5158691
1: -30.7478714, 51.6129456, -27.8080616, 46.5959473, -77.3438187, 79.4210052
2: -26.0317593, 54.6833229, -23.5447140, 49.3333931, -75.3651352, 78.2280350
3: -41.7778282, 52.7051926, -37.7599602, 47.6195831, -89.3974152, 90.4651489
4: -30.2181091, 59.0113335, -27.3529243, 53.2658768, -83.4839859, 86.3642578

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2923841, upper bound: 110.2939535
time: 0.76 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2923841, upper bound: 110.2972776
time: 1.26 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -38.2779274, 55.9137802, -34.8739777, 50.9410629, -89.2189941, 90.7877426
1: -30.7478714, 51.6129456, -28.0718346, 47.0488205, -77.7966919, 79.6847839
2: -26.0317593, 54.6833229, -23.7538452, 49.7960548, -75.8278122, 78.4371643
3: -41.7778282, 52.7051926, -38.0938187, 48.1170311, -89.8948593, 90.7989960
4: -30.2181091, 59.0113335, -27.6374874, 53.7609291, -83.9790192, 86.6488190

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2923841, upper bound: 110.2945320
time: 0.93 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2923841, upper bound: 110.2973840
time: 0.99 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -34.6021156, 50.4781685, -24.1472416, 34.9470558, -69.5491562, 74.6254120
1: -27.8080616, 46.5959473, -19.8914948, 32.2284966, -60.0365601, 66.4874420
2: -23.5447140, 49.3333931, -16.8092842, 34.1107864, -57.6554947, 66.1426773
3: -37.7599602, 47.6195831, -27.4348984, 33.0227242, -70.7826767, 75.0544815
4: -27.3529243, 53.2658768, -19.4530334, 37.3334503, -64.6863708, 72.7189102

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_A2_B1_B2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -34.8739777, 50.9410629, -24.1472416, 34.9470558, -69.8210220, 75.0883026
1: -28.0718346, 47.0488205, -19.8914948, 32.2284966, -60.3003273, 66.9403152
2: -23.7538452, 49.7960548, -16.8092842, 34.1107864, -57.8646278, 66.6053391
3: -38.0938187, 48.1170311, -27.4348984, 33.0227242, -71.1165085, 75.5519257
4: -27.6374874, 53.7609291, -19.4530334, 37.3334503, -64.9709396, 73.2139511

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_A2_B1_B2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -40.2609520, 58.5737915, -24.1472416, 34.9470558, -75.2079849, 82.7210312
1: -32.3201103, 54.0791168, -19.8914948, 32.2284966, -64.5486069, 73.9706039
2: -27.3807240, 57.4484787, -16.8092842, 34.1107864, -61.4914970, 74.2577438
3: -44.2037888, 55.0638237, -27.4348984, 33.0227242, -77.2265091, 82.4987183
4: -31.6159039, 62.1161880, -19.4530334, 37.3334503, -68.9493561, 81.5692139

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_A2_B1_B2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -40.2895660, 58.7787056, -24.1472416, 34.9470558, -75.2365952, 82.9259491
1: -32.3486443, 54.2659302, -19.8914948, 32.2284966, -64.5771408, 74.1574020
2: -27.3884315, 57.5934296, -16.8092842, 34.1107864, -61.4992104, 74.4027023
3: -44.1501656, 55.3143311, -27.4348984, 33.0227242, -77.1728745, 82.7492218
4: -31.6658173, 62.2130051, -19.4530334, 37.3334503, -68.9992676, 81.6660309

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_A2_B1_B2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -34.6021156, 50.4781685, -40.2609520, 58.5737915, -93.1758957, 90.7391052
1: -27.8080616, 46.5959473, -32.3201103, 54.0791168, -81.8871765, 78.9160538
2: -23.5447140, 49.3333931, -27.3807240, 57.4484787, -80.9931870, 76.7141190
3: -37.7599602, 47.6195831, -44.2037888, 55.0638237, -92.8237839, 91.8233719
4: -27.3529243, 53.2658768, -31.6159039, 62.1161880, -89.4691086, 84.8817825

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A1_A2_B1_B2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -40.2609520, 58.5737915, -40.2609520, 58.5737915, -98.8347321, 98.8347321
1: -32.3201103, 54.0791168, -32.3201103, 54.0791168, -86.3992233, 86.3992233
2: -27.3807240, 57.4484787, -27.3807240, 57.4484787, -84.8291931, 84.8291931
3: -44.2037888, 55.0638237, -44.2037888, 55.0638237, -99.2676086, 99.2676086
4: -31.6159039, 62.1161880, -31.6159039, 62.1161880, -93.7320862, 93.7320862

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A1_A2_B1_B2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -34.8652649, 50.9253311, -40.2609520, 58.5737915, -93.4390564, 91.1862717
1: -28.0652142, 47.0314369, -32.3201103, 54.0791168, -82.1443253, 79.3515320
2: -23.7472534, 49.7781372, -27.3807240, 57.4484787, -81.1957321, 77.1588593
3: -38.0845451, 48.1008568, -44.2037888, 55.0638237, -93.1483688, 92.3046417
4: -27.6302910, 53.7431412, -31.6159039, 62.1161880, -89.7464676, 85.3590469

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

## BFS NS instance: NS_A1_A2_B1_B2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -40.3779411, 58.8986969, -40.2609520, 58.5737915, -98.9517288, 99.1596451
1: -32.4151039, 54.3724442, -32.3201103, 54.0791168, -86.4942169, 86.6925507
2: -27.4445686, 57.7033768, -27.3807240, 57.4484787, -84.8930511, 85.0840988
3: -44.2361183, 55.4284019, -44.2037888, 55.0638237, -99.2999420, 99.6321869
4: -31.7356205, 62.3324547, -31.6159039, 62.1161880, -93.8518066, 93.9483490

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

## BFS NS instance: NS_A1_A2_B1_B2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -34.6021156, 50.4781685, -24.5585728, 35.6476822, -70.2498016, 75.0367432
1: -27.8080616, 46.5959473, -20.1751938, 32.7953796, -60.6034393, 66.7711334
2: -23.5447140, 49.3333931, -17.0628605, 34.6541252, -58.1988335, 66.3962555
3: -37.7599602, 47.6195831, -27.6626434, 33.6470490, -71.4069901, 75.2822266
4: -27.3529243, 53.2658768, -19.8094215, 37.8483315, -65.2012482, 73.0753021

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_A2_B1_B2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -34.8739777, 50.9410629, -24.5585728, 35.6476822, -70.5216599, 75.4996262
1: -28.0718346, 47.0488205, -20.1751938, 32.7953796, -60.8672104, 67.2240143
2: -23.7538452, 49.7960548, -17.0628605, 34.6541252, -58.4079704, 66.8589172
3: -38.0938187, 48.1170311, -27.6626434, 33.6470490, -71.7408218, 75.7796783
4: -27.6374874, 53.7609291, -19.8094215, 37.8483315, -65.4858093, 73.5703430

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_A2_B1_B2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -40.2609520, 58.5737915, -24.5585728, 35.6476822, -75.9086304, 83.1323624
1: -32.3201103, 54.0791168, -20.1751938, 32.7953796, -65.1154861, 74.2543030
2: -27.3807240, 57.4484787, -17.0628605, 34.6541252, -62.0348473, 74.5113373
3: -44.2037888, 55.0638237, -27.6626434, 33.6470490, -77.8508377, 82.7264709
4: -31.6159039, 62.1161880, -19.8094215, 37.8483315, -69.4642334, 81.9255981

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_A2_B1_B2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -40.2895660, 58.7787056, -24.5585728, 35.6476822, -75.9372482, 83.3372803
1: -32.3486443, 54.2659302, -20.1751938, 32.7953796, -65.1440277, 74.4411011
2: -27.3884315, 57.5934296, -17.0628605, 34.6541252, -62.0425529, 74.6562881
3: -44.1501656, 55.3143311, -27.6626434, 33.6470490, -77.7971878, 82.9769745
4: -31.6658173, 62.2130051, -19.8094215, 37.8483315, -69.5141449, 82.0224152

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 0

## BFS NS instance: NS_A1_A2_B1_B2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -39.3334885, 57.1704597, -40.2895660, 58.7787056, -98.1121979, 97.4599915
1: -31.5721359, 52.8182869, -32.3486443, 54.2659302, -85.8380432, 85.1669312
2: -26.7320309, 56.1400452, -27.3884315, 57.5934296, -84.3254623, 83.5284729
3: -43.2408867, 53.7558174, -44.1501656, 55.3143311, -98.5551910, 97.9059830
4: -30.8398476, 60.7224808, -31.6658173, 62.2130051, -93.0528488, 92.3882980

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A1_A2_B1_B2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2978410, upper bound: 110.2976478
time: 0.84 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A1_A1_B2

### Relational analysis result of NS_A1_A2_B1_B2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2978410, upper bound: 110.2980630
time: 0.83 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -39.4026718, 57.4415474, -40.2895660, 58.7787056, -98.1813812, 97.7310715
1: -31.6260357, 53.0438347, -32.3486443, 54.2659302, -85.8919373, 85.3924789
2: -26.7575607, 56.3398781, -27.3884315, 57.5934296, -84.3509827, 83.7283020
3: -43.2231560, 54.0440063, -44.1501656, 55.3143311, -98.5374908, 98.1941681
4: -30.9101753, 60.8750534, -31.6658173, 62.2130051, -93.1231842, 92.5408707

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1_B2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2978410, upper bound: 110.2976478
time: 1.02 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_A2_B1_B2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -37.7216606, 54.9729843, -40.2895660, 58.7787056, -96.5003662, 95.2625198
1: -30.2750683, 50.7156181, -32.3486443, 54.2659302, -84.5409851, 83.0642624
2: -25.6327324, 53.8032494, -27.3884315, 57.5934296, -83.2261581, 81.1916809
3: -41.2501068, 51.7479668, -44.1501656, 55.3143311, -96.5644379, 95.8981247
4: -29.6809120, 58.1034470, -31.6658173, 62.2130051, -91.8939209, 89.7692642

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A2_A1_A1

### Relational analysis result of NS_A1_A2_B1_B2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2917622, upper bound: 110.2924764
time: 1.16 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A2_A1_A2

### Relational analysis result of NS_A1_A2_B1_B2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2917622, upper bound: 110.2982361
time: 0.92 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -38.2779274, 55.9137802, -40.2895660, 58.7787056, -97.0566330, 96.2033081
1: -30.7478714, 51.6129456, -32.3486443, 54.2659302, -85.0137863, 83.9615936
2: -26.0317593, 54.6833229, -27.3884315, 57.5934296, -83.6251755, 82.0717545
3: -41.7778282, 52.7051926, -44.1501656, 55.3143311, -97.0921478, 96.8553619
4: -30.2181091, 59.0113335, -31.6658173, 62.2130051, -92.4311066, 90.6771545

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A2_A2_A1

### Relational analysis result of NS_A1_A2_B1_B2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2917622, upper bound: 110.2946677
time: 1.05 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2_B2_A2_A2_A2

### Relational analysis result of NS_A1_A2_B1_B2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -110.2917622, upper bound: 110.2982361
time: 1.14 seconds

## BFS NS instance: NS_A1_A2_B2_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -35.9560547, 52.0968666, -46.8781281, 67.6992264, -103.6552811, 98.9749908
1: -28.8900509, 47.9917946, -37.3850899, 62.4281769, -91.3182220, 85.3768768
2: -24.4437256, 50.9698906, -31.7776775, 66.1662445, -90.6099701, 82.7475662
3: -39.4713860, 48.9712791, -51.0553131, 63.7152748, -103.1866608, 100.0265884
4: -28.2736664, 55.1733856, -36.8398056, 71.5378647, -99.8115234, 92.0131836

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A1_A2_B2_A1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -35.9560547, 52.0968666, -47.5451851, 68.7545547, -104.7106094, 99.6420441
1: -28.8900509, 47.9917946, -37.9440422, 63.3302460, -92.2202911, 85.9358368
2: -24.4437256, 50.9698906, -32.2653580, 67.1647644, -91.6084900, 83.2352448
3: -39.4713860, 48.9712791, -51.7972107, 64.6090240, -104.0804138, 100.7684937
4: -28.2736664, 55.1733856, -37.4488029, 72.5966644, -100.8703156, 92.6221695

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 38

## BFS NS instance: NS_A1_A2_B2_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -35.9560547, 52.0968666, -50.4869919, 73.1812134, -109.1372681, 102.5838623
1: -28.8900509, 47.9917946, -40.2471542, 67.4857788, -96.3758163, 88.2389450
2: -24.4437256, 50.9698906, -34.2501564, 71.3996735, -95.8433990, 85.2200470
3: -39.4713860, 48.9712791, -54.7409172, 68.9038696, -108.3752594, 103.7121964
4: -28.2736664, 55.1733856, -39.7669754, 77.0777740, -105.3514328, 94.9403305

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A1_A2_B2_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -35.9560547, 52.0968666, -51.6174545, 74.9559326, -110.9119873, 103.7143173
1: -28.8900509, 47.9917946, -41.1845093, 69.1003036, -97.9903488, 89.1763000
2: -24.4437256, 50.9698906, -35.0547943, 73.1368637, -97.5805893, 86.0246811
3: -39.4713860, 48.9712791, -55.9992218, 70.5187149, -109.9900970, 104.9704895
4: -28.2736664, 55.1733856, -40.7542191, 78.9219513, -107.1956100, 95.9276047

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 38

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.70 + 310.58 = 314.28 seconds
