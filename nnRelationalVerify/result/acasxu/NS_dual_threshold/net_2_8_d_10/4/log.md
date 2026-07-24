## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 53.8414279856


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954)
1: (-20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342)
2: (-34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566)
3: (-40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698)
4: (-30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.77 + 1.56 = 4.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -54.3304016, upper bound: 54.3304016

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.8257853, upper bound: 54.1946070
time: 0.49 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8168306, upper bound: 53.8168306
time: 0.52 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.24 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 4, lower bound: -53.8257853, upper bound: 54.1946070
NS_A2, status: Status.VERIFIED, split count: 1, time: 1.24
Output dim: 4, lower bound: -53.8168306, upper bound: 53.8168306

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -204.7570190, 206.6748505, -208.9272308, 212.0743713, -416.8313904, 415.6020203
1: -20.3960381, 15.5670252, -20.8862324, 15.8918066, -36.2878418, 36.4532509
2: -33.9861832, 38.0697670, -34.6868706, 39.0282860, -73.0144653, 72.7566376
3: -39.5753365, 25.5940475, -40.3503723, 26.2559071, -65.8312454, 65.9444199
4: -29.7705994, 31.4544239, -30.3582592, 32.2099609, -61.9805603, 61.8126793

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.8257853, upper bound: 54.1946070
time: 0.47 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.8227465, upper bound: 54.1000779
time: 0.51 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.26 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.26
Output dim: 4, lower bound: -53.8257853, upper bound: 54.1946070
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.26
Output dim: 4, lower bound: -53.8227465, upper bound: 54.1000779

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -201.1277924, 202.2205658, -208.9272308, 212.0743713, -413.2021484, 411.1477661
1: -20.0000896, 15.2751503, -20.8862324, 15.8918066, -35.8918953, 36.1613846
2: -33.3960190, 37.2743912, -34.6868706, 39.0282860, -72.4243011, 71.9612579
3: -38.9535866, 25.0402565, -40.3503723, 26.2559071, -65.2094879, 65.3906250
4: -29.2669163, 30.8637314, -30.3582592, 32.2099609, -61.4768677, 61.2219925

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6112085, upper bound: 54.0910418
time: 0.48 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6112085, upper bound: 54.1000779
time: 0.55 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -229.8283234, 239.6914062, -206.8325500, 209.2644043, -439.0927124, 446.5239563
1: -23.1948261, 17.4775448, -20.6432190, 15.7288771, -38.9237022, 38.1207657
2: -37.8304062, 43.6338005, -34.3319397, 38.5532150, -76.3835907, 77.9657440
3: -43.0711975, 29.7514858, -39.9569626, 25.9197121, -68.9909058, 69.7084503
4: -32.7904396, 35.6039162, -30.0564499, 31.8423252, -64.6327667, 65.6603622

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_A1

### Relational analysis result of NS_A1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8128563, upper bound: 53.7879211
time: 0.56 seconds

## Relational analysis of NS_A1_A2_A2

### Relational analysis result of NS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.8189465, upper bound: 54.0962175
time: 0.54 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.88 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.88
Output dim: 4, lower bound: -53.6112085, upper bound: 54.0910418
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.88
Output dim: 4, lower bound: -53.6112085, upper bound: 54.1000779
NS_A1_A2_A1, status: Status.VERIFIED, split count: 3, time: 3.88
Output dim: 4, lower bound: -53.8128563, upper bound: 53.7879211
NS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.88
Output dim: 4, lower bound: -53.8189465, upper bound: 54.0962175

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -201.1277924, 202.2205658, -205.1965942, 207.5162201, -408.6439819, 407.4171753
1: -20.0000896, 15.2751503, -20.4815578, 15.5937052, -35.5937958, 35.7567062
2: -33.3960190, 37.2743912, -34.0846062, 38.2102547, -71.6062775, 71.3589935
3: -38.9535866, 25.0402565, -39.7204514, 25.6877327, -64.6413193, 64.7606888
4: -29.2669163, 30.8637314, -29.8492126, 31.5948086, -60.8617210, 60.7129440

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6135298, upper bound: 54.1624420
time: 0.53 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6135298, upper bound: 54.1624420
time: 0.51 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -201.1277924, 202.2205658, -235.1039276, 246.2913666, -447.4191284, 437.3244934
1: -20.0000896, 15.2751503, -23.7828350, 17.8729362, -37.8730202, 39.0579834
2: -33.3960190, 37.2743912, -38.6518555, 44.8501358, -78.2461548, 75.9262390
3: -38.9535866, 25.0402565, -43.9290009, 30.5741119, -69.5276947, 68.9692535
4: -29.2669163, 30.8637314, -33.4317856, 36.6602097, -65.9271240, 64.2955170

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6135298, upper bound: 54.1946070
time: 0.55 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6135298, upper bound: 54.1946070
time: 0.52 seconds

## BFS NS instance: NS_A1_A2_A2

### Backsubstitution after applying NS history:
0: -228.5599213, 238.0851898, -206.8325500, 209.2644043, -437.8243408, 444.9177246
1: -23.0520573, 17.3757534, -20.6432190, 15.7288771, -38.7809334, 38.0189743
2: -37.6229324, 43.3493080, -34.3319397, 38.5532150, -76.1761322, 77.6812439
3: -42.8509712, 29.5538826, -39.9569626, 25.9197121, -68.7706833, 69.5108414
4: -32.6137810, 35.3890038, -30.0564499, 31.8423252, -64.4561081, 65.4454498

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_A2_B1

### Relational analysis result of NS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6107314, upper bound: 54.0874685
time: 0.47 seconds

## Relational analysis of NS_A1_A2_A2_B2

### Relational analysis result of NS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6107314, upper bound: 54.0874685
time: 0.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.87 seconds
NS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 4, lower bound: -53.6135298, upper bound: 54.1624420
NS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 4, lower bound: -53.6135298, upper bound: 54.1624420
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 4, lower bound: -53.6135298, upper bound: 54.1946070
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 4, lower bound: -53.6135298, upper bound: 54.1946070
NS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 4, lower bound: -53.6107314, upper bound: 54.0874685
NS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.87
Output dim: 4, lower bound: -53.6107314, upper bound: 54.0874685

## BFS NS instance: NS_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -201.1277924, 202.2205658, -201.1277924, 202.2205658, -403.3483582, 403.3483582
1: -20.0000896, 15.2751503, -20.0000896, 15.2751503, -35.2752380, 35.2752380
2: -33.3960190, 37.2743912, -33.3960190, 37.2743912, -70.6704102, 70.6704102
3: -38.9535866, 25.0402565, -38.9535866, 25.0402565, -63.9938316, 63.9938431
4: -29.2669163, 30.8637314, -29.2669163, 30.8637314, -60.1306458, 60.1306458

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -201.1277924, 202.2205658, -145.7483215, 121.6899109, -322.8176575, 347.9688110
1: -20.0000896, 15.2751503, -12.8442450, 10.9235306, -30.9236145, 28.1193962
2: -33.3960190, 37.2743912, -24.4200821, 23.4116135, -56.8076324, 61.6944618
3: -38.9535866, 25.0402565, -29.6215401, 15.4392090, -54.3927956, 54.6617889
4: -29.2669163, 30.8637314, -21.9733162, 19.4988098, -48.7657127, 52.8370438

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B2_A1

### Relational analysis result of NS_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.5369750, upper bound: 54.1370882
time: 0.47 seconds

## Relational analysis of NS_A1_A1_B1_B2_A2

### Relational analysis result of NS_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.5345565, upper bound: 54.0567534
time: 0.54 seconds

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -201.1277924, 202.2205658, -230.9022980, 240.7675323, -441.8953247, 433.1228638
1: -20.0000896, 15.2751503, -23.2837906, 17.5435219, -37.5436096, 38.5589409
2: -33.3960190, 37.2743912, -37.9511795, 43.8691788, -77.2651978, 75.2255630
3: -38.9535866, 25.0402565, -43.1572227, 29.9001389, -68.8537292, 68.1974716
4: -29.2669163, 30.8637314, -32.8406258, 35.8758469, -65.1427536, 63.7043571

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.8257853, upper bound: 54.1946070
time: 0.54 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.8235779, upper bound: 54.1158953
time: 0.55 seconds

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -201.1277924, 202.2205658, -160.1356201, 140.8509674, -341.9787598, 362.3562012
1: -20.0000896, 15.2751503, -14.4603577, 12.1470957, -32.1471863, 29.7355080
2: -33.3960190, 37.2743912, -26.3998222, 26.6745338, -60.0705414, 63.6742058
3: -38.9535866, 25.0402565, -31.1123238, 17.8478298, -56.8014069, 56.1525650
4: -29.2669163, 30.8637314, -23.4536324, 22.0673199, -51.3342361, 54.3173637

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.8048732, upper bound: 54.1930342
time: 0.51 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6984828, upper bound: 54.1848772
time: 0.55 seconds

## BFS NS instance: NS_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -228.5599213, 238.0851898, -205.1965942, 207.5162201, -436.0761414, 443.2817993
1: -23.0520573, 17.3757534, -20.4815578, 15.5937052, -38.6457634, 37.8573074
2: -37.6229324, 43.3493080, -34.0846062, 38.2102547, -75.8331909, 77.4339142
3: -42.8509712, 29.5538826, -39.7204514, 25.6877327, -68.5386887, 69.2743149
4: -32.6137810, 35.3890038, -29.8492126, 31.5948086, -64.2085876, 65.2382050

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_A2_B1_B1

### Relational analysis result of NS_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6107314, upper bound: 54.0874685
time: 0.54 seconds

## Relational analysis of NS_A1_A2_A2_B1_B2

### Relational analysis result of NS_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6107314, upper bound: 54.0874685
time: 0.54 seconds

## BFS NS instance: NS_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -228.5599213, 238.0851898, -235.1039276, 246.2913666, -474.8512878, 473.1890869
1: -23.0520573, 17.3757534, -23.7828350, 17.8729362, -40.9249954, 41.1585770
2: -37.6229324, 43.3493080, -38.6518555, 44.8501358, -82.4730682, 82.0011597
3: -42.8509712, 29.5538826, -43.9290009, 30.5741119, -73.4250793, 73.4828720
4: -32.6137810, 35.3890038, -33.4317856, 36.6602097, -69.2739868, 68.8207855

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_A2_B2_B1

### Relational analysis result of NS_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6046183, upper bound: 54.0874685
time: 0.59 seconds

## Relational analysis of NS_A1_A2_A2_B2_B2

### Relational analysis result of NS_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6046183, upper bound: 54.0874685
time: 0.59 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.61 seconds
NS_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.61
Output dim: 4, lower bound: -53.5369750, upper bound: 54.1370882
NS_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.61
Output dim: 4, lower bound: -53.5345565, upper bound: 54.0567534
NS_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.61
Output dim: 4, lower bound: -53.8257853, upper bound: 54.1946070
NS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.61
Output dim: 4, lower bound: -53.8235779, upper bound: 54.1158953
NS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.61
Output dim: 4, lower bound: -53.8048732, upper bound: 54.1930342
NS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.61
Output dim: 4, lower bound: -53.6984828, upper bound: 54.1848772
NS_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.61
Output dim: 4, lower bound: -53.6107314, upper bound: 54.0874685
NS_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.61
Output dim: 4, lower bound: -53.6107314, upper bound: 54.0874685
NS_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.61
Output dim: 4, lower bound: -53.6046183, upper bound: 54.0874685
NS_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.61
Output dim: 4, lower bound: -53.6046183, upper bound: 54.0874685

## BFS NS instance: NS_A1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -188.5525970, 187.3345642, -145.7483215, 121.6899109, -310.2424622, 333.0827637
1: -18.6621628, 14.2894564, -12.8442450, 10.9235306, -29.5856934, 27.1337013
2: -31.3127098, 34.6222191, -24.4200821, 23.4116135, -54.7243233, 59.0423012
3: -36.7261963, 23.2074509, -29.6215401, 15.4392090, -52.1654053, 52.8289833
4: -27.5246277, 28.8546772, -21.9733162, 19.4988098, -47.0234299, 50.8279953

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -168.8072662, 165.4194641, -140.2997284, 115.5167770, -284.3240356, 305.7191162
1: -16.3793564, 12.8702412, -12.2533693, 10.4889851, -26.8683414, 25.1236115
2: -27.7593670, 30.4771137, -23.5046730, 22.2810936, -50.0404587, 53.9817734
3: -32.0839424, 20.4941921, -28.5613327, 14.6548624, -46.7388039, 49.0555267
4: -24.4444008, 24.5533714, -21.1733627, 18.5067101, -42.9511108, 45.7267342

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.5108019, upper bound: 54.0300584
time: 0.53 seconds

## Relational analysis of NS_A1_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.5093732, upper bound: 53.9397993
time: 0.52 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -182.1478119, 179.0897827, -230.9022980, 240.7675323, -422.9153442, 409.9920654
1: -17.9017239, 13.7745075, -23.2837906, 17.5435219, -35.4452438, 37.0582962
2: -30.2998810, 33.1248741, -37.9511795, 43.8691788, -74.1690521, 71.0760498
3: -35.5966110, 22.1574287, -43.1572227, 29.9001389, -65.4967422, 65.3146439
4: -26.6115284, 27.6934547, -32.8406258, 35.8758469, -62.4873657, 60.5340805

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0104986, upper bound: 54.1217562
time: 0.52 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0104986, upper bound: 54.1217680
time: 0.49 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -571.6734009, 684.3543701, -228.9132996, 238.3629761, -810.0363770, 913.2677002
1: -63.7892113, 43.8944435, -23.0705471, 17.3859253, -81.1751404, 66.9649887
2: -94.5700760, 121.6911087, -37.6295395, 43.4351730, -138.0052185, 159.3206482
3: -103.8410187, 84.0843735, -42.8105125, 29.5987778, -133.4397888, 126.8948669
4: -79.7502213, 98.0364609, -32.5637779, 35.5491486, -115.2993698, 130.6002197

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0104986, upper bound: 54.1217562
time: 0.56 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0104986, upper bound: 54.1217680
time: 0.51 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -187.0753479, 183.1617279, -114.2380371, 84.3780136, -271.4533691, 297.3997192
1: -18.3204231, 14.1766882, -9.3639755, 8.2793512, -26.5997734, 23.5406647
2: -31.0967579, 33.9892426, -19.0470047, 16.8700485, -47.9668045, 53.0362473
3: -36.5664940, 22.7303982, -23.4604416, 10.9370203, -47.5035133, 46.1908340
4: -27.3840771, 28.3140430, -17.1426678, 14.2672577, -41.6513290, 45.4567032

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.7701046, upper bound: 54.1858179
time: 0.55 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.7646860, upper bound: 53.9472422
time: 0.50 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -201.1277924, 202.2205658, -156.0186157, 135.3240356, -336.4518127, 358.2391968
1: -20.0000896, 15.2751503, -13.9542274, 11.8206205, -31.8207092, 29.2293777
2: -33.3960190, 37.2743912, -25.7565575, 25.7007389, -59.0967560, 63.0309372
3: -38.9535866, 25.0402565, -30.4332581, 17.1891632, -56.1427460, 55.4735146
4: -29.2669163, 30.8637314, -22.9141693, 21.2864532, -50.5533676, 53.7778969

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6494036, upper bound: 54.1632102
time: 0.56 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.6441196, upper bound: 53.9438304
time: 0.56 seconds

## BFS NS instance: NS_A1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -228.5599213, 238.0851898, -201.1277924, 202.2205658, -430.7804871, 439.2129822
1: -23.0520573, 17.3757534, -20.0000896, 15.2751503, -38.3272095, 37.3758392
2: -37.6229324, 43.3493080, -33.3960190, 37.2743912, -74.8973160, 76.7453308
3: -42.8509712, 29.5538826, -38.9535866, 25.0402565, -67.8912201, 68.5074692
4: -32.6137810, 35.3890038, -29.2669163, 30.8637314, -63.4775124, 64.6559143

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -228.5599213, 238.0851898, -145.7483215, 121.6899109, -350.2498169, 383.8334351
1: -23.0520573, 17.3757534, -12.8442450, 10.9235306, -33.9755859, 30.2199974
2: -37.6229324, 43.3493080, -24.4200821, 23.4116135, -61.0345345, 67.7693939
3: -42.8509712, 29.5538826, -29.6215401, 15.4392090, -58.2901726, 59.1754227
4: -32.6137810, 35.3890038, -21.9733162, 19.4988098, -52.1125755, 57.3623161

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

## BFS NS instance: NS_A1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -228.5599213, 238.0851898, -230.9022980, 240.7675323, -469.3274536, 468.9874878
1: -23.0520573, 17.3757534, -23.2837906, 17.5435219, -40.5955811, 40.6595383
2: -37.6229324, 43.3493080, -37.9511795, 43.8691788, -81.4921112, 81.3004913
3: -42.8509712, 29.5538826, -43.1572227, 29.9001389, -72.7511139, 72.7110901
4: -32.6137810, 35.3890038, -32.8406258, 35.8758469, -68.4896164, 68.2296295

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -228.5599213, 238.0851898, -160.1356201, 140.8509674, -369.4108887, 398.2207947
1: -23.0520573, 17.3757534, -14.4603577, 12.1470957, -35.1991539, 31.8361092
2: -37.6229324, 43.3493080, -26.3998222, 26.6745338, -64.2974548, 69.7491302
3: -42.8509712, 29.5538826, -31.1123238, 17.8478298, -60.6987877, 60.6662064
4: -32.6137810, 35.3890038, -23.4536324, 22.0673199, -54.6810989, 58.8426361

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.33 + 96.99 = 101.32 seconds
