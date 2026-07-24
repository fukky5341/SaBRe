## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 655.396249165482


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729)
1: (-107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362)
2: (-66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734)
3: (-134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476)
4: (-73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.13 + 2.06 = 4.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -655.4355753, upper bound: 655.4355753

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4351509, upper bound: 655.4355341
time: 0.72 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4355717, upper bound: 655.4355717
time: 0.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.63 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 0, lower bound: -655.4351509, upper bound: 655.4355341
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 0, lower bound: -655.4355717, upper bound: 655.4355717

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -303.7320862, 607.9927368, -304.5673828, 610.6347656, -914.3668213, 912.5601196
1: -106.8568878, 219.1014099, -107.1243286, 219.6566925, -326.5135193, 326.2257385
2: -66.2196808, 224.8518066, -66.3665009, 225.4796295, -291.6992188, 291.2182922
3: -133.7228088, 263.3948975, -134.0784149, 264.2018738, -397.9245911, 397.4732971
4: -72.8595505, 220.4355011, -73.0650711, 221.0344849, -293.8939819, 293.5005798

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343831, upper bound: 655.4340708
time: 0.70 seconds

## Relational analysis of NS_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4346031, upper bound: 655.4345667
time: 0.71 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4340295, upper bound: 655.4347700
time: 0.79 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -305.5349121, 611.6073608, -303.2554016, 607.1478882, -912.6828003, 914.8627930
1: -107.4886780, 220.3997650, -106.6945190, 218.7716675, -326.2602539, 327.0942688
2: -66.6060410, 226.1742706, -66.1188812, 224.5136871, -291.1197205, 292.2931213
3: -134.5018005, 264.9532471, -133.5181427, 263.0096741, -397.5114746, 398.4713745
4: -73.2891235, 221.7262726, -72.7492981, 220.1071320, -293.3962402, 294.4755859

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_B1

### Relational analysis result of NS_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4346161, upper bound: 655.4346429
time: 0.82 seconds

## Relational analysis of NS_B2_B2

### Relational analysis result of NS_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4345995, upper bound: 655.4345995
time: 0.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.74 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 3.74
Output dim: 0, lower bound: -655.4346031, upper bound: 655.4345667
NS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 3.74
Output dim: 0, lower bound: -655.4340295, upper bound: 655.4347700
NS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 3.74
Output dim: 0, lower bound: -655.4346161, upper bound: 655.4346429
NS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 3.74
Output dim: 0, lower bound: -655.4345995, upper bound: 655.4345995

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: -297.2562561, 595.1660767, -285.4097595, 572.6933594, -869.9495239, 880.5758057
1: -104.6777496, 214.7907410, -100.6768265, 206.9187622, -311.5964355, 315.4675598
2: -64.8571091, 220.3884125, -62.3336220, 212.2898407, -277.1469421, 282.7220459
3: -130.9979858, 258.1552734, -126.0137863, 248.7289429, -379.7269287, 384.1690369
4: -71.3542099, 215.9783325, -68.6126862, 207.8574677, -279.2116394, 284.5909729

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_B1_A1

### Relational analysis result of NS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4338778, upper bound: 655.4345668
time: 0.74 seconds

## Relational analysis of NS_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4338778, upper bound: 655.4345668
time: 0.78 seconds

## BFS NS instance: NS_B1_B2

### Backsubstitution after applying NS history:
0: -303.0835571, 606.6955566, -365.2921143, 739.5451660, -1042.6286621, 971.9876709
1: -106.6318512, 218.6608124, -129.9187622, 266.6718140, -373.3036194, 348.5794678
2: -66.0823288, 224.3912811, -80.5317078, 273.9836121, -339.8304138, 304.9229431
3: -133.4481354, 262.8658447, -161.7942505, 320.9751892, -454.1537170, 424.6600342
4: -72.7072830, 219.9798889, -88.7595062, 267.6143494, -340.1715088, 308.7393799

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_A1

### Relational analysis result of NS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4331061, upper bound: 655.4347532
time: 0.77 seconds

## Relational analysis of NS_B1_B2_A2

### Relational analysis result of NS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333243, upper bound: 655.4342583
time: 0.75 seconds

## BFS NS instance: NS_B2_B1

### Backsubstitution after applying NS history:
0: -289.3557739, 579.2822266, -279.8775330, 562.6428223, -851.9985352, 859.1597900
1: -101.8772202, 208.9060516, -98.7732239, 202.7090912, -304.5863037, 307.6791687
2: -63.1496277, 214.4232635, -61.1096191, 208.2153778, -271.3649902, 275.5328674
3: -127.4989700, 251.0850677, -123.4561081, 244.1199799, -371.6189575, 374.5411072
4: -69.4420090, 210.2756958, -67.3258362, 204.0345764, -273.4765930, 277.6015015

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_B1_A1

### Relational analysis result of NS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4345774, upper bound: 655.4345774
time: 0.78 seconds

## Relational analysis of NS_B2_B1_A2

### Relational analysis result of NS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4345774, upper bound: 655.4345974
time: 0.82 seconds

## BFS NS instance: NS_B2_B2

### Backsubstitution after applying NS history:
0: -304.4783630, 609.4874268, -294.6537170, 589.8226318, -894.3009644, 904.1411133
1: -107.1167450, 219.6549988, -103.6562271, 212.6954803, -319.8121643, 323.3112183
2: -66.3787079, 225.4057770, -64.2663574, 218.2355347, -284.6141968, 289.6721191
3: -134.0392609, 264.0542297, -129.7195282, 255.6669159, -389.7061768, 393.7737122
4: -73.0353470, 220.9709930, -70.6764679, 213.9241943, -286.9595337, 291.6474304

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_B2_A1

### Relational analysis result of NS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4345974, upper bound: 655.4345799
time: 0.83 seconds

## Relational analysis of NS_B2_B2_A2

### Relational analysis result of NS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4345974, upper bound: 655.4345995
time: 0.89 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.86 seconds
NS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 0, lower bound: -655.4338778, upper bound: 655.4345668
NS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 0, lower bound: -655.4338778, upper bound: 655.4345668
NS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 0, lower bound: -655.4331061, upper bound: 655.4347532
NS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 0, lower bound: -655.4333243, upper bound: 655.4342583
NS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 0, lower bound: -655.4345774, upper bound: 655.4345774
NS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 0, lower bound: -655.4345774, upper bound: 655.4345974
NS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 0, lower bound: -655.4345974, upper bound: 655.4345799
NS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 0, lower bound: -655.4345974, upper bound: 655.4345995

## BFS NS instance: NS_B1_B1_A1

### Backsubstitution after applying NS history:
0: -284.4118652, 569.6688232, -285.4097595, 572.6933594, -857.1050415, 855.0786133
1: -100.3417130, 206.2233734, -100.6768265, 206.9187622, -307.2604675, 306.9002075
2: -62.1472015, 211.5144958, -62.3336220, 212.2898407, -274.4370422, 273.8481140
3: -125.5792084, 247.7417755, -126.0137863, 248.7289429, -374.3081055, 373.7555542
4: -68.3601074, 207.1183472, -68.6126862, 207.8574677, -276.2175293, 275.7310181

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_B1_A1_B1

### Relational analysis result of NS_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4332024, upper bound: 655.4337833
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A1_B2

### Relational analysis result of NS_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336519, upper bound: 655.4341964
time: 0.68 seconds

## BFS NS instance: NS_B1_B1_A2

### Backsubstitution after applying NS history:
0: -368.6683960, 745.2770996, -285.4097595, 572.6933594, -941.3617554, 1030.6867676
1: -131.1792755, 269.1217346, -100.6768265, 206.9187622, -338.0980225, 369.7985535
2: -81.2746506, 276.4298706, -62.3336220, 212.2898407, -293.5644836, 338.5209656
3: -163.2350769, 323.7264709, -126.0137863, 248.7289429, -411.9639893, 449.4883423
4: -89.5345154, 270.0223389, -68.6126862, 207.8574677, -297.3919373, 338.4839783

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_A2_B1

### Relational analysis result of NS_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343918, upper bound: 655.4345380
time: 0.78 seconds

## Relational analysis of NS_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_B1_A2_B1

### Relational analysis result of NS_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4332541, upper bound: 655.4338742
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A2_B2

### Relational analysis result of NS_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4345953, upper bound: 655.4345642
time: 0.66 seconds

## BFS NS instance: NS_B1_B2_A1

### Backsubstitution after applying NS history:
0: -270.2115173, 541.6589355, -362.1427612, 733.4139404, -1003.6254272, 903.8016357
1: -95.4805145, 196.3548126, -128.8690796, 264.5390625, -360.0195923, 325.2238770
2: -59.1623268, 201.3903961, -79.8721771, 271.7865906, -330.7025757, 281.2625732
3: -119.4600983, 235.8445587, -160.4729919, 318.3883972, -437.5698242, 396.3175049
4: -65.0648193, 197.0860291, -88.0344162, 265.4254150, -330.3365784, 285.1204224

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B2_A1_B1

### Relational analysis result of NS_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4327840, upper bound: 655.4328925
time: 0.76 seconds

## Relational analysis of NS_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_A1_B1

### Relational analysis result of NS_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4331061, upper bound: 655.4340119
time: 0.76 seconds

## Relational analysis of NS_B1_B2_A1_B2

### Relational analysis result of NS_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4331061, upper bound: 655.4342583
time: 0.71 seconds

## BFS NS instance: NS_B1_B2_A2

### Backsubstitution after applying NS history:
0: -506.7702942, 1024.2611084, -358.1487122, 725.7060547, -1232.4763184, 1378.9379883
1: -179.8924713, 365.2241516, -127.4161072, 261.5541382, -441.1301880, 492.3838501
2: -111.6563873, 375.7283020, -78.9984436, 268.7794189, -380.0345459, 453.5758667
3: -222.5174713, 440.9206238, -158.6037292, 314.8477478, -537.0664062, 598.4150391
4: -123.4393768, 367.3726807, -87.0474243, 262.5301208, -385.6633911, 453.6320496

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B2_A2_B1

### Relational analysis result of NS_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333243, upper bound: 655.4340119
time: 0.72 seconds

## Relational analysis of NS_B1_B2_A2_B2

### Relational analysis result of NS_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333243, upper bound: 655.4342583
time: 0.70 seconds

## BFS NS instance: NS_B2_B1_A1

### Backsubstitution after applying NS history:
0: -281.3200073, 565.4867554, -279.8775330, 562.6428223, -843.9628296, 845.3642578
1: -99.2849655, 203.7628479, -98.7732239, 202.7090912, -301.9940491, 302.5360107
2: -61.4176140, 209.2827911, -61.1096191, 208.2153778, -269.6329651, 270.3923950
3: -124.0811539, 245.3663025, -123.4561081, 244.1199799, -368.2011414, 368.8224182
4: -67.6681137, 205.0659180, -67.3258362, 204.0345764, -271.7026367, 272.3917236

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_B1_A1_A1

### Relational analysis result of NS_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334271, upper bound: 655.4343293
time: 0.78 seconds

## Relational analysis of NS_B2_B1_A1_A2

### Relational analysis result of NS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334271, upper bound: 655.4335257
time: 0.85 seconds

## BFS NS instance: NS_B2_B1_A2

### Backsubstitution after applying NS history:
0: -296.6877136, 593.7878418, -279.8775330, 562.6428223, -859.3305664, 873.6654053
1: -104.3657379, 214.1592712, -98.7732239, 202.7090912, -307.0748291, 312.9324036
2: -64.7006531, 219.7236328, -61.1096191, 208.2153778, -272.9160156, 280.8332520
3: -130.6006317, 257.4122925, -123.4561081, 244.1199799, -374.7206116, 380.8684082
4: -71.1577072, 215.3730621, -67.3258362, 204.0345764, -275.1922607, 282.6989136

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_B1_A2_B1

### Relational analysis result of NS_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343214, upper bound: 655.4335553
time: 1.02 seconds

## Relational analysis of NS_B2_B1_A2_B2

### Relational analysis result of NS_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334828, upper bound: 655.4335257
time: 0.99 seconds

## BFS NS instance: NS_B2_B2_A1

### Backsubstitution after applying NS history:
0: -281.3200073, 565.4867554, -294.6537170, 589.8226318, -871.1426392, 860.1405029
1: -99.2849655, 203.7628479, -103.6562271, 212.6954803, -311.9803772, 307.4190674
2: -61.4176140, 209.2827911, -64.2663574, 218.2355347, -279.6531067, 273.5491333
3: -124.0811539, 245.3663025, -129.7195282, 255.6669159, -379.7479858, 375.0858154
4: -67.6681137, 205.0659180, -70.6764679, 213.9241943, -281.5922241, 275.7423401

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_B2_A1_A1

### Relational analysis result of NS_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334234, upper bound: 655.4343227
time: 0.77 seconds

## Relational analysis of NS_B2_B2_A1_A2

### Relational analysis result of NS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334784, upper bound: 655.4334793
time: 0.75 seconds

## BFS NS instance: NS_B2_B2_A2

### Backsubstitution after applying NS history:
0: -296.6877136, 593.7878418, -294.6537170, 589.8226318, -886.5103760, 888.4415283
1: -104.3657379, 214.1592712, -103.6562271, 212.6954803, -317.0612183, 317.8154907
2: -64.7006531, 219.7236328, -64.2663574, 218.2355347, -282.9361877, 283.9899902
3: -130.6006317, 257.4122925, -129.7195282, 255.6669159, -386.2674866, 387.1318359
4: -71.1577072, 215.3730621, -70.6764679, 213.9241943, -285.0818787, 286.0495300

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_B2_A2_A1

### Relational analysis result of NS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334234, upper bound: 655.4343947
time: 0.79 seconds

## Relational analysis of NS_B2_B2_A2_A2

### Relational analysis result of NS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334784, upper bound: 655.4336230
time: 0.88 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.84 seconds
NS_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4332024, upper bound: 655.4337833
NS_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4336519, upper bound: 655.4341964
NS_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4332541, upper bound: 655.4338742
NS_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4345953, upper bound: 655.4345642
NS_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4331061, upper bound: 655.4340119
NS_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4331061, upper bound: 655.4342583
NS_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4333243, upper bound: 655.4340119
NS_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4333243, upper bound: 655.4342583
NS_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4334271, upper bound: 655.4343293
NS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4334271, upper bound: 655.4335257
NS_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4343214, upper bound: 655.4335553
NS_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4334828, upper bound: 655.4335257
NS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4334234, upper bound: 655.4343227
NS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4334784, upper bound: 655.4334793
NS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4334234, upper bound: 655.4343947
NS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -655.4334784, upper bound: 655.4336230

## BFS NS instance: NS_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -278.7882080, 558.6371460, -276.1640015, 554.7743530, -833.5625610, 834.8011475
1: -98.4210129, 202.3205566, -97.5296555, 200.5030365, -298.9240417, 299.8501892
2: -60.9690666, 207.5259857, -60.3963089, 205.7541199, -266.7231750, 267.9223022
3: -123.1782150, 243.0384979, -122.0703049, 241.0009613, -364.1791687, 365.1086731
4: -67.0460205, 203.1841583, -66.4561844, 201.4024658, -268.4484863, 269.6403503

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_B1_A1_B1_A1

### Relational analysis result of NS_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337670, upper bound: 655.4343458
time: 0.71 seconds

## Relational analysis of NS_B1_B1_A1_B1_A2

### Relational analysis result of NS_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337615, upper bound: 655.4345730
time: 0.79 seconds

## BFS NS instance: NS_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -281.3729858, 563.5347290, -285.5978699, 574.3397217, -855.7126465, 849.1325684
1: -99.2903824, 204.1332703, -100.9692001, 207.2597809, -306.5501709, 305.1024780
2: -61.5006409, 209.3455811, -62.4961395, 212.8603058, -274.3609619, 271.8417053
3: -124.2858810, 245.2305450, -126.2862091, 249.1243439, -373.4102173, 371.5167236
4: -67.6393738, 204.9850922, -68.7750702, 208.4643250, -276.1036987, 273.7601013

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_B1_A1_B2_B1

### Relational analysis result of NS_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336503, upper bound: 655.4346997
time: 0.67 seconds

## Relational analysis of NS_B1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_B2_A1

### Relational analysis result of NS_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333794, upper bound: 655.4342750
time: 0.71 seconds

## Relational analysis of NS_B1_B1_A1_B2_A2

### Relational analysis result of NS_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336216, upper bound: 655.4345502
time: 0.80 seconds

## BFS NS instance: NS_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -367.4974060, 742.8956299, -281.4480286, 564.6060181, -932.1033936, 1024.3436279
1: -130.7707062, 268.2885437, -99.2784348, 203.9435272, -334.7141724, 367.5669861
2: -81.0260010, 275.5747681, -61.4865608, 209.2844849, -290.3104553, 336.8175964
3: -162.7293854, 322.7210999, -124.2763367, 245.1490173, -407.8783875, 446.7494507
4: -89.2571487, 269.1846924, -67.6569748, 204.9340820, -294.1912231, 336.6910706

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_B1_A2_B1_B1

### Relational analysis result of NS_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316607, upper bound: 655.4329315
time: 0.80 seconds

## Relational analysis of NS_B1_B1_A2_B1_B2

### Relational analysis result of NS_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316552, upper bound: 655.4331052
time: 0.81 seconds

## BFS NS instance: NS_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -368.6683960, 745.2770996, -285.1825256, 572.2634888, -940.9318848, 1030.4594727
1: -131.1792755, 269.1217346, -100.6005325, 206.7611237, -337.9403992, 369.7221985
2: -81.2746506, 276.4298706, -62.2852516, 212.1304016, -293.4050598, 338.4715271
3: -163.2350769, 323.7264709, -125.9161072, 248.5384216, -411.7734070, 449.3875732
4: -89.5345154, 270.0223389, -68.5595551, 207.7013092, -297.2358093, 338.4301453

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_B1_A2_B2_B1

### Relational analysis result of NS_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330992, upper bound: 655.4337730
time: 0.73 seconds

## Relational analysis of NS_B1_B1_A2_B2_B2

### Relational analysis result of NS_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335760, upper bound: 655.4341885
time: 0.75 seconds

## BFS NS instance: NS_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -270.2115173, 541.6589355, -335.8776245, 682.2878418, -952.4993896, 877.5365601
1: -95.4805145, 196.3548126, -120.1094589, 246.7636414, -342.2441406, 316.4642639
2: -59.1623268, 201.3903961, -74.3869476, 253.4926453, -312.3336792, 275.7773438
3: -119.4600983, 235.8445587, -149.4456329, 296.8889465, -415.9708862, 385.2901917
4: -65.0648193, 197.0860291, -82.0037994, 247.2059021, -312.0609436, 279.0898438

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_B2_A1_B1_A1

### Relational analysis result of NS_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4329146, upper bound: 655.4323934
time: 0.81 seconds

## Relational analysis of NS_B1_B2_A1_B1_A2

### Relational analysis result of NS_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4331057, upper bound: 655.4345405
time: 0.75 seconds

## BFS NS instance: NS_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -270.2115173, 541.6589355, -510.4286804, 1040.9388428, -1303.5120850, 1052.0876465
1: -95.4805145, 196.3548126, -183.3463745, 374.7771606, -469.0671692, 379.7011719
2: -59.1623268, 201.3903961, -113.6671982, 385.4181824, -442.1118469, 315.0575867
3: -119.4600983, 235.8445587, -226.7258759, 451.7924194, -568.4875488, 462.5704346
4: -65.0648193, 197.0860291, -125.6145477, 375.6672058, -438.9745789, 322.7005615

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_A1_B2_B1

### Relational analysis result of NS_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4315465, upper bound: 655.4336217
time: 0.73 seconds

## Relational analysis of NS_B1_B2_A1_B2_B2

### Relational analysis result of NS_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323813, upper bound: 655.4341992
time: 0.76 seconds

## BFS NS instance: NS_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -504.7594604, 1020.5460205, -335.8776245, 682.2878418, -1187.0468750, 1352.8946533
1: -179.2101898, 363.8814392, -120.1094589, 246.7636414, -425.6136169, 483.5840759
2: -111.2300949, 374.3564758, -74.3869476, 253.4926453, -364.2330322, 447.5697632
3: -221.6486511, 439.3168335, -149.4456329, 296.8889465, -518.1287231, 587.5361328
4: -122.9680405, 366.0145569, -82.0037994, 247.2059021, -369.8014832, 447.2056580

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_B2_A2_B1_A1

### Relational analysis result of NS_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330956, upper bound: 655.4319616
time: 0.85 seconds

## Relational analysis of NS_B1_B2_A2_B1_A2

### Relational analysis result of NS_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333230, upper bound: 655.4340119
time: 0.81 seconds

## BFS NS instance: NS_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -507.2344666, 1025.1188965, -520.8622437, 1062.6779785, -1561.9393311, 1542.4350586
1: -180.0500031, 365.5339661, -187.0806274, 381.9378967, -560.1875610, 551.5573120
2: -111.7548523, 376.0448608, -115.9496002, 392.8047485, -501.9152222, 490.6101379
3: -222.7181091, 441.2908020, -231.3587189, 460.4232178, -680.3192749, 671.2518921
4: -123.5482254, 367.6861267, -128.1483612, 382.9768677, -504.5975647, 494.8048706

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_B2_A2_B2_A1

### Relational analysis result of NS_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330956, upper bound: 655.4319616
time: 0.76 seconds

## Relational analysis of NS_B1_B2_A2_B2_A2

### Relational analysis result of NS_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333230, upper bound: 655.4340119
time: 0.72 seconds

## BFS NS instance: NS_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -249.7840729, 503.2324829, -276.4178772, 555.8069458, -805.5910034, 779.6503906
1: -88.5857239, 182.3641815, -97.6014023, 200.3663483, -288.9520264, 279.9655762
2: -54.7852821, 187.2775269, -60.3834152, 205.8044281, -260.5897217, 247.6609497
3: -110.6447220, 219.5446472, -121.9826736, 241.2917328, -351.9363708, 341.5272217
4: -60.3495255, 183.1372833, -66.5231628, 201.6296387, -261.9791565, 249.6604462

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_B1_A1_A1_B1

### Relational analysis result of NS_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334474, upper bound: 655.4334474
time: 0.80 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2

### Relational analysis result of NS_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4328469, upper bound: 655.4335302
time: 0.83 seconds

## BFS NS instance: NS_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -489.9229736, 992.4478149, -278.7102661, 560.3480225, -1050.2709961, 1265.9733887
1: -174.3223877, 354.0904236, -98.3684235, 201.8787842, -376.2011719, 452.1736450
2: -107.9831543, 364.4405212, -60.8557243, 207.3693542, -315.3525085, 424.0178223
3: -215.3241119, 427.7023621, -122.9572449, 243.1223297, -458.4464417, 549.2540283
4: -119.5247650, 356.2318115, -67.0528107, 203.2060699, -322.7307739, 422.3576050

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_B1_A1_A2_B1

### Relational analysis result of NS_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335301, upper bound: 655.4334474
time: 0.75 seconds

## Relational analysis of NS_B2_B1_A1_A2_B2

### Relational analysis result of NS_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335301, upper bound: 655.4335301
time: 0.95 seconds

## BFS NS instance: NS_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -293.1340637, 586.7754517, -248.3945618, 500.5333252, -793.6673584, 835.1700439
1: -103.1628876, 211.7566528, -88.0960999, 181.3596039, -284.5224915, 299.8527527
2: -63.9546928, 217.2475433, -54.4891930, 186.2583008, -250.2129822, 271.7366943
3: -129.0911560, 254.5047760, -110.0451050, 218.3574219, -347.4485779, 364.5498352
4: -70.3340759, 212.9055939, -60.0209389, 182.1485138, -252.4825592, 272.9265137

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_B1_A2_B1_A1

### Relational analysis result of NS_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4327217, upper bound: 655.4335553
time: 0.84 seconds

## Relational analysis of NS_B2_B1_A2_B1_A2

### Relational analysis result of NS_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334315, upper bound: 655.4335553
time: 0.84 seconds

## BFS NS instance: NS_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -295.4877014, 591.4110718, -453.4024963, 919.7541504, -1212.1588135, 1044.8134766
1: -103.9471588, 213.2993927, -161.5505219, 329.8564148, -433.6744690, 374.8499146
2: -64.4393692, 218.8468475, -100.2757950, 339.1453247, -402.6053162, 319.1226501
3: -130.0830994, 256.3812866, -200.2138977, 398.4988708, -527.3710327, 456.5951843
4: -70.8760834, 214.5123291, -110.9012985, 331.2800293, -401.4280090, 325.4136353

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_B1_A2_B2_A1

### Relational analysis result of NS_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334315, upper bound: 655.4336263
time: 0.75 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2

### Relational analysis result of NS_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334315, upper bound: 655.4336263
time: 0.84 seconds

## BFS NS instance: NS_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -249.7840729, 503.2324829, -291.0944519, 582.7980347, -832.5820312, 794.3269043
1: -88.5857239, 182.3641815, -102.4513092, 210.2886963, -298.8744202, 284.8154907
2: -54.7852821, 187.2775269, -63.5191689, 215.7552490, -270.5405273, 250.7966766
3: -110.6447220, 219.5446472, -128.2076416, 252.7545166, -363.3992310, 347.7521973
4: -60.3495255, 183.1372833, -69.8515244, 211.4528503, -271.8023376, 252.9887848

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_B2_A1_A1_B1

### Relational analysis result of NS_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335553, upper bound: 655.4334315
time: 0.83 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2

### Relational analysis result of NS_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335553, upper bound: 655.4334837
time: 0.81 seconds

## BFS NS instance: NS_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -490.4326477, 993.3973999, -293.4486084, 587.4358521, -1077.8685303, 1281.8023682
1: -174.4966431, 354.4351196, -103.2359848, 211.8317566, -386.3283997, 457.4754333
2: -108.0913544, 364.7931519, -64.0042038, 217.3549805, -325.4463501, 427.5553284
3: -215.5454254, 428.1123047, -129.1994019, 254.6311493, -470.1765442, 556.0016479
4: -119.6445236, 356.5802612, -70.3936310, 213.0593262, -332.7037659, 426.0719299

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_B2_A1_A2_B1

### Relational analysis result of NS_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336263, upper bound: 655.4334315
time: 0.77 seconds

## Relational analysis of NS_B2_B2_A1_A2_B2

### Relational analysis result of NS_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336263, upper bound: 655.4334837
time: 0.89 seconds

## BFS NS instance: NS_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -264.3528748, 529.9644165, -291.0944519, 582.7980347, -847.1508789, 821.0588379
1: -93.4128571, 192.2550049, -102.4513092, 210.2886963, -303.7015381, 294.7062683
2: -57.8978233, 197.1433105, -63.5191689, 215.7552490, -273.6530762, 260.6624756
3: -116.8486252, 230.8878174, -128.2076416, 252.7545166, -369.6031494, 359.0953979
4: -63.6478119, 192.8829803, -69.8515244, 211.4528503, -275.1005859, 262.7344971

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_B2_A2_A1_B1

### Relational analysis result of NS_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335264, upper bound: 655.4335603
time: 0.75 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2

### Relational analysis result of NS_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335263, upper bound: 655.4336230
time: 0.78 seconds

## BFS NS instance: NS_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -505.0922852, 1020.1849365, -293.4486084, 587.4358521, -1092.5280762, 1310.4403076
1: -179.1806793, 363.9656982, -103.2359848, 211.8317566, -391.0124512, 467.2016296
2: -111.2490387, 374.3659973, -64.0042038, 217.3549805, -328.6040039, 437.3450623
3: -221.6428680, 439.3942566, -129.1994019, 254.6311493, -476.2739868, 567.4923706
4: -122.9792328, 366.0240173, -70.3936310, 213.0593262, -336.0385742, 435.7186279

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_B2_A2_A2_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335808, upper bound: 655.4335603
time: 0.81 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2

### Relational analysis result of NS_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335808, upper bound: 655.4336230
time: 0.88 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.00 seconds
NS_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4337670, upper bound: 655.4343458
NS_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4337615, upper bound: 655.4345730
NS_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4333794, upper bound: 655.4342750
NS_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4336216, upper bound: 655.4345502
NS_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4316607, upper bound: 655.4329315
NS_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4316552, upper bound: 655.4331052
NS_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4330992, upper bound: 655.4337730
NS_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4335760, upper bound: 655.4341885
NS_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4329146, upper bound: 655.4323934
NS_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4331057, upper bound: 655.4345405
NS_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4315465, upper bound: 655.4336217
NS_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4323813, upper bound: 655.4341992
NS_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4330956, upper bound: 655.4319616
NS_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4333230, upper bound: 655.4340119
NS_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4330956, upper bound: 655.4319616
NS_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4333230, upper bound: 655.4340119
NS_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4334474, upper bound: 655.4334474
NS_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4328469, upper bound: 655.4335302
NS_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4335301, upper bound: 655.4334474
NS_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4335301, upper bound: 655.4335301
NS_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4327217, upper bound: 655.4335553
NS_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4334315, upper bound: 655.4335553
NS_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4334315, upper bound: 655.4336263
NS_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4334315, upper bound: 655.4336263
NS_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4335553, upper bound: 655.4334315
NS_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4335553, upper bound: 655.4334837
NS_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4336263, upper bound: 655.4334315
NS_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4336263, upper bound: 655.4334837
NS_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4335264, upper bound: 655.4335603
NS_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4335263, upper bound: 655.4336230
NS_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4335808, upper bound: 655.4335603
NS_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -655.4335808, upper bound: 655.4336230

## BFS NS instance: NS_B1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -259.0079956, 520.1624756, -269.0493164, 540.3302612, -799.3381348, 789.2117920
1: -91.5379181, 188.3399658, -95.0219498, 195.4265594, -286.9644775, 283.3619080
2: -56.5923157, 193.0878143, -58.7869034, 200.5212708, -257.1135864, 251.8747253
3: -114.6557922, 225.9959564, -118.9760284, 234.7494812, -349.4052429, 344.9719849
4: -62.2515297, 189.0608063, -64.6940460, 196.3048553, -258.5563660, 253.7548370

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_B1_A1_B1_A1_B1

### Relational analysis result of NS_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336705, upper bound: 655.4343458
time: 0.78 seconds

## Relational analysis of NS_B1_B1_A1_B1_A1_B2

### Relational analysis result of NS_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336705, upper bound: 655.4343458
time: 0.81 seconds

## BFS NS instance: NS_B1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -273.7376404, 548.6153564, -274.5762024, 551.6284790, -825.3660889, 823.1913452
1: -96.6459045, 198.6981964, -96.9722061, 199.3620453, -296.0079346, 295.6704102
2: -59.8632851, 203.8120270, -60.0481606, 204.5856018, -264.4488220, 263.8601379
3: -120.9368057, 238.6856995, -121.3617935, 239.6322784, -360.5690613, 360.0474854
4: -65.8255768, 199.5302277, -66.0722122, 200.2509155, -266.0764465, 265.6024475

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336705, upper bound: 655.4345730
time: 0.84 seconds

## Relational analysis of NS_B1_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336705, upper bound: 655.4345730
time: 0.73 seconds

## BFS NS instance: NS_B1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -272.7237854, 546.4223633, -282.0245056, 567.1750488, -839.8988037, 828.4468994
1: -96.2939987, 198.1040344, -99.7318497, 204.7720795, -301.0660706, 297.8358765
2: -59.6535110, 203.1464233, -61.7259979, 210.2901001, -269.9436035, 264.8724365
3: -120.5567093, 237.9588470, -124.7343826, 246.1270447, -366.6837463, 362.6931763
4: -65.5925903, 198.8743744, -67.9258194, 205.9217072, -271.5142822, 266.8001709

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_B2_A1_B1

### Relational analysis result of NS_B1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4324389, upper bound: 655.4336459
time: 0.70 seconds

## Relational analysis of NS_B1_B1_A1_B2_A1_B2

### Relational analysis result of NS_B1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4324389, upper bound: 655.4342750
time: 0.87 seconds

## BFS NS instance: NS_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -282.0775757, 566.3452148, -282.9638062, 569.2690430, -851.3465576, 849.3088379
1: -99.7751923, 205.5013123, -100.0638885, 205.3867645, -305.1619568, 305.5651245
2: -61.8409309, 210.7101898, -61.9457817, 210.9547424, -272.7956848, 272.6558838
3: -124.7443085, 246.9560089, -125.1670685, 246.8773651, -371.6216736, 372.1229858
4: -68.0225296, 206.0715027, -68.1615295, 206.6119537, -274.6344910, 274.2330322

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_B1_A1_B2_A2_B1

### Relational analysis result of NS_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316330, upper bound: 655.4336787
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A1_B2_A2_B2

### Relational analysis result of NS_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335330, upper bound: 655.4345439
time: 0.80 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -361.7532654, 731.6751099, -272.1047058, 546.5646362, -908.3178101, 1003.7797852
1: -128.8310394, 264.3013000, -96.1026306, 197.4705658, -326.3016052, 360.4039307
2: -79.8301392, 271.4966431, -59.5293617, 202.6873322, -282.5174255, 330.7686157
3: -160.2961121, 317.9150391, -120.2908859, 237.3561096, -397.6522217, 437.9429016
4: -87.9271240, 265.1660767, -65.4800186, 198.4188385, -286.3459473, 330.4862061

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B1_B1_B1

### Relational analysis result of NS_B1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4315270, upper bound: 655.4322515
time: 0.75 seconds

## Relational analysis of NS_B1_B1_A2_B1_B1_B2

### Relational analysis result of NS_B1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316562, upper bound: 655.4328699
time: 0.75 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -364.5426025, 737.0339966, -281.5440674, 566.0229492, -930.5654297, 1018.5780640
1: -129.7509308, 266.2591248, -99.5141525, 204.2042999, -333.9551697, 365.7732849
2: -80.4010544, 273.4817200, -61.6279831, 209.7723846, -290.1733398, 334.8553772
3: -161.4765930, 320.2839050, -124.5073853, 245.4479523, -406.9245605, 444.5647583
4: -88.5619202, 267.1253967, -67.7951736, 205.4680481, -294.0299377, 334.7646484

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B1_B2_B1

### Relational analysis result of NS_B1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316210, upper bound: 655.4323870
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A2_B1_B2_B2

### Relational analysis result of NS_B1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316514, upper bound: 655.4330682
time: 0.77 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -362.9481506, 734.1021118, -275.9361572, 554.3456421, -917.2937622, 1010.0381470
1: -129.2476959, 265.1510315, -97.4532166, 200.3450775, -329.5927734, 362.6042480
2: -80.0834045, 272.3686523, -60.3477936, 205.5942993, -285.6777039, 332.4593811
3: -160.8116302, 318.9396973, -121.9723511, 240.8100433, -401.6216736, 440.6419067
4: -88.2095566, 266.0202637, -66.4029160, 201.2459869, -289.4555054, 332.2620544

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B2_B1_B1

### Relational analysis result of NS_B1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314608, upper bound: 655.4322163
time: 0.82 seconds

## Relational analysis of NS_B1_B1_A2_B2_B1_B2

### Relational analysis result of NS_B1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330919, upper bound: 655.4337063
time: 0.97 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -365.6893616, 739.3649902, -285.3501892, 573.8672485, -939.5566406, 1024.7150879
1: -130.1509399, 267.0756836, -100.8853302, 207.0863495, -337.2372437, 367.9609985
2: -80.6443481, 274.3194885, -62.4432831, 212.6849976, -293.3293457, 336.5081787
3: -161.9719391, 321.2691040, -126.1796265, 248.9146881, -410.8866272, 447.2149658
4: -88.8332977, 267.9460144, -68.7169113, 208.2928925, -297.1260986, 336.5065002

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B2_B2_B1

### Relational analysis result of NS_B1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320927, upper bound: 655.4326825
time: 0.83 seconds

## Relational analysis of NS_B1_B1_A2_B2_B2_B2

### Relational analysis result of NS_B1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335633, upper bound: 655.4341287
time: 0.83 seconds

## BFS NS instance: NS_B1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -267.8843384, 536.9671021, -334.6813049, 679.8351440, -947.7194824, 871.6484375
1: -94.5999146, 194.5491791, -119.6903687, 245.9018707, -340.5017700, 314.2395630
2: -58.6682587, 199.5718689, -74.1331329, 252.6116028, -310.9532471, 273.7050171
3: -118.4066162, 233.7367554, -148.9267578, 295.8482666, -413.8750916, 382.6634827
4: -64.5073624, 195.3087616, -81.7196350, 246.3467255, -310.6460266, 277.0283813

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_B2_A1_B1_A1_A1

### Relational analysis result of NS_B1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336411, upper bound: 655.4323934
time: 0.73 seconds

## Relational analysis of NS_B1_B2_A1_B1_A1_A2

### Relational analysis result of NS_B1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336411, upper bound: 655.4323934
time: 0.80 seconds

## BFS NS instance: NS_B1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -269.9541626, 541.1636353, -335.8776245, 682.2878418, -952.2419434, 877.0412598
1: -95.3928070, 196.1741791, -120.1094589, 246.7636414, -342.1564331, 316.2836304
2: -59.1073990, 201.2070160, -74.3869476, 253.4926453, -312.2776794, 275.5939636
3: -119.3490982, 235.6255646, -149.4456329, 296.8889465, -415.8565979, 385.0711975
4: -65.0042038, 196.9067993, -82.0037994, 247.2059021, -311.9995117, 278.9105835

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_B2_A1_B1_A2_A1

### Relational analysis result of NS_B1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337635, upper bound: 655.4337635
time: 0.75 seconds

## Relational analysis of NS_B1_B2_A1_B1_A2_A2

### Relational analysis result of NS_B1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337635, upper bound: 655.4345405
time: 0.75 seconds

## BFS NS instance: NS_B1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -265.1582031, 531.8320923, -505.3384094, 1032.1496582, -1289.4555664, 1037.1705322
1: -93.7428894, 192.8541260, -181.6755524, 371.4797974, -463.9808350, 374.5296021
2: -58.0848389, 197.8056641, -112.5978012, 382.0594177, -437.6427612, 310.4034729
3: -117.2886047, 231.6255493, -224.5699463, 447.9581604, -562.2998047, 456.1954346
4: -63.8761444, 193.5596771, -124.4611664, 372.3149719, -434.4153137, 318.0208435

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_B2_A1_B2_B1_A1

### Relational analysis result of NS_B1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4315123, upper bound: 655.4331096
time: 0.77 seconds

## Relational analysis of NS_B1_B2_A1_B2_B1_A2

### Relational analysis result of NS_B1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4315122, upper bound: 655.4336217
time: 0.74 seconds

## BFS NS instance: NS_B1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -269.0924683, 539.5418701, -503.2910461, 1027.2694092, -1288.6888428, 1042.8326416
1: -95.0919113, 195.5868530, -180.8528748, 369.8277893, -463.7218323, 376.4396362
2: -58.9197578, 200.5987244, -112.1324158, 380.3213501, -436.7654419, 312.7311401
3: -118.9713058, 234.9206696, -223.6034241, 445.8825989, -562.0484619, 458.5241089
4: -64.8010864, 196.2995758, -123.9344864, 370.6390076, -433.6778870, 320.2339478

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_B2_A1_B2_B2_A1

### Relational analysis result of NS_B1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323115, upper bound: 655.4332176
time: 0.87 seconds

## Relational analysis of NS_B1_B2_A1_B2_B2_A2

### Relational analysis result of NS_B1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323115, upper bound: 655.4341992
time: 0.74 seconds

## BFS NS instance: NS_B1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -501.8479919, 1015.0072632, -334.6813049, 679.8351440, -1181.6829834, 1345.7353516
1: -178.1308289, 361.6298218, -119.6903687, 245.9018707, -423.6668701, 480.8760986
2: -110.6156235, 372.1081238, -74.1331329, 252.6116028, -362.7289124, 445.0001526
3: -220.3226929, 436.7128296, -148.9267578, 295.8482666, -515.7502441, 584.3848267
4: -122.2789612, 363.7914124, -81.7196350, 246.3467255, -368.2505188, 444.6497498

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_A2_B1_A1_A1

### Relational analysis result of NS_B1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330174, upper bound: 655.4311967
time: 0.70 seconds

## Relational analysis of NS_B1_B2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_B2_A2_B1_A1_A1

### Relational analysis result of NS_B1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337976, upper bound: 655.4319616
time: 0.79 seconds

## Relational analysis of NS_B1_B2_A2_B1_A1_A2

### Relational analysis result of NS_B1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337976, upper bound: 655.4319616
time: 0.72 seconds

## BFS NS instance: NS_B1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -504.4192810, 1019.8952026, -335.8776245, 682.2878418, -1186.7067871, 1352.2293701
1: -179.0941010, 363.6441345, -120.1094589, 246.7636414, -425.4944153, 483.3460693
2: -111.1576614, 374.1158752, -74.3869476, 253.4926453, -364.1591492, 447.3267212
3: -221.5022583, 439.0308838, -149.4456329, 296.8889465, -517.9785767, 587.2504272
4: -122.8879395, 365.7792358, -82.0037994, 247.2059021, -369.7200623, 446.9684448

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_A2_B1_A2_A1

### Relational analysis result of NS_B1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330702, upper bound: 655.4314333
time: 0.74 seconds

## Relational analysis of NS_B1_B2_A2_B1_A2_A2

### Relational analysis result of NS_B1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4332522, upper bound: 655.4334386
time: 0.77 seconds

## BFS NS instance: NS_B1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -504.3713989, 1019.6704712, -519.6886597, 1060.2667236, -1556.6480713, 1535.3867188
1: -178.9853821, 363.3126526, -186.6672668, 381.0796509, -558.2590332, 548.8855591
2: -111.1502457, 373.8285522, -115.6994629, 391.9320679, -500.4244995, 488.0751038
3: -221.4120026, 438.7256470, -230.8485565, 459.3915405, -677.9714966, 668.1472778
4: -122.8700638, 365.4941711, -127.8684616, 382.1246033, -503.0613098, 492.2836914

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_B2_A2_B2_A1_B1

### Relational analysis result of NS_B1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320782, upper bound: 655.4318387
time: 0.82 seconds

## Relational analysis of NS_B1_B2_A2_B2_A1_B2

### Relational analysis result of NS_B1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320782, upper bound: 655.4319616
time: 0.76 seconds

## BFS NS instance: NS_B1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -506.9811401, 1024.6284180, -520.8622437, 1062.6779785, -1561.6840820, 1541.9296875
1: -179.9634552, 365.3546753, -187.0806274, 381.9378967, -560.0979004, 551.3773804
2: -111.7008667, 375.8636475, -115.9496002, 392.8047485, -501.8598633, 490.4264526
3: -222.6092529, 441.0742493, -231.3587189, 460.4232178, -680.2065430, 671.0357666
4: -123.4885254, 367.5095215, -128.1483612, 382.9768677, -504.5366211, 494.6264343

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_A2_B2_A2_B1

### Relational analysis result of NS_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4317431, upper bound: 655.4327806
time: 0.78 seconds

## Relational analysis of NS_B1_B2_A2_B2_A2_B2

### Relational analysis result of NS_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4325814, upper bound: 655.4334386
time: 0.76 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -249.7840729, 503.2324829, -248.3945618, 500.5333252, -750.3173218, 751.6270752
1: -88.5857239, 182.3641815, -88.0960999, 181.3596039, -269.9453125, 270.4602661
2: -54.7852821, 187.2775269, -54.4891930, 186.2583008, -241.0435791, 241.7667084
3: -110.6447220, 219.5446472, -110.0451050, 218.3574219, -329.0021362, 329.5896912
4: -60.3495255, 183.1372833, -60.0209389, 182.1485138, -242.4980469, 243.1582184

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B1_A1_A1_B1_A1

### Relational analysis result of NS_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314558, upper bound: 655.4337541
time: 0.70 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_A2

### Relational analysis result of NS_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309353, upper bound: 655.4316939
time: 0.76 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -249.7840729, 503.2324829, -450.0046692, 913.3039551, -1159.8280029, 953.2371826
1: -88.5857239, 182.3641815, -160.3698730, 327.5220337, -415.8839722, 342.7340088
2: -54.7852821, 187.2775269, -99.5443115, 336.7535095, -390.5212708, 286.8218384
3: -110.6447220, 219.5446472, -198.7207642, 395.7313538, -505.0733643, 418.2653198
4: -60.3495255, 183.1372833, -110.0928879, 328.9214478, -388.5215149, 293.2301636

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A1_B2_A1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4332275, upper bound: 655.4341485
time: 0.75 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333499, upper bound: 655.4340034
time: 0.77 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -487.0251465, 987.0439453, -248.3945618, 500.5333252, -987.5584106, 1230.1982422
1: -173.3317719, 352.1291809, -88.0960999, 181.3596039, -354.6913147, 439.9343567
2: -107.3682175, 362.4349060, -54.4891930, 186.2583008, -293.6265259, 415.6387329
3: -214.0665741, 425.3705750, -110.0451050, 218.3574219, -432.4240112, 534.0122681
4: -118.8440094, 354.2496033, -60.0209389, 182.1485138, -300.9924622, 413.3464966

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A1_A2_B1_A1

### Relational analysis result of NS_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334425, upper bound: 655.4330330
time: 0.84 seconds

## Relational analysis of NS_B2_B1_A1_A2_B1_A2

### Relational analysis result of NS_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334403, upper bound: 655.4333605
time: 0.91 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -493.2630615, 998.6672974, -456.7290649, 926.0529785, -1416.0998535, 1450.1230469
1: -175.4642792, 356.3484497, -162.7059784, 332.1381226, -506.6705322, 518.3323975
2: -108.6924744, 366.7500305, -100.9918518, 341.4832153, -448.9780273, 466.3092957
3: -216.7750244, 430.3876343, -201.6750641, 401.2048340, -616.6233521, 630.6933594
4: -120.3095779, 358.5146484, -111.6923676, 333.5849304, -452.9650879, 469.1451416

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B1_A1_A2_B2_A1

### Relational analysis result of NS_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4315200, upper bound: 655.4328832
time: 0.99 seconds

## Relational analysis of NS_B2_B1_A1_A2_B2_A2

### Relational analysis result of NS_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309813, upper bound: 655.4309353
time: 0.81 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -264.3528748, 529.9644165, -248.3945618, 500.5333252, -764.8861694, 778.3590088
1: -93.4128571, 192.2550049, -88.0960999, 181.3596039, -274.7724609, 280.3511047
2: -57.8978233, 197.1433105, -54.4891930, 186.2583008, -244.1561279, 251.6325073
3: -116.8486252, 230.8878174, -110.0451050, 218.3574219, -335.2060547, 340.9328918
4: -63.6478119, 192.8829803, -60.0209389, 182.1485138, -245.7963104, 252.9039154

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_B1_A2_B1_A1_B1

### Relational analysis result of NS_B2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4342961, upper bound: 655.4335416
time: 0.92 seconds

## Relational analysis of NS_B2_B1_A2_B1_A1_B2

### Relational analysis result of NS_B2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4321963, upper bound: 655.4330161
time: 0.74 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -501.9806824, 1014.4064941, -248.3945618, 500.5333252, -1002.5140381, 1257.9420166
1: -178.1192627, 361.8738403, -88.0960999, 181.3596039, -359.4788818, 449.7160950
2: -110.5875626, 372.2257996, -54.4891930, 186.2583008, -296.8458557, 425.4264832
3: -220.2953186, 436.8955078, -110.0451050, 218.3574219, -438.6527405, 545.5739746
4: -122.2480774, 363.9082642, -60.0209389, 182.1485138, -304.3965454, 423.0328064

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_B1_A2_B1

### Relational analysis result of NS_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4341179, upper bound: 655.4333256
time: 0.84 seconds

## Relational analysis of NS_B2_B1_A2_B1_A2_B2

### Relational analysis result of NS_B2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4338252, upper bound: 655.4334627
time: 0.90 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -264.3528748, 529.9644165, -450.6017761, 914.4380493, -1175.6459961, 980.5661621
1: -93.4128571, 192.2550049, -160.5772705, 327.9323425, -421.2115784, 352.8322754
2: -57.8978233, 197.1433105, -99.6728363, 337.1738586, -394.0881958, 296.8161316
3: -116.8486252, 230.8878174, -198.9831543, 396.2178040, -511.8540039, 429.8709106
4: -63.6478119, 192.8829803, -110.2349701, 329.3359985, -392.2610779, 303.1179504

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_B1_A2_B2_A1_B1

### Relational analysis result of NS_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333028, upper bound: 655.4332216
time: 0.75 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A2_B2_A1_B1

### Relational analysis result of NS_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330109, upper bound: 655.4334852
time: 0.77 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_B2

### Relational analysis result of NS_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333455, upper bound: 655.4336178
time: 0.88 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -507.7007446, 1025.0267334, -456.7290649, 926.0529785, -1430.6707764, 1476.8780518
1: -180.0705872, 365.7184143, -162.7059784, 332.1381226, -511.4443970, 527.7424927
2: -111.8039627, 376.1591492, -100.9918518, 341.4832153, -452.1255188, 475.7180481
3: -222.7735596, 441.4889221, -201.6750641, 401.2048340, -622.7187500, 641.8252563
4: -123.5925446, 367.7968445, -111.6923676, 333.5849304, -456.2814026, 478.4624634

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_B1_A2_B2_A2_B1

### Relational analysis result of NS_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334239, upper bound: 655.4335416
time: 0.76 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2_B2

### Relational analysis result of NS_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314142, upper bound: 655.4330161
time: 0.96 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -249.7840729, 503.2324829, -262.2931519, 525.9716187, -775.7556763, 765.5256348
1: -88.5857239, 182.3641815, -92.6951981, 190.7752838, -279.3609619, 275.0593872
2: -54.7852821, 187.2775269, -57.4585457, 195.6372528, -250.4225311, 244.7360535
3: -110.6447220, 219.5446472, -115.9573059, 229.1246796, -339.7692871, 335.5018616
4: -60.3495255, 183.1372833, -63.1609802, 191.4156494, -251.7651672, 246.2982483

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_A1_B1_A1

### Relational analysis result of NS_B2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335415, upper bound: 655.4342961
time: 0.78 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_A2

### Relational analysis result of NS_B2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330161, upper bound: 655.4321964
time: 0.82 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -249.7840729, 503.2324829, -499.1403809, 1008.9745483, -1253.8874512, 1002.3728638
1: -88.5857239, 182.3641815, -177.1448822, 359.8768616, -448.2133179, 359.5089722
2: -54.7852821, 187.2775269, -109.9872894, 370.1892395, -423.6871033, 297.2648315
3: -110.6447220, 219.5446472, -219.0831909, 434.5091553, -543.7944946, 438.6278381
4: -60.3495255, 183.1372833, -121.5837021, 361.9176636, -421.3707886, 304.7209778

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A1_A1_B2_A1

### Relational analysis result of NS_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333256, upper bound: 655.4341351
time: 0.82 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2_A2

### Relational analysis result of NS_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334627, upper bound: 655.4339442
time: 0.80 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -487.6687012, 988.2446289, -262.2931519, 525.9716187, -1013.6403198, 1245.4157715
1: -173.5517731, 352.5648499, -92.6951981, 190.7752838, -364.3269653, 445.0606384
2: -107.5047684, 362.8803711, -57.4585457, 195.6372528, -303.1419678, 419.0885925
3: -214.3457489, 425.8885498, -115.9573059, 229.1246796, -443.4703674, 540.5342407
4: -118.9951706, 354.6898499, -63.1609802, 191.4156494, -310.4107056, 416.9539185

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_A2_B1_A1

### Relational analysis result of NS_B2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335777, upper bound: 655.4334239
time: 0.83 seconds

## Relational analysis of NS_B2_B2_A1_A2_B1_A2

### Relational analysis result of NS_B2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330691, upper bound: 655.4314142
time: 0.82 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -493.2630615, 998.6672974, -505.6468506, 1021.0565796, -1509.5505371, 1499.2940674
1: -175.4642792, 356.3484497, -179.3644562, 364.2495728, -538.7588501, 534.8841553
2: -108.6924744, 366.7500305, -111.3708878, 374.6630249, -481.9017334, 476.7012634
3: -216.7750244, 430.3876343, -221.9020538, 439.7334595, -655.0886230, 650.9287720
4: -120.3095779, 358.5146484, -123.1129456, 366.3409119, -485.5834351, 480.5588989

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_A2_B2_A1

### Relational analysis result of NS_B2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335777, upper bound: 655.4334239
time: 0.85 seconds

## Relational analysis of NS_B2_B2_A1_A2_B2_A2

### Relational analysis result of NS_B2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330691, upper bound: 655.4314142
time: 1.05 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -264.3528748, 529.9644165, -262.2931519, 525.9716187, -790.3244629, 792.2575684
1: -93.4128571, 192.2550049, -92.6951981, 190.7752838, -284.1881409, 284.9501953
2: -57.8978233, 197.1433105, -57.4585457, 195.6372528, -253.5350800, 254.6018524
3: -116.8486252, 230.8878174, -115.9573059, 229.1246796, -345.9732666, 346.8450623
4: -63.6478119, 192.8829803, -63.1609802, 191.4156494, -255.0634613, 256.0439453

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A1_B1_A1

### Relational analysis result of NS_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4332931, upper bound: 655.4342050
time: 0.76 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_A2

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333326, upper bound: 655.4339572
time: 0.84 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -264.3528748, 529.9644165, -499.7129517, 1010.0386963, -1271.0836182, 1029.6773682
1: -93.4128571, 192.2550049, -177.3401794, 360.2615356, -453.6743774, 369.5951843
2: -57.8978233, 197.1433105, -110.1090012, 370.5830078, -427.4510193, 307.2522583
3: -116.8486252, 230.8878174, -219.3310699, 434.9685974, -550.7223511, 450.2188721
4: -63.6478119, 192.8829803, -121.7181702, 362.3070068, -425.2600403, 314.6011353

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A1_B2_A1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4332931, upper bound: 655.4342075
time: 0.85 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A2

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334360, upper bound: 655.4340758
time: 0.83 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -502.5676270, 1015.4968262, -262.2931519, 525.9716187, -1028.5391846, 1274.4855957
1: -178.3194427, 362.2681274, -92.6951981, 190.7752838, -369.0946960, 454.9633179
2: -110.7122574, 372.6293335, -57.4585457, 195.6372528, -306.3494568, 429.0542908
3: -220.5494385, 437.3664246, -115.9573059, 229.1246796, -449.6740417, 552.2232056
4: -122.3858948, 364.3073425, -63.1609802, 191.4156494, -313.8015137, 426.7721252

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_B2_A2_A2_B1_A1

### Relational analysis result of NS_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4100332, upper bound: 655.4099891
time: 0.76 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_A2

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4094657, upper bound: 655.4096068
time: 1.12 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -507.7007446, 1025.0267334, -505.6468506, 1021.0565796, -1525.6143799, 1527.5343018
1: -180.0705872, 365.7184143, -179.3644562, 364.2495728, -543.7593994, 544.5196533
2: -111.8039627, 376.1591492, -111.3708878, 374.6630249, -485.2691040, 486.3272400
3: -222.7735596, 441.4889221, -221.9020538, 439.7334595, -661.3806763, 662.2565308
4: -123.5925446, 367.7968445, -123.1129456, 366.3409119, -489.0791016, 490.0538940

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4100332, upper bound: 655.4101318
time: 0.81 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4094657, upper bound: 655.4097294
time: 0.83 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.10 seconds
NS_B1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4336705, upper bound: 655.4343458
NS_B1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4336705, upper bound: 655.4343458
NS_B1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4336705, upper bound: 655.4345730
NS_B1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4336705, upper bound: 655.4345730
NS_B1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4324389, upper bound: 655.4336459
NS_B1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4324389, upper bound: 655.4342750
NS_B1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4316330, upper bound: 655.4336787
NS_B1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4335330, upper bound: 655.4345439
NS_B1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4315270, upper bound: 655.4322515
NS_B1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4316562, upper bound: 655.4328699
NS_B1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4316210, upper bound: 655.4323870
NS_B1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4316514, upper bound: 655.4330682
NS_B1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4314608, upper bound: 655.4322163
NS_B1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4330919, upper bound: 655.4337063
NS_B1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4320927, upper bound: 655.4326825
NS_B1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4335633, upper bound: 655.4341287
NS_B1_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4336411, upper bound: 655.4323934
NS_B1_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4336411, upper bound: 655.4323934
NS_B1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4337635, upper bound: 655.4337635
NS_B1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4337635, upper bound: 655.4345405
NS_B1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4315123, upper bound: 655.4331096
NS_B1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4315122, upper bound: 655.4336217
NS_B1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4323115, upper bound: 655.4332176
NS_B1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4323115, upper bound: 655.4341992
NS_B1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4337976, upper bound: 655.4319616
NS_B1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4337976, upper bound: 655.4319616
NS_B1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4330702, upper bound: 655.4314333
NS_B1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4332522, upper bound: 655.4334386
NS_B1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4320782, upper bound: 655.4318387
NS_B1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4320782, upper bound: 655.4319616
NS_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4317431, upper bound: 655.4327806
NS_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4325814, upper bound: 655.4334386
NS_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4314558, upper bound: 655.4337541
NS_B2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4309353, upper bound: 655.4316939
NS_B2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4332275, upper bound: 655.4341485
NS_B2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4333499, upper bound: 655.4340034
NS_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4334425, upper bound: 655.4330330
NS_B2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4334403, upper bound: 655.4333605
NS_B2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4315200, upper bound: 655.4328832
NS_B2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4309813, upper bound: 655.4309353
NS_B2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4342961, upper bound: 655.4335416
NS_B2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4321963, upper bound: 655.4330161
NS_B2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4341179, upper bound: 655.4333256
NS_B2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4338252, upper bound: 655.4334627
NS_B2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4330109, upper bound: 655.4334852
NS_B2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4333455, upper bound: 655.4336178
NS_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4334239, upper bound: 655.4335416
NS_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4314142, upper bound: 655.4330161
NS_B2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4335415, upper bound: 655.4342961
NS_B2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4330161, upper bound: 655.4321964
NS_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4333256, upper bound: 655.4341351
NS_B2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4334627, upper bound: 655.4339442
NS_B2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4335777, upper bound: 655.4334239
NS_B2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4330691, upper bound: 655.4314142
NS_B2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4335777, upper bound: 655.4334239
NS_B2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4330691, upper bound: 655.4314142
NS_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4332931, upper bound: 655.4342050
NS_B2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4333326, upper bound: 655.4339572
NS_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4332931, upper bound: 655.4342075
NS_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4334360, upper bound: 655.4340758
NS_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4100332, upper bound: 655.4099891
NS_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4094657, upper bound: 655.4096068
NS_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4100332, upper bound: 655.4101318
NS_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -655.4094657, upper bound: 655.4097294

## BFS NS instance: NS_B1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -259.0079956, 520.1624756, -260.3574524, 524.3128052, -783.3207397, 780.5197754
1: -91.5379181, 188.3399658, -92.0557709, 189.2300568, -280.7679749, 280.3956909
2: -56.5923157, 193.0878143, -56.8497543, 194.1286926, -250.7210083, 249.9375610
3: -114.6557922, 225.9959564, -115.2148514, 227.2112427, -341.8670349, 341.2108154
4: -62.2515297, 189.0608063, -62.5818367, 190.0472717, -252.2987823, 251.6426392

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_B1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333458, upper bound: 655.4328091
time: 0.82 seconds

## Relational analysis of NS_B1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_B1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336365, upper bound: 655.4342449
time: 0.79 seconds

## BFS NS instance: NS_B1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -259.0079956, 520.1624756, -271.6228943, 545.7624512, -804.7703247, 791.7853394
1: -91.5379181, 188.3399658, -95.9324493, 197.2268372, -288.7647705, 284.2723694
2: -56.5923157, 193.0878143, -59.3999634, 202.4023285, -258.9946289, 252.4877319
3: -114.6557922, 225.9959564, -120.0419235, 237.0750580, -351.7308350, 346.0378723
4: -62.2515297, 189.0608063, -65.3572845, 198.1024780, -260.3540039, 254.4180450

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_B1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333458, upper bound: 655.4328091
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_B1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336365, upper bound: 655.4342449
time: 0.72 seconds

## BFS NS instance: NS_B1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -273.7376404, 548.6153564, -260.3574524, 524.3128052, -798.0504150, 808.9725952
1: -96.6459045, 198.6981964, -92.0557709, 189.2300568, -285.8759766, 290.7539673
2: -59.8632851, 203.8120270, -56.8497543, 194.1286926, -253.9919739, 260.6617126
3: -120.9368057, 238.6856995, -115.2148514, 227.2112427, -348.1480408, 353.9005432
4: -65.8255768, 199.5302277, -62.5818367, 190.0472717, -255.8728485, 262.1120605

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_B1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4332683, upper bound: 655.4341960
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_B1_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336123, upper bound: 655.4343863
time: 0.71 seconds

## BFS NS instance: NS_B1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -273.7376404, 548.6153564, -271.6228943, 545.7624512, -819.5001221, 820.2382202
1: -96.6459045, 198.6981964, -95.9324493, 197.2268372, -293.8727417, 294.6306458
2: -59.8632851, 203.8120270, -59.3999634, 202.4023285, -262.2656250, 263.2119751
3: -120.9368057, 238.6856995, -120.0419235, 237.0750580, -358.0118103, 358.7276306
4: -65.8255768, 199.5302277, -65.3572845, 198.1024780, -263.9280396, 264.8875122

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_B1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333958, upper bound: 655.4341978
time: 0.81 seconds

## Relational analysis of NS_B1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_B1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336123, upper bound: 655.4343863
time: 1.05 seconds

## BFS NS instance: NS_B1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -272.7237854, 546.4223633, -278.8186646, 560.7500000, -833.4737549, 825.2408447
1: -96.2939987, 198.1040344, -98.6309891, 202.5439148, -298.8379211, 296.7349854
2: -59.6535110, 203.1464233, -61.0399284, 207.9895935, -267.6430969, 264.1863403
3: -120.5567093, 237.9588470, -123.3501282, 243.4347382, -363.9914246, 361.3089600
4: -65.5925903, 198.8743744, -67.1681137, 203.6446381, -269.2372437, 266.0424500

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_B1_A1_B2_A1_B1_B1

### Relational analysis result of NS_B1_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314053, upper bound: 655.4330759
time: 0.95 seconds

## Relational analysis of NS_B1_B1_A1_B2_A1_B1_B2

### Relational analysis result of NS_B1_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323409, upper bound: 655.4336187
time: 0.84 seconds

## BFS NS instance: NS_B1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -272.7237854, 546.4223633, -289.1075134, 582.6945801, -855.4183350, 835.5296631
1: -96.2939987, 198.1040344, -102.5586472, 210.4208221, -306.7148132, 300.6626892
2: -59.6535110, 203.1464233, -63.4853439, 216.1141663, -275.7676697, 266.6317749
3: -120.5567093, 237.9588470, -128.0482178, 252.9547729, -373.5114441, 366.0070801
4: -65.5925903, 198.8743744, -69.8493881, 211.4669800, -277.0595703, 268.7237244

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_B1_A1_B2_A1_B2_B1

### Relational analysis result of NS_B1_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314053, upper bound: 655.4335178
time: 0.73 seconds

## Relational analysis of NS_B1_B1_A1_B2_A1_B2_B2

### Relational analysis result of NS_B1_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323409, upper bound: 655.4342554
time: 0.80 seconds

## BFS NS instance: NS_B1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -281.0158386, 564.1822510, -278.9174805, 560.9663086, -841.9821777, 843.0997314
1: -99.4029770, 204.7459869, -98.6133804, 202.3361206, -301.7391052, 303.3593445
2: -61.6157951, 209.9355011, -61.0793839, 207.8713379, -269.4871216, 271.0148926
3: -124.2859650, 246.0430298, -123.3926010, 243.2054138, -367.4913330, 369.4356384
4: -67.7704926, 205.3149567, -67.1828232, 203.6216278, -271.3920898, 272.4977112

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_B2_A2_B1_B1

### Relational analysis result of NS_B1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316302, upper bound: 655.4331452
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A1_B2_A2_B1_B2

### Relational analysis result of NS_B1_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316302, upper bound: 655.4335764
time: 0.77 seconds

## BFS NS instance: NS_B1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -282.0775757, 566.3452148, -282.7156067, 568.7954102, -850.8729248, 849.0607910
1: -99.7751923, 205.5013123, -99.9797974, 205.2134399, -304.9886169, 305.4810486
2: -61.8409309, 210.7101898, -61.8927994, 210.7790985, -272.6200256, 272.6029358
3: -124.7443085, 246.9560089, -125.0602951, 246.6674500, -371.4117432, 372.0162354
4: -68.0225296, 206.0715027, -68.1032715, 206.4401855, -274.4626465, 274.1747131

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_B2_A2_B2_B1

### Relational analysis result of NS_B1_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4325584, upper bound: 655.4338029
time: 0.76 seconds

## Relational analysis of NS_B1_B1_A1_B2_A2_B2_B2

### Relational analysis result of NS_B1_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4325583, upper bound: 655.4344426
time: 0.72 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -358.0892639, 724.5586548, -264.2203064, 531.0654297, -889.1546631, 988.7788086
1: -127.5953369, 261.7943726, -93.3792038, 192.0017548, -319.5971069, 355.1735535
2: -79.0576553, 268.9154358, -57.8497429, 197.0623932, -276.1200562, 326.4978943
3: -158.7411652, 314.8903503, -116.8970184, 230.7603912, -389.5015564, 431.5182495
4: -87.0757523, 262.6092224, -63.6204910, 192.8621521, -279.9378967, 326.0623779

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_B1_A2_B1_B1_B1_A1

### Relational analysis result of NS_B1_B1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4303777, upper bound: 655.4322161
time: 0.69 seconds

## Relational analysis of NS_B1_B1_A2_B1_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4303777, upper bound: 655.4322515
time: 0.79 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -356.8777771, 721.9350586, -272.8213196, 549.2100830, -906.0878296, 994.7563477
1: -127.0806503, 260.7735596, -96.5865402, 198.7811432, -325.8617859, 357.3600769
2: -78.7605515, 267.8851929, -59.8684807, 203.9937744, -282.7543335, 327.4808960
3: -158.1207275, 313.7088318, -120.7469559, 239.0092468, -397.1299744, 434.2022400
4: -86.7475281, 261.6387024, -65.8625336, 199.4704437, -286.2179260, 327.3419189

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_B1_A2_B1_B1_B2_A1

### Relational analysis result of NS_B1_B1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316562, upper bound: 655.4325908
time: 0.79 seconds

## Relational analysis of NS_B1_B1_A2_B1_B1_B2_A2

### Relational analysis result of NS_B1_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316562, upper bound: 655.4328699
time: 0.76 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -360.8366699, 729.8001099, -274.8222656, 552.5510864, -913.3877563, 1004.6222534
1: -128.4980316, 263.7129211, -97.1951218, 199.5280609, -328.0260925, 360.9080505
2: -79.6198196, 270.8587646, -60.1831360, 204.9396973, -284.5594482, 330.7768860
3: -159.8999329, 317.2086792, -121.5920181, 239.8030701, -399.7030029, 438.5664062
4: -87.6998444, 264.5288391, -66.2012787, 200.6823578, -288.3822021, 330.5642395

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_B1_A2_B1_B2_B1_A1

### Relational analysis result of NS_B1_B1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316210, upper bound: 655.4323827
time: 0.72 seconds

## Relational analysis of NS_B1_B1_A2_B1_B2_B1_A2

### Relational analysis result of NS_B1_B1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316210, upper bound: 655.4323870
time: 0.74 seconds

## BFS NS instance: NS_B1_B1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -359.7346497, 727.4193726, -284.8130188, 573.9434814, -933.6781006, 1012.2323608
1: -128.0227966, 262.7738342, -101.0001450, 207.1848907, -335.2077026, 363.7739868
2: -79.3469696, 269.9114685, -62.5658722, 212.8457031, -292.1926575, 332.2055054
3: -159.3301849, 316.1292114, -126.1684113, 249.0541992, -408.3843994, 442.0726318
4: -87.3981857, 263.6412964, -68.8084869, 208.2987213, -295.6968994, 332.2814636

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_B1_A2_B1_B2_B2_A1

### Relational analysis result of NS_B1_B1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316514, upper bound: 655.4327720
time: 0.74 seconds

## Relational analysis of NS_B1_B1_A2_B1_B2_B2_A2

### Relational analysis result of NS_B1_B1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316514, upper bound: 655.4330682
time: 0.75 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -359.2760315, 726.9649048, -267.9685669, 538.6351929, -897.9112549, 994.9333496
1: -128.0086670, 262.6367798, -94.6975174, 194.8126526, -322.8213196, 357.3342896
2: -79.3092499, 269.7794189, -58.6487007, 199.9044189, -279.2136230, 328.1611633
3: -159.2525940, 315.9053040, -118.5426865, 234.1375275, -393.3901367, 434.1716919
4: -87.3562088, 263.4555054, -64.5218048, 195.6266785, -282.9828796, 327.8086243

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_B1_A2_B2_B1_B1_A1

### Relational analysis result of NS_B1_B1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4300484, upper bound: 655.4321810
time: 0.74 seconds

## Relational analysis of NS_B1_B1_A2_B2_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4300484, upper bound: 655.4322163
time: 0.90 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -358.0590515, 724.3337402, -276.6891785, 557.0789185, -915.1379395, 1001.0228882
1: -127.4921112, 261.6124878, -97.9491196, 201.6930237, -329.1851196, 359.5615540
2: -79.0107956, 268.7461853, -60.6916618, 206.9447784, -285.9555664, 329.1652222
3: -158.6297913, 314.7205811, -122.4410248, 242.5167542, -401.1464844, 436.9016724
4: -87.0264893, 262.4822388, -66.7936172, 202.3331146, -289.3596191, 329.1158142

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B1_A2_B2_B1_B2_A1

### Relational analysis result of NS_B1_B1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4318953, upper bound: 655.4310731
time: 0.79 seconds

## Relational analysis of NS_B1_B1_A2_B2_B1_B2_A2

### Relational analysis result of NS_B1_B1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4321830, upper bound: 655.4330203
time: 0.96 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -361.9813538, 732.1271362, -278.5673523, 560.2707520, -922.2520752, 1010.6943359
1: -128.8971252, 264.5269165, -98.5458145, 202.3677521, -331.2648926, 363.0727234
2: -79.8624878, 271.6940613, -60.9862442, 207.8115540, -287.6739197, 332.4153748
3: -160.3941803, 318.1905518, -123.2419128, 243.2218018, -403.6159363, 441.1907349
4: -87.9704132, 265.3473511, -67.1090775, 203.4705963, -291.4410095, 332.2900696

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_B1_A2_B2_B2_B1_A1

### Relational analysis result of NS_B1_B1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320715, upper bound: 655.4325083
time: 0.74 seconds

## Relational analysis of NS_B1_B1_A2_B2_B2_B1_A2

### Relational analysis result of NS_B1_B1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4320715, upper bound: 655.4326825
time: 0.86 seconds

## BFS NS instance: NS_B1_B1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -360.8804321, 729.7465210, -288.8639221, 582.2291870, -943.1095581, 1018.6103516
1: -128.4223328, 263.5925903, -102.4757614, 210.2490692, -338.6713867, 366.0683289
2: -79.5902939, 270.7486572, -63.4330521, 215.9407196, -295.5310059, 333.9102783
3: -159.8252716, 317.1156616, -127.9428864, 252.7472534, -412.5725098, 444.8318176
4: -87.6697083, 264.4605713, -69.7919235, 211.2976685, -298.9672852, 334.0840759

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B1_A2_B2_B2_B2_A1

### Relational analysis result of NS_B1_B1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4324690, upper bound: 655.4313463
time: 0.96 seconds

## Relational analysis of NS_B1_B1_A2_B2_B2_B2_A2

### Relational analysis result of NS_B1_B1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4327072, upper bound: 655.4335386
time: 0.81 seconds

## BFS NS instance: NS_B1_B2_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -268.7559204, 539.6497192, -334.6813049, 679.8351440, -948.5910645, 874.3310547
1: -94.9280472, 195.1009216, -119.6903687, 245.9018707, -340.8299255, 314.7912903
2: -58.8233414, 200.2166901, -74.1331329, 252.6116028, -311.1163940, 274.3498230
3: -118.7789154, 234.5239563, -148.9267578, 295.8482666, -414.2677307, 383.4506531
4: -64.7233734, 195.9302063, -81.7196350, 246.3467255, -310.8638916, 277.6498413

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_B1

### Relational analysis result of NS_B1_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323258, upper bound: 655.4322715
time: 0.80 seconds

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_B2

### Relational analysis result of NS_B1_B2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323258, upper bound: 655.4323934
time: 0.80 seconds

## BFS NS instance: NS_B1_B2_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -267.6928406, 536.7030640, -334.6813049, 679.8351440, -947.5279541, 871.3842773
1: -94.5441666, 194.4291229, -119.6903687, 245.9018707, -340.4460144, 314.1195068
2: -58.6300850, 199.4470978, -74.1331329, 252.6116028, -310.9124146, 273.5802307
3: -118.3338470, 233.6017151, -148.9267578, 295.8482666, -413.7951965, 382.5284119
4: -64.4664764, 195.1890411, -81.7196350, 246.3467255, -310.6037292, 276.9086304

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_B1

### Relational analysis result of NS_B1_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323258, upper bound: 655.4322715
time: 0.75 seconds

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_B2

### Relational analysis result of NS_B1_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4323258, upper bound: 655.4323934
time: 0.74 seconds

## BFS NS instance: NS_B1_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -271.0392151, 544.4196777, -335.8776245, 682.2878418, -953.3270264, 880.2973022
1: -95.7547150, 196.9387054, -120.1094589, 246.7636414, -342.5183716, 317.0481567
2: -59.3096428, 202.0564575, -74.3869476, 253.4926453, -312.4859009, 276.4434204
3: -119.8125153, 236.6984711, -149.4456329, 296.8889465, -416.3399963, 386.1441040
4: -65.2738190, 197.7051849, -82.0037994, 247.2059021, -312.2731934, 279.7089844

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_A1_B1_A2_A1_B1

### Relational analysis result of NS_B1_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4321175, upper bound: 655.4328924
time: 0.78 seconds

## Relational analysis of NS_B1_B2_A1_B1_A2_A1_B2

### Relational analysis result of NS_B1_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330005, upper bound: 655.4330005
time: 0.77 seconds

## BFS NS instance: NS_B1_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -269.4585876, 540.2991943, -335.8776245, 682.2878418, -951.7463989, 876.1768188
1: -95.2246552, 195.8273163, -120.1094589, 246.7636414, -341.9882812, 315.9367676
2: -59.0033798, 200.8528900, -74.3869476, 253.4926453, -312.1709595, 275.2398376
3: -119.1378708, 235.2214050, -149.4456329, 296.8889465, -415.6389465, 384.6670227
4: -64.8904037, 196.5644226, -82.0037994, 247.2059021, -311.8843079, 278.5682373

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_A1_B1_A2_A2_B1

### Relational analysis result of NS_B1_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4321175, upper bound: 655.4334467
time: 0.75 seconds

## Relational analysis of NS_B1_B2_A1_B1_A2_A2_B2

### Relational analysis result of NS_B1_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330005, upper bound: 655.4339942
time: 0.79 seconds

## BFS NS instance: NS_B1_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -266.4229736, 535.4568481, -504.4446411, 1030.4354248, -1289.0963135, 1039.9014893
1: -94.1628113, 193.7279510, -181.3674316, 370.8659363, -463.8123779, 375.0953979
2: -58.3254967, 198.7800446, -112.4062195, 381.4328308, -437.2626038, 311.1861877
3: -117.8213120, 232.8428650, -224.1789856, 447.2264404, -562.1238403, 457.0218506
4: -64.1855774, 194.4808502, -124.2494507, 371.6995239, -434.1139832, 318.7302856

Time for backsubstitution: 2.20 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.19 + 416.61 = 420.80 seconds
