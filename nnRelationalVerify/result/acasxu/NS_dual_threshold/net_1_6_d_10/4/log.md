## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 27.198190501800003


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.9076786, 12.5877399, -4.9076786, 12.5877399, -17.4954185, 17.4954185)
1: (-13.2235098, 19.0510464, -13.2235098, 19.0510464, -32.2745552, 32.2745476)
2: (-8.4530544, 17.2152138, -8.4530544, 17.2152138, -25.6682663, 25.6682663)
3: (-14.9454336, 21.4810410, -14.9454336, 21.4810410, -36.4264717, 36.4264717)
4: (-10.1239004, 20.9261990, -10.1239004, 20.9261990, -31.0500984, 31.0500984)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.72 + 1.89 = 2.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -27.2117964, upper bound: 27.2117964

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1974623, upper bound: 27.1946267
time: 1.65 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1974623, upper bound: 27.2117964
time: 0.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.32 seconds
NS_B1, status: Status.VERIFIED, split count: 1, time: 2.32
Output dim: 4, lower bound: -27.1974623, upper bound: 27.1946267
NS_B2, status: Status.UNKNOWN, split count: 1, time: 2.32
Output dim: 4, lower bound: -27.1974623, upper bound: 27.2117964

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -4.8985047, 12.5633020, -4.7271943, 12.1057043, -17.0042095, 17.2904968
1: -13.1985741, 19.0151119, -12.7328453, 18.3434982, -31.5420723, 31.7479553
2: -8.4372826, 17.1821251, -8.1425447, 16.5627804, -25.0000629, 25.3246651
3: -14.9168396, 21.4397507, -14.3829765, 20.6675854, -35.5844231, 35.8227272
4: -10.1049128, 20.8857403, -9.7501955, 20.1286144, -30.2335224, 30.6359329

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2096359, upper bound: 27.2091887
time: 0.65 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2091721, upper bound: 27.2091721
time: 0.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.07 seconds
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 4, lower bound: -27.2096359, upper bound: 27.2091887
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 4, lower bound: -27.2091721, upper bound: 27.2091721

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -4.5814300, 11.7579050, -4.7271943, 12.1057043, -16.6871338, 16.4850998
1: -12.4122257, 17.9194889, -12.7328453, 18.3434982, -30.7557220, 30.6523342
2: -7.9391966, 16.1215668, -8.1425447, 16.5627804, -24.5019760, 24.2641106
3: -13.9956961, 20.1080360, -14.3829765, 20.6675854, -34.6632805, 34.4910126
4: -9.4861622, 19.5885639, -9.7501955, 20.1286144, -29.6147709, 29.3387566

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2091694, upper bound: 27.2091694
time: 0.63 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2091694, upper bound: 27.2091694
time: 0.91 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -4.6628623, 11.8867149, -4.7164869, 12.0766678, -16.7395306, 16.6032028
1: -12.5339441, 18.0045204, -12.7029562, 18.3003063, -30.8342495, 30.7074776
2: -7.9853606, 16.2999954, -8.1233330, 16.5233879, -24.5087490, 24.4233284
3: -14.1708450, 20.3100052, -14.3490124, 20.6180573, -34.7889023, 34.6590118
4: -9.5720787, 19.8279953, -9.7269678, 20.0820217, -29.6540966, 29.5549622

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1905461, upper bound: 27.2091721
time: 0.73 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1905461, upper bound: 27.2091721
time: 0.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.07 seconds
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 4, lower bound: -27.2091694, upper bound: 27.2091694
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 4, lower bound: -27.2091694, upper bound: 27.2091694
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 4, lower bound: -27.1905461, upper bound: 27.2091721
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 4, lower bound: -27.1905461, upper bound: 27.2091721

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.5814300, 11.7579050, -4.4116325, 11.3064547, -15.8878841, 16.1695366
1: -12.4122257, 17.9194889, -11.9514399, 17.2588844, -29.6711063, 29.8709297
2: -7.9391966, 16.1215668, -7.6472921, 15.5107203, -23.4499168, 23.7688599
3: -13.9956961, 20.1080360, -13.4661083, 19.3460236, -33.3417168, 33.5741425
4: -9.4861622, 19.5885639, -9.1350269, 18.8462715, -28.3324337, 28.7235909

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1791607, upper bound: 27.1937847
time: 0.59 seconds

## Relational analysis of NS_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1791607, upper bound: 27.2082043
time: 1.00 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.5814300, 11.7579050, -4.4915919, 11.4322100, -16.0136395, 16.2494965
1: -12.4122257, 17.9194889, -12.0714865, 17.3388920, -29.7511158, 29.9909744
2: -7.9391966, 16.1215668, -7.6917596, 15.6831150, -23.6223106, 23.8133259
3: -13.9956961, 20.1080360, -13.6411037, 19.5463009, -33.5419960, 33.7491379
4: -9.4861622, 19.5885639, -9.2179928, 19.0775623, -28.5637207, 28.8065567

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1791607, upper bound: 27.1937847
time: 0.96 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1791607, upper bound: 27.1752221
time: 0.59 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.6628623, 11.8867149, -4.4116325, 11.3064547, -15.9693165, 16.2983475
1: -12.5339441, 18.0045204, -11.9514399, 17.2588844, -29.7928276, 29.9559593
2: -7.9853606, 16.2999954, -7.6472921, 15.5107203, -23.4960804, 23.9472885
3: -14.1708450, 20.3100052, -13.4661083, 19.3460236, -33.5168648, 33.7761116
4: -9.5720787, 19.8279953, -9.1350269, 18.8462715, -28.4183502, 28.9630222

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1736874, upper bound: 27.1904783
time: 0.60 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1736874, upper bound: 27.1736874
time: 0.93 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.6628623, 11.8867149, -4.4915919, 11.4322100, -16.0950718, 16.3783073
1: -12.5339441, 18.0045204, -12.0714865, 17.3388920, -29.8728371, 30.0760078
2: -7.9853606, 16.2999954, -7.6917596, 15.6831150, -23.6684761, 23.9917545
3: -14.1708450, 20.3100052, -13.6411037, 19.5463009, -33.7171478, 33.9510994
4: -9.5720787, 19.8279953, -9.2179928, 19.0775623, -28.6496372, 29.0459862

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1736874, upper bound: 27.1904783
time: 0.93 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1736874, upper bound: 27.2083343
time: 0.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.68 seconds
NS_B2_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 4, lower bound: -27.1791607, upper bound: 27.1937847
NS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 4, lower bound: -27.1791607, upper bound: 27.2082043
NS_B2_A1_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 4, lower bound: -27.1791607, upper bound: 27.1937847
NS_B2_A1_B2_A2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 4, lower bound: -27.1791607, upper bound: 27.1752221
NS_B2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 4, lower bound: -27.1736874, upper bound: 27.1904783
NS_B2_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 4, lower bound: -27.1736874, upper bound: 27.1736874
NS_B2_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 4, lower bound: -27.1736874, upper bound: 27.1904783
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 4, lower bound: -27.1736874, upper bound: 27.2083343

## BFS NS instance: NS_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.4116325, 11.3064547, -4.4116325, 11.3064547, -15.7180872, 15.7180872
1: -11.9514399, 17.2588844, -11.9514399, 17.2588844, -29.2103233, 29.2103233
2: -7.6472921, 15.5107203, -7.6472921, 15.5107203, -23.1580124, 23.1580124
3: -13.4661083, 19.3460236, -13.4661083, 19.3460236, -32.8121300, 32.8121300
4: -9.1350269, 18.8462715, -9.1350269, 18.8462715, -27.9812984, 27.9812984

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_B1_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1794016, upper bound: 27.1802377
time: 0.58 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2

### Relational analysis result of NS_B2_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1794276, upper bound: 27.1794276
time: 0.64 seconds

## BFS NS instance: NS_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.4915919, 11.4322100, -4.4915919, 11.4322100, -15.9237986, 15.9237995
1: -12.0714865, 17.3388920, -12.0714865, 17.3388920, -29.4103775, 29.4103775
2: -7.6917596, 15.6831150, -7.6917596, 15.6831150, -23.3748741, 23.3748741
3: -13.6411037, 19.5463009, -13.6411037, 19.5463009, -33.1874046, 33.1874046
4: -9.2179928, 19.0775623, -9.2179928, 19.0775623, -28.2955532, 28.2955532

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1734399, upper bound: 27.1727492
time: 0.62 seconds

## Relational analysis of NS_B2_A2_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1721118, upper bound: 27.1725018
time: 0.84 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.87 seconds
NS_B2_A1_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 2.87
Output dim: 4, lower bound: -27.1794016, upper bound: 27.1802377
NS_B2_A1_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 2.87
Output dim: 4, lower bound: -27.1794276, upper bound: 27.1794276
NS_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.87
Output dim: 4, lower bound: -27.1734399, upper bound: 27.1727492
NS_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.87
Output dim: 4, lower bound: -27.1721118, upper bound: 27.1725018

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 2.60 + 24.61 = 27.21 seconds
