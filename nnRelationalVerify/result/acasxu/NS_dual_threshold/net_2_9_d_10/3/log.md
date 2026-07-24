## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.005605665000000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0132039, 0.0242686, 0.0132039, 0.0242686, -0.0110647, 0.0110647)
1: (-0.0417597, -0.0198232, -0.0417597, -0.0198232, -0.0219365, 0.0219365)
2: (-0.0483784, 0.0230394, -0.0483784, 0.0230394, -0.0714179, 0.0714179)
3: (-0.0934542, -0.0138781, -0.0934542, -0.0138781, -0.0795761, 0.0795761)
4: (-0.0491619, 0.0238405, -0.0491619, 0.0238405, -0.0730024, 0.0730024)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.95 + 0.69 = 3.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0086241, upper bound: 0.0086241

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0086013, upper bound: 0.0070428
time: 0.30 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074224, upper bound: 0.0074224
time: 0.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.84 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -0.0086013, upper bound: 0.0070428
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -0.0074224, upper bound: 0.0074224

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0137908, 0.0234117, 0.0132039, 0.0242686, -0.0104777, 0.0102078
1: -0.0411373, -0.0198371, -0.0417597, -0.0198232, -0.0213140, 0.0219226
2: -0.0464529, 0.0216638, -0.0483784, 0.0230394, -0.0694924, 0.0700423
3: -0.0912642, -0.0147501, -0.0934542, -0.0138781, -0.0773862, 0.0787041
4: -0.0471789, 0.0224950, -0.0491619, 0.0238405, -0.0710194, 0.0716570

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085848, upper bound: 0.0053132
time: 0.30 seconds

## Relational analysis of NS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0070428, upper bound: 0.0070428
time: 0.30 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0070428, upper bound: 0.0070428
time: 0.30 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0026806, 0.0246433, 0.0132039, 0.0242686, -0.0215879, 0.0114394
1: -0.0605749, -0.0195432, -0.0417597, -0.0198232, -0.0407517, 0.0222165
2: -0.1077064, 0.0230471, -0.0483784, 0.0230394, -0.1307458, 0.0714255
3: -0.1609610, -0.0138797, -0.0934542, -0.0138781, -0.1470829, 0.0795746
4: -0.1101796, 0.0238355, -0.0491619, 0.0238405, -0.1340201, 0.0729974

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074224, upper bound: 0.0056785
time: 0.30 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0057169, upper bound: 0.0057169
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.57 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 0, lower bound: -0.0070428, upper bound: 0.0070428
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 0, lower bound: -0.0070428, upper bound: 0.0070428
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 0, lower bound: -0.0074224, upper bound: 0.0056785
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 0, lower bound: -0.0057169, upper bound: 0.0057169

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0137908, 0.0234117, 0.0137908, 0.0234117, -0.0096209, 0.0096209
1: -0.0411373, -0.0198371, -0.0411373, -0.0198371, -0.0213002, 0.0213002
2: -0.0464529, 0.0216638, -0.0464529, 0.0216638, -0.0681168, 0.0681168
3: -0.0912642, -0.0147501, -0.0912642, -0.0147501, -0.0765141, 0.0765141
4: -0.0471789, 0.0224950, -0.0471789, 0.0224950, -0.0696740, 0.0696740

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0084370, upper bound: 0.0068692
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0082915, upper bound: 0.0070412
time: 0.30 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.0137908, 0.0234117, 0.0026806, 0.0246433, -0.0108525, 0.0207311
1: -0.0411373, -0.0198371, -0.0605749, -0.0195432, -0.0215941, 0.0407378
2: -0.0464529, 0.0216638, -0.1077064, 0.0230471, -0.0695000, 0.1293702
3: -0.0912642, -0.0147501, -0.1609610, -0.0138797, -0.0773846, 0.1462108
4: -0.0471789, 0.0224950, -0.1101796, 0.0238355, -0.0710144, 0.1326746

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0084304, upper bound: 0.0069963
time: 0.30 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0084304, upper bound: 0.0069234
time: 0.30 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: 0.0028268, 0.0244372, 0.0132039, 0.0242686, -0.0214418, 0.0112333
1: -0.0604125, -0.0195467, -0.0417597, -0.0198232, -0.0405892, 0.0222130
2: -0.1071979, 0.0228221, -0.0483784, 0.0230394, -0.1302373, 0.0712005
3: -0.1603829, -0.0140183, -0.0934542, -0.0138781, -0.1465048, 0.0794359
4: -0.1096565, 0.0236184, -0.0491619, 0.0238405, -0.1334970, 0.0727804

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0073758, upper bound: 0.0055936
time: 0.30 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0073029, upper bound: 0.0055936
time: 0.30 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.0334639, 0.0259453, 0.0132039, 0.0242686, -0.0577324, 0.0127414
1: -0.1134679, -0.0184130, -0.0417597, -0.0198232, -0.0936447, 0.0233467
2: -0.2750255, 0.0231532, -0.0483784, 0.0230394, -0.2980649, 0.0715316
3: -0.3512858, -0.0138200, -0.0934542, -0.0138781, -0.3374077, 0.0796342
4: -0.2822623, 0.0239249, -0.0491619, 0.0238405, -0.3061028, 0.0730868

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0056497, upper bound: 0.0056240
time: 0.30 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055567, upper bound: 0.0055567
time: 0.30 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.66 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -0.0084370, upper bound: 0.0068692
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -0.0082915, upper bound: 0.0070412
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -0.0084304, upper bound: 0.0069963
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -0.0084304, upper bound: 0.0069234
NS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -0.0073758, upper bound: 0.0055936
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -0.0073029, upper bound: 0.0055936
NS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -0.0056497, upper bound: 0.0056240
NS_A2_A2_A2, status: Status.VERIFIED, split count: 3, time: 4.66
Output dim: 0, lower bound: -0.0055567, upper bound: 0.0055567

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0151135, 0.0231104, 0.0137908, 0.0234117, -0.0082982, 0.0093195
1: -0.0381770, -0.0198591, -0.0411373, -0.0198371, -0.0183399, 0.0212781
2: -0.0371663, 0.0212939, -0.0464529, 0.0216638, -0.0588302, 0.0677468
3: -0.0806948, -0.0149684, -0.0912642, -0.0147501, -0.0659447, 0.0762958
4: -0.0376278, 0.0221459, -0.0471789, 0.0224950, -0.0601228, 0.0693248

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081196, upper bound: 0.0081196
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081196, upper bound: 0.0081196
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0138521, 0.0233853, 0.0137908, 0.0234117, -0.0095596, 0.0095945
1: -0.0410483, -0.0198391, -0.0411373, -0.0198371, -0.0212112, 0.0212982
2: -0.0461732, 0.0216600, -0.0464529, 0.0216638, -0.0678370, 0.0681130
3: -0.0909458, -0.0147526, -0.0912642, -0.0147501, -0.0761956, 0.0765116
4: -0.0468910, 0.0224914, -0.0471789, 0.0224950, -0.0693861, 0.0696703

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081196, upper bound: 0.0082915
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081196, upper bound: 0.0082915
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0137908, 0.0234117, 0.0059873, 0.0241804, -0.0103895, 0.0174244
1: -0.0411373, -0.0198371, -0.0552418, -0.0196470, -0.0214903, 0.0354047
2: -0.0464529, 0.0216638, -0.0908982, 0.0227108, -0.0691638, 0.1125620
3: -0.0912642, -0.0147501, -0.1418434, -0.0141165, -0.0771478, 0.1270933
4: -0.0471789, 0.0224950, -0.0928955, 0.0235024, -0.0706813, 0.1153906

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0084117, upper bound: 0.0066236
time: 0.30 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085773, upper bound: 0.0069908
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0137908, 0.0234117, 0.0027399, 0.0246412, -0.0108503, 0.0206718
1: -0.0411373, -0.0198371, -0.0603256, -0.0195441, -0.0215932, 0.0404885
2: -0.0464529, 0.0216638, -0.1069151, 0.0230461, -0.0694990, 0.1285790
3: -0.0912642, -0.0147501, -0.1600597, -0.0138851, -0.0773792, 0.1453096
4: -0.0471789, 0.0224950, -0.1093657, 0.0238346, -0.0710135, 0.1318608

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0084117, upper bound: 0.0065507
time: 0.32 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085779, upper bound: 0.0069179
time: 0.31 seconds

## BFS NS instance: NS_A2_A1_A1

### Backsubstitution after applying NS history:
0: 0.0061268, 0.0239658, 0.0132039, 0.0242686, -0.0181418, 0.0107619
1: -0.0550959, -0.0196506, -0.0417597, -0.0198232, -0.0352727, 0.0221091
2: -0.0904407, 0.0224827, -0.0483784, 0.0230394, -0.1134802, 0.0708611
3: -0.1413234, -0.0142590, -0.0934542, -0.0138781, -0.1274453, 0.0791952
4: -0.0924248, 0.0232809, -0.0491619, 0.0238405, -0.1162653, 0.0724429

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_A1_B1

### Relational analysis result of NS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0073758, upper bound: 0.0055530
time: 0.30 seconds

## Relational analysis of NS_A2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_A1_A1

### Relational analysis result of NS_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0073646, upper bound: 0.0055457
time: 0.30 seconds

## Relational analysis of NS_A2_A1_A1_A2

### Relational analysis result of NS_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062465, upper bound: 0.0055783
time: 0.31 seconds

## BFS NS instance: NS_A2_A1_A2

### Backsubstitution after applying NS history:
0: 0.0028725, 0.0244353, 0.0132039, 0.0242686, -0.0213961, 0.0112314
1: -0.0601751, -0.0195475, -0.0417597, -0.0198232, -0.0403518, 0.0222122
2: -0.1064619, 0.0228211, -0.0483784, 0.0230394, -0.1295013, 0.0711996
3: -0.1595444, -0.0140252, -0.0934542, -0.0138781, -0.1456664, 0.0794290
4: -0.1088992, 0.0236165, -0.0491619, 0.0238405, -0.1327397, 0.0727785

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0073029, upper bound: 0.0053873
time: 0.30 seconds

## Relational analysis of NS_A2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_A2_A1

### Relational analysis result of NS_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0072916, upper bound: 0.0055544
time: 0.30 seconds

## Relational analysis of NS_A2_A1_A2_A2

### Relational analysis result of NS_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0060439, upper bound: 0.0055783
time: 0.30 seconds

## BFS NS instance: NS_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0332136, 0.0255500, 0.0132039, 0.0242686, -0.0574822, 0.0123461
1: -0.1121544, -0.0184184, -0.0417597, -0.0198232, -0.0923312, 0.0233413
2: -0.2709423, 0.0228106, -0.0483784, 0.0230394, -0.2939818, 0.0711890
3: -0.3466368, -0.0140281, -0.0934542, -0.0138781, -0.3327588, 0.0794261
4: -0.2780625, 0.0235877, -0.0491619, 0.0238405, -0.3019029, 0.0727496

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A2_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055567, upper bound: 0.0055567
time: 0.30 seconds

## Relational analysis of NS_A2_A2_A1_B2

### Relational analysis result of NS_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0055567, upper bound: 0.0055567
time: 0.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.85 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.85
Output dim: 0, lower bound: -0.0081196, upper bound: 0.0081196
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.85
Output dim: 0, lower bound: -0.0081196, upper bound: 0.0081196
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.85
Output dim: 0, lower bound: -0.0081196, upper bound: 0.0082915
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.85
Output dim: 0, lower bound: -0.0081196, upper bound: 0.0082915
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.85
Output dim: 0, lower bound: -0.0084117, upper bound: 0.0066236
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.85
Output dim: 0, lower bound: -0.0085773, upper bound: 0.0069908
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.85
Output dim: 0, lower bound: -0.0084117, upper bound: 0.0065507
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 6.85
Output dim: 0, lower bound: -0.0085779, upper bound: 0.0069179
NS_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 6.85
Output dim: 0, lower bound: -0.0073646, upper bound: 0.0055457
NS_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 6.85
Output dim: 0, lower bound: -0.0062465, upper bound: 0.0055783
NS_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.85
Output dim: 0, lower bound: -0.0072916, upper bound: 0.0055544
NS_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 6.85
Output dim: 0, lower bound: -0.0060439, upper bound: 0.0055783
NS_A2_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 6.85
Output dim: 0, lower bound: -0.0055567, upper bound: 0.0055567
NS_A2_A2_A1_B2, status: Status.VERIFIED, split count: 4, time: 6.85
Output dim: 0, lower bound: -0.0055567, upper bound: 0.0055567

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0151135, 0.0231104, 0.0151135, 0.0231104, -0.0079969, 0.0079969
1: -0.0381770, -0.0198591, -0.0381770, -0.0198591, -0.0183179, 0.0183179
2: -0.0371663, 0.0212939, -0.0371663, 0.0212939, -0.0584602, 0.0584602
3: -0.0806948, -0.0149684, -0.0806948, -0.0149684, -0.0657264, 0.0657264
4: -0.0376278, 0.0221459, -0.0376278, 0.0221459, -0.0597737, 0.0597737

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081738, upper bound: 0.0078021
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0082513, upper bound: 0.0081059
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0151135, 0.0231104, 0.0138521, 0.0233853, -0.0082718, 0.0092583
1: -0.0381770, -0.0198591, -0.0410483, -0.0198391, -0.0183379, 0.0211891
2: -0.0371663, 0.0212939, -0.0461732, 0.0216600, -0.0588264, 0.0674670
3: -0.0806948, -0.0149684, -0.0909458, -0.0147526, -0.0659422, 0.0759774
4: -0.0376278, 0.0221459, -0.0468910, 0.0224914, -0.0601192, 0.0690369

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079476, upper bound: 0.0079916
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0082513, upper bound: 0.0081059
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0138521, 0.0233853, 0.0151135, 0.0231104, -0.0092583, 0.0082718
1: -0.0410483, -0.0198391, -0.0381770, -0.0198591, -0.0211891, 0.0183379
2: -0.0461732, 0.0216600, -0.0371663, 0.0212939, -0.0674670, 0.0588264
3: -0.0909458, -0.0147526, -0.0806948, -0.0149684, -0.0759774, 0.0659422
4: -0.0468910, 0.0224914, -0.0376278, 0.0221459, -0.0690369, 0.0601192

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079916, upper bound: 0.0078876
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081059, upper bound: 0.0082860
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0138521, 0.0233853, 0.0138521, 0.0233853, -0.0095332, 0.0095332
1: -0.0410483, -0.0198391, -0.0410483, -0.0198391, -0.0212092, 0.0212092
2: -0.0461732, 0.0216600, -0.0461732, 0.0216600, -0.0678332, 0.0678332
3: -0.0909458, -0.0147526, -0.0909458, -0.0147526, -0.0761932, 0.0761932
4: -0.0468910, 0.0224914, -0.0468910, 0.0224914, -0.0693825, 0.0693825

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079916, upper bound: 0.0078876
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081059, upper bound: 0.0082860
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0128208, 0.0227821, 0.0059873, 0.0241804, -0.0113596, 0.0167947
1: -0.0433761, -0.0198199, -0.0552418, -0.0196470, -0.0237292, 0.0354219
2: -0.0535134, 0.0211087, -0.0908982, 0.0227108, -0.0762242, 0.1120069
3: -0.0992958, -0.0150780, -0.1418434, -0.0141165, -0.0851793, 0.1267654
4: -0.0544364, 0.0219662, -0.0928955, 0.0235024, -0.0779387, 0.1148617

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081025, upper bound: 0.0065881
time: 0.31 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081906, upper bound: 0.0065563
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0138056, 0.0233843, 0.0059873, 0.0241804, -0.0103748, 0.0173970
1: -0.0411188, -0.0198374, -0.0552418, -0.0196470, -0.0214719, 0.0354044
2: -0.0463957, 0.0216158, -0.0908982, 0.0227108, -0.0691065, 0.1125140
3: -0.0911990, -0.0147792, -0.1418434, -0.0141165, -0.0770825, 0.1270642
4: -0.0471199, 0.0224490, -0.0928955, 0.0235024, -0.0706223, 0.1153445

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085436, upper bound: 0.0069390
time: 0.31 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085433, upper bound: 0.0069614
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0128208, 0.0227821, 0.0027399, 0.0246412, -0.0118204, 0.0200422
1: -0.0433761, -0.0198199, -0.0603256, -0.0195441, -0.0238320, 0.0405057
2: -0.0535134, 0.0211087, -0.1069151, 0.0230461, -0.0765594, 0.1280238
3: -0.0992958, -0.0150780, -0.1600597, -0.0138851, -0.0854107, 0.1449817
4: -0.0544364, 0.0219662, -0.1093657, 0.0238346, -0.0782709, 0.1313319

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080273, upper bound: 0.0065237
time: 0.31 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081906, upper bound: 0.0064893
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0138056, 0.0233843, 0.0027399, 0.0246412, -0.0108355, 0.0206445
1: -0.0411188, -0.0198374, -0.0603256, -0.0195441, -0.0215747, 0.0404881
2: -0.0463957, 0.0216158, -0.1069151, 0.0230461, -0.0694418, 0.1285309
3: -0.0911990, -0.0147792, -0.1600597, -0.0138851, -0.0773139, 0.1452804
4: -0.0471199, 0.0224490, -0.1093657, 0.0238346, -0.0709545, 0.1318147

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080273, upper bound: 0.0068894
time: 0.31 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0082003, upper bound: 0.0068550
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_A1_A1

### Backsubstitution after applying NS history:
0: 0.0061984, 0.0238467, 0.0132039, 0.0242686, -0.0180702, 0.0106428
1: -0.0550196, -0.0196523, -0.0417597, -0.0198232, -0.0351964, 0.0221074
2: -0.0902050, 0.0223220, -0.0483784, 0.0230394, -0.1132444, 0.0707004
3: -0.1410551, -0.0143647, -0.0934542, -0.0138781, -0.1271770, 0.0790895
4: -0.0921818, 0.0231219, -0.0491619, 0.0238405, -0.1160222, 0.0722839

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_A1_A1_B1

### Relational analysis result of NS_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0073630, upper bound: 0.0055193
time: 0.31 seconds

## Relational analysis of NS_A2_A1_A1_A1_B2

### Relational analysis result of NS_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0073630, upper bound: 0.0055457
time: 0.31 seconds

## BFS NS instance: NS_A2_A1_A1_A2

### Backsubstitution after applying NS history:
0: 0.0077262, 0.0238715, 0.0132039, 0.0242686, -0.0165424, 0.0106676
1: -0.0517102, -0.0196804, -0.0417597, -0.0198232, -0.0318869, 0.0220793
2: -0.0797982, 0.0223627, -0.0483784, 0.0230394, -0.1028377, 0.0707412
3: -0.1292162, -0.0143354, -0.0934542, -0.0138781, -0.1153381, 0.0791188
4: -0.0814825, 0.0231642, -0.0491619, 0.0238405, -0.1053230, 0.0723261

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_A1_A2_B1

### Relational analysis result of NS_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062449, upper bound: 0.0055519
time: 0.31 seconds

## Relational analysis of NS_A2_A1_A1_A2_B2

### Relational analysis result of NS_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062449, upper bound: 0.0055783
time: 0.30 seconds

## BFS NS instance: NS_A2_A1_A2_A1

### Backsubstitution after applying NS history:
0: 0.0029260, 0.0243077, 0.0132039, 0.0242686, -0.0213426, 0.0111038
1: -0.0601382, -0.0195490, -0.0417597, -0.0198232, -0.0403150, 0.0222107
2: -0.1063517, 0.0226419, -0.0483784, 0.0230394, -0.1293911, 0.0710204
3: -0.1594193, -0.0141431, -0.0934542, -0.0138781, -0.1455413, 0.0793111
4: -0.1087855, 0.0234392, -0.0491619, 0.0238405, -0.1326259, 0.0726011

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_A2_A1_B1

### Relational analysis result of NS_A2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0072900, upper bound: 0.0055279
time: 0.31 seconds

## Relational analysis of NS_A2_A1_A2_A1_B2

### Relational analysis result of NS_A2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0072900, upper bound: 0.0055415
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_A2_A2

### Backsubstitution after applying NS history:
0: 0.0045775, 0.0243572, 0.0132039, 0.0242686, -0.0196911, 0.0111533
1: -0.0567702, -0.0195787, -0.0417597, -0.0198232, -0.0369469, 0.0221811
2: -0.0957376, 0.0227800, -0.0483784, 0.0230394, -0.1187770, 0.0711584
3: -0.1473465, -0.0140503, -0.0934542, -0.0138781, -0.1334684, 0.0794039
4: -0.0978744, 0.0235783, -0.0491619, 0.0238405, -0.1217148, 0.0727402

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_A2_A2_B1

### Relational analysis result of NS_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0060423, upper bound: 0.0055519
time: 0.32 seconds

## Relational analysis of NS_A2_A1_A2_A2_B2

### Relational analysis result of NS_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0060423, upper bound: 0.0055626
time: 0.31 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.68 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0081738, upper bound: 0.0078021
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0082513, upper bound: 0.0081059
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0079476, upper bound: 0.0079916
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0082513, upper bound: 0.0081059
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0079916, upper bound: 0.0078876
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0081059, upper bound: 0.0082860
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0079916, upper bound: 0.0078876
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0081059, upper bound: 0.0082860
NS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0081025, upper bound: 0.0065881
NS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0081906, upper bound: 0.0065563
NS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0085436, upper bound: 0.0069390
NS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0085433, upper bound: 0.0069614
NS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0080273, upper bound: 0.0065237
NS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0081906, upper bound: 0.0064893
NS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0080273, upper bound: 0.0068894
NS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0082003, upper bound: 0.0068550
NS_A2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0073630, upper bound: 0.0055193
NS_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0073630, upper bound: 0.0055457
NS_A2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0062449, upper bound: 0.0055519
NS_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0062449, upper bound: 0.0055783
NS_A2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0072900, upper bound: 0.0055279
NS_A2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0072900, upper bound: 0.0055415
NS_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0060423, upper bound: 0.0055519
NS_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0060423, upper bound: 0.0055626

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0147416, 0.0225598, 0.0151135, 0.0231104, -0.0083687, 0.0074463
1: -0.0391509, -0.0198525, -0.0381770, -0.0198591, -0.0192918, 0.0183246
2: -0.0402318, 0.0207041, -0.0371663, 0.0212939, -0.0615256, 0.0578705
3: -0.0841832, -0.0153145, -0.0806948, -0.0149684, -0.0692148, 0.0653803
4: -0.0407803, 0.0215854, -0.0376278, 0.0221459, -0.0629261, 0.0592131

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078700, upper bound: 0.0078700
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078700, upper bound: 0.0079476
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0151251, 0.0230883, 0.0151135, 0.0231104, -0.0079853, 0.0079748
1: -0.0381634, -0.0198594, -0.0381770, -0.0198591, -0.0183043, 0.0183176
2: -0.0371241, 0.0212632, -0.0371663, 0.0212939, -0.0584180, 0.0584295
3: -0.0806467, -0.0149875, -0.0806948, -0.0149684, -0.0656783, 0.0657074
4: -0.0375842, 0.0221168, -0.0376278, 0.0221459, -0.0597301, 0.0597446

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079476, upper bound: 0.0081738
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079476, upper bound: 0.0082513
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0151135, 0.0231104, 0.0128621, 0.0227716, -0.0076581, 0.0102483
1: -0.0381770, -0.0198591, -0.0433120, -0.0198215, -0.0183555, 0.0234528
2: -0.0371663, 0.0212939, -0.0533105, 0.0211051, -0.0582715, 0.0746043
3: -0.0806948, -0.0149684, -0.0990650, -0.0150804, -0.0656144, 0.0840966
4: -0.0376278, 0.0221459, -0.0542279, 0.0219627, -0.0595904, 0.0763737

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079555, upper bound: 0.0076878
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079555, upper bound: 0.0079916
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0151135, 0.0231104, 0.0138665, 0.0233580, -0.0082445, 0.0092438
1: -0.0381770, -0.0198591, -0.0410304, -0.0198394, -0.0183376, 0.0211712
2: -0.0371663, 0.0212939, -0.0461178, 0.0216120, -0.0587783, 0.0674116
3: -0.0806948, -0.0149684, -0.0908826, -0.0147817, -0.0659131, 0.0759142
4: -0.0376278, 0.0221459, -0.0468338, 0.0224454, -0.0600731, 0.0689797

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0083539, upper bound: 0.0078021
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0083539, upper bound: 0.0081059
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0128621, 0.0227716, 0.0151135, 0.0231104, -0.0102483, 0.0076581
1: -0.0433120, -0.0198215, -0.0381770, -0.0198591, -0.0234528, 0.0183555
2: -0.0533105, 0.0211051, -0.0371663, 0.0212939, -0.0746043, 0.0582715
3: -0.0990650, -0.0150804, -0.0806948, -0.0149684, -0.0840966, 0.0656144
4: -0.0542279, 0.0219627, -0.0376278, 0.0221459, -0.0763737, 0.0595904

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076878, upper bound: 0.0079555
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076878, upper bound: 0.0080330
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0138665, 0.0233580, 0.0151135, 0.0231104, -0.0092438, 0.0082445
1: -0.0410304, -0.0198394, -0.0381770, -0.0198591, -0.0211712, 0.0183376
2: -0.0461178, 0.0216120, -0.0371663, 0.0212939, -0.0674116, 0.0587783
3: -0.0908826, -0.0147817, -0.0806948, -0.0149684, -0.0759142, 0.0659131
4: -0.0468338, 0.0224454, -0.0376278, 0.0221459, -0.0689797, 0.0600731

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078021, upper bound: 0.0083539
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078021, upper bound: 0.0084315
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0128621, 0.0227716, 0.0138521, 0.0233853, -0.0105232, 0.0089195
1: -0.0433120, -0.0198215, -0.0410483, -0.0198391, -0.0234729, 0.0212268
2: -0.0533105, 0.0211051, -0.0461732, 0.0216600, -0.0749705, 0.0672783
3: -0.0990650, -0.0150804, -0.0909458, -0.0147526, -0.0843124, 0.0758653
4: -0.0542279, 0.0219627, -0.0468910, 0.0224914, -0.0767193, 0.0688537

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076878, upper bound: 0.0077732
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076878, upper bound: 0.0078876
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0138665, 0.0233580, 0.0138521, 0.0233853, -0.0095188, 0.0095059
1: -0.0410304, -0.0198394, -0.0410483, -0.0198391, -0.0211913, 0.0212089
2: -0.0461178, 0.0216120, -0.0461732, 0.0216600, -0.0677778, 0.0677852
3: -0.0908826, -0.0147817, -0.0909458, -0.0147526, -0.0761300, 0.0761640
4: -0.0468338, 0.0224454, -0.0468910, 0.0224914, -0.0693252, 0.0693364

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078021, upper bound: 0.0081717
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078021, upper bound: 0.0082860
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0128208, 0.0227821, 0.0060306, 0.0241725, -0.0113518, 0.0167515
1: -0.0433761, -0.0198199, -0.0551499, -0.0196477, -0.0237284, 0.0353299
2: -0.0535134, 0.0211087, -0.0906109, 0.0227088, -0.0762222, 0.1117196
3: -0.0992958, -0.0150780, -0.1415163, -0.0141184, -0.0851774, 0.1264383
4: -0.0544364, 0.0219662, -0.0925998, 0.0235002, -0.0779366, 0.1145660

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081025, upper bound: 0.0063775
time: 0.31 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081025, upper bound: 0.0065563
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0128208, 0.0227821, 0.0072537, 0.0240102, -0.0111894, 0.0155283
1: -0.0433761, -0.0198199, -0.0520465, -0.0196621, -0.0237140, 0.0322266
2: -0.0535134, 0.0211087, -0.0808753, 0.0223941, -0.0759074, 0.1019840
3: -0.0992958, -0.0150780, -0.1304372, -0.0143211, -0.0849747, 0.1153591
4: -0.0544364, 0.0219662, -0.0825893, 0.0231918, -0.0776281, 0.1045555

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B1_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079699, upper bound: 0.0060222
time: 0.32 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081535, upper bound: 0.0065244
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.0141266, 0.0231744, 0.0059873, 0.0241804, -0.0100538, 0.0171871
1: -0.0397210, -0.0198428, -0.0552418, -0.0196470, -0.0200740, 0.0353990
2: -0.0420416, 0.0215086, -0.0908982, 0.0227108, -0.0647524, 0.1124068
3: -0.0862416, -0.0148620, -0.1418434, -0.0141165, -0.0721251, 0.1269814
4: -0.0426416, 0.0223385, -0.0928955, 0.0235024, -0.0661440, 0.1152341

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080030, upper bound: 0.0069024
time: 0.32 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081087, upper bound: 0.0068706
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.0139794, 0.0231922, 0.0059873, 0.0241804, -0.0102009, 0.0172048
1: -0.0402945, -0.0198331, -0.0552418, -0.0196470, -0.0206475, 0.0354087
2: -0.0438326, 0.0215255, -0.0908982, 0.0227108, -0.0665434, 0.1124238
3: -0.0882807, -0.0148390, -0.1418434, -0.0141165, -0.0741642, 0.1270044
4: -0.0444835, 0.0223602, -0.0928955, 0.0235024, -0.0679859, 0.1152557

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080770, upper bound: 0.0069244
time: 0.32 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081615, upper bound: 0.0068926
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0128208, 0.0227821, 0.0029307, 0.0246302, -0.0118094, 0.0198514
1: -0.0433761, -0.0198199, -0.0601128, -0.0195467, -0.0238294, 0.0402929
2: -0.0535134, 0.0211087, -0.1062435, 0.0230436, -0.0765570, 0.1273521
3: -0.0992958, -0.0150780, -0.1592974, -0.0138872, -0.0854086, 0.1442194
4: -0.0544364, 0.0219662, -0.1086764, 0.0238321, -0.0782685, 0.1306426

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080273, upper bound: 0.0063106
time: 0.33 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080273, upper bound: 0.0064893
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0128208, 0.0227821, 0.0038588, 0.0245134, -0.0116926, 0.0189232
1: -0.0433761, -0.0198199, -0.0575248, -0.0195580, -0.0238181, 0.0377049
2: -0.0535134, 0.0211087, -0.0981476, 0.0228597, -0.0763730, 0.1192563
3: -0.0992958, -0.0150780, -0.1500821, -0.0140112, -0.0852846, 0.1350040
4: -0.0544364, 0.0219662, -0.1003500, 0.0236494, -0.0780858, 0.1223162

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081906, upper bound: 0.0063106
time: 0.31 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081906, upper bound: 0.0064893
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0138056, 0.0233843, 0.0029307, 0.0246302, -0.0108245, 0.0204537
1: -0.0411188, -0.0198374, -0.0601128, -0.0195467, -0.0215721, 0.0402754
2: -0.0463957, 0.0216158, -0.1062435, 0.0230436, -0.0694393, 0.1278593
3: -0.0911990, -0.0147792, -0.1592974, -0.0138872, -0.0773118, 0.1445182
4: -0.0471199, 0.0224490, -0.1086764, 0.0238321, -0.0709520, 0.1311254

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078684, upper bound: 0.0068380
time: 0.32 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079905, upper bound: 0.0068600
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0138056, 0.0233843, 0.0038588, 0.0245134, -0.0107078, 0.0195255
1: -0.0411188, -0.0198374, -0.0575248, -0.0195580, -0.0215608, 0.0376874
2: -0.0463957, 0.0216158, -0.0981476, 0.0228597, -0.0692554, 0.1197634
3: -0.0911990, -0.0147792, -0.1500821, -0.0140112, -0.0771878, 0.1353028
4: -0.0471199, 0.0224490, -0.1003500, 0.0236494, -0.0707693, 0.1227990

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080501, upper bound: 0.0068037
time: 0.32 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081615, upper bound: 0.0068257
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0061984, 0.0238467, 0.0161660, 0.0238971, -0.0176987, 0.0076806
1: -0.0550196, -0.0196523, -0.0373209, -0.0199202, -0.0350995, 0.0176686
2: -0.0902050, 0.0223220, -0.0343814, 0.0227455, -0.1129505, 0.0567035
3: -0.1410551, -0.0143647, -0.0775367, -0.0140851, -0.1269700, 0.0631720
4: -0.0921818, 0.0231219, -0.0347685, 0.0235457, -0.1157274, 0.0578904

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A1_A1_B1_B1

### Relational analysis result of NS_A2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0073236, upper bound: 0.0055193
time: 0.32 seconds

## Relational analysis of NS_A2_A1_A1_A1_B1_B2

### Relational analysis result of NS_A2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0073236, upper bound: 0.0055193
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0061984, 0.0238467, 0.0133833, 0.0242625, -0.0180641, 0.0104633
1: -0.0550196, -0.0196523, -0.0415461, -0.0198255, -0.0351942, 0.0218939
2: -0.0902050, 0.0223220, -0.0476973, 0.0230384, -0.1132434, 0.0700193
3: -0.1410551, -0.0143647, -0.0926802, -0.0138786, -0.1271765, 0.0783155
4: -0.0921818, 0.0231219, -0.0484618, 0.0238397, -0.1160214, 0.0715838

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A1_A1_B2_B1

### Relational analysis result of NS_A2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0073236, upper bound: 0.0055457
time: 0.33 seconds

## Relational analysis of NS_A2_A1_A1_A1_B2_B2

### Relational analysis result of NS_A2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0073236, upper bound: 0.0055457
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0077262, 0.0238715, 0.0161660, 0.0238971, -0.0161710, 0.0077054
1: -0.0517102, -0.0196804, -0.0373209, -0.0199202, -0.0317900, 0.0176405
2: -0.0797982, 0.0223627, -0.0343814, 0.0227455, -0.1025438, 0.0567442
3: -0.1292162, -0.0143354, -0.0775367, -0.0140851, -0.1151311, 0.0632012
4: -0.0814825, 0.0231642, -0.0347685, 0.0235457, -0.1050282, 0.0579327

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A1_A2_B1_B1

### Relational analysis result of NS_A2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062056, upper bound: 0.0055519
time: 0.32 seconds

## Relational analysis of NS_A2_A1_A1_A2_B1_B2

### Relational analysis result of NS_A2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062056, upper bound: 0.0055519
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0077262, 0.0238715, 0.0133833, 0.0242625, -0.0165363, 0.0104881
1: -0.0517102, -0.0196804, -0.0415461, -0.0198255, -0.0318847, 0.0218657
2: -0.0797982, 0.0223627, -0.0476973, 0.0230384, -0.1028367, 0.0700600
3: -0.1292162, -0.0143354, -0.0926802, -0.0138786, -0.1153376, 0.0783448
4: -0.0814825, 0.0231642, -0.0484618, 0.0238397, -0.1053222, 0.0716260

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A1_A2_B2_B1

### Relational analysis result of NS_A2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062056, upper bound: 0.0055783
time: 0.32 seconds

## Relational analysis of NS_A2_A1_A1_A2_B2_B2

### Relational analysis result of NS_A2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062056, upper bound: 0.0055783
time: 0.31 seconds

## BFS NS instance: NS_A2_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0029260, 0.0243077, 0.0161660, 0.0238971, -0.0209712, 0.0081417
1: -0.0601382, -0.0195490, -0.0373209, -0.0199202, -0.0402180, 0.0177719
2: -0.1063517, 0.0226419, -0.0343814, 0.0227455, -0.1290972, 0.0570234
3: -0.1594193, -0.0141431, -0.0775367, -0.0140851, -0.1453342, 0.0633936
4: -0.1087855, 0.0234392, -0.0347685, 0.0235457, -0.1323311, 0.0582077

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0072506, upper bound: 0.0055279
time: 0.33 seconds

## Relational analysis of NS_A2_A1_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0072506, upper bound: 0.0055279
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0029260, 0.0243077, 0.0133833, 0.0242625, -0.0213365, 0.0109244
1: -0.0601382, -0.0195490, -0.0415461, -0.0198255, -0.0403127, 0.0219972
2: -0.1063517, 0.0226419, -0.0476973, 0.0230384, -0.1293902, 0.0703392
3: -0.1594193, -0.0141431, -0.0926802, -0.0138786, -0.1455407, 0.0785371
4: -0.1087855, 0.0234392, -0.0484618, 0.0238397, -0.1326251, 0.0719010

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0072506, upper bound: 0.0055415
time: 0.33 seconds

## Relational analysis of NS_A2_A1_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0072506, upper bound: 0.0055415
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0045775, 0.0243572, 0.0161660, 0.0238971, -0.0193196, 0.0081912
1: -0.0567702, -0.0195787, -0.0373209, -0.0199202, -0.0368500, 0.0177422
2: -0.0957376, 0.0227800, -0.0343814, 0.0227455, -0.1184831, 0.0571614
3: -0.1473465, -0.0140503, -0.0775367, -0.0140851, -0.1332614, 0.0634864
4: -0.0978744, 0.0235783, -0.0347685, 0.0235457, -0.1214200, 0.0583468

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A2_A2_B1_B1

### Relational analysis result of NS_A2_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0060030, upper bound: 0.0055519
time: 0.32 seconds

## Relational analysis of NS_A2_A1_A2_A2_B1_B2

### Relational analysis result of NS_A2_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0060030, upper bound: 0.0055519
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0045775, 0.0243572, 0.0133833, 0.0242625, -0.0196849, 0.0109739
1: -0.0567702, -0.0195787, -0.0415461, -0.0198255, -0.0369447, 0.0219675
2: -0.0957376, 0.0227800, -0.0476973, 0.0230384, -0.1187760, 0.0704772
3: -0.1473465, -0.0140503, -0.0926802, -0.0138786, -0.1334679, 0.0786299
4: -0.0978744, 0.0235783, -0.0484618, 0.0238397, -0.1217140, 0.0720401

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A2_A2_B2_B1

### Relational analysis result of NS_A2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0060030, upper bound: 0.0055626
time: 0.33 seconds

## Relational analysis of NS_A2_A1_A2_A2_B2_B2

### Relational analysis result of NS_A2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0060030, upper bound: 0.0055626
time: 0.33 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.41 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0078700, upper bound: 0.0078700
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0078700, upper bound: 0.0079476
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0079476, upper bound: 0.0081738
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0079476, upper bound: 0.0082513
NS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0079555, upper bound: 0.0076878
NS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0079555, upper bound: 0.0079916
NS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0083539, upper bound: 0.0078021
NS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0083539, upper bound: 0.0081059
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0076878, upper bound: 0.0079555
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0076878, upper bound: 0.0080330
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0078021, upper bound: 0.0083539
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0078021, upper bound: 0.0084315
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0076878, upper bound: 0.0077732
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0076878, upper bound: 0.0078876
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0078021, upper bound: 0.0081717
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0078021, upper bound: 0.0082860
NS_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0081025, upper bound: 0.0063775
NS_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0081025, upper bound: 0.0065563
NS_A1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0079699, upper bound: 0.0060222
NS_A1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0081535, upper bound: 0.0065244
NS_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0080030, upper bound: 0.0069024
NS_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0081087, upper bound: 0.0068706
NS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0080770, upper bound: 0.0069244
NS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0081615, upper bound: 0.0068926
NS_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0080273, upper bound: 0.0063106
NS_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0080273, upper bound: 0.0064893
NS_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0081906, upper bound: 0.0063106
NS_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0081906, upper bound: 0.0064893
NS_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0078684, upper bound: 0.0068380
NS_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0079905, upper bound: 0.0068600
NS_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0080501, upper bound: 0.0068037
NS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0081615, upper bound: 0.0068257
NS_A2_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0073236, upper bound: 0.0055193
NS_A2_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0073236, upper bound: 0.0055193
NS_A2_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0073236, upper bound: 0.0055457
NS_A2_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0073236, upper bound: 0.0055457
NS_A2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0062056, upper bound: 0.0055519
NS_A2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0062056, upper bound: 0.0055519
NS_A2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0062056, upper bound: 0.0055783
NS_A2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0062056, upper bound: 0.0055783
NS_A2_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0072506, upper bound: 0.0055279
NS_A2_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0072506, upper bound: 0.0055279
NS_A2_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0072506, upper bound: 0.0055415
NS_A2_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0072506, upper bound: 0.0055415
NS_A2_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0060030, upper bound: 0.0055519
NS_A2_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0060030, upper bound: 0.0055519
NS_A2_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0060030, upper bound: 0.0055626
NS_A2_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -0.0060030, upper bound: 0.0055626

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0147416, 0.0225598, 0.0147416, 0.0225598, -0.0078181, 0.0078181
1: -0.0391509, -0.0198525, -0.0391509, -0.0198525, -0.0192984, 0.0192984
2: -0.0402318, 0.0207041, -0.0402318, 0.0207041, -0.0609359, 0.0609359
3: -0.0841832, -0.0153145, -0.0841832, -0.0153145, -0.0688687, 0.0688687
4: -0.0407803, 0.0215854, -0.0407803, 0.0215854, -0.0623656, 0.0623656

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078700, upper bound: 0.0075111
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076366, upper bound: 0.0075957
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0147416, 0.0225598, 0.0151251, 0.0230883, -0.0083467, 0.0074346
1: -0.0391509, -0.0198525, -0.0381634, -0.0198594, -0.0192915, 0.0183110
2: -0.0402318, 0.0207041, -0.0371241, 0.0212632, -0.0614950, 0.0578282
3: -0.0841832, -0.0153145, -0.0806467, -0.0149875, -0.0691958, 0.0653322
4: -0.0407803, 0.0215854, -0.0375842, 0.0221168, -0.0628971, 0.0591696

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0075519, upper bound: 0.0079476
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076366, upper bound: 0.0075957
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0151251, 0.0230883, 0.0147416, 0.0225598, -0.0074346, 0.0083467
1: -0.0381634, -0.0198594, -0.0391509, -0.0198525, -0.0183110, 0.0192915
2: -0.0371241, 0.0212632, -0.0402318, 0.0207041, -0.0578282, 0.0614950
3: -0.0806467, -0.0149875, -0.0841832, -0.0153145, -0.0653322, 0.0691958
4: -0.0375842, 0.0221168, -0.0407803, 0.0215854, -0.0591696, 0.0628971

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079259, upper bound: 0.0078348
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079274, upper bound: 0.0081593
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0151251, 0.0230883, 0.0151251, 0.0230883, -0.0079632, 0.0079632
1: -0.0381634, -0.0198594, -0.0381634, -0.0198594, -0.0183041, 0.0183041
2: -0.0371241, 0.0212632, -0.0371241, 0.0212632, -0.0583873, 0.0583873
3: -0.0806467, -0.0149875, -0.0806467, -0.0149875, -0.0656593, 0.0656593
4: -0.0375842, 0.0221168, -0.0375842, 0.0221168, -0.0597010, 0.0597010

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079259, upper bound: 0.0079123
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079274, upper bound: 0.0082368
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0147416, 0.0225598, 0.0128621, 0.0227716, -0.0080299, 0.0096977
1: -0.0391509, -0.0198525, -0.0433120, -0.0198215, -0.0193294, 0.0234595
2: -0.0402318, 0.0207041, -0.0533105, 0.0211051, -0.0613369, 0.0740146
3: -0.0841832, -0.0153145, -0.0990650, -0.0150804, -0.0691028, 0.0837505
4: -0.0407803, 0.0215854, -0.0542279, 0.0219627, -0.0627429, 0.0758132

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079555, upper bound: 0.0075200
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076812, upper bound: 0.0076046
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0151251, 0.0230883, 0.0128621, 0.0227716, -0.0076464, 0.0102262
1: -0.0381634, -0.0198594, -0.0433120, -0.0198215, -0.0183420, 0.0234526
2: -0.0371241, 0.0212632, -0.0533105, 0.0211051, -0.0582292, 0.0745737
3: -0.0806467, -0.0149875, -0.0990650, -0.0150804, -0.0655663, 0.0840776
4: -0.0375842, 0.0221168, -0.0542279, 0.0219627, -0.0595469, 0.0763447

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079555, upper bound: 0.0078938
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076812, upper bound: 0.0079084
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0147416, 0.0225598, 0.0138665, 0.0233580, -0.0086163, 0.0086932
1: -0.0391509, -0.0198525, -0.0410304, -0.0198394, -0.0193115, 0.0211779
2: -0.0402318, 0.0207041, -0.0461178, 0.0216120, -0.0618438, 0.0668219
3: -0.0841832, -0.0153145, -0.0908826, -0.0147817, -0.0694015, 0.0755681
4: -0.0407803, 0.0215854, -0.0468338, 0.0224454, -0.0632256, 0.0684192

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0072930, upper bound: 0.0078021
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076812, upper bound: 0.0075909
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0151251, 0.0230883, 0.0138665, 0.0233580, -0.0082329, 0.0092218
1: -0.0381634, -0.0198594, -0.0410304, -0.0198394, -0.0183240, 0.0211710
2: -0.0371241, 0.0212632, -0.0461178, 0.0216120, -0.0587361, 0.0673809
3: -0.0806467, -0.0149875, -0.0908826, -0.0147817, -0.0658650, 0.0758951
4: -0.0375842, 0.0221168, -0.0468338, 0.0224454, -0.0600296, 0.0689506

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0072930, upper bound: 0.0076878
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076812, upper bound: 0.0078947
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0128621, 0.0227716, 0.0147416, 0.0225598, -0.0096977, 0.0080299
1: -0.0433120, -0.0198215, -0.0391509, -0.0198525, -0.0234595, 0.0193294
2: -0.0533105, 0.0211051, -0.0402318, 0.0207041, -0.0740146, 0.0613369
3: -0.0990650, -0.0150804, -0.0841832, -0.0153145, -0.0837505, 0.0691028
4: -0.0542279, 0.0219627, -0.0407803, 0.0215854, -0.0758132, 0.0627429

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0075200, upper bound: 0.0079555
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076046, upper bound: 0.0076812
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0128621, 0.0227716, 0.0151251, 0.0230883, -0.0102262, 0.0076464
1: -0.0433120, -0.0198215, -0.0381634, -0.0198594, -0.0234526, 0.0183420
2: -0.0533105, 0.0211051, -0.0371241, 0.0212632, -0.0745737, 0.0582292
3: -0.0990650, -0.0150804, -0.0806467, -0.0149875, -0.0840776, 0.0655663
4: -0.0542279, 0.0219627, -0.0375842, 0.0221168, -0.0763447, 0.0595469

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0075200, upper bound: 0.0080330
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076046, upper bound: 0.0076812
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0138665, 0.0233580, 0.0147416, 0.0225598, -0.0086932, 0.0086163
1: -0.0410304, -0.0198394, -0.0391509, -0.0198525, -0.0211779, 0.0193115
2: -0.0461178, 0.0216120, -0.0402318, 0.0207041, -0.0668219, 0.0618438
3: -0.0908826, -0.0147817, -0.0841832, -0.0153145, -0.0755681, 0.0694015
4: -0.0468338, 0.0224454, -0.0407803, 0.0215854, -0.0684192, 0.0632256

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077819, upper bound: 0.0083025
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077235, upper bound: 0.0083246
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0138665, 0.0233580, 0.0151251, 0.0230883, -0.0092218, 0.0082329
1: -0.0410304, -0.0198394, -0.0381634, -0.0198594, -0.0211710, 0.0183240
2: -0.0461178, 0.0216120, -0.0371241, 0.0212632, -0.0673809, 0.0587361
3: -0.0908826, -0.0147817, -0.0806467, -0.0149875, -0.0758951, 0.0658650
4: -0.0468338, 0.0224454, -0.0375842, 0.0221168, -0.0689506, 0.0600296

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077819, upper bound: 0.0083801
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077235, upper bound: 0.0084021
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0128621, 0.0227716, 0.0128621, 0.0227716, -0.0099095, 0.0099095
1: -0.0433120, -0.0198215, -0.0433120, -0.0198215, -0.0234905, 0.0234905
2: -0.0533105, 0.0211051, -0.0533105, 0.0211051, -0.0744156, 0.0744156
3: -0.0990650, -0.0150804, -0.0990650, -0.0150804, -0.0839846, 0.0839846
4: -0.0542279, 0.0219627, -0.0542279, 0.0219627, -0.0761905, 0.0761905

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076878, upper bound: 0.0072882
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076046, upper bound: 0.0076763
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0128621, 0.0227716, 0.0138665, 0.0233580, -0.0104959, 0.0089050
1: -0.0433120, -0.0198215, -0.0410304, -0.0198394, -0.0234725, 0.0212089
2: -0.0533105, 0.0211051, -0.0461178, 0.0216120, -0.0749225, 0.0672229
3: -0.0990650, -0.0150804, -0.0908826, -0.0147817, -0.0842833, 0.0758022
4: -0.0542279, 0.0219627, -0.0468338, 0.0224454, -0.0766732, 0.0687965

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0073019, upper bound: 0.0078876
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076046, upper bound: 0.0076763
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0138665, 0.0233580, 0.0128621, 0.0227716, -0.0089050, 0.0104959
1: -0.0410304, -0.0198394, -0.0433120, -0.0198215, -0.0212089, 0.0234725
2: -0.0461178, 0.0216120, -0.0533105, 0.0211051, -0.0672229, 0.0749225
3: -0.0908826, -0.0147817, -0.0990650, -0.0150804, -0.0758022, 0.0842833
4: -0.0468338, 0.0224454, -0.0542279, 0.0219627, -0.0687965, 0.0766732

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077819, upper bound: 0.0081203
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077235, upper bound: 0.0081424
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0138665, 0.0233580, 0.0138665, 0.0233580, -0.0094914, 0.0094914
1: -0.0410304, -0.0198394, -0.0410304, -0.0198394, -0.0211910, 0.0211910
2: -0.0461178, 0.0216120, -0.0461178, 0.0216120, -0.0677297, 0.0677297
3: -0.0908826, -0.0147817, -0.0908826, -0.0147817, -0.0761009, 0.0761009
4: -0.0468338, 0.0224454, -0.0468338, 0.0224454, -0.0692792, 0.0692792

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077819, upper bound: 0.0081762
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077235, upper bound: 0.0081983
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0128647, 0.0227751, 0.0060306, 0.0241725, -0.0113079, 0.0167445
1: -0.0432709, -0.0198207, -0.0551499, -0.0196477, -0.0236232, 0.0353292
2: -0.0531830, 0.0211061, -0.0906109, 0.0227088, -0.0758919, 0.1117169
3: -0.0989199, -0.0150804, -0.1415163, -0.0141184, -0.0848015, 0.1264359
4: -0.0540968, 0.0219634, -0.0925998, 0.0235002, -0.0775970, 0.1145632

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0141849, 0.0226788, 0.0060306, 0.0241725, -0.0099877, 0.0166482
1: -0.0400524, -0.0198381, -0.0551499, -0.0196477, -0.0204046, 0.0353118
2: -0.0430786, 0.0208736, -0.0906109, 0.0227088, -0.0657874, 0.1114845
3: -0.0874207, -0.0152276, -0.1415163, -0.0141184, -0.0733023, 0.1262887
4: -0.0437065, 0.0217373, -0.0925998, 0.0235002, -0.0672067, 0.1143371

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0128208, 0.0227821, 0.0075897, 0.0237686, -0.0109478, 0.0151924
1: -0.0433761, -0.0198199, -0.0506101, -0.0196680, -0.0237081, 0.0307901
2: -0.0535134, 0.0211087, -0.0764010, 0.0221813, -0.0756947, 0.0975097
3: -0.0992958, -0.0150780, -0.1253428, -0.0144694, -0.0848264, 0.1102648
4: -0.0544364, 0.0219662, -0.0779874, 0.0229785, -0.0774149, 0.0999535

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0128208, 0.0227821, 0.0073947, 0.0237081, -0.0108873, 0.0153874
1: -0.0433761, -0.0198199, -0.0512448, -0.0196685, -0.0237076, 0.0314249
2: -0.0535134, 0.0211087, -0.0783873, 0.0220174, -0.0755308, 0.0994960
3: -0.0992958, -0.0150780, -0.1276041, -0.0145656, -0.0847302, 0.1125261
4: -0.0544364, 0.0219662, -0.0800302, 0.0228215, -0.0772579, 0.1019963

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080770, upper bound: 0.0063720
time: 0.33 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080770, upper bound: 0.0065244
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0141266, 0.0231744, 0.0060306, 0.0241725, -0.0100460, 0.0171438
1: -0.0397210, -0.0198428, -0.0551499, -0.0196477, -0.0200732, 0.0353070
2: -0.0420416, 0.0215086, -0.0906109, 0.0227088, -0.0647504, 0.1121194
3: -0.0862416, -0.0148620, -0.1415163, -0.0141184, -0.0721232, 0.1266543
4: -0.0426416, 0.0223385, -0.0925998, 0.0235002, -0.0661418, 0.1149383

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080030, upper bound: 0.0068407
time: 0.33 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080030, upper bound: 0.0068706
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0141266, 0.0231744, 0.0072537, 0.0240102, -0.0098836, 0.0159206
1: -0.0397210, -0.0198428, -0.0520465, -0.0196621, -0.0200588, 0.0322037
2: -0.0420416, 0.0215086, -0.0808753, 0.0223941, -0.0644356, 0.1023839
3: -0.0862416, -0.0148620, -0.1304372, -0.0143211, -0.0719204, 0.1155752
4: -0.0426416, 0.0223385, -0.0825893, 0.0231918, -0.0658334, 0.1049279

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081087, upper bound: 0.0068407
time: 0.33 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081087, upper bound: 0.0068706
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0139794, 0.0231922, 0.0060306, 0.0241725, -0.0101931, 0.0171616
1: -0.0402945, -0.0198331, -0.0551499, -0.0196477, -0.0206468, 0.0353167
2: -0.0438326, 0.0215255, -0.0906109, 0.0227088, -0.0665414, 0.1121364
3: -0.0882807, -0.0148390, -0.1415163, -0.0141184, -0.0741623, 0.1266773
4: -0.0444835, 0.0223602, -0.0925998, 0.0235002, -0.0679837, 0.1149600

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080770, upper bound: 0.0068474
time: 0.33 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080770, upper bound: 0.0068926
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0139794, 0.0231922, 0.0072537, 0.0240102, -0.0100307, 0.0159384
1: -0.0402945, -0.0198331, -0.0520465, -0.0196621, -0.0206323, 0.0322134
2: -0.0438326, 0.0215255, -0.0808753, 0.0223941, -0.0662266, 0.1024009
3: -0.0882807, -0.0148390, -0.1304372, -0.0143211, -0.0739596, 0.1155982
4: -0.0444835, 0.0223602, -0.0825893, 0.0231918, -0.0676753, 0.1049495

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081615, upper bound: 0.0068474
time: 0.33 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081615, upper bound: 0.0068926
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0128647, 0.0227751, 0.0029307, 0.0246302, -0.0117655, 0.0198444
1: -0.0432709, -0.0198207, -0.0601128, -0.0195467, -0.0237242, 0.0402922
2: -0.0531830, 0.0211061, -0.1062435, 0.0230436, -0.0762267, 0.1273495
3: -0.0989199, -0.0150804, -0.1592974, -0.0138872, -0.0850327, 0.1442171
4: -0.0540968, 0.0219634, -0.1086764, 0.0238321, -0.0779289, 0.1306398

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0141849, 0.0226788, 0.0029307, 0.0246302, -0.0104453, 0.0197481
1: -0.0400524, -0.0198381, -0.0601128, -0.0195467, -0.0205056, 0.0402747
2: -0.0430786, 0.0208736, -0.1062435, 0.0230436, -0.0661222, 0.1271171
3: -0.0874207, -0.0152276, -0.1592974, -0.0138872, -0.0735335, 0.1440698
4: -0.0437065, 0.0217373, -0.1086764, 0.0238321, -0.0675386, 0.1304138

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0128647, 0.0227751, 0.0038588, 0.0245134, -0.0116487, 0.0189163
1: -0.0432709, -0.0198207, -0.0575248, -0.0195580, -0.0237129, 0.0377041
2: -0.0531830, 0.0211061, -0.0981476, 0.0228597, -0.0760427, 0.1192537
3: -0.0989199, -0.0150804, -0.1500821, -0.0140112, -0.0849087, 0.1350017
4: -0.0540968, 0.0219634, -0.1003500, 0.0236494, -0.0777462, 0.1223134

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0141849, 0.0226788, 0.0038588, 0.0245134, -0.0103285, 0.0188200
1: -0.0400524, -0.0198381, -0.0575248, -0.0195580, -0.0204944, 0.0376867
2: -0.0430786, 0.0208736, -0.0981476, 0.0228597, -0.0659382, 0.1190213
3: -0.0874207, -0.0152276, -0.1500821, -0.0140112, -0.0734094, 0.1348545
4: -0.0437065, 0.0217373, -0.1003500, 0.0236494, -0.0673559, 0.1220874

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0141266, 0.0231744, 0.0029307, 0.0246302, -0.0105036, 0.0202437
1: -0.0397210, -0.0198428, -0.0601128, -0.0195467, -0.0201742, 0.0402700
2: -0.0420416, 0.0215086, -0.1062435, 0.0230436, -0.0650852, 0.1277520
3: -0.0862416, -0.0148620, -0.1592974, -0.0138872, -0.0723544, 0.1444355
4: -0.0426416, 0.0223385, -0.1086764, 0.0238321, -0.0664737, 0.1310150

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_A2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078684, upper bound: 0.0068081
time: 0.33 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078684, upper bound: 0.0068380
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0139794, 0.0231922, 0.0029307, 0.0246302, -0.0106507, 0.0202615
1: -0.0402945, -0.0198331, -0.0601128, -0.0195467, -0.0207478, 0.0402797
2: -0.0438326, 0.0215255, -0.1062435, 0.0230436, -0.0668762, 0.1277690
3: -0.0882807, -0.0148390, -0.1592974, -0.0138872, -0.0743935, 0.1444585
4: -0.0444835, 0.0223602, -0.1086764, 0.0238321, -0.0683156, 0.1310366

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079756, upper bound: 0.0068600
time: 0.33 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079756, upper bound: 0.0068600
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0141266, 0.0231744, 0.0038588, 0.0245134, -0.0103868, 0.0193155
1: -0.0397210, -0.0198428, -0.0575248, -0.0195580, -0.0201630, 0.0376820
2: -0.0420416, 0.0215086, -0.0981476, 0.0228597, -0.0649012, 0.1196562
3: -0.0862416, -0.0148620, -0.1500821, -0.0140112, -0.0722303, 0.1352201
4: -0.0426416, 0.0223385, -0.1003500, 0.0236494, -0.0662910, 0.1226886

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078684, upper bound: 0.0067738
time: 0.33 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078684, upper bound: 0.0068037
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0139794, 0.0231922, 0.0038588, 0.0245134, -0.0105340, 0.0193333
1: -0.0402945, -0.0198331, -0.0575248, -0.0195580, -0.0207365, 0.0376917
2: -0.0438326, 0.0215255, -0.0981476, 0.0228597, -0.0666922, 0.1196732
3: -0.0882807, -0.0148390, -0.1500821, -0.0140112, -0.0742695, 0.1352431
4: -0.0444835, 0.0223602, -0.1003500, 0.0236494, -0.0681329, 0.1227102

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079905, upper bound: 0.0067805
time: 0.35 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079905, upper bound: 0.0068257
time: 0.34 seconds

## BFS NS instance: NS_A2_A1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0061984, 0.0238467, 0.0163204, 0.0237225, -0.0175242, 0.0075263
1: -0.0550196, -0.0196523, -0.0371442, -0.0199242, -0.0350954, 0.0174919
2: -0.0902050, 0.0223220, -0.0338259, 0.0225228, -0.1127278, 0.0561479
3: -0.1410551, -0.0143647, -0.0769051, -0.0142250, -0.1268301, 0.0625404
4: -0.0921818, 0.0231219, -0.0341971, 0.0233281, -0.1155098, 0.0573190

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A2_A1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0061984, 0.0238467, -0.0182050, 0.0248955, -0.0186971, 0.0420516
1: -0.0550196, -0.0196523, -0.0870651, -0.0188278, -0.0361918, 0.0674128
2: -0.0902050, 0.0223220, -0.1918367, 0.0228660, -0.1130710, 0.2141587
3: -0.1410551, -0.0143647, -0.2566160, -0.0140209, -0.1270342, 0.2422513
4: -0.0921818, 0.0231219, -0.1966949, 0.0236467, -0.1158285, 0.2198168

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_A1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0061984, 0.0238467, 0.0135545, 0.0240871, -0.0178887, 0.0102921
1: -0.0550196, -0.0196523, -0.0413420, -0.0198296, -0.0351900, 0.0216897
2: -0.0902050, 0.0223220, -0.0470616, 0.0228182, -0.1130231, 0.0693836
3: -0.1410551, -0.0143647, -0.0919576, -0.0140163, -0.1270388, 0.0775929
4: -0.0921818, 0.0231219, -0.0478079, 0.0236252, -0.1158070, 0.0709299

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A2_A1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0061984, 0.0238467, -0.0212213, 0.0254332, -0.0192348, 0.0450680
1: -0.0550196, -0.0196523, -0.0923657, -0.0187425, -0.0362771, 0.0727134
2: -0.0902050, 0.0223220, -0.2084725, 0.0231583, -0.1133633, 0.2307945
3: -0.1410551, -0.0143647, -0.2755580, -0.0138183, -0.1272368, 0.2611933
4: -0.0921818, 0.0231219, -0.2138128, 0.0239335, -0.1161152, 0.2369348

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_A1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0077262, 0.0238715, 0.0163204, 0.0237225, -0.0159964, 0.0075511
1: -0.0517102, -0.0196804, -0.0371442, -0.0199242, -0.0317860, 0.0174638
2: -0.0797982, 0.0223627, -0.0338259, 0.0225228, -0.1023211, 0.0561887
3: -0.1292162, -0.0143354, -0.0769051, -0.0142250, -0.1149912, 0.0625696
4: -0.0814825, 0.0231642, -0.0341971, 0.0233281, -0.1048106, 0.0573612

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A2_A1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0077262, 0.0238715, -0.0182050, 0.0248955, -0.0171694, 0.0420764
1: -0.0517102, -0.0196804, -0.0870651, -0.0188278, -0.0328824, 0.0673847
2: -0.0797982, 0.0223627, -0.1918367, 0.0228660, -0.1026643, 0.2141995
3: -0.1292162, -0.0143354, -0.2566160, -0.0140209, -0.1151953, 0.2422806
4: -0.0814825, 0.0231642, -0.1966949, 0.0236467, -0.1051293, 0.2198591

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_A1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0077262, 0.0238715, 0.0135545, 0.0240871, -0.0163609, 0.0103169
1: -0.0517102, -0.0196804, -0.0413420, -0.0198296, -0.0318806, 0.0216616
2: -0.0797982, 0.0223627, -0.0470616, 0.0228182, -0.1026164, 0.0694243
3: -0.1292162, -0.0143354, -0.0919576, -0.0140163, -0.1151998, 0.0776222
4: -0.0814825, 0.0231642, -0.0478079, 0.0236252, -0.1051077, 0.0709721

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A2_A1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0077262, 0.0238715, -0.0212213, 0.0254332, -0.0177071, 0.0450928
1: -0.0517102, -0.0196804, -0.0923657, -0.0187425, -0.0329676, 0.0726853
2: -0.0797982, 0.0223627, -0.2084725, 0.0231583, -0.1029565, 0.2308352
3: -0.1292162, -0.0143354, -0.2755580, -0.0138183, -0.1153979, 0.2612225
4: -0.0814825, 0.0231642, -0.2138128, 0.0239335, -0.1054160, 0.2369770

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_A1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0029260, 0.0243077, 0.0163204, 0.0237225, -0.0207966, 0.0079873
1: -0.0601382, -0.0195490, -0.0371442, -0.0199242, -0.0402140, 0.0175952
2: -0.1063517, 0.0226419, -0.0338259, 0.0225228, -0.1288745, 0.0564679
3: -0.1594193, -0.0141431, -0.0769051, -0.0142250, -0.1451943, 0.0627620
4: -0.1087855, 0.0234392, -0.0341971, 0.0233281, -0.1321135, 0.0576363

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A2_A1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0029260, 0.0243077, -0.0182050, 0.0248955, -0.0219696, 0.0425127
1: -0.0601382, -0.0195490, -0.0870651, -0.0188278, -0.0413104, 0.0675161
2: -0.1063517, 0.0226419, -0.1918367, 0.0228660, -0.1292177, 0.2144787
3: -0.1594193, -0.0141431, -0.2566160, -0.0140209, -0.1453984, 0.2424729
4: -0.1087855, 0.0234392, -0.1966949, 0.0236467, -0.1324322, 0.2201341

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_A1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0029260, 0.0243077, 0.0135545, 0.0240871, -0.0211611, 0.0107532
1: -0.0601382, -0.0195490, -0.0413420, -0.0198296, -0.0403086, 0.0217930
2: -0.1063517, 0.0226419, -0.0470616, 0.0228182, -0.1291699, 0.0697036
3: -0.1594193, -0.0141431, -0.0919576, -0.0140163, -0.1454030, 0.0778145
4: -0.1087855, 0.0234392, -0.0478079, 0.0236252, -0.1324107, 0.0712471

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A2_A1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0029260, 0.0243077, -0.0212213, 0.0254332, -0.0225073, 0.0455290
1: -0.0601382, -0.0195490, -0.0923657, -0.0187425, -0.0413957, 0.0728167
2: -0.1063517, 0.0226419, -0.2084725, 0.0231583, -0.1295100, 0.2311144
3: -0.1594193, -0.0141431, -0.2755580, -0.0138183, -0.1456010, 0.2614149
4: -0.1087855, 0.0234392, -0.2138128, 0.0239335, -0.1327190, 0.2372520

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_A1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0045775, 0.0243572, 0.0163204, 0.0237225, -0.0191450, 0.0080369
1: -0.0567702, -0.0195787, -0.0371442, -0.0199242, -0.0368460, 0.0175655
2: -0.0957376, 0.0227800, -0.0338259, 0.0225228, -0.1182604, 0.0566059
3: -0.1473465, -0.0140503, -0.0769051, -0.0142250, -0.1331215, 0.0628548
4: -0.0978744, 0.0235783, -0.0341971, 0.0233281, -0.1212024, 0.0577753

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A2_A1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0045775, 0.0243572, -0.0182050, 0.0248955, -0.0203180, 0.0425622
1: -0.0567702, -0.0195787, -0.0870651, -0.0188278, -0.0379424, 0.0674864
2: -0.0957376, 0.0227800, -0.1918367, 0.0228660, -0.1186036, 0.2146167
3: -0.1473465, -0.0140503, -0.2566160, -0.0140209, -0.1333256, 0.2425657
4: -0.0978744, 0.0235783, -0.1966949, 0.0236467, -0.1215211, 0.2202732

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_A1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0045775, 0.0243572, 0.0135545, 0.0240871, -0.0195096, 0.0108027
1: -0.0567702, -0.0195787, -0.0413420, -0.0198296, -0.0369406, 0.0217633
2: -0.0957376, 0.0227800, -0.0470616, 0.0228182, -0.1185557, 0.0698416
3: -0.1473465, -0.0140503, -0.0919576, -0.0140163, -0.1333302, 0.0779073
4: -0.0978744, 0.0235783, -0.0478079, 0.0236252, -0.1214996, 0.0713862

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A2_A1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0045775, 0.0243572, -0.0212213, 0.0254332, -0.0208557, 0.0455786
1: -0.0567702, -0.0195787, -0.0923657, -0.0187425, -0.0380276, 0.0727871
2: -0.0957376, 0.0227800, -0.2084725, 0.0231583, -0.1188959, 0.2312525
3: -0.1473465, -0.0140503, -0.2755580, -0.0138183, -0.1335282, 0.2615077
4: -0.0978744, 0.0235783, -0.2138128, 0.0239335, -0.1218079, 0.2373911

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 35

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.64 + 394.27 = 397.90 seconds
