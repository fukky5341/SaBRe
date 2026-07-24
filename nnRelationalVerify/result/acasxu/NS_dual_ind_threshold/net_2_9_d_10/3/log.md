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
execution time: IAR + RelationalAnalysis = 3.02 + 0.70 = 3.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0086241, upper bound: 0.0086241

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

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
time: 0.30 seconds

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

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

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
time: 0.31 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0026806, 0.0246433, 0.0132039, 0.0242686, -0.0215879, 0.0114394
1: -0.0605749, -0.0195432, -0.0417597, -0.0198232, -0.0407517, 0.0222165
2: -0.1077064, 0.0230471, -0.0483784, 0.0230394, -0.1307458, 0.0714255
3: -0.1609610, -0.0138797, -0.0934542, -0.0138781, -0.1470829, 0.0795746
4: -0.1101796, 0.0238355, -0.0491619, 0.0238405, -0.1340201, 0.0729974

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0070428, upper bound: 0.0074224
time: 0.30 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0070428, upper bound: 0.0074224
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.66 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 0, lower bound: -0.0070428, upper bound: 0.0070428
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 0, lower bound: -0.0070428, upper bound: 0.0070428
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 0, lower bound: -0.0070428, upper bound: 0.0074224
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 0, lower bound: -0.0070428, upper bound: 0.0074224

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0137908, 0.0234117, 0.0137908, 0.0234117, -0.0096209, 0.0096209
1: -0.0411373, -0.0198371, -0.0411373, -0.0198371, -0.0213002, 0.0213002
2: -0.0464529, 0.0216638, -0.0464529, 0.0216638, -0.0681168, 0.0681168
3: -0.0912642, -0.0147501, -0.0912642, -0.0147501, -0.0765141, 0.0765141
4: -0.0471789, 0.0224950, -0.0471789, 0.0224950, -0.0696740, 0.0696740

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080042, upper bound: 0.0069234
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0084304, upper bound: 0.0069234
time: 0.31 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.0137908, 0.0234117, 0.0026806, 0.0246433, -0.0108525, 0.0207311
1: -0.0411373, -0.0198371, -0.0605749, -0.0195432, -0.0215941, 0.0407378
2: -0.0464529, 0.0216638, -0.1077064, 0.0230471, -0.0695000, 0.1293702
3: -0.0912642, -0.0147501, -0.1609610, -0.0138797, -0.0773846, 0.1462108
4: -0.0471789, 0.0224950, -0.1101796, 0.0238355, -0.0710144, 0.1326746

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080042, upper bound: 0.0069234
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0084304, upper bound: 0.0069234
time: 0.31 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0026806, 0.0246433, 0.0137908, 0.0234117, -0.0207311, 0.0108525
1: -0.0605749, -0.0195432, -0.0411373, -0.0198371, -0.0407378, 0.0215941
2: -0.1077064, 0.0230471, -0.0464529, 0.0216638, -0.1293702, 0.0695000
3: -0.1609610, -0.0138797, -0.0912642, -0.0147501, -0.1462108, 0.0773846
4: -0.1101796, 0.0238355, -0.0471789, 0.0224950, -0.1326746, 0.0710144

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069963, upper bound: 0.0073014
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0073029
time: 0.31 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0026806, 0.0246433, 0.0026806, 0.0246433, -0.0219627, 0.0219627
1: -0.0605749, -0.0195432, -0.0605749, -0.0195432, -0.0410317, 0.0410317
2: -0.1077064, 0.0230471, -0.1077064, 0.0230471, -0.1307535, 0.1307535
3: -0.1609610, -0.0138797, -0.1609610, -0.0138797, -0.1470813, 0.1470813
4: -0.1101796, 0.0238355, -0.1101796, 0.0238355, -0.1340151, 0.1340151

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069963, upper bound: 0.0073014
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0073029
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.22 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -0.0080042, upper bound: 0.0069234
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -0.0084304, upper bound: 0.0069234
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -0.0080042, upper bound: 0.0069234
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -0.0084304, upper bound: 0.0069234
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -0.0069963, upper bound: 0.0073014
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0073029
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -0.0069963, upper bound: 0.0073014
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0073029

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0166793, 0.0230950, 0.0137908, 0.0234117, -0.0067324, 0.0093042
1: -0.0367899, -0.0199326, -0.0411373, -0.0198371, -0.0169527, 0.0212047
2: -0.0327252, 0.0215493, -0.0464529, 0.0216638, -0.0543891, 0.0680022
3: -0.0756531, -0.0148507, -0.0912642, -0.0147501, -0.0609029, 0.0764135
4: -0.0330632, 0.0223704, -0.0471789, 0.0224950, -0.0555582, 0.0695494

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080042, upper bound: 0.0080042
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080042, upper bound: 0.0084304
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0139741, 0.0234078, 0.0137908, 0.0234117, -0.0094376, 0.0096169
1: -0.0409187, -0.0198393, -0.0411373, -0.0198371, -0.0210815, 0.0212980
2: -0.0457554, 0.0216631, -0.0464529, 0.0216638, -0.0674193, 0.0681160
3: -0.0904716, -0.0147605, -0.0912642, -0.0147501, -0.0757215, 0.0765037
4: -0.0464621, 0.0224886, -0.0471789, 0.0224950, -0.0689572, 0.0696675

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0084304, upper bound: 0.0080042
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0084304, upper bound: 0.0084304
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0166793, 0.0230950, 0.0026806, 0.0246433, -0.0079640, 0.0204144
1: -0.0367899, -0.0199326, -0.0605749, -0.0195432, -0.0172467, 0.0406423
2: -0.0327252, 0.0215493, -0.1077064, 0.0230471, -0.0557723, 0.1292557
3: -0.0756531, -0.0148507, -0.1609610, -0.0138797, -0.0617734, 0.1461103
4: -0.0330632, 0.0223704, -0.1101796, 0.0238355, -0.0568987, 0.1325500

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080904, upper bound: 0.0069234
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080904, upper bound: 0.0069234
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0139741, 0.0234078, 0.0026806, 0.0246433, -0.0106693, 0.0207271
1: -0.0409187, -0.0198393, -0.0605749, -0.0195432, -0.0213755, 0.0407357
2: -0.0457554, 0.0216631, -0.1077064, 0.0230471, -0.0688025, 0.1293695
3: -0.0904716, -0.0147605, -0.1609610, -0.0138797, -0.0765920, 0.1462005
4: -0.0464621, 0.0224886, -0.1101796, 0.0238355, -0.0702976, 0.1326682

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085799, upper bound: 0.0069234
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085799, upper bound: 0.0069234
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0059873, 0.0241804, 0.0137908, 0.0234117, -0.0174244, 0.0103895
1: -0.0552418, -0.0196470, -0.0411373, -0.0198371, -0.0354047, 0.0214903
2: -0.0908982, 0.0227108, -0.0464529, 0.0216638, -0.1125620, 0.0691638
3: -0.1418434, -0.0141165, -0.0912642, -0.0147501, -0.1270933, 0.0771478
4: -0.0928955, 0.0235024, -0.0471789, 0.0224950, -0.1153906, 0.0706813

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0080904
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0085799
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0027399, 0.0246412, 0.0137908, 0.0234117, -0.0206718, 0.0108503
1: -0.0603256, -0.0195441, -0.0411373, -0.0198371, -0.0404885, 0.0215932
2: -0.1069151, 0.0230461, -0.0464529, 0.0216638, -0.1285790, 0.0694990
3: -0.1600597, -0.0138851, -0.0912642, -0.0147501, -0.1453096, 0.0773792
4: -0.1093657, 0.0238346, -0.0471789, 0.0224950, -0.1318608, 0.0710135

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0080904
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0085805
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0059873, 0.0241804, 0.0026806, 0.0246433, -0.0186560, 0.0214997
1: -0.0552418, -0.0196470, -0.0605749, -0.0195432, -0.0356986, 0.0409280
2: -0.0908982, 0.0227108, -0.1077064, 0.0230471, -0.1139453, 0.1304172
3: -0.1418434, -0.0141165, -0.1609610, -0.0138797, -0.1279638, 0.1468445
4: -0.0928955, 0.0235024, -0.1101796, 0.0238355, -0.1167310, 0.1336820

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0073014
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0073014
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0027399, 0.0246412, 0.0026806, 0.0246433, -0.0219035, 0.0219605
1: -0.0603256, -0.0195441, -0.0605749, -0.0195432, -0.0407824, 0.0410308
2: -0.1069151, 0.0230461, -0.1077064, 0.0230471, -0.1299622, 0.1307525
3: -0.1600597, -0.0138851, -0.1609610, -0.0138797, -0.1461800, 0.1470759
4: -0.1093657, 0.0238346, -0.1101796, 0.0238355, -0.1332012, 0.1340142

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0073029
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0073029
time: 0.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.75 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0080042, upper bound: 0.0080042
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0080042, upper bound: 0.0084304
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0084304, upper bound: 0.0080042
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0084304, upper bound: 0.0084304
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0080904, upper bound: 0.0069234
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0080904, upper bound: 0.0069234
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0085799, upper bound: 0.0069234
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0085799, upper bound: 0.0069234
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0080904
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0085799
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0080904
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0085805
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0073014
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0073014
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0073029
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 0, lower bound: -0.0069234, upper bound: 0.0073029

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0166793, 0.0230950, 0.0166793, 0.0230950, -0.0064157, 0.0064157
1: -0.0367899, -0.0199326, -0.0367899, -0.0199326, -0.0168573, 0.0168573
2: -0.0327252, 0.0215493, -0.0327252, 0.0215493, -0.0542745, 0.0542745
3: -0.0756531, -0.0148507, -0.0756531, -0.0148507, -0.0608024, 0.0608024
4: -0.0330632, 0.0223704, -0.0330632, 0.0223704, -0.0554336, 0.0554336

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080026, upper bound: 0.0074828
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076548, upper bound: 0.0076548
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0166793, 0.0230950, 0.0139741, 0.0234078, -0.0067285, 0.0091210
1: -0.0367899, -0.0199326, -0.0409187, -0.0198393, -0.0169506, 0.0209860
2: -0.0327252, 0.0215493, -0.0457554, 0.0216631, -0.0543884, 0.0673047
3: -0.0756531, -0.0148507, -0.0904716, -0.0147605, -0.0608926, 0.0756209
4: -0.0330632, 0.0223704, -0.0464621, 0.0224886, -0.0555517, 0.0688326

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080026, upper bound: 0.0081161
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076548, upper bound: 0.0082880
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0139741, 0.0234078, 0.0166793, 0.0230950, -0.0091210, 0.0067285
1: -0.0409187, -0.0198393, -0.0367899, -0.0199326, -0.0209860, 0.0169506
2: -0.0457554, 0.0216631, -0.0327252, 0.0215493, -0.0673047, 0.0543884
3: -0.0904716, -0.0147605, -0.0756531, -0.0148507, -0.0756209, 0.0608926
4: -0.0464621, 0.0224886, -0.0330632, 0.0223704, -0.0688326, 0.0555517

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0084289, upper bound: 0.0074033
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0082880, upper bound: 0.0076548
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0139741, 0.0234078, 0.0139741, 0.0234078, -0.0094337, 0.0094337
1: -0.0409187, -0.0198393, -0.0409187, -0.0198393, -0.0210794, 0.0210794
2: -0.0457554, 0.0216631, -0.0457554, 0.0216631, -0.0674185, 0.0674185
3: -0.0904716, -0.0147605, -0.0904716, -0.0147605, -0.0757111, 0.0757111
4: -0.0464621, 0.0224886, -0.0464621, 0.0224886, -0.0689507, 0.0689507

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0084289, upper bound: 0.0074903
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0082880, upper bound: 0.0080505
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0166793, 0.0230950, 0.0059873, 0.0241804, -0.0075011, 0.0171077
1: -0.0367899, -0.0199326, -0.0552418, -0.0196470, -0.0171429, 0.0353092
2: -0.0327252, 0.0215493, -0.0908982, 0.0227108, -0.0554361, 0.1124475
3: -0.0756531, -0.0148507, -0.1418434, -0.0141165, -0.0615366, 0.1269927
4: -0.0330632, 0.0223704, -0.0928955, 0.0235024, -0.0565655, 0.1152660

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080893, upper bound: 0.0067499
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079564, upper bound: 0.0069218
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0166793, 0.0230950, 0.0027399, 0.0246412, -0.0079619, 0.0203552
1: -0.0367899, -0.0199326, -0.0603256, -0.0195441, -0.0172458, 0.0403930
2: -0.0327252, 0.0215493, -0.1069151, 0.0230461, -0.0557713, 0.1284644
3: -0.0756531, -0.0148507, -0.1600597, -0.0138851, -0.0617680, 0.1452090
4: -0.0330632, 0.0223704, -0.1093657, 0.0238346, -0.0568977, 0.1317362

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080893, upper bound: 0.0067499
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079564, upper bound: 0.0069218
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0139741, 0.0234078, 0.0059873, 0.0241804, -0.0102063, 0.0174204
1: -0.0409187, -0.0198393, -0.0552418, -0.0196470, -0.0212717, 0.0354026
2: -0.0457554, 0.0216631, -0.0908982, 0.0227108, -0.0684663, 0.1125613
3: -0.0904716, -0.0147605, -0.1418434, -0.0141165, -0.0763552, 0.1270829
4: -0.0464621, 0.0224886, -0.0928955, 0.0235024, -0.0699645, 0.1153841

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085793, upper bound: 0.0066703
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085709, upper bound: 0.0069218
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0139741, 0.0234078, 0.0027399, 0.0246412, -0.0106671, 0.0206679
1: -0.0409187, -0.0198393, -0.0603256, -0.0195441, -0.0213746, 0.0404863
2: -0.0457554, 0.0216631, -0.1069151, 0.0230461, -0.0688015, 0.1285782
3: -0.0904716, -0.0147605, -0.1600597, -0.0138851, -0.0765866, 0.1452992
4: -0.0464621, 0.0224886, -0.1093657, 0.0238346, -0.0702967, 0.1318543

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085793, upper bound: 0.0066703
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085709, upper bound: 0.0069218
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0059873, 0.0241804, 0.0166793, 0.0230950, -0.0171077, 0.0075011
1: -0.0552418, -0.0196470, -0.0367899, -0.0199326, -0.0353092, 0.0171429
2: -0.0908982, 0.0227108, -0.0327252, 0.0215493, -0.1124475, 0.0554361
3: -0.1418434, -0.0141165, -0.0756531, -0.0148507, -0.1269927, 0.0615366
4: -0.0928955, 0.0235024, -0.0330632, 0.0223704, -0.1152660, 0.0565655

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069593, upper bound: 0.0077329
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069275, upper bound: 0.0078164
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0059873, 0.0241804, 0.0139741, 0.0234078, -0.0174204, 0.0102063
1: -0.0552418, -0.0196470, -0.0409187, -0.0198393, -0.0354026, 0.0212717
2: -0.0908982, 0.0227108, -0.0457554, 0.0216631, -0.1125613, 0.0684663
3: -0.1418434, -0.0141165, -0.0904716, -0.0147605, -0.1270829, 0.0763552
4: -0.0928955, 0.0235024, -0.0464621, 0.0224886, -0.1153841, 0.0699645

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069593, upper bound: 0.0081063
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069275, upper bound: 0.0082044
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0027399, 0.0246412, 0.0166793, 0.0230950, -0.0203552, 0.0079619
1: -0.0603256, -0.0195441, -0.0367899, -0.0199326, -0.0403930, 0.0172458
2: -0.1069151, 0.0230461, -0.0327252, 0.0215493, -0.1284644, 0.0557713
3: -0.1600597, -0.0138851, -0.0756531, -0.0148507, -0.1452090, 0.0617680
4: -0.1093657, 0.0238346, -0.0330632, 0.0223704, -0.1317362, 0.0568977

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068938, upper bound: 0.0080441
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068941, upper bound: 0.0080527
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0027399, 0.0246412, 0.0139741, 0.0234078, -0.0206679, 0.0106671
1: -0.0603256, -0.0195441, -0.0409187, -0.0198393, -0.0404863, 0.0213746
2: -0.1069151, 0.0230461, -0.0457554, 0.0216631, -0.1285782, 0.0688015
3: -0.1600597, -0.0138851, -0.0904716, -0.0147605, -0.1452992, 0.0765866
4: -0.1093657, 0.0238346, -0.0464621, 0.0224886, -0.1318543, 0.0702967

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068938, upper bound: 0.0081892
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068941, upper bound: 0.0081592
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0059873, 0.0241804, 0.0059873, 0.0241804, -0.0181931, 0.0181931
1: -0.0552418, -0.0196470, -0.0552418, -0.0196470, -0.0355949, 0.0355949
2: -0.0908982, 0.0227108, -0.0908982, 0.0227108, -0.1136090, 0.1136090
3: -0.1418434, -0.0141165, -0.1418434, -0.0141165, -0.1277269, 0.1277269
4: -0.0928955, 0.0235024, -0.0928955, 0.0235024, -0.1163979, 0.1163979

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069593, upper bound: 0.0071507
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069275, upper bound: 0.0072397
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0059873, 0.0241804, 0.0027399, 0.0246412, -0.0186538, 0.0214405
1: -0.0552418, -0.0196470, -0.0603256, -0.0195441, -0.0356977, 0.0406786
2: -0.0908982, 0.0227108, -0.1069151, 0.0230461, -0.1139443, 0.1296260
3: -0.1418434, -0.0141165, -0.1600597, -0.0138851, -0.1279583, 0.1459432
4: -0.0928955, 0.0235024, -0.1093657, 0.0238346, -0.1167301, 0.1328681

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069593, upper bound: 0.0071507
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069275, upper bound: 0.0072397
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0027399, 0.0246412, 0.0059873, 0.0241804, -0.0214405, 0.0186538
1: -0.0603256, -0.0195441, -0.0552418, -0.0196470, -0.0406786, 0.0356977
2: -0.1069151, 0.0230461, -0.0908982, 0.0227108, -0.1296260, 0.1139443
3: -0.1600597, -0.0138851, -0.1418434, -0.0141165, -0.1459432, 0.1279583
4: -0.1093657, 0.0238346, -0.0928955, 0.0235024, -0.1328681, 0.1167301

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068938, upper bound: 0.0073029
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068941, upper bound: 0.0070877
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0027399, 0.0246412, 0.0027399, 0.0246412, -0.0219013, 0.0219013
1: -0.0603256, -0.0195441, -0.0603256, -0.0195441, -0.0407815, 0.0407815
2: -0.1069151, 0.0230461, -0.1069151, 0.0230461, -0.1299612, 0.1299612
3: -0.1600597, -0.0138851, -0.1600597, -0.0138851, -0.1461746, 0.1461746
4: -0.1093657, 0.0238346, -0.1093657, 0.0238346, -0.1332003, 0.1332003

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068938, upper bound: 0.0073029
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068941, upper bound: 0.0070877
time: 0.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.96 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0080026, upper bound: 0.0074828
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0076548, upper bound: 0.0076548
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0080026, upper bound: 0.0081161
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0076548, upper bound: 0.0082880
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0084289, upper bound: 0.0074033
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0082880, upper bound: 0.0076548
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0084289, upper bound: 0.0074903
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0082880, upper bound: 0.0080505
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0080893, upper bound: 0.0067499
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0079564, upper bound: 0.0069218
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0080893, upper bound: 0.0067499
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0079564, upper bound: 0.0069218
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0085793, upper bound: 0.0066703
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0085709, upper bound: 0.0069218
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0085793, upper bound: 0.0066703
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0085709, upper bound: 0.0069218
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0069593, upper bound: 0.0077329
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0069275, upper bound: 0.0078164
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0069593, upper bound: 0.0081063
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0069275, upper bound: 0.0082044
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0068938, upper bound: 0.0080441
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0068941, upper bound: 0.0080527
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0068938, upper bound: 0.0081892
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0068941, upper bound: 0.0081592
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0069593, upper bound: 0.0071507
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0069275, upper bound: 0.0072397
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0069593, upper bound: 0.0071507
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0069275, upper bound: 0.0072397
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0068938, upper bound: 0.0073029
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0068941, upper bound: 0.0070877
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0068938, upper bound: 0.0073029
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.96
Output dim: 0, lower bound: -0.0068941, upper bound: 0.0070877

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0183990, 0.0227445, 0.0166793, 0.0230950, -0.0046960, 0.0060652
1: -0.0333436, -0.0199625, -0.0367899, -0.0199326, -0.0134110, 0.0168274
2: -0.0218794, 0.0211079, -0.0327252, 0.0215493, -0.0434287, 0.0538332
3: -0.0633138, -0.0151102, -0.0756531, -0.0148507, -0.0484631, 0.0605428
4: -0.0219115, 0.0219520, -0.0330632, 0.0223704, -0.0442819, 0.0550151

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074828, upper bound: 0.0074828
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074828, upper bound: 0.0074828
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0167226, 0.0230932, 0.0166793, 0.0230950, -0.0063724, 0.0064139
1: -0.0367231, -0.0199342, -0.0367899, -0.0199326, -0.0167905, 0.0168556
2: -0.0325141, 0.0215467, -0.0327252, 0.0215493, -0.0540634, 0.0542719
3: -0.0754130, -0.0148524, -0.0756531, -0.0148507, -0.0605623, 0.0608006
4: -0.0328460, 0.0223679, -0.0330632, 0.0223704, -0.0552165, 0.0554310

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074828, upper bound: 0.0076548
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074828, upper bound: 0.0076548
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0183990, 0.0227445, 0.0139741, 0.0234078, -0.0050088, 0.0087704
1: -0.0333436, -0.0199625, -0.0409187, -0.0198393, -0.0135043, 0.0209562
2: -0.0218794, 0.0211079, -0.0457554, 0.0216631, -0.0435425, 0.0668634
3: -0.0633138, -0.0151102, -0.0904716, -0.0147605, -0.0485533, 0.0753614
4: -0.0219115, 0.0219520, -0.0464621, 0.0224886, -0.0444001, 0.0684141

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074033, upper bound: 0.0081161
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074033, upper bound: 0.0081161
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0167226, 0.0230932, 0.0139741, 0.0234078, -0.0066851, 0.0091191
1: -0.0367231, -0.0199342, -0.0409187, -0.0198393, -0.0168838, 0.0209844
2: -0.0325141, 0.0215467, -0.0457554, 0.0216631, -0.0541772, 0.0673021
3: -0.0754130, -0.0148524, -0.0904716, -0.0147605, -0.0606525, 0.0756192
4: -0.0328460, 0.0223679, -0.0464621, 0.0224886, -0.0553346, 0.0688300

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074033, upper bound: 0.0082880
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074033, upper bound: 0.0082880
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0151334, 0.0231086, 0.0166793, 0.0230950, -0.0079617, 0.0064293
1: -0.0381432, -0.0198599, -0.0367899, -0.0199326, -0.0182106, 0.0169299
2: -0.0370599, 0.0212926, -0.0327252, 0.0215493, -0.0586092, 0.0540178
3: -0.0805737, -0.0149785, -0.0756531, -0.0148507, -0.0657230, 0.0606746
4: -0.0375182, 0.0221389, -0.0330632, 0.0223704, -0.0598887, 0.0552021

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081161, upper bound: 0.0074033
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081161, upper bound: 0.0074033
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0140358, 0.0233815, 0.0166793, 0.0230950, -0.0090592, 0.0067022
1: -0.0408275, -0.0198412, -0.0367899, -0.0199326, -0.0208949, 0.0169487
2: -0.0454689, 0.0216594, -0.0327252, 0.0215493, -0.0670182, 0.0543846
3: -0.0901453, -0.0147628, -0.0756531, -0.0148507, -0.0752946, 0.0608902
4: -0.0461672, 0.0224851, -0.0330632, 0.0223704, -0.0685377, 0.0555483

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081161, upper bound: 0.0076548
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081161, upper bound: 0.0076548
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0151334, 0.0231086, 0.0139741, 0.0234078, -0.0082744, 0.0091346
1: -0.0381432, -0.0198599, -0.0409187, -0.0198393, -0.0183040, 0.0210587
2: -0.0370599, 0.0212926, -0.0457554, 0.0216631, -0.0587230, 0.0670480
3: -0.0805737, -0.0149785, -0.0904716, -0.0147605, -0.0658132, 0.0754931
4: -0.0375182, 0.0221389, -0.0464621, 0.0224886, -0.0600068, 0.0686010

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080365, upper bound: 0.0074903
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080365, upper bound: 0.0074903
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0140358, 0.0233815, 0.0139741, 0.0234078, -0.0093719, 0.0094074
1: -0.0408275, -0.0198412, -0.0409187, -0.0198393, -0.0209883, 0.0210775
2: -0.0454689, 0.0216594, -0.0457554, 0.0216631, -0.0671320, 0.0674148
3: -0.0901453, -0.0147628, -0.0904716, -0.0147605, -0.0753848, 0.0757088
4: -0.0461672, 0.0224851, -0.0464621, 0.0224886, -0.0686558, 0.0689472

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080365, upper bound: 0.0080505
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080365, upper bound: 0.0080505
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0183990, 0.0227445, 0.0059873, 0.0241804, -0.0057814, 0.0167572
1: -0.0333436, -0.0199625, -0.0552418, -0.0196470, -0.0136966, 0.0352793
2: -0.0218794, 0.0211079, -0.0908982, 0.0227108, -0.0445902, 0.1120062
3: -0.0633138, -0.0151102, -0.1418434, -0.0141165, -0.0491974, 0.1267332
4: -0.0219115, 0.0219520, -0.0928955, 0.0235024, -0.0454139, 0.1148475

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077311, upper bound: 0.0067848
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078152, upper bound: 0.0067530
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0167226, 0.0230932, 0.0059873, 0.0241804, -0.0074578, 0.0171059
1: -0.0367231, -0.0199342, -0.0552418, -0.0196470, -0.0170761, 0.0353076
2: -0.0325141, 0.0215467, -0.0908982, 0.0227108, -0.0552250, 0.1124449
3: -0.0754130, -0.0148524, -0.1418434, -0.0141165, -0.0612965, 0.1269910
4: -0.0328460, 0.0223679, -0.0928955, 0.0235024, -0.0563484, 0.1152634

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076853, upper bound: 0.0069577
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077906, upper bound: 0.0069259
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0183990, 0.0227445, 0.0027399, 0.0246412, -0.0062422, 0.0200046
1: -0.0333436, -0.0199625, -0.0603256, -0.0195441, -0.0137995, 0.0403631
2: -0.0218794, 0.0211079, -0.1069151, 0.0230461, -0.0449255, 0.1280231
3: -0.0633138, -0.0151102, -0.1600597, -0.0138851, -0.0494288, 0.1449495
4: -0.0219115, 0.0219520, -0.1093657, 0.0238346, -0.0457461, 0.1313177

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080431, upper bound: 0.0067379
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080516, upper bound: 0.0067382
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0167226, 0.0230932, 0.0027399, 0.0246412, -0.0079185, 0.0203533
1: -0.0367231, -0.0199342, -0.0603256, -0.0195441, -0.0171790, 0.0403914
2: -0.0325141, 0.0215467, -0.1069151, 0.0230461, -0.0555602, 0.1284618
3: -0.0754130, -0.0148524, -0.1600597, -0.0138851, -0.0615279, 0.1452073
4: -0.0328460, 0.0223679, -0.1093657, 0.0238346, -0.0566806, 0.1317336

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079386, upper bound: 0.0068922
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078197, upper bound: 0.0068925
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0151334, 0.0231086, 0.0059873, 0.0241804, -0.0090470, 0.0171213
1: -0.0381432, -0.0198599, -0.0552418, -0.0196470, -0.0184963, 0.0353819
2: -0.0370599, 0.0212926, -0.0908982, 0.0227108, -0.0597707, 0.1121908
3: -0.0805737, -0.0149785, -0.1418434, -0.0141165, -0.0664572, 0.1268649
4: -0.0375182, 0.0221389, -0.0928955, 0.0235024, -0.0610206, 0.1150344

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081046, upper bound: 0.0066883
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081979, upper bound: 0.0066565
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0140358, 0.0233815, 0.0059873, 0.0241804, -0.0101445, 0.0173941
1: -0.0408275, -0.0198412, -0.0552418, -0.0196470, -0.0211806, 0.0354006
2: -0.0454689, 0.0216594, -0.0908982, 0.0227108, -0.0681798, 0.1125576
3: -0.0901453, -0.0147628, -0.1418434, -0.0141165, -0.0760289, 0.1270806
4: -0.0461672, 0.0224851, -0.0928955, 0.0235024, -0.0696696, 0.1153806

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081039, upper bound: 0.0069577
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0082031, upper bound: 0.0069259
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0151334, 0.0231086, 0.0027399, 0.0246412, -0.0095078, 0.0203687
1: -0.0381432, -0.0198599, -0.0603256, -0.0195441, -0.0185992, 0.0404657
2: -0.0370599, 0.0212926, -0.1069151, 0.0230461, -0.0601060, 0.1282077
3: -0.0805737, -0.0149785, -0.1600597, -0.0138851, -0.0666886, 0.1450812
4: -0.0375182, 0.0221389, -0.1093657, 0.0238346, -0.0613528, 0.1315046

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085510, upper bound: 0.0066314
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085343, upper bound: 0.0066403
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0140358, 0.0233815, 0.0027399, 0.0246412, -0.0106053, 0.0206416
1: -0.0408275, -0.0198412, -0.0603256, -0.0195441, -0.0212835, 0.0404844
2: -0.0454689, 0.0216594, -0.1069151, 0.0230461, -0.0685150, 0.1285745
3: -0.0901453, -0.0147628, -0.1600597, -0.0138851, -0.0762603, 0.1452969
4: -0.0461672, 0.0224851, -0.1093657, 0.0238346, -0.0700018, 0.1318508

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0084013, upper bound: 0.0068917
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0083789, upper bound: 0.0068925
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0060306, 0.0241725, 0.0166793, 0.0230950, -0.0170644, 0.0074933
1: -0.0551499, -0.0196477, -0.0367899, -0.0199326, -0.0352173, 0.0171421
2: -0.0906109, 0.0227088, -0.0327252, 0.0215493, -0.1121602, 0.0554341
3: -0.1415163, -0.0141184, -0.0756531, -0.0148507, -0.1266656, 0.0615346
4: -0.0925998, 0.0235002, -0.0330632, 0.0223704, -0.1149702, 0.0565634

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0077329
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0077329
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0072537, 0.0240102, 0.0166793, 0.0230950, -0.0158413, 0.0073309
1: -0.0520465, -0.0196621, -0.0367899, -0.0199326, -0.0321139, 0.0171277
2: -0.0808753, 0.0223941, -0.0327252, 0.0215493, -0.1024246, 0.0551193
3: -0.1304372, -0.0143211, -0.0756531, -0.0148507, -0.1155865, 0.0613319
4: -0.0825893, 0.0231918, -0.0330632, 0.0223704, -0.1049598, 0.0562549

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0078164
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0078164
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0060306, 0.0241725, 0.0139741, 0.0234078, -0.0173772, 0.0101985
1: -0.0551499, -0.0196477, -0.0409187, -0.0198393, -0.0353106, 0.0212709
2: -0.0906109, 0.0227088, -0.0457554, 0.0216631, -0.1122740, 0.0684643
3: -0.1415163, -0.0141184, -0.0904716, -0.0147605, -0.1267558, 0.0763532
4: -0.0925998, 0.0235002, -0.0464621, 0.0224886, -0.1150884, 0.0699623

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069080, upper bound: 0.0080057
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069299, upper bound: 0.0080808
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0072537, 0.0240102, 0.0139741, 0.0234078, -0.0161540, 0.0100361
1: -0.0520465, -0.0196621, -0.0409187, -0.0198393, -0.0322073, 0.0212565
2: -0.0808753, 0.0223941, -0.0457554, 0.0216631, -0.1025384, 0.0681495
3: -0.1304372, -0.0143211, -0.0904716, -0.0147605, -0.1156767, 0.0761505
4: -0.0825893, 0.0231918, -0.0464621, 0.0224886, -0.1050779, 0.0696539

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068762, upper bound: 0.0081116
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068981, upper bound: 0.0081659
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0030208, 0.0243556, 0.0166793, 0.0230950, -0.0200742, 0.0076764
1: -0.0592655, -0.0195496, -0.0367899, -0.0199326, -0.0393329, 0.0172403
2: -0.1036336, 0.0227146, -0.0327252, 0.0215493, -0.1251829, 0.0554399
3: -0.1563238, -0.0141121, -0.0756531, -0.0148507, -0.1414731, 0.0615410
4: -0.1059904, 0.0235039, -0.0330632, 0.0223704, -0.1283608, 0.0565671

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068344, upper bound: 0.0069521
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068344, upper bound: 0.0080441
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0029301, 0.0243821, 0.0166793, 0.0230950, -0.0201649, 0.0077028
1: -0.0595034, -0.0195402, -0.0367899, -0.0199326, -0.0395707, 0.0172496
2: -0.1043574, 0.0227430, -0.0327252, 0.0215493, -0.1259066, 0.0554682
3: -0.1571495, -0.0140831, -0.0756531, -0.0148507, -0.1422988, 0.0615699
4: -0.1067372, 0.0235363, -0.0330632, 0.0223704, -0.1291076, 0.0565995

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068347, upper bound: 0.0069521
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068347, upper bound: 0.0080527
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0030208, 0.0243556, 0.0139741, 0.0234078, -0.0203870, 0.0103816
1: -0.0592655, -0.0195496, -0.0409187, -0.0198393, -0.0394262, 0.0213691
2: -0.1036336, 0.0227146, -0.0457554, 0.0216631, -0.1252967, 0.0684701
3: -0.1563238, -0.0141121, -0.0904716, -0.0147605, -0.1415633, 0.0763596
4: -0.1059904, 0.0235039, -0.0464621, 0.0224886, -0.1284789, 0.0699661

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0076344
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0081583
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0029301, 0.0243821, 0.0139741, 0.0234078, -0.0204776, 0.0104080
1: -0.0595034, -0.0195402, -0.0409187, -0.0198393, -0.0396641, 0.0213784
2: -0.1043574, 0.0227430, -0.0457554, 0.0216631, -0.1260205, 0.0684984
3: -0.1571495, -0.0140831, -0.0904716, -0.0147605, -0.1423890, 0.0763885
4: -0.1067372, 0.0235363, -0.0464621, 0.0224886, -0.1292258, 0.0699984

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0076344
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0081592
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0060306, 0.0241725, 0.0059873, 0.0241804, -0.0181498, 0.0181852
1: -0.0551499, -0.0196477, -0.0552418, -0.0196470, -0.0355029, 0.0355941
2: -0.0906109, 0.0227088, -0.0908982, 0.0227108, -0.1133217, 0.1136071
3: -0.1415163, -0.0141184, -0.1418434, -0.0141165, -0.1273998, 0.1277250
4: -0.0925998, 0.0235002, -0.0928955, 0.0235024, -0.1161021, 0.1163957

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0072177
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0072177
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0072537, 0.0240102, 0.0059873, 0.0241804, -0.0169266, 0.0180228
1: -0.0520465, -0.0196621, -0.0552418, -0.0196470, -0.0323996, 0.0355797
2: -0.0808753, 0.0223941, -0.0908982, 0.0227108, -0.1035862, 0.1132923
3: -0.1304372, -0.0143211, -0.1418434, -0.0141165, -0.1163207, 0.1275223
4: -0.0825893, 0.0231918, -0.0928955, 0.0235024, -0.1060917, 0.1160873

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0073067
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0073067
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0060306, 0.0241725, 0.0027399, 0.0246412, -0.0186106, 0.0214327
1: -0.0551499, -0.0196477, -0.0603256, -0.0195441, -0.0356058, 0.0406779
2: -0.0906109, 0.0227088, -0.1069151, 0.0230461, -0.1136570, 0.1296240
3: -0.1415163, -0.0141184, -0.1600597, -0.0138851, -0.1276312, 0.1459413
4: -0.0925998, 0.0235002, -0.1093657, 0.0238346, -0.1164343, 0.1328659

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069080, upper bound: 0.0070354
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069299, upper bound: 0.0070508
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0072537, 0.0240102, 0.0027399, 0.0246412, -0.0173874, 0.0212703
1: -0.0520465, -0.0196621, -0.0603256, -0.0195441, -0.0325025, 0.0406634
2: -0.0808753, 0.0223941, -0.1069151, 0.0230461, -0.1039214, 0.1293092
3: -0.1304372, -0.0143211, -0.1600597, -0.0138851, -0.1165521, 0.1457386
4: -0.0825893, 0.0231918, -0.1093657, 0.0238346, -0.1064239, 0.1325575

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068762, upper bound: 0.0071594
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068981, upper bound: 0.0071748
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0030208, 0.0243556, 0.0059873, 0.0241804, -0.0211596, 0.0183683
1: -0.0592655, -0.0195496, -0.0552418, -0.0196470, -0.0396185, 0.0356922
2: -0.1036336, 0.0227146, -0.0908982, 0.0227108, -0.1263444, 0.1136129
3: -0.1563238, -0.0141121, -0.1418434, -0.0141165, -0.1422073, 0.1277314
4: -0.1059904, 0.0235039, -0.0928955, 0.0235024, -0.1294927, 0.1163995

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067732, upper bound: 0.0073400
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068158, upper bound: 0.0073082
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0029301, 0.0243821, 0.0059873, 0.0241804, -0.0212502, 0.0183947
1: -0.0595034, -0.0195402, -0.0552418, -0.0196470, -0.0398564, 0.0357016
2: -0.1043574, 0.0227430, -0.0908982, 0.0227108, -0.1270682, 0.1136412
3: -0.1571495, -0.0140831, -0.1418434, -0.0141165, -0.1430331, 0.1277603
4: -0.1067372, 0.0235363, -0.0928955, 0.0235024, -0.1302395, 0.1164318

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067885, upper bound: 0.0071242
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068312, upper bound: 0.0070924
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0030208, 0.0243556, 0.0027399, 0.0246412, -0.0216204, 0.0216158
1: -0.0592655, -0.0195496, -0.0603256, -0.0195441, -0.0397214, 0.0407760
2: -0.1036336, 0.0227146, -0.1069151, 0.0230461, -0.1266797, 0.1296298
3: -0.1563238, -0.0141121, -0.1600597, -0.0138851, -0.1424387, 0.1459476
4: -0.1059904, 0.0235039, -0.1093657, 0.0238346, -0.1298249, 0.1328697

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0070789
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0070877
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0029301, 0.0243821, 0.0027399, 0.0246412, -0.0217110, 0.0216422
1: -0.0595034, -0.0195402, -0.0603256, -0.0195441, -0.0399593, 0.0407854
2: -0.1043574, 0.0227430, -0.1069151, 0.0230461, -0.1274034, 0.1296581
3: -0.1571495, -0.0140831, -0.1600597, -0.0138851, -0.1432645, 0.1459765
4: -0.1067372, 0.0235363, -0.1093657, 0.0238346, -0.1305717, 0.1329020

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0070789
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0070877
time: 0.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.96 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0074828, upper bound: 0.0074828
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0074828, upper bound: 0.0074828
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0074828, upper bound: 0.0076548
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0074828, upper bound: 0.0076548
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0074033, upper bound: 0.0081161
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0074033, upper bound: 0.0081161
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0074033, upper bound: 0.0082880
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0074033, upper bound: 0.0082880
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0081161, upper bound: 0.0074033
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0081161, upper bound: 0.0074033
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0081161, upper bound: 0.0076548
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0081161, upper bound: 0.0076548
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0080365, upper bound: 0.0074903
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0080365, upper bound: 0.0074903
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0080365, upper bound: 0.0080505
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0080365, upper bound: 0.0080505
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0077311, upper bound: 0.0067848
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0078152, upper bound: 0.0067530
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0076853, upper bound: 0.0069577
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0077906, upper bound: 0.0069259
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0080431, upper bound: 0.0067379
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0080516, upper bound: 0.0067382
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0079386, upper bound: 0.0068922
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0078197, upper bound: 0.0068925
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0081046, upper bound: 0.0066883
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0081979, upper bound: 0.0066565
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0081039, upper bound: 0.0069577
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0082031, upper bound: 0.0069259
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0085510, upper bound: 0.0066314
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0085343, upper bound: 0.0066403
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0084013, upper bound: 0.0068917
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0083789, upper bound: 0.0068925
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0077329
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0077329
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0078164
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0078164
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0069080, upper bound: 0.0080057
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0069299, upper bound: 0.0080808
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068762, upper bound: 0.0081116
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068981, upper bound: 0.0081659
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068344, upper bound: 0.0069521
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068344, upper bound: 0.0080441
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068347, upper bound: 0.0069521
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068347, upper bound: 0.0080527
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0076344
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0081583
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0076344
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0081592
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0072177
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0072177
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0073067
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068893, upper bound: 0.0073067
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0069080, upper bound: 0.0070354
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0069299, upper bound: 0.0070508
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068762, upper bound: 0.0071594
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068981, upper bound: 0.0071748
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0067732, upper bound: 0.0073400
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068158, upper bound: 0.0073082
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0067885, upper bound: 0.0071242
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068312, upper bound: 0.0070924
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0070789
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0070877
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0070789
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0068363, upper bound: 0.0070877

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0183990, 0.0227445, 0.0183990, 0.0227445, -0.0043455, 0.0043455
1: -0.0333436, -0.0199625, -0.0333436, -0.0199625, -0.0133811, 0.0133811
2: -0.0218794, 0.0211079, -0.0218794, 0.0211079, -0.0429873, 0.0429873
3: -0.0633138, -0.0151102, -0.0633138, -0.0151102, -0.0482036, 0.0482036
4: -0.0219115, 0.0219520, -0.0219115, 0.0219520, -0.0438635, 0.0438635

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078277, upper bound: 0.0073752
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0075455, upper bound: 0.0073871
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0183990, 0.0227445, 0.0167226, 0.0230932, -0.0046942, 0.0060219
1: -0.0333436, -0.0199625, -0.0367231, -0.0199342, -0.0134093, 0.0167606
2: -0.0218794, 0.0211079, -0.0325141, 0.0215467, -0.0434261, 0.0536221
3: -0.0633138, -0.0151102, -0.0754130, -0.0148524, -0.0484614, 0.0603027
4: -0.0219115, 0.0219520, -0.0328460, 0.0223679, -0.0442794, 0.0547980

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078277, upper bound: 0.0073752
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0075455, upper bound: 0.0073871
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0167226, 0.0230932, 0.0183990, 0.0227445, -0.0060219, 0.0046942
1: -0.0367231, -0.0199342, -0.0333436, -0.0199625, -0.0167606, 0.0134093
2: -0.0325141, 0.0215467, -0.0218794, 0.0211079, -0.0536221, 0.0434261
3: -0.0754130, -0.0148524, -0.0633138, -0.0151102, -0.0603027, 0.0484614
4: -0.0328460, 0.0223679, -0.0219115, 0.0219520, -0.0547980, 0.0442794

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066026, upper bound: 0.0075659
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074711, upper bound: 0.0076254
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0167226, 0.0230932, 0.0167226, 0.0230932, -0.0063706, 0.0063706
1: -0.0367231, -0.0199342, -0.0367231, -0.0199342, -0.0167888, 0.0167888
2: -0.0325141, 0.0215467, -0.0325141, 0.0215467, -0.0540608, 0.0540608
3: -0.0754130, -0.0148524, -0.0754130, -0.0148524, -0.0605605, 0.0605605
4: -0.0328460, 0.0223679, -0.0328460, 0.0223679, -0.0552139, 0.0552139

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066026, upper bound: 0.0075659
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074711, upper bound: 0.0076254
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0183990, 0.0227445, 0.0151334, 0.0231086, -0.0047096, 0.0076111
1: -0.0333436, -0.0199625, -0.0381432, -0.0198599, -0.0134836, 0.0181808
2: -0.0218794, 0.0211079, -0.0370599, 0.0212926, -0.0431720, 0.0581678
3: -0.0633138, -0.0151102, -0.0805737, -0.0149785, -0.0483353, 0.0654634
4: -0.0219115, 0.0219520, -0.0375182, 0.0221389, -0.0440504, 0.0594702

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077312, upper bound: 0.0078732
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074490, upper bound: 0.0078851
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0183990, 0.0227445, 0.0140358, 0.0233815, -0.0049825, 0.0087086
1: -0.0333436, -0.0199625, -0.0408275, -0.0198412, -0.0135024, 0.0208651
2: -0.0218794, 0.0211079, -0.0454689, 0.0216594, -0.0435388, 0.0665769
3: -0.0633138, -0.0151102, -0.0901453, -0.0147628, -0.0485510, 0.0750351
4: -0.0219115, 0.0219520, -0.0461672, 0.0224851, -0.0443966, 0.0681192

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077312, upper bound: 0.0078732
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0074490, upper bound: 0.0078851
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0167226, 0.0230932, 0.0151334, 0.0231086, -0.0063860, 0.0079598
1: -0.0367231, -0.0199342, -0.0381432, -0.0198599, -0.0168631, 0.0182090
2: -0.0325141, 0.0215467, -0.0370599, 0.0212926, -0.0538067, 0.0586066
3: -0.0754130, -0.0148524, -0.0805737, -0.0149785, -0.0604345, 0.0657212
4: -0.0328460, 0.0223679, -0.0375182, 0.0221389, -0.0549849, 0.0598861

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065047, upper bound: 0.0081293
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0073732, upper bound: 0.0081888
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0167226, 0.0230932, 0.0140358, 0.0233815, -0.0066588, 0.0090574
1: -0.0367231, -0.0199342, -0.0408275, -0.0198412, -0.0168819, 0.0208933
2: -0.0325141, 0.0215467, -0.0454689, 0.0216594, -0.0541735, 0.0670156
3: -0.0754130, -0.0148524, -0.0901453, -0.0147628, -0.0606501, 0.0752929
4: -0.0328460, 0.0223679, -0.0461672, 0.0224851, -0.0553312, 0.0685351

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065047, upper bound: 0.0081293
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0073732, upper bound: 0.0081888
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0151334, 0.0231086, 0.0183990, 0.0227445, -0.0076111, 0.0047096
1: -0.0381432, -0.0198599, -0.0333436, -0.0199625, -0.0181808, 0.0134836
2: -0.0370599, 0.0212926, -0.0218794, 0.0211079, -0.0581678, 0.0431720
3: -0.0805737, -0.0149785, -0.0633138, -0.0151102, -0.0654634, 0.0483353
4: -0.0375182, 0.0221389, -0.0219115, 0.0219520, -0.0594702, 0.0440504

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081647, upper bound: 0.0071654
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0082432, upper bound: 0.0073826
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0151334, 0.0231086, 0.0167226, 0.0230932, -0.0079598, 0.0063860
1: -0.0381432, -0.0198599, -0.0367231, -0.0199342, -0.0182090, 0.0168631
2: -0.0370599, 0.0212926, -0.0325141, 0.0215467, -0.0586066, 0.0538067
3: -0.0805737, -0.0149785, -0.0754130, -0.0148524, -0.0657212, 0.0604345
4: -0.0375182, 0.0221389, -0.0328460, 0.0223679, -0.0598861, 0.0549849

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081647, upper bound: 0.0071654
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0082432, upper bound: 0.0073826
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0140358, 0.0233815, 0.0183990, 0.0227445, -0.0087086, 0.0049825
1: -0.0408275, -0.0198412, -0.0333436, -0.0199625, -0.0208651, 0.0135024
2: -0.0454689, 0.0216594, -0.0218794, 0.0211079, -0.0665769, 0.0435388
3: -0.0901453, -0.0147628, -0.0633138, -0.0151102, -0.0750351, 0.0485510
4: -0.0461672, 0.0224851, -0.0219115, 0.0219520, -0.0681192, 0.0443966

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079674, upper bound: 0.0072508
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081024, upper bound: 0.0076493
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0140358, 0.0233815, 0.0167226, 0.0230932, -0.0090574, 0.0066588
1: -0.0408275, -0.0198412, -0.0367231, -0.0199342, -0.0208933, 0.0168819
2: -0.0454689, 0.0216594, -0.0325141, 0.0215467, -0.0670156, 0.0541735
3: -0.0901453, -0.0147628, -0.0754130, -0.0148524, -0.0752929, 0.0606501
4: -0.0461672, 0.0224851, -0.0328460, 0.0223679, -0.0685351, 0.0553312

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0079674, upper bound: 0.0072508
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081024, upper bound: 0.0076493
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0151334, 0.0231086, 0.0151334, 0.0231086, -0.0079752, 0.0079752
1: -0.0381432, -0.0198599, -0.0381432, -0.0198599, -0.0182833, 0.0182833
2: -0.0370599, 0.0212926, -0.0370599, 0.0212926, -0.0583525, 0.0583525
3: -0.0805737, -0.0149785, -0.0805737, -0.0149785, -0.0655952, 0.0655952
4: -0.0375182, 0.0221389, -0.0375182, 0.0221389, -0.0596571, 0.0596571

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080781, upper bound: 0.0073727
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081566, upper bound: 0.0074733
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0151334, 0.0231086, 0.0140358, 0.0233815, -0.0082481, 0.0090728
1: -0.0381432, -0.0198599, -0.0408275, -0.0198412, -0.0183020, 0.0209676
2: -0.0370599, 0.0212926, -0.0454689, 0.0216594, -0.0587193, 0.0667615
3: -0.0805737, -0.0149785, -0.0901453, -0.0147628, -0.0658108, 0.0751668
4: -0.0375182, 0.0221389, -0.0461672, 0.0224851, -0.0600033, 0.0683061

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080781, upper bound: 0.0073727
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081566, upper bound: 0.0074733
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0140358, 0.0233815, 0.0151334, 0.0231086, -0.0090728, 0.0082481
1: -0.0408275, -0.0198412, -0.0381432, -0.0198599, -0.0209676, 0.0183020
2: -0.0454689, 0.0216594, -0.0370599, 0.0212926, -0.0667615, 0.0587193
3: -0.0901453, -0.0147628, -0.0805737, -0.0149785, -0.0751668, 0.0658108
4: -0.0461672, 0.0224851, -0.0375182, 0.0221389, -0.0683061, 0.0600033

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078809, upper bound: 0.0075521
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080158, upper bound: 0.0080491
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0140358, 0.0233815, 0.0140358, 0.0233815, -0.0093456, 0.0093456
1: -0.0408275, -0.0198412, -0.0408275, -0.0198412, -0.0209863, 0.0209863
2: -0.0454689, 0.0216594, -0.0454689, 0.0216594, -0.0671283, 0.0671283
3: -0.0901453, -0.0147628, -0.0901453, -0.0147628, -0.0753825, 0.0753825
4: -0.0461672, 0.0224851, -0.0461672, 0.0224851, -0.0686524, 0.0686524

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078809, upper bound: 0.0075521
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080158, upper bound: 0.0080491
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0183990, 0.0227445, 0.0060306, 0.0241725, -0.0057735, 0.0167139
1: -0.0333436, -0.0199625, -0.0551499, -0.0196477, -0.0136959, 0.0351874
2: -0.0218794, 0.0211079, -0.0906109, 0.0227088, -0.0445882, 0.1117188
3: -0.0633138, -0.0151102, -0.1415163, -0.0141184, -0.0491954, 0.1264061
4: -0.0219115, 0.0219520, -0.0925998, 0.0235002, -0.0454117, 0.1145518

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077311, upper bound: 0.0067411
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077311, upper bound: 0.0067530
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0183990, 0.0227445, 0.0072537, 0.0240102, -0.0056112, 0.0154907
1: -0.0333436, -0.0199625, -0.0520465, -0.0196621, -0.0136814, 0.0320841
2: -0.0218794, 0.0211079, -0.0808753, 0.0223941, -0.0442735, 0.1019833
3: -0.0633138, -0.0151102, -0.1304372, -0.0143211, -0.0489927, 0.1153269
4: -0.0219115, 0.0219520, -0.0825893, 0.0231918, -0.0451033, 0.1045413

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078152, upper bound: 0.0067411
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0078152, upper bound: 0.0067530
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0167226, 0.0230932, 0.0060306, 0.0241725, -0.0074499, 0.0170626
1: -0.0367231, -0.0199342, -0.0551499, -0.0196477, -0.0170753, 0.0352156
2: -0.0325141, 0.0215467, -0.0906109, 0.0227088, -0.0552230, 0.1121576
3: -0.0754130, -0.0148524, -0.1415163, -0.0141184, -0.0612946, 0.1266639
4: -0.0328460, 0.0223679, -0.0925998, 0.0235002, -0.0563462, 0.1149677

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076853, upper bound: 0.0068876
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0076853, upper bound: 0.0069259
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0167226, 0.0230932, 0.0072537, 0.0240102, -0.0072875, 0.0158394
1: -0.0367231, -0.0199342, -0.0520465, -0.0196621, -0.0170609, 0.0321123
2: -0.0325141, 0.0215467, -0.0808753, 0.0223941, -0.0549082, 0.1024220
3: -0.0754130, -0.0148524, -0.1304372, -0.0143211, -0.0610918, 0.1155848
4: -0.0328460, 0.0223679, -0.0825893, 0.0231918, -0.0560378, 0.1049572

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077906, upper bound: 0.0068876
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077906, upper bound: 0.0069259
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0183990, 0.0227445, 0.0030208, 0.0243556, -0.0059566, 0.0197237
1: -0.0333436, -0.0199625, -0.0592655, -0.0195496, -0.0137940, 0.0393030
2: -0.0218794, 0.0211079, -0.1036336, 0.0227146, -0.0445940, 0.1247415
3: -0.0633138, -0.0151102, -0.1563238, -0.0141121, -0.0492018, 0.1412135
4: -0.0219115, 0.0219520, -0.1059904, 0.0235039, -0.0454154, 0.1279423

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080339, upper bound: 0.0066459
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077747, upper bound: 0.0066596
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0183990, 0.0227445, 0.0029301, 0.0243821, -0.0059831, 0.0198143
1: -0.0333436, -0.0199625, -0.0595034, -0.0195402, -0.0138033, 0.0395409
2: -0.0218794, 0.0211079, -0.1043574, 0.0227430, -0.0446224, 0.1254653
3: -0.0633138, -0.0151102, -0.1571495, -0.0140831, -0.0492307, 0.1420393
4: -0.0219115, 0.0219520, -0.1067372, 0.0235363, -0.0454478, 0.1286892

Time for backsubstitution: 3.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080456, upper bound: 0.0066613
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0077761, upper bound: 0.0066750
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0167226, 0.0230932, 0.0030208, 0.0243556, -0.0076330, 0.0200724
1: -0.0367231, -0.0199342, -0.0592655, -0.0195496, -0.0171735, 0.0393312
2: -0.0325141, 0.0215467, -0.1036336, 0.0227146, -0.0552288, 0.1251803
3: -0.0754130, -0.0148524, -0.1563238, -0.0141121, -0.0613009, 0.1414713
4: -0.0328460, 0.0223679, -0.1059904, 0.0235039, -0.0563500, 0.1283582

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069511, upper bound: 0.0068327
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069511, upper bound: 0.0068922
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0167226, 0.0230932, 0.0029301, 0.0243821, -0.0076594, 0.0201630
1: -0.0367231, -0.0199342, -0.0595034, -0.0195402, -0.0171828, 0.0395691
2: -0.0325141, 0.0215467, -0.1043574, 0.0227430, -0.0552571, 0.1259041
3: -0.0754130, -0.0148524, -0.1571495, -0.0140831, -0.0613298, 0.1422971
4: -0.0328460, 0.0223679, -0.1067372, 0.0235363, -0.0563824, 0.1291050

Time for backsubstitution: 3.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069511, upper bound: 0.0068330
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0069511, upper bound: 0.0068925
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0151334, 0.0231086, 0.0060306, 0.0241725, -0.0090392, 0.0170780
1: -0.0381432, -0.0198599, -0.0551499, -0.0196477, -0.0184955, 0.0352899
2: -0.0370599, 0.0212926, -0.0906109, 0.0227088, -0.0597687, 0.1119035
3: -0.0805737, -0.0149785, -0.1415163, -0.0141184, -0.0664553, 0.1265378
4: -0.0375182, 0.0221389, -0.0925998, 0.0235002, -0.0610184, 0.1147387

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081046, upper bound: 0.0066565
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081046, upper bound: 0.0066565
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0151334, 0.0231086, 0.0072537, 0.0240102, -0.0088768, 0.0158549
1: -0.0381432, -0.0198599, -0.0520465, -0.0196621, -0.0184811, 0.0321866
2: -0.0370599, 0.0212926, -0.0808753, 0.0223941, -0.0594540, 0.1021679
3: -0.0805737, -0.0149785, -0.1304372, -0.0143211, -0.0662526, 0.1154587
4: -0.0375182, 0.0221389, -0.0825893, 0.0231918, -0.0607100, 0.1047282

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081979, upper bound: 0.0066565
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081979, upper bound: 0.0066565
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0140358, 0.0233815, 0.0060306, 0.0241725, -0.0101367, 0.0173509
1: -0.0408275, -0.0198412, -0.0551499, -0.0196477, -0.0211798, 0.0353087
2: -0.0454689, 0.0216594, -0.0906109, 0.0227088, -0.0681778, 0.1122703
3: -0.0901453, -0.0147628, -0.1415163, -0.0141184, -0.0760269, 0.1267535
4: -0.0461672, 0.0224851, -0.0925998, 0.0235002, -0.0696674, 0.1150849

Time for backsubstitution: 3.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080044, upper bound: 0.0069068
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0080775, upper bound: 0.0069283
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0140358, 0.0233815, 0.0072537, 0.0240102, -0.0099743, 0.0161277
1: -0.0408275, -0.0198412, -0.0520465, -0.0196621, -0.0211654, 0.0322053
2: -0.0454689, 0.0216594, -0.0808753, 0.0223941, -0.0678630, 0.1025347
3: -0.0901453, -0.0147628, -0.1304372, -0.0143211, -0.0758242, 0.1156744
4: -0.0461672, 0.0224851, -0.0825893, 0.0231918, -0.0693590, 0.1050744

Time for backsubstitution: 3.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081103, upper bound: 0.0068750
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0081650, upper bound: 0.0068965
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0151334, 0.0231086, 0.0030208, 0.0243556, -0.0092223, 0.0200878
1: -0.0381432, -0.0198599, -0.0592655, -0.0195496, -0.0185937, 0.0394055
2: -0.0370599, 0.0212926, -0.1036336, 0.0227146, -0.0597745, 0.1249262
3: -0.0805737, -0.0149785, -0.1563238, -0.0141121, -0.0664616, 0.1413453
4: -0.0375182, 0.0221389, -0.1059904, 0.0235039, -0.0610222, 0.1281293

Time for backsubstitution: 3.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085343, upper bound: 0.0064141
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085343, upper bound: 0.0066314
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0151334, 0.0231086, 0.0029301, 0.0243821, -0.0092487, 0.0201785
1: -0.0381432, -0.0198599, -0.0595034, -0.0195402, -0.0186030, 0.0396434
2: -0.0370599, 0.0212926, -0.1043574, 0.0227430, -0.0598029, 0.1256500
3: -0.0805737, -0.0149785, -0.1571495, -0.0140831, -0.0664905, 0.1421710
4: -0.0375182, 0.0221389, -0.1067372, 0.0235363, -0.0610545, 0.1288761

Time for backsubstitution: 3.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085343, upper bound: 0.0064144
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0085343, upper bound: 0.0066403
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0140358, 0.0233815, 0.0030208, 0.0243556, -0.0103198, 0.0203607
1: -0.0408275, -0.0198412, -0.0592655, -0.0195496, -0.0212780, 0.0394243
2: -0.0454689, 0.0216594, -0.1036336, 0.0227146, -0.0681836, 0.1252930
3: -0.0901453, -0.0147628, -0.1563238, -0.0141121, -0.0760333, 0.1415609
4: -0.0461672, 0.0224851, -0.1059904, 0.0235039, -0.0696712, 0.1284755

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0083789, upper bound: 0.0068702
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0083789, upper bound: 0.0068917
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0140358, 0.0233815, 0.0029301, 0.0243821, -0.0103462, 0.0204513
1: -0.0408275, -0.0198412, -0.0595034, -0.0195402, -0.0212873, 0.0396622
2: -0.0454689, 0.0216594, -0.1043574, 0.0227430, -0.0682119, 0.1260168
3: -0.0901453, -0.0147628, -0.1571495, -0.0140831, -0.0760622, 0.1423867
4: -0.0461672, 0.0224851, -0.1067372, 0.0235363, -0.0697035, 0.1292223

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0083789, upper bound: 0.0068705
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0083789, upper bound: 0.0068925
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0060306, 0.0241725, 0.0167044, 0.0230915, -0.0170609, 0.0074682
1: -0.0551499, -0.0196477, -0.0367468, -0.0199332, -0.0352167, 0.0170991
2: -0.0906109, 0.0227088, -0.0325901, 0.0215477, -0.1121585, 0.0552989
3: -0.1415163, -0.0141184, -0.0754993, -0.0148525, -0.1266638, 0.0613809
4: -0.0925998, 0.0235002, -0.0329242, 0.0223685, -0.1149683, 0.0564244

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0060306, 0.0241725, 0.0179307, 0.0229756, -0.0169450, 0.0062419
1: -0.0551499, -0.0196477, -0.0338168, -0.0199495, -0.0352004, 0.0141691
2: -0.0906109, 0.0227088, -0.0233918, 0.0213817, -0.1119926, 0.0461006
3: -0.1415163, -0.0141184, -0.0650313, -0.0149578, -0.1265585, 0.0509129
4: -0.0925998, 0.0235002, -0.0234653, 0.0222070, -0.1148068, 0.0469655

Time for backsubstitution: 3.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0072537, 0.0240102, 0.0167044, 0.0230915, -0.0158378, 0.0073058
1: -0.0520465, -0.0196621, -0.0367468, -0.0199332, -0.0321134, 0.0170847
2: -0.0808753, 0.0223941, -0.0325901, 0.0215477, -0.1024230, 0.0549842
3: -0.1304372, -0.0143211, -0.0754993, -0.0148525, -0.1155847, 0.0611782
4: -0.0825893, 0.0231918, -0.0329242, 0.0223685, -0.1049578, 0.0561159

Time for backsubstitution: 3.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0072537, 0.0240102, 0.0179307, 0.0229756, -0.0157219, 0.0060795
1: -0.0520465, -0.0196621, -0.0338168, -0.0199495, -0.0320971, 0.0141547
2: -0.0808753, 0.0223941, -0.0233918, 0.0213817, -0.1022570, 0.0457859
3: -0.1304372, -0.0143211, -0.0650313, -0.0149578, -0.1154794, 0.0507102
4: -0.0825893, 0.0231918, -0.0234653, 0.0222070, -0.1047963, 0.0466570

Time for backsubstitution: 3.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0060306, 0.0241725, 0.0142645, 0.0231998, -0.0171692, 0.0099080
1: -0.0551499, -0.0196477, -0.0395153, -0.0198443, -0.0353056, 0.0198675
2: -0.0906109, 0.0227088, -0.0413835, 0.0215544, -0.1121653, 0.0640924
3: -0.1415163, -0.0141184, -0.0854943, -0.0148455, -0.1266708, 0.0713759
4: -0.0925998, 0.0235002, -0.0419661, 0.0223756, -0.1149754, 0.0654663

Time for backsubstitution: 3.11 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.72 + 418.97 = 422.70 seconds
