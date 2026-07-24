## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 10.310653145


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540)
1: (-4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358)
2: (-5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447)
3: (-6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520)
4: (-6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002)
5: (-5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936)
6: (-4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792)
7: (-5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823)
8: (-7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416)
9: (-4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.00 + 5.33 = 6.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -10.8533189, upper bound: 10.8533189

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8494828, upper bound: 10.8477722
time: 29.37 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8476955, upper bound: 10.8476954
time: 3.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 33.31 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 33.31
Output dim: 8, lower bound: -10.8494828, upper bound: 10.8477722
NS_B2, status: Status.UNKNOWN, split count: 1, time: 33.31
Output dim: 8, lower bound: -10.8476955, upper bound: 10.8476954

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -5.1619492, 4.2683043, -4.8091030, 3.9809673, -9.1429167, 9.0774078
1: -4.5150156, 4.0219212, -4.1900167, 3.7616804, -8.2766953, 8.2119370
2: -5.8974509, 4.1410961, -5.4816332, 3.8969140, -9.7943640, 9.6227293
3: -6.4188118, 3.7199407, -5.9486446, 3.4754472, -9.8942585, 9.6685848
4: -6.0448523, 4.4138484, -5.6237993, 4.1171808, -10.1620331, 10.0376472
5: -5.1020646, 4.2303286, -4.7429023, 3.9461298, -9.0481949, 8.9732304
6: -4.8802671, 4.7729130, -4.5476189, 4.4447222, -9.3249874, 9.3205318
7: -5.2257671, 5.0835156, -4.8612185, 4.7321424, -9.9579096, 9.9447346
8: -7.9439735, 4.0656700, -7.4275122, 3.8547773, -11.7987480, 11.4931812
9: -4.5706940, 4.8500509, -4.2459145, 4.5239472, -9.0946398, 9.0959644

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4588523, upper bound: 10.6268714
time: 1.52 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8439266, upper bound: 10.8439987
time: 1.74 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -4.3596220, 3.6176953, -4.8939848, 4.0257545, -8.3853760, 8.5116796
1: -3.7762024, 3.4348392, -4.2509437, 3.7665713, -7.5427737, 7.6857829
2: -4.9514799, 3.5895605, -5.5441694, 3.9399633, -8.8914433, 9.1337299
3: -5.3584423, 3.1831036, -5.8212814, 3.2330990, -8.5915403, 9.0043850
4: -5.0857267, 3.7469468, -5.6733103, 4.1140733, -9.1998005, 9.4202557
5: -4.2851748, 3.5885410, -4.7948580, 3.9627540, -8.2479286, 8.3833981
6: -4.1274104, 4.0326872, -4.5430226, 4.4767122, -8.6041222, 8.5757103
7: -4.3985171, 4.2867756, -4.9111648, 4.7876940, -9.1862106, 9.1979399
8: -6.7713509, 3.6085205, -7.5964932, 3.8544226, -10.6257734, 11.2050133
9: -3.8378708, 4.1155744, -4.2639904, 4.5530343, -8.3909035, 8.3795633

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4347758, upper bound: 10.5259260
time: 1.72 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8406201, upper bound: 10.8406201
time: 2.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.15 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 5.15
Output dim: 8, lower bound: -10.4588523, upper bound: 10.6268714
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 5.15
Output dim: 8, lower bound: -10.8439266, upper bound: 10.8439987
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 5.15
Output dim: 8, lower bound: -10.4347758, upper bound: 10.5259260
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 5.15
Output dim: 8, lower bound: -10.8406201, upper bound: 10.8406201

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -1.0109360, 1.0661721, -2.9171205, 2.4092093, -3.4201453, 3.9832926
1: -0.8210928, 0.9583632, -2.3849864, 2.3493662, -3.1704590, 3.3433495
2: -0.9250798, 1.2859070, -3.1864502, 2.5465593, -3.4716392, 4.4723573
3: -0.9066604, 1.0202264, -3.3806784, 2.1769562, -3.0836167, 4.4009047
4: -1.0571092, 0.9829249, -3.3000588, 2.5169952, -3.5741043, 4.2829838
5: -0.9172652, 1.0346648, -2.7793574, 2.4250696, -3.3423347, 3.8140221
6: -0.9817039, 0.9888433, -2.7171481, 2.6823256, -3.6640296, 3.7059913
7: -0.9717048, 1.0027300, -2.9073620, 2.8174405, -3.7891452, 3.9100919
8: -1.3888791, 2.6229784, -4.5574760, 2.9962656, -4.3851447, 7.1804543
9: -0.9480655, 1.1377145, -2.4900370, 2.7667885, -3.7148540, 3.6277514

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4492942, upper bound: 10.5779446
time: 2.25 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4492942, upper bound: 10.6268714
time: 2.08 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -3.7270885, 3.0890789, -4.2669077, 3.5424657, -7.2695541, 7.3559847
1: -3.1653371, 2.9619009, -3.6857367, 3.3653643, -6.5307016, 6.6476374
2: -4.1831040, 3.1329398, -4.8414631, 3.5199237, -7.7030277, 7.9744029
3: -4.5280991, 2.7820382, -5.2430515, 3.1252191, -7.6533184, 8.0250893
4: -4.3120480, 3.2161894, -4.9768176, 3.6704021, -7.9824500, 8.1930065
5: -3.6234443, 3.0854201, -4.1910505, 3.5149186, -7.1383629, 7.2764707
6: -3.5185735, 3.4486907, -4.0395508, 3.9490709, -7.4676447, 7.4882412
7: -3.7487261, 3.6495347, -4.3064013, 4.1961169, -7.9448433, 7.9559345
8: -5.8063211, 3.2942400, -6.6314263, 3.5575144, -9.3638344, 9.9256668
9: -3.2495794, 3.5275011, -3.7539837, 4.0308790, -7.2804585, 7.2814846

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6902139, upper bound: 10.6706181
time: 2.63 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6902139, upper bound: 10.6706181
time: 3.51 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -0.6166275, 0.7989485, -3.2797279, 2.7439437, -3.3605711, 4.0786762
1: -0.5099363, 0.6488933, -2.7206697, 2.5822659, -3.0922022, 3.3695631
2: -0.5919858, 0.9725808, -3.6041501, 2.8034062, -3.3953919, 4.5767307
3: -0.4385257, 0.7381055, -3.7181687, 2.2313666, -2.6698923, 4.4562740
4: -0.6488171, 0.6450522, -3.7274528, 2.7856221, -3.4344392, 4.3725052
5: -0.5853552, 0.7166393, -3.1374860, 2.6709588, -3.2563138, 3.8541253
6: -0.6016781, 0.6667634, -3.0243926, 3.0063031, -3.6079812, 3.6911559
7: -0.6159548, 0.6319572, -3.2546813, 3.1649461, -3.7809010, 3.8866386
8: -0.6625817, 2.5692225, -5.1391368, 3.1918633, -3.8544450, 7.7083592
9: -0.7083535, 0.8158872, -2.8064964, 3.0744371, -3.7827907, 3.6223836

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 126

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3262735, upper bound: 10.3842496
time: 2.01 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2340819, upper bound: 10.3269938
time: 1.75 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -3.0279815, 2.5012472, -4.4447494, 3.6675324, -6.6955137, 6.9459963
1: -2.4909463, 2.4402728, -3.8298991, 3.4398050, -5.9307513, 6.2701721
2: -3.3267605, 2.6378677, -5.0120091, 3.6230164, -6.9497762, 7.6498766
3: -3.5649118, 2.3094866, -5.2473907, 2.9641776, -6.5290890, 7.5568771
4: -3.4349449, 2.6212492, -5.1350389, 3.7489035, -7.1838484, 7.7562871
5: -2.8936615, 2.5241451, -4.3345184, 3.6049585, -6.4986191, 6.8586636
6: -2.8353622, 2.7921808, -4.1258402, 4.0685873, -6.9039497, 6.9180212
7: -3.0196805, 2.9321327, -4.4470344, 4.3408108, -7.3604913, 7.3791661
8: -4.7286358, 3.0354779, -6.9228234, 3.6378598, -8.3664951, 9.9583015
9: -2.5949152, 2.8791103, -3.8593097, 4.1436839, -6.7385974, 6.7384200

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5259260, upper bound: 10.4347758
time: 1.99 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5259260, upper bound: 10.8406201
time: 1.86 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.75 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.75
Output dim: 8, lower bound: -10.4492942, upper bound: 10.5779446
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.75
Output dim: 8, lower bound: -10.4492942, upper bound: 10.6268714
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.75
Output dim: 8, lower bound: -10.6902139, upper bound: 10.6706181
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.75
Output dim: 8, lower bound: -10.6902139, upper bound: 10.6706181
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.75
Output dim: 8, lower bound: -10.3262735, upper bound: 10.3842496
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.75
Output dim: 8, lower bound: -10.2340819, upper bound: 10.3269938
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.75
Output dim: 8, lower bound: -10.5259260, upper bound: 10.4347758
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.75
Output dim: 8, lower bound: -10.5259260, upper bound: 10.8406201

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.0109360, 1.0661721, -0.8224976, 0.9526591, -1.9635952, 1.8886697
1: -0.8210928, 0.9583632, -0.6783867, 0.8205066, -1.6415994, 1.6367500
2: -0.9250798, 1.2859070, -0.7437838, 1.1596640, -2.0847437, 2.0296907
3: -0.9066604, 1.0202264, -0.6810015, 0.9041843, -1.8108448, 1.7012279
4: -1.0571092, 0.9829249, -0.8567464, 0.8332369, -1.8903461, 1.8396714
5: -0.9172652, 1.0346648, -0.7553797, 0.8981278, -1.8153930, 1.7900444
6: -0.9817039, 0.9888433, -0.8005887, 0.8345573, -1.8162613, 1.7894320
7: -0.9717048, 1.0027300, -0.8075277, 0.8357326, -1.8074374, 1.8102577
8: -1.3888791, 2.6229784, -1.0656095, 2.6009374, -3.9898164, 3.6885879
9: -0.9480655, 1.1377145, -0.8306347, 0.9904779, -1.9385433, 1.9683492

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4068180, upper bound: 10.5138878
time: 2.57 seconds

## Relational analysis of NS_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2893873, upper bound: 10.4274707
time: 1.69 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.0109360, 1.0661721, -3.3821220, 2.7721789, -3.7831149, 4.4482942
1: -0.8210928, 0.9583632, -2.8427665, 2.6969657, -3.5180585, 3.8011298
2: -0.9250798, 1.2859070, -3.7620134, 2.8720245, -3.7971044, 5.0479202
3: -0.9066604, 1.0202264, -4.0491743, 2.5488682, -3.4555287, 5.0694008
4: -1.0571092, 0.9829249, -3.8139873, 2.9232626, -3.9803720, 4.7969122
5: -0.9172652, 1.0346648, -3.2598171, 2.7886219, -3.7058871, 4.2944818
6: -0.9817039, 0.9888433, -3.1692996, 3.1174283, -4.0991321, 4.1581430
7: -0.9717048, 1.0027300, -3.3878736, 3.3019173, -4.2736220, 4.3906035
8: -1.3888791, 2.6229784, -5.2810340, 3.1540082, -4.5428872, 7.9040127
9: -0.9480655, 1.1377145, -2.9147217, 3.1973007, -4.1453662, 4.0524364

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_B1_A1_B2_B1

### Relational analysis result of NS_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2894134, upper bound: 10.4737945
time: 2.01 seconds

## Relational analysis of NS_B1_A1_B2_B2

### Relational analysis result of NS_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2893873, upper bound: 10.4738321
time: 2.72 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.7270885, 3.0890789, -0.8224976, 0.9526591, -4.6797476, 3.9115765
1: -3.1653371, 2.9619009, -0.6783867, 0.8205066, -3.9858437, 3.6402876
2: -4.1831040, 3.1329398, -0.7437838, 1.1596640, -5.3427682, 3.8767238
3: -4.5280991, 2.7820382, -0.6810015, 0.9041843, -5.4322834, 3.4630396
4: -4.3120480, 3.2161894, -0.8567464, 0.8332369, -5.1452847, 4.0729361
5: -3.6234443, 3.0854201, -0.7553797, 0.8981278, -4.5215721, 3.8407998
6: -3.5185735, 3.4486907, -0.8005887, 0.8345573, -4.3531308, 4.2492795
7: -3.7487261, 3.6495347, -0.8075277, 0.8357326, -4.5844588, 4.4570622
8: -5.8063211, 3.2942400, -1.0656095, 2.6009374, -8.4072590, 4.3598495
9: -3.2495794, 3.5275011, -0.8306347, 0.9904779, -4.2400575, 4.3581357

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6421804, upper bound: 10.6272372
time: 2.19 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6288355, upper bound: 10.6066275
time: 4.06 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.7270885, 3.0890789, -3.4052205, 2.8175251, -6.5446138, 6.4942994
1: -3.1653371, 2.9619009, -2.8558207, 2.7208309, -5.8861680, 5.8177214
2: -4.1831040, 3.1329398, -3.7890821, 2.9046569, -7.0877609, 6.9220219
3: -4.5280991, 2.7820382, -4.0843210, 2.5608428, -7.0889416, 6.8663592
4: -4.3120480, 3.2161894, -3.9089458, 2.9427168, -7.2547646, 7.1251354
5: -3.6234443, 3.0854201, -3.2855368, 2.8261533, -6.4495974, 6.3709569
6: -3.5185735, 3.4486907, -3.2041881, 3.1468105, -6.6653843, 6.6528788
7: -3.7487261, 3.6495347, -3.4125867, 3.3201020, -7.0688281, 7.0621214
8: -5.8063211, 3.2942400, -5.3120508, 3.1667218, -8.9730434, 8.6062908
9: -3.2495794, 3.5275011, -2.9472203, 3.2273698, -6.4769492, 6.4747214

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6421805, upper bound: 10.8415949
time: 2.29 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6288356, upper bound: 10.8377708
time: 2.60 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.4399655, 0.6378927, -3.0567014, 2.5617790, -3.0017445, 3.6945941
1: -0.3653784, 0.4780297, -2.5034494, 2.4206457, -2.7860241, 2.9814792
2: -0.5046318, 0.7774706, -3.3322976, 2.6494820, -3.1541138, 4.1097679
3: -0.2791503, 0.5823917, -3.4266276, 2.1030025, -2.3821528, 4.0090194
4: -0.4951183, 0.4719853, -3.4517324, 2.5978360, -3.0929544, 3.9237177
5: -0.4196697, 0.5732877, -2.9084036, 2.4962811, -2.9159508, 3.4816914
6: -0.4193625, 0.5160049, -2.8123510, 2.8000989, -3.2194614, 3.3283558
7: -0.4618016, 0.4638760, -3.0223713, 2.9365859, -3.3983874, 3.4862473
8: -0.3309709, 2.4645641, -4.7895365, 3.1105521, -3.4415231, 7.2541008
9: -0.6216207, 0.6466268, -2.6092424, 2.8690686, -3.4906893, 3.2558694

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0467749, upper bound: 10.0726482
time: 1.74 seconds

## Relational analysis of NS_B2_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0942922, upper bound: 10.1571122
time: 1.49 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -1.2261815, 1.3062198, -2.9545913, 2.4783230, -3.7045045, 4.2608109
1: -0.9791979, 1.1733789, -2.4030476, 2.3472841, -3.3264818, 3.5764265
2: -0.9199095, 1.6271781, -3.2065151, 2.5789423, -3.4988518, 4.8336930
3: -1.0922583, 1.1746113, -3.2927847, 2.0425429, -3.1348014, 4.4673958
4: -1.1806741, 1.2008911, -3.3244524, 2.5110183, -3.6916924, 4.5253434
5: -1.0925370, 1.1990054, -2.8029633, 2.4170940, -3.5096312, 4.0019684
6: -1.1249063, 1.1679790, -2.7152414, 2.7055042, -3.8304105, 3.8832204
7: -1.1833377, 1.1946260, -2.9153070, 2.8318524, -4.0151901, 4.1099329
8: -1.7498575, 2.5849457, -4.6279354, 3.0671742, -4.8170319, 7.2128811
9: -1.0911226, 1.3417611, -2.5209908, 2.7739432, -3.8650658, 3.8627520

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2093583, upper bound: 10.2093583
time: 2.09 seconds

## Relational analysis of NS_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2093583, upper bound: 10.3269938
time: 1.63 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.0279815, 2.5012472, -1.2885556, 1.2915748, -4.3195562, 3.7898028
1: -2.4909463, 2.4402728, -0.9978373, 1.1210864, -3.6120327, 3.4381101
2: -3.3267605, 2.6378677, -1.1817958, 1.4661613, -4.7929220, 3.8196635
3: -3.5649118, 2.3094866, -1.2152059, 1.0104641, -4.5753756, 3.5246925
4: -3.4349449, 2.6212492, -1.3690012, 1.1674459, -4.6023908, 3.9902503
5: -2.8936615, 2.5241451, -1.1490538, 1.2066720, -4.1003332, 3.6731989
6: -2.8353622, 2.7921808, -1.1961780, 1.2265388, -4.0619011, 3.9883587
7: -3.0196805, 2.9321327, -1.2147151, 1.2315087, -4.2511892, 4.1468477
8: -4.7286358, 3.0354779, -1.8933592, 2.8310375, -7.5596733, 4.9288368
9: -2.5949152, 2.8791103, -1.1436851, 1.3561463, -3.9510615, 4.0227957

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3679715, upper bound: 10.2743488
time: 1.60 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
time: 2.22 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.0279815, 2.5012472, -3.6654465, 3.0519638, -6.0799456, 6.1666927
1: -2.4909463, 2.4402728, -3.0935688, 2.8719342, -5.3628807, 5.5338416
2: -3.3267605, 2.6378677, -4.0813146, 3.0818248, -6.4085855, 6.7191820
3: -3.5649118, 2.3094866, -4.2469306, 2.4954326, -6.0603447, 6.5564175
4: -3.4349449, 2.6212492, -4.2021399, 3.1118362, -6.5467811, 6.8233891
5: -2.8936615, 2.5241451, -3.5362399, 2.9846327, -5.8782940, 6.0603848
6: -2.8353622, 2.7921808, -3.4004350, 3.3648591, -6.2002211, 6.1926155
7: -3.0196805, 2.9321327, -3.6567678, 3.5623693, -6.5820498, 6.5889006
8: -4.7286358, 3.0354779, -5.7447891, 3.3173430, -8.0459785, 8.7802668
9: -2.5949152, 2.8791103, -3.1585369, 3.4368827, -6.0317979, 6.0376472

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 167

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3679716, upper bound: 10.8014411
time: 1.83 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.8002446
time: 3.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 9.49 seconds
NS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.4068180, upper bound: 10.5138878
NS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.2893873, upper bound: 10.4274707
NS_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.2894134, upper bound: 10.4737945
NS_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.2893873, upper bound: 10.4738321
NS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.6421804, upper bound: 10.6272372
NS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.6288355, upper bound: 10.6066275
NS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.6421805, upper bound: 10.8415949
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.6288356, upper bound: 10.8377708
NS_B2_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.0467749, upper bound: 10.0726482
NS_B2_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.0942922, upper bound: 10.1571122
NS_B2_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.2093583, upper bound: 10.2093583
NS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.2093583, upper bound: 10.3269938
NS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.3679715, upper bound: 10.2743488
NS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
NS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.3679716, upper bound: 10.8014411
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 9.49
Output dim: 8, lower bound: -10.3639818, upper bound: 10.8002446

## BFS NS instance: NS_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.5823044, 0.7623762, -0.6612955, 0.8342103, -1.4165146, 1.4236717
1: -0.4856315, 0.6175168, -0.5503117, 0.6839608, -1.1695924, 1.1678284
2: -0.5682570, 0.9213098, -0.6182889, 1.0145186, -1.5827756, 1.5395987
3: -0.4081050, 0.7298506, -0.4953497, 0.7858531, -1.1939582, 1.2252003
4: -0.6276305, 0.6173378, -0.7009859, 0.6903644, -1.3179948, 1.3183237
5: -0.5609183, 0.6906986, -0.6246184, 0.7620667, -1.3229849, 1.3153170
6: -0.5705864, 0.6374269, -0.6492186, 0.6990575, -1.2696440, 1.2866454
7: -0.5943890, 0.6131480, -0.6666925, 0.6861237, -1.2805127, 1.2798405
8: -0.5885465, 2.5068948, -0.7507786, 2.5520401, -3.1405866, 3.2576735
9: -0.6914485, 0.7806575, -0.7313614, 0.8553208, -1.5467693, 1.5120189

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_A1_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1441971, upper bound: 10.2156970
time: 6.60 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1835947, upper bound: 10.2971485
time: 2.13 seconds

## BFS NS instance: NS_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.9637101, 1.0354692, -0.6780854, 0.8468478, -1.8105578, 1.7135546
1: -0.7814004, 0.9083995, -0.5643547, 0.6980037, -1.4794042, 1.4727542
2: -0.8783789, 1.2366799, -0.6298130, 1.0325100, -1.9108889, 1.8664929
3: -0.8607405, 0.9741442, -0.5138784, 0.7996606, -1.6604011, 1.4880226
4: -1.0190847, 0.9430204, -0.7166800, 0.7061533, -1.7252380, 1.6597004
5: -0.8752270, 0.9973388, -0.6383705, 0.7770494, -1.6522763, 1.6357093
6: -0.9339508, 0.9458307, -0.6658063, 0.7115808, -1.6455317, 1.6116370
7: -0.9377082, 0.9708058, -0.6815936, 0.7025371, -1.6402452, 1.6523993
8: -1.3006881, 2.5960696, -0.7836897, 2.5595200, -3.8602080, 3.3797593
9: -0.9164696, 1.0967474, -0.7409934, 0.8701606, -1.7866302, 1.8377408

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0290989, upper bound: 10.1379494
time: 1.94 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0496060, upper bound: 10.2000592
time: 1.85 seconds

## BFS NS instance: NS_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.7887654, 0.9257174, -2.6256793, 2.1719642, -2.9607296, 3.5513966
1: -0.6532334, 0.7889489, -2.1261425, 2.1281936, -2.7814269, 2.9150915
2: -0.7158680, 1.1270115, -2.8324234, 2.3360937, -3.0519617, 3.9594350
3: -0.6480175, 0.8900136, -3.0545502, 2.0663738, -2.7143912, 3.9445639
4: -0.8309270, 0.8052841, -2.9825959, 2.2827604, -3.1136873, 3.7878799
5: -0.7283072, 0.8741019, -2.4819634, 2.2193336, -2.9476409, 3.3560653
6: -0.7714838, 0.8040364, -2.4621646, 2.4203858, -3.1918697, 3.2662010
7: -0.7835889, 0.8117791, -2.6212811, 2.5360684, -3.3196573, 3.4330602
8: -1.0010104, 2.5708289, -4.0910535, 2.8534055, -3.8544159, 6.6618824
9: -0.8102062, 0.9621032, -2.2357306, 2.5189247, -3.3291309, 3.1978340

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B1_A1_B2_B1_A1

### Relational analysis result of NS_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
time: 2.40 seconds

## Relational analysis of NS_B1_A1_B2_B1_A2

### Relational analysis result of NS_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
time: 2.44 seconds

## BFS NS instance: NS_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.8113918, 0.9408661, -3.2581463, 2.6739514, -3.4853432, 4.1990123
1: -0.6715765, 0.8060303, -2.7026410, 2.5836129, -3.2551894, 3.5086713
2: -0.7354344, 1.1455388, -3.6069539, 2.7651370, -3.5005713, 4.7524929
3: -0.6744518, 0.9045583, -3.8830571, 2.4436963, -3.1181481, 4.7876153
4: -0.8525327, 0.8247910, -3.6885266, 2.8131232, -3.6656559, 4.5133176
5: -0.7470209, 0.8913012, -3.1369815, 2.6878891, -3.4349101, 4.0282826
6: -0.7924621, 0.8231571, -3.0517068, 2.9951763, -3.7876384, 3.8748639
7: -0.8026726, 0.8314747, -3.2679448, 3.1787872, -3.9814599, 4.0994196
8: -1.0425594, 2.5784807, -5.0800896, 3.0789616, -4.1215210, 7.6585703
9: -0.8232400, 0.9810954, -2.8009467, 3.0785897, -3.9018297, 3.7820420

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B1_A1_B2_B2_A1

### Relational analysis result of NS_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4738320
time: 1.96 seconds

## Relational analysis of NS_B1_A1_B2_B2_A2

### Relational analysis result of NS_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4738320
time: 2.03 seconds

## BFS NS instance: NS_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.9328561, 2.4127960, -0.6612955, 0.8342103, -3.7670665, 3.0740914
1: -2.3821862, 2.3555315, -0.5503117, 0.6839608, -3.0661469, 2.9058433
2: -3.2099998, 2.5388985, -0.6182889, 1.0145186, -4.2245183, 3.1571875
3: -3.4837213, 2.2754459, -0.4953497, 0.7858531, -4.2695742, 2.7707956
4: -3.3391724, 2.5442429, -0.7009859, 0.6903644, -4.0295367, 3.2452288
5: -2.8007469, 2.4521141, -0.6246184, 0.7620667, -3.5628135, 3.0767326
6: -2.7534752, 2.7012711, -0.6492186, 0.6990575, -3.4525328, 3.3504896
7: -2.9355774, 2.8502209, -0.6666925, 0.6861237, -3.6217012, 3.5169134
8: -4.5668526, 2.9443765, -0.7507786, 2.5520401, -7.1188927, 3.6951551
9: -2.5102952, 2.7934465, -0.7313614, 0.8553208, -3.3656158, 3.5248079

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B1_A2_B1_A1_A1

### Relational analysis result of NS_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3330525, upper bound: 10.3526300
time: 1.88 seconds

## Relational analysis of NS_B1_A2_B1_A1_A2

### Relational analysis result of NS_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4414304, upper bound: 10.4247140
time: 2.68 seconds

## BFS NS instance: NS_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.6088955, 2.9799416, -0.6780854, 0.8468478, -4.4557433, 3.6580272
1: -3.0334468, 2.8518283, -0.5643547, 0.6980037, -3.7314506, 3.4161830
2: -4.0335865, 3.0179672, -0.6298130, 1.0325100, -5.0660963, 3.6477802
3: -4.3737674, 2.6712751, -0.5138784, 0.7996606, -5.1734281, 3.1851535
4: -4.1703949, 3.1077013, -0.7166800, 0.7061533, -4.8765483, 3.8243814
5: -3.5007262, 2.9792786, -0.6383705, 0.7770494, -4.2777758, 3.6176491
6: -3.3946471, 3.3272810, -0.6658063, 0.7115808, -4.1062279, 3.9930873
7: -3.6294036, 3.5355167, -0.6815936, 0.7025371, -4.3319407, 4.2171102
8: -5.6099930, 3.2119558, -0.7836897, 2.5595200, -8.1695127, 3.9956455
9: -3.1330554, 3.4108317, -0.7409934, 0.8701606, -4.0032158, 4.1518250

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B1_A2_B1_A2_A1

### Relational analysis result of NS_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3131469, upper bound: 10.3267432
time: 2.01 seconds

## Relational analysis of NS_B1_A2_B1_A2_A2

### Relational analysis result of NS_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4279370, upper bound: 10.4030513
time: 2.57 seconds

## BFS NS instance: NS_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.9328561, 2.4127960, -3.0883577, 2.5467279, -5.4795837, 5.5011539
1: -2.3821862, 2.3555315, -2.5388188, 2.4750776, -4.8572636, 4.8943501
2: -3.2099998, 2.5388985, -3.4013002, 2.6653361, -5.8753357, 5.9401989
3: -3.4837213, 2.2754459, -3.6682503, 2.3589551, -5.8426762, 5.9436960
4: -3.3391724, 2.5442429, -3.5202122, 2.6734188, -6.0125914, 6.0644550
5: -2.8007469, 2.4521141, -2.9583192, 2.5731437, -5.3738909, 5.4104333
6: -2.7534752, 2.7012711, -2.8972917, 2.8470900, -5.6005650, 5.5985627
7: -2.9355774, 2.8502209, -3.0888228, 3.0008030, -5.9363804, 5.9390440
8: -4.5668526, 2.9443765, -4.8144937, 3.0287681, -7.5956206, 7.7588701
9: -2.5102952, 2.7934465, -2.6515465, 2.9349995, -5.4452944, 5.4449930

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_B1_A2_B2_A1_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377654
time: 2.26 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377707
time: 3.27 seconds

## BFS NS instance: NS_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.6088955, 2.9799416, -3.1183994, 2.5732698, -6.1821651, 6.0983410
1: -3.0334468, 2.8518283, -2.5712657, 2.5010595, -5.5345063, 5.4230938
2: -4.0335865, 3.0179672, -3.4382429, 2.6907985, -6.7243853, 6.4562101
3: -4.3737674, 2.6712751, -3.7073720, 2.3794723, -6.7532396, 6.3786469
4: -4.1703949, 3.1077013, -3.5562634, 2.6996932, -6.8700881, 6.6639647
5: -3.5007262, 2.9792786, -2.9893138, 2.5976183, -6.0983448, 5.9685926
6: -3.3946471, 3.3272810, -2.9270360, 2.8762727, -6.2709198, 6.2543173
7: -3.6294036, 3.5355167, -3.1186116, 3.0306363, -6.6600399, 6.6541281
8: -5.6099930, 3.2119558, -4.8624253, 3.0457692, -8.6557617, 8.0743809
9: -3.1330554, 3.4108317, -2.6803725, 2.9632652, -6.0963206, 6.0912042

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_B1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377654
time: 2.56 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377708
time: 2.65 seconds

## BFS NS instance: NS_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -1.2261815, 1.3062198, -3.3253064, 2.7745113, -4.0006928, 4.6315260
1: -0.9791979, 1.1733789, -2.7683277, 2.6246052, -3.6038032, 3.9417067
2: -0.9199095, 1.6271781, -3.6687129, 2.8438945, -3.7638040, 5.2958908
3: -1.0922583, 1.1746113, -3.8050508, 2.3003664, -3.3926249, 4.9796619
4: -1.1806741, 1.2008911, -3.7885990, 2.8257647, -4.0064387, 4.9894900
5: -1.0925370, 1.1990054, -3.1907544, 2.7119479, -3.8044848, 4.3897600
6: -1.1249063, 1.1679790, -3.0704439, 3.0525312, -4.1774378, 4.2384229
7: -1.1833377, 1.1946260, -3.3012130, 3.2198911, -4.4032288, 4.4958391
8: -1.7498575, 2.5849457, -5.2129774, 3.1725335, -4.9223909, 7.7979231
9: -1.0911226, 1.3417611, -2.8529987, 3.1251621, -4.2162848, 4.1947598

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B2_A1_A2_B2_B1

### Relational analysis result of NS_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2093583, upper bound: 10.3269938
time: 1.51 seconds

## Relational analysis of NS_B2_A1_A2_B2_B2

### Relational analysis result of NS_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2093583, upper bound: 10.3269938
time: 2.03 seconds

## BFS NS instance: NS_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.2877104, 1.9087899, -1.0731872, 1.1607205, -3.4484310, 2.9819770
1: -1.8476956, 1.8796308, -0.8373573, 0.9601887, -2.8078842, 2.7169881
2: -2.4105849, 2.1123986, -0.9714405, 1.3179774, -3.7285624, 3.0838392
3: -2.5813680, 1.8356632, -0.9692649, 0.9019313, -3.4832993, 2.8049281
4: -2.5926938, 1.9941437, -1.1269585, 1.0061640, -3.5988579, 3.1211023
5: -2.1247373, 1.9644263, -0.9575926, 1.0531656, -3.1779027, 2.9220190
6: -2.1501877, 2.1079900, -1.0043336, 1.0486889, -3.1988766, 3.1123238
7: -2.2694120, 2.1918299, -1.0183607, 1.0607555, -3.3301675, 3.2101908
8: -3.5469000, 2.7682288, -1.5200527, 2.7787313, -6.3256311, 4.2882814
9: -1.9375224, 2.2168264, -0.9959816, 1.1858504, -3.1233728, 3.2128081

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_B2_A2_B1_A1_B1

### Relational analysis result of NS_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
time: 1.99 seconds

## Relational analysis of NS_B2_A2_B1_A1_B2

### Relational analysis result of NS_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
time: 1.99 seconds

## BFS NS instance: NS_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.9069414, 2.3972228, -1.0983660, 1.1752684, -4.0822096, 3.4955888
1: -2.3556957, 2.3274298, -0.8553005, 0.9820984, -3.3377941, 3.1827302
2: -3.1762242, 2.5260589, -0.9955319, 1.3360803, -4.5123043, 3.5215907
3: -3.4031565, 2.2061355, -0.9972059, 0.9149535, -4.3181100, 3.2033415
4: -3.2982602, 2.5139987, -1.1532403, 1.0256796, -4.3239398, 3.6672392
5: -2.7725687, 2.4218564, -0.9797614, 1.0711894, -3.8437581, 3.4016178
6: -2.7182167, 2.6712651, -1.0268713, 1.0697291, -3.7879457, 3.6981363
7: -2.9044781, 2.8140368, -1.0407318, 1.0799347, -3.9844127, 3.8547688
8: -4.5336490, 2.9664898, -1.5650713, 2.7871346, -7.3207836, 4.5315609
9: -2.4802496, 2.7612855, -1.0132402, 1.2059033, -3.6861529, 3.7745256

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_B2_A2_B1_A2_B1

### Relational analysis result of NS_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
time: 1.82 seconds

## Relational analysis of NS_B2_A2_B1_A2_B2

### Relational analysis result of NS_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
time: 1.88 seconds

## BFS NS instance: NS_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.2877104, 1.9087899, -3.3875391, 2.8261530, -5.1138635, 5.2963290
1: -1.8476956, 1.8796308, -2.8144519, 2.6556194, -4.5033150, 4.6940827
2: -2.4105849, 2.1123986, -3.7427049, 2.8758640, -5.2864490, 5.8551035
3: -2.5813680, 1.8356632, -3.8901303, 2.3231845, -4.9045525, 5.7257934
4: -2.5926938, 1.9941437, -3.8720617, 2.8785496, -5.4712434, 5.8662052
5: -2.1247373, 1.9644263, -3.2553036, 2.7616289, -4.8863659, 5.2197299
6: -2.1501877, 2.1079900, -3.1378045, 3.1051850, -5.2553730, 5.2457943
7: -2.2694120, 2.1918299, -3.3748651, 3.2849381, -5.5543499, 5.5666952
8: -3.5469000, 2.7682288, -5.3076253, 3.1945670, -6.7414670, 8.0758543
9: -1.9375224, 2.2168264, -2.9076352, 3.1786747, -5.1161971, 5.1244617

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_B2_A2_B2_A1_B1

### Relational analysis result of NS_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445
time: 1.42 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2

### Relational analysis result of NS_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445
time: 1.54 seconds

## BFS NS instance: NS_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.9069414, 2.3972228, -3.4131951, 2.8467779, -5.7537193, 5.8104181
1: -2.3556957, 2.3274298, -2.8427856, 2.6781769, -5.0338726, 5.1702156
2: -3.1762242, 2.5260589, -3.7738323, 2.8958693, -6.0720935, 6.2998915
3: -3.4031565, 2.2061355, -3.9218144, 2.3386483, -5.7418051, 6.1279497
4: -3.2982602, 2.5139987, -3.9016438, 2.9006772, -6.1989374, 6.4156427
5: -2.7725687, 2.4218564, -3.2805617, 2.7825580, -5.5551267, 5.7024183
6: -2.7182167, 2.6712651, -3.1618438, 3.1299157, -5.8481321, 5.8331089
7: -2.9044781, 2.8140368, -3.4003630, 3.3101737, -6.2146521, 6.2143998
8: -4.5336490, 2.9664898, -5.3490891, 3.2094810, -7.7431297, 8.3155785
9: -2.4802496, 2.7612855, -2.9307275, 3.2028346, -5.6830845, 5.6920128

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_B2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445
time: 1.66 seconds

## Relational analysis of NS_B2_A2_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445
time: 2.92 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.46 seconds
NS_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.1441971, upper bound: 10.2156970
NS_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.1835947, upper bound: 10.2971485
NS_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.0290989, upper bound: 10.1379494
NS_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.0496060, upper bound: 10.2000592
NS_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
NS_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
NS_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4738320
NS_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4738320
NS_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.3330525, upper bound: 10.3526300
NS_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.4414304, upper bound: 10.4247140
NS_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.3131469, upper bound: 10.3267432
NS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.4279370, upper bound: 10.4030513
NS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377654
NS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377707
NS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377654
NS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377708
NS_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.2093583, upper bound: 10.3269938
NS_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.2093583, upper bound: 10.3269938
NS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
NS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
NS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
NS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
NS_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445
NS_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445
NS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445
NS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.46
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445

## BFS NS instance: NS_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5823044, 0.7623762, -2.6256793, 2.1719642, -2.7542686, 3.3880553
1: -0.4856315, 0.6175168, -2.1261425, 2.1281936, -2.6138251, 2.7436593
2: -0.5682570, 0.9213098, -2.8324234, 2.3360937, -2.9043508, 3.7537332
3: -0.4081050, 0.7298506, -3.0545502, 2.0663738, -2.4744787, 3.7844007
4: -0.6276305, 0.6173378, -2.9825959, 2.2827604, -2.9103909, 3.5999336
5: -0.5609183, 0.6906986, -2.4819634, 2.2193336, -2.7802520, 3.1726620
6: -0.5705864, 0.6374269, -2.4621646, 2.4203858, -2.9909723, 3.0995915
7: -0.5943890, 0.6131480, -2.6212811, 2.5360684, -3.1304574, 3.2344291
8: -0.5885465, 2.5068948, -4.0910535, 2.8534055, -3.4419520, 6.5979481
9: -0.6914485, 0.7806575, -2.2357306, 2.5189247, -3.2103732, 3.0163882

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_A1_B2_B1_A1_B1

### Relational analysis result of NS_B1_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0379905, upper bound: 10.1752205
time: 1.46 seconds

## Relational analysis of NS_B1_A1_B2_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0622791, upper bound: 10.2518600
time: 1.81 seconds

## BFS NS instance: NS_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.9398836, 1.0319059, -2.6256793, 2.1719642, -3.1118479, 3.6575851
1: -0.7659042, 0.8978448, -2.1261425, 2.1281936, -2.8940978, 3.0239873
2: -0.8471785, 1.2306426, -2.8324234, 2.3360937, -3.1832721, 4.0630660
3: -0.8361225, 0.9697104, -3.0545502, 2.0663738, -2.9024963, 4.0242605
4: -0.9810215, 0.9360934, -2.9825959, 2.2827604, -3.2637818, 3.9186893
5: -0.8448707, 0.9894065, -2.4819634, 2.2193336, -3.0642045, 3.4713700
6: -0.9088482, 0.9264999, -2.4621646, 2.4203858, -3.3292341, 3.3886645
7: -0.9186203, 0.9539633, -2.6212811, 2.5360684, -3.4546888, 3.5752444
8: -1.2739576, 2.5951953, -4.0910535, 2.8534055, -4.1273632, 6.6862488
9: -0.9020308, 1.0831215, -2.2357306, 2.5189247, -3.4209557, 3.3188522

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_B1_A1_B2_B1_A2_A1

### Relational analysis result of NS_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2176350, upper bound: 10.3236621
time: 2.26 seconds

## Relational analysis of NS_B1_A1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_B1_A1_B2_B1_A2_B1

### Relational analysis result of NS_B1_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1325542, upper bound: 10.3064628
time: 3.32 seconds

## Relational analysis of NS_B1_A1_B2_B1_A2_B2

### Relational analysis result of NS_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.1340738, upper bound: 10.3132061
time: 2.39 seconds

## BFS NS instance: NS_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5823044, 0.7623762, -3.2581463, 2.6739514, -3.2562559, 4.0205226
1: -0.4856315, 0.6175168, -2.7026410, 2.5836129, -3.0692444, 3.3201578
2: -0.5682570, 0.9213098, -3.6069539, 2.7651370, -3.3333941, 4.5282636
3: -0.4081050, 0.7298506, -3.8830571, 2.4436963, -2.8518014, 4.6129079
4: -0.6276305, 0.6173378, -3.6885266, 2.8131232, -3.4407537, 4.3058643
5: -0.5609183, 0.6906986, -3.1369815, 2.6878891, -3.2488074, 3.8276801
6: -0.5705864, 0.6374269, -3.0517068, 2.9951763, -3.5657628, 3.6891336
7: -0.5943890, 0.6131480, -3.2679448, 3.1787872, -3.7731762, 3.8810928
8: -0.5885465, 2.5068948, -5.0800896, 3.0789616, -3.6675081, 7.5869846
9: -0.6914485, 0.7806575, -2.8009467, 3.0785897, -3.7700381, 3.5816042

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_A1_B2_B2_A1_B1

### Relational analysis result of NS_B1_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0376586, upper bound: 10.1745094
time: 2.04 seconds

## Relational analysis of NS_B1_A1_B2_B2_A1_B2

### Relational analysis result of NS_B1_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0622790, upper bound: 10.2521523
time: 1.63 seconds

## BFS NS instance: NS_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.9421180, 1.0320566, -3.2581463, 2.6739514, -3.6160693, 4.2902031
1: -0.7760738, 0.8978448, -2.7026410, 2.5836129, -3.3596866, 3.6004858
2: -0.8541905, 1.2306426, -3.6069539, 2.7651370, -3.6193275, 4.8375964
3: -0.8372067, 0.9717373, -3.8830571, 2.4436963, -3.2809029, 4.8547945
4: -0.9941624, 0.9360934, -3.6885266, 2.8131232, -3.8072855, 4.6246200
5: -0.8557203, 0.9894065, -3.1369815, 2.6878891, -3.5436094, 4.1263881
6: -0.9100110, 0.9366938, -3.0517068, 2.9951763, -3.9051874, 3.9884007
7: -0.9200113, 0.9590390, -3.2679448, 3.1787872, -4.0987988, 4.2269840
8: -1.2825596, 2.5954096, -5.0800896, 3.0789616, -4.3615212, 7.6754990
9: -0.9054558, 1.0903533, -2.8009467, 3.0785897, -3.9840455, 3.8913000

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_B1_A1_B2_B2_A2_A1

### Relational analysis result of NS_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2143202, upper bound: 10.3190469
time: 2.16 seconds

## Relational analysis of NS_B1_A1_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of NS_B1_A1_B2_B2_A2_A1

### Relational analysis result of NS_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4735759
time: 1.65 seconds

## Relational analysis of NS_B1_A1_B2_B2_A2_A2

### Relational analysis result of NS_B1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
time: 2.41 seconds

## BFS NS instance: NS_B1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.5681466, 0.7484194, -0.3530665, 0.5474774, -1.1156240, 1.1014860
1: -0.5009052, 0.6149521, -0.2814330, 0.3940346, -0.8949398, 0.8963851
2: -0.5307309, 0.8295106, -0.4576199, 0.5924907, -1.1232216, 1.2871305
3: -0.4029301, 0.7945162, -0.2134947, 0.4570267, -0.8599569, 1.0080109
4: -0.6048436, 0.6257648, -0.4229846, 0.3691886, -0.9740322, 1.0487494
5: -0.5558202, 0.6852477, -0.3434045, 0.4702854, -1.0261056, 1.0286522
6: -0.5501204, 0.6316398, -0.3112675, 0.4268531, -0.9769734, 0.9429073
7: -0.5907705, 0.6063630, -0.3910034, 0.3669192, -0.9576896, 0.9973664
8: -0.5369560, 2.4385779, -0.0828341, 2.4528990, -2.9898548, 2.5214119
9: -0.6703390, 0.7592965, -0.5785313, 0.5378324, -1.2081714, 1.3378279

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_B1_A2_B1_A1_A1_A1

### Relational analysis result of NS_B1_A2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.9099598, upper bound: 9.8949006
time: 2.38 seconds

## Relational analysis of NS_B1_A2_B1_A1_A1_A2

### Relational analysis result of NS_B1_A2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.6570093, upper bound: 9.6246803
time: 2.06 seconds

## BFS NS instance: NS_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -1.8043153, 1.5551851, -0.4838380, 0.6839600, -2.4882753, 2.0390229
1: -1.4490350, 1.5350487, -0.4027784, 0.5167377, -1.9657727, 1.9378271
2: -1.7930186, 1.7759449, -0.5273594, 0.8013871, -2.5944057, 2.3033042
3: -1.9157420, 1.5462267, -0.3098397, 0.6330364, -2.5487785, 1.8560665
4: -2.0220578, 1.6002737, -0.5277487, 0.5282099, -2.5502677, 2.1280224
5: -1.6570227, 1.6066689, -0.4677826, 0.6096593, -2.2666821, 2.0744514
6: -1.7033873, 1.6698742, -0.4667977, 0.5552092, -2.2585964, 2.1366720
7: -1.7596881, 1.7187676, -0.5046337, 0.5103711, -2.2700593, 2.2234013
8: -2.7422221, 2.6589584, -0.4034616, 2.5158687, -5.2580910, 3.0624199
9: -1.5202951, 1.7920907, -0.6481612, 0.6853438, -2.2056389, 2.4402518

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_B1_A2_B1_A1_A2_A1

### Relational analysis result of NS_B1_A2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0143492, upper bound: 9.9681528
time: 1.59 seconds

## Relational analysis of NS_B1_A2_B1_A1_A2_A2

### Relational analysis result of NS_B1_A2_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.7670893, upper bound: 9.7298962
time: 4.21 seconds

## BFS NS instance: NS_B1_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.9397333, 1.0135541, -0.3592169, 0.5538517, -1.4935851, 1.3727710
1: -0.7848649, 0.9064604, -0.2877711, 0.3992085, -1.1840734, 1.1942315
2: -0.8320580, 1.1362298, -0.4607743, 0.6035067, -1.4355646, 1.5970041
3: -0.8558954, 1.0292697, -0.2169175, 0.4671372, -1.3230326, 1.2461872
4: -0.9955692, 0.9431753, -0.4278531, 0.3770877, -1.3726569, 1.3710284
5: -0.8706967, 0.9766896, -0.3492087, 0.4764962, -1.3471929, 1.3258983
6: -0.8978168, 0.9306795, -0.3176564, 0.4337308, -1.3315476, 1.2483358
7: -0.9297712, 0.9544949, -0.3960524, 0.3742004, -1.3039716, 1.3505473
8: -1.2426062, 2.5246916, -0.1007057, 2.4595163, -3.7021224, 2.6253972
9: -0.8968986, 1.0774174, -0.5817349, 0.5451272, -1.4420257, 1.6591523

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_B1_A2_B1_A2_A1_A1

### Relational analysis result of NS_B1_A2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.8736466, upper bound: 9.8460038
time: 5.66 seconds

## Relational analysis of NS_B1_A2_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B1_A2_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1972492, upper bound: 10.2144567
time: 2.35 seconds

## Relational analysis of NS_B1_A2_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1835089, upper bound: 10.1964884
time: 1.88 seconds

## BFS NS instance: NS_B1_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -2.3790596, 1.9928509, -0.4954976, 0.6949489, -3.0740085, 2.4883485
1: -1.9283799, 1.9499545, -0.4139854, 0.5292872, -2.4576671, 2.3639398
2: -2.5205500, 2.1615653, -0.5327615, 0.8181286, -3.3386786, 2.6943269
3: -2.7045591, 1.8751864, -0.3199161, 0.6444191, -3.3489783, 2.1951025
4: -2.7054448, 2.0727735, -0.5378658, 0.5396143, -3.2450590, 2.6106391
5: -2.2313783, 2.0340221, -0.4790093, 0.6195303, -2.8509088, 2.5130315
6: -2.2252462, 2.1977654, -0.4804957, 0.5656820, -2.7909281, 2.6782610
7: -2.3680527, 2.2879741, -0.5144389, 0.5211884, -2.8892412, 2.8024130
8: -3.7105646, 2.8064680, -0.4280767, 2.5231318, -6.2336965, 3.2345448
9: -2.0230145, 2.3016105, -0.6532571, 0.6978922, -2.7209067, 2.9548676

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of NS_B1_A2_B1_A2_A2_A1

### Relational analysis result of NS_B1_A2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.9733021, upper bound: 9.9212605
time: 1.96 seconds

## Relational analysis of NS_B1_A2_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B1_A2_A2_A1

### Relational analysis result of NS_B1_A2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1688281, upper bound: 10.1393709
time: 2.27 seconds

## Relational analysis of NS_B1_A2_B1_A2_A2_A2

### Relational analysis result of NS_B1_A2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1385483, upper bound: 10.1150341
time: 2.55 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.9328561, 2.4127960, -2.6259260, 2.1722479, -5.1051040, 5.0387220
1: -2.3821862, 2.3555315, -2.1264014, 2.1283784, -4.5105648, 4.4819326
2: -3.2099998, 2.5388985, -2.8327496, 2.3362224, -5.5462222, 5.3716478
3: -3.4837213, 2.2754459, -3.0550425, 2.0664408, -5.5501623, 5.3304882
4: -3.3391724, 2.5442429, -2.9829547, 2.2829671, -5.6221395, 5.5271978
5: -2.8007469, 2.4521141, -2.4822688, 2.2194552, -5.0202022, 4.9343829
6: -2.7534752, 2.7012711, -2.4624128, 2.4206142, -5.1740894, 5.1636839
7: -2.9355774, 2.8502209, -2.6215091, 2.5364044, -5.4719820, 5.4717302
8: -4.5668526, 2.9443765, -4.0913000, 2.8535833, -7.4204359, 7.0356765
9: -2.5102952, 2.7934465, -2.2360466, 2.5191741, -5.0294695, 5.0294933

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B1_A2_B2_A1_B1_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7422154, upper bound: 10.8245306
time: 2.57 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7405626, upper bound: 10.8241305
time: 2.39 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.9328561, 2.4127960, -3.2755766, 2.6998014, -5.6326575, 5.6883726
1: -2.3821862, 2.3555315, -2.7131920, 2.6016662, -4.9838524, 5.0687237
2: -3.2099998, 2.5388985, -3.6265893, 2.7814317, -5.9914312, 6.1654878
3: -3.4837213, 2.2754459, -3.9138446, 2.4478853, -5.9316063, 6.1892905
4: -3.3391724, 2.5442429, -3.7541409, 2.8261204, -6.1652927, 6.2983837
5: -2.8007469, 2.4521141, -3.1519122, 2.7134638, -5.5142107, 5.6040263
6: -2.7534752, 2.7012711, -3.0710707, 3.0156674, -5.7691426, 5.7723417
7: -2.9355774, 2.8502209, -3.2830110, 3.1946421, -6.1302195, 6.1332321
8: -4.5668526, 2.9443765, -5.0991282, 3.0879221, -7.6547747, 8.0435047
9: -2.5102952, 2.7934465, -2.8212910, 3.1019900, -5.6122851, 5.6147375

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B1_A2_B2_A1_B2_B1

### Relational analysis result of NS_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7422154, upper bound: 10.8245307
time: 3.81 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7405626, upper bound: 10.8241305
time: 2.36 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.6088955, 2.9799416, -2.6259260, 2.1722479, -5.7811432, 5.6058674
1: -3.0334468, 2.8518283, -2.1264014, 2.1283784, -5.1618252, 4.9782295
2: -4.0335865, 3.0179672, -2.8327496, 2.3362224, -6.3698092, 5.8507166
3: -4.3737674, 2.6712751, -3.0550425, 2.0664408, -6.4402084, 5.7263174
4: -4.1703949, 3.1077013, -2.9829547, 2.2829671, -6.4533620, 6.0906563
5: -3.5007262, 2.9792786, -2.4822688, 2.2194552, -5.7201815, 5.4615474
6: -3.3946471, 3.3272810, -2.4624128, 2.4206142, -5.8152614, 5.7896938
7: -3.6294036, 3.5355167, -2.6215091, 2.5364044, -6.1658077, 6.1570258
8: -5.6099930, 3.2119558, -4.0913000, 2.8535833, -8.4635763, 7.3032560
9: -3.1330554, 3.4108317, -2.2360466, 2.5191741, -5.6522293, 5.6468782

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B1_A2_B2_A2_B1_B1

### Relational analysis result of NS_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7179373, upper bound: 10.8199559
time: 2.19 seconds

## Relational analysis of NS_B1_A2_B2_A2_B1_B2

### Relational analysis result of NS_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7164918, upper bound: 10.8194116
time: 2.05 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.6088955, 2.9799416, -3.2755766, 2.6998014, -6.3086967, 6.2555180
1: -3.0334468, 2.8518283, -2.7131920, 2.6016662, -5.6351128, 5.5650206
2: -4.0335865, 3.0179672, -3.6265893, 2.7814317, -6.8150182, 6.6445565
3: -4.3737674, 2.6712751, -3.9138446, 2.4478853, -6.8216524, 6.5851197
4: -4.1703949, 3.1077013, -3.7541409, 2.8261204, -6.9965153, 6.8618422
5: -3.5007262, 2.9792786, -3.1519122, 2.7134638, -6.2141900, 6.1311908
6: -3.3946471, 3.3272810, -3.0710707, 3.0156674, -6.4103146, 6.3983517
7: -3.6294036, 3.5355167, -3.2830110, 3.1946421, -6.8240457, 6.8185277
8: -5.6099930, 3.2119558, -5.0991282, 3.0879221, -8.6979151, 8.3110838
9: -3.1330554, 3.4108317, -2.8212910, 3.1019900, -6.2350454, 6.2321224

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of NS_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377654
time: 2.36 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089027, upper bound: 10.8377565
time: 2.55 seconds

## BFS NS instance: NS_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1.2261815, 1.3062198, -2.9841256, 2.4975467, -3.7237282, 4.2903452
1: -0.9791979, 1.1733789, -2.4317844, 2.3783789, -3.3575768, 3.6051633
2: -0.9199095, 1.6271781, -3.2512221, 2.6125007, -3.5324101, 4.8784003
3: -1.0922583, 1.1746113, -3.3600059, 2.1035519, -3.1958103, 4.5346174
4: -1.1806741, 1.2008911, -3.3625844, 2.5391955, -3.7198696, 4.5634756
5: -1.0925370, 1.1990054, -2.8358202, 2.4502070, -3.5427442, 4.0348253
6: -1.1249063, 1.1679790, -2.7543998, 2.7360237, -3.8609300, 3.9223788
7: -1.1833377, 1.1946260, -2.9482336, 2.8677421, -4.0510798, 4.1428595
8: -1.7498575, 2.5849457, -4.6782603, 3.0512009, -4.8010583, 7.2632060
9: -1.0911226, 1.3417611, -2.5482907, 2.8110876, -3.9022102, 3.8900518

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_B2_A1_A2_B2_B1_A1

### Relational analysis result of NS_B2_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0765472, upper bound: 10.1218231
time: 1.77 seconds

## Relational analysis of NS_B2_A1_A2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B2_A1_A2_B2_B1_B1

### Relational analysis result of NS_B2_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.9643424, upper bound: 10.0240646
time: 1.48 seconds

## Relational analysis of NS_B2_A1_A2_B2_B1_B2

### Relational analysis result of NS_B2_A1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.9914298, upper bound: 10.0968363
time: 1.55 seconds

## BFS NS instance: NS_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -1.2261815, 1.3062198, -4.9549704, 4.1017799, -5.3279614, 6.2611904
1: -0.9791979, 1.1733789, -4.3657265, 3.7865992, -4.7657971, 5.5391054
2: -0.9199095, 1.6271781, -5.6395326, 3.9404986, -4.8604083, 7.2667108
3: -1.0922583, 1.1746113, -5.9149270, 3.2190866, -4.3113451, 7.0895386
4: -1.1806741, 1.2008911, -5.8374100, 4.1625233, -5.3431973, 7.0383010
5: -1.0925370, 1.1990054, -4.8901324, 3.9499094, -5.0424466, 6.0891380
6: -1.1249063, 1.1679790, -4.5576596, 4.5601029, -5.6850090, 5.7256384
7: -1.1833377, 1.1946260, -4.9786892, 4.9098926, -6.0932302, 6.1733150
8: -1.7498575, 2.5849457, -7.7505288, 3.7638307, -5.5136881, 10.3354740
9: -1.0911226, 1.3417611, -4.3454690, 4.6057243, -5.6968470, 5.6872301

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_B2_A1_A2_B2_B2_A1

### Relational analysis result of NS_B2_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0765472, upper bound: 10.1218231
time: 1.34 seconds

## Relational analysis of NS_B2_A1_A2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_B2_A1_A2_B2_B2_A1

### Relational analysis result of NS_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2340818, upper bound: 10.3269938
time: 2.54 seconds

## Relational analysis of NS_B2_A1_A2_B2_B2_A2

### Relational analysis result of NS_B2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2340818, upper bound: 10.3269938
time: 1.61 seconds

## BFS NS instance: NS_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.2877104, 1.9087899, -0.7839077, 0.9813707, -3.2690811, 2.6926975
1: -1.8476956, 1.8796308, -0.6236843, 0.7426473, -2.5903430, 2.5033150
2: -2.4105849, 2.1123986, -0.7004970, 1.1132182, -3.5238032, 2.8128958
3: -2.5813680, 1.8356632, -0.6336464, 0.7548249, -3.3361928, 2.4693096
4: -2.5926938, 1.9941437, -0.8255264, 0.7804300, -3.3731236, 2.8196702
5: -2.1247373, 1.9644263, -0.7093499, 0.8492496, -2.9739869, 2.6737761
6: -2.1501877, 2.1079900, -0.7357227, 0.8105178, -2.9607055, 2.8437128
7: -2.2694120, 2.1918299, -0.7704170, 0.8206292, -3.0900412, 2.9622469
8: -3.5469000, 2.7682288, -1.0128505, 2.7122550, -6.2591553, 3.7810793
9: -1.9375224, 2.2168264, -0.8165429, 0.9571456, -2.8946681, 3.0333693

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0655433, upper bound: 10.0057429
time: 2.07 seconds

## Relational analysis of NS_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1373336, upper bound: 10.0318994
time: 2.53 seconds

## BFS NS instance: NS_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.2877104, 1.9087899, -1.2681804, 1.2783751, -3.5660856, 3.1769702
1: -1.8476956, 1.8796308, -0.9806322, 1.0860441, -2.9337397, 2.8602629
2: -2.4105849, 2.1123986, -1.1637746, 1.4336338, -3.8442187, 3.2761731
3: -2.5813680, 1.8356632, -1.2028930, 0.9989486, -3.5803165, 3.0385561
4: -2.5926938, 1.9941437, -1.3576442, 1.1508827, -3.7435765, 3.3517880
5: -2.1247373, 1.9644263, -1.1365323, 1.1857440, -3.3104813, 3.1009586
6: -2.1501877, 2.1079900, -1.1778419, 1.2055669, -3.3557546, 3.2858319
7: -2.2694120, 2.1918299, -1.2093580, 1.2229133, -3.4923253, 3.4011879
8: -3.5469000, 2.7682288, -1.8562834, 2.7976198, -6.3445196, 4.6245122
9: -1.9375224, 2.2168264, -1.1308748, 1.3397750, -3.2772975, 3.3477011

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0655433, upper bound: 10.0057429
time: 1.47 seconds

## Relational analysis of NS_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1373336, upper bound: 10.0318994
time: 2.15 seconds

## BFS NS instance: NS_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.9069414, 2.3972228, -0.7839077, 0.9813707, -3.8883121, 3.1811304
1: -2.3556957, 2.3274298, -0.6236843, 0.7426473, -3.0983431, 2.9511142
2: -3.1762242, 2.5260589, -0.7004970, 1.1132182, -4.2894425, 3.2265558
3: -3.4031565, 2.2061355, -0.6336464, 0.7548249, -4.1579814, 2.8397820
4: -3.2982602, 2.5139987, -0.8255264, 0.7804300, -4.0786901, 3.3395252
5: -2.7725687, 2.4218564, -0.7093499, 0.8492496, -3.6218183, 3.1312063
6: -2.7182167, 2.6712651, -0.7357227, 0.8105178, -3.5287344, 3.4069879
7: -2.9044781, 2.8140368, -0.7704170, 0.8206292, -3.7251072, 3.5844538
8: -4.5336490, 2.9664898, -1.0128505, 2.7122550, -7.2459040, 3.9793403
9: -2.4802496, 2.7612855, -0.8165429, 0.9571456, -3.4373953, 3.5778284

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0584648, upper bound: 10.0025910
time: 1.47 seconds

## Relational analysis of NS_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1346575, upper bound: 10.0310663
time: 2.40 seconds

## BFS NS instance: NS_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.9069414, 2.3972228, -1.2716371, 1.2783751, -4.1853166, 3.6688600
1: -2.3556957, 2.3274298, -0.9825920, 1.0880443, -3.4437399, 3.3100219
2: -3.1762242, 2.5260589, -1.1652203, 1.4337060, -4.6099300, 3.6912792
3: -3.4031565, 2.2061355, -1.2057397, 0.9989486, -4.4021053, 3.4118752
4: -3.2982602, 2.5139987, -1.3631992, 1.1508827, -4.4491429, 3.8771980
5: -2.7725687, 2.4218564, -1.1365323, 1.1917937, -3.9643624, 3.5583887
6: -2.7182167, 2.6712651, -1.1790874, 1.2079697, -3.9261863, 3.8503525
7: -2.9044781, 2.8140368, -1.2097311, 1.2234243, -4.1279025, 4.0237679
8: -4.5336490, 2.9664898, -1.8572013, 2.7976356, -7.3312845, 4.8236914
9: -2.4802496, 2.7612855, -1.1308748, 1.3415459, -3.8217955, 3.8921604

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 103

## Relational analysis of NS_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of NS_B2_A2_B1_A2_B2_B1

### Relational analysis result of NS_B2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3620760, upper bound: 10.2732645
time: 1.50 seconds

## Relational analysis of NS_B2_A2_B1_A2_B2_B2

### Relational analysis result of NS_B2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732645
time: 1.64 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.2877104, 1.9087899, -2.9766254, 2.4911520, -4.7788625, 4.8854151
1: -1.8476956, 1.8796308, -2.3980727, 2.3344359, -4.1821318, 4.2777033
2: -2.4105849, 2.1123986, -3.2412877, 2.5686948, -4.9792795, 5.3536863
3: -2.5813680, 1.8356632, -3.3584819, 2.0692980, -4.6506662, 5.1941452
4: -2.5926938, 1.9941437, -3.3781281, 2.5308957, -5.1235895, 5.3722715
5: -2.1247373, 1.9644263, -2.8386552, 2.4334557, -4.5581932, 4.8030815
6: -2.1501877, 2.1079900, -2.7456985, 2.7186759, -4.8688636, 4.8536882
7: -2.2694120, 2.1918299, -2.9558308, 2.8710127, -5.1404247, 5.1476607
8: -3.5469000, 2.7682288, -4.6541281, 3.0337334, -6.5806332, 7.4223566
9: -1.9375224, 2.2168264, -2.5429280, 2.7944522, -4.7319746, 4.7597542

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6544690, upper bound: 10.6530265
time: 1.68 seconds

## Relational analysis of NS_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6699786, upper bound: 10.6596350
time: 1.86 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.2877104, 1.9087899, -3.5835872, 2.9864328, -5.2741432, 5.4923773
1: -1.8476956, 1.8796308, -2.9990077, 2.7892370, -4.6369324, 4.8786383
2: -2.4105849, 2.1123986, -3.9776998, 2.9991190, -5.4097042, 6.0900984
3: -2.5813680, 1.8356632, -4.1446934, 2.4241414, -5.0055094, 5.9803567
4: -2.5926938, 1.9941437, -4.1187057, 3.0391212, -5.6318150, 6.1128492
5: -2.1247373, 1.9644263, -3.4591312, 2.9108064, -5.0355434, 5.4235573
6: -2.1501877, 2.1079900, -3.3210511, 3.2834110, -5.4335985, 5.4290409
7: -2.2694120, 2.1918299, -3.5799434, 3.4870443, -5.7564564, 5.7717733
8: -3.5469000, 2.7682288, -5.6093845, 3.2501209, -6.7970209, 8.3776131
9: -1.9375224, 2.2168264, -3.0849566, 3.3533344, -5.2908568, 5.3017831

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5893956, upper bound: 10.5314564
time: 1.72 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5424432, upper bound: 10.5230275
time: 1.56 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.9069414, 2.3972228, -2.9766254, 2.4911520, -5.3980932, 5.3738480
1: -2.3556957, 2.3274298, -2.3980727, 2.3344359, -4.6901317, 4.7255025
2: -3.1762242, 2.5260589, -3.2412877, 2.5686948, -5.7449188, 5.7673464
3: -3.4031565, 2.2061355, -3.3584819, 2.0692980, -5.4724545, 5.5646172
4: -3.2982602, 2.5139987, -3.3781281, 2.5308957, -5.8291559, 5.8921270
5: -2.7725687, 2.4218564, -2.8386552, 2.4334557, -5.2060242, 5.2605114
6: -2.7182167, 2.6712651, -2.7456985, 2.7186759, -5.4368925, 5.4169636
7: -2.9044781, 2.8140368, -2.9558308, 2.8710127, -5.7754908, 5.7698679
8: -4.5336490, 2.9664898, -4.6541281, 3.0337334, -7.5673823, 7.6206179
9: -2.4802496, 2.7612855, -2.5429280, 2.7944522, -5.2747021, 5.3042135

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6418901, upper bound: 10.6512603
time: 1.36 seconds

## Relational analysis of NS_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6583344, upper bound: 10.6583345
time: 1.48 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.9069414, 2.3972228, -3.5835872, 2.9864328, -5.8933744, 5.9808102
1: -2.3556957, 2.3274298, -2.9990077, 2.7892370, -5.1449327, 5.3264375
2: -3.1762242, 2.5260589, -3.9776998, 2.9991190, -6.1753435, 6.5037584
3: -3.4031565, 2.2061355, -4.1446934, 2.4241414, -5.8272982, 6.3508291
4: -3.2982602, 2.5139987, -4.1187057, 3.0391212, -6.3373814, 6.6327047
5: -2.7725687, 2.4218564, -3.4591312, 2.9108064, -5.6833754, 5.8809876
6: -2.7182167, 2.6712651, -3.3210511, 3.2834110, -6.0016279, 5.9923162
7: -2.9044781, 2.8140368, -3.5799434, 3.4870443, -6.3915224, 6.3939800
8: -4.5336490, 2.9664898, -5.6093845, 3.2501209, -7.7837696, 8.5758743
9: -2.4802496, 2.7612855, -3.0849566, 3.3533344, -5.8335838, 5.8462420

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 210

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of NS_B2_A2_B2_A2_B2_B1

### Relational analysis result of NS_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8002297, upper bound: 10.8002388
time: 2.14 seconds

## Relational analysis of NS_B2_A2_B2_A2_B2_B2

### Relational analysis result of NS_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8002297, upper bound: 10.8002301
time: 1.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 8.47 seconds
NS_B1_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.0379905, upper bound: 10.1752205
NS_B1_A1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.0622791, upper bound: 10.2518600
NS_B1_A1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.1325542, upper bound: 10.3064628
NS_B1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.1340738, upper bound: 10.3132061
NS_B1_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.0376586, upper bound: 10.1745094
NS_B1_A1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.0622790, upper bound: 10.2521523
NS_B1_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4735759
NS_B1_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
NS_B1_A2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -9.9099598, upper bound: 9.8949006
NS_B1_A2_B1_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -9.6570093, upper bound: 9.6246803
NS_B1_A2_B1_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.0143492, upper bound: 9.9681528
NS_B1_A2_B1_A1_A2_A2, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -9.7670893, upper bound: 9.7298962
NS_B1_A2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.1972492, upper bound: 10.2144567
NS_B1_A2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.1835089, upper bound: 10.1964884
NS_B1_A2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.1688281, upper bound: 10.1393709
NS_B1_A2_B1_A2_A2_A2, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.1385483, upper bound: 10.1150341
NS_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.7422154, upper bound: 10.8245306
NS_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.7405626, upper bound: 10.8241305
NS_B1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.7422154, upper bound: 10.8245307
NS_B1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.7405626, upper bound: 10.8241305
NS_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.7179373, upper bound: 10.8199559
NS_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.7164918, upper bound: 10.8194116
NS_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377654
NS_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.8089027, upper bound: 10.8377565
NS_B2_A1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -9.9643424, upper bound: 10.0240646
NS_B2_A1_A2_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -9.9914298, upper bound: 10.0968363
NS_B2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.2340818, upper bound: 10.3269938
NS_B2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.2340818, upper bound: 10.3269938
NS_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.0655433, upper bound: 10.0057429
NS_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.1373336, upper bound: 10.0318994
NS_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.0655433, upper bound: 10.0057429
NS_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.1373336, upper bound: 10.0318994
NS_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.0584648, upper bound: 10.0025910
NS_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.1346575, upper bound: 10.0310663
NS_B2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.3620760, upper bound: 10.2732645
NS_B2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732645
NS_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.6544690, upper bound: 10.6530265
NS_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.6699786, upper bound: 10.6596350
NS_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.5893956, upper bound: 10.5314564
NS_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.5424432, upper bound: 10.5230275
NS_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.6418901, upper bound: 10.6512603
NS_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.6583344, upper bound: 10.6583345
NS_B2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.8002297, upper bound: 10.8002388
NS_B2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.47
Output dim: 8, lower bound: -10.8002297, upper bound: 10.8002301

## BFS NS instance: NS_B1_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5862559, 0.7808629, -2.1799507, 1.8533878, -2.4396439, 2.9608135
1: -0.4860865, 0.6125175, -1.7554476, 1.7853987, -2.2714853, 2.3679652
2: -0.5710031, 0.9099085, -2.2499206, 1.9828224, -2.5538254, 3.1598291
3: -0.4159561, 0.7140951, -2.4165840, 1.7091947, -2.1251507, 3.1306791
4: -0.6315936, 0.6232990, -2.4846978, 1.8995643, -2.5311580, 3.1079969
5: -0.5675339, 0.6910769, -2.0243671, 1.8699455, -2.4374795, 2.7154441
6: -0.5686164, 0.6424640, -2.0425169, 2.0099623, -2.5785787, 2.6849809
7: -0.6008333, 0.6197610, -2.1633117, 2.0890050, -2.6898384, 2.7830727
8: -0.5952435, 2.5331864, -3.3789003, 2.7294757, -3.3247192, 5.9120865
9: -0.6943408, 0.7857733, -1.8466189, 2.1136732, -2.8080139, 2.6323922

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B1_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_B1_A1_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.8056850, upper bound: 10.0209070
time: 1.93 seconds

## Relational analysis of NS_B1_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_B1_A1_B2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.8940397, upper bound: 10.0921307
time: 1.66 seconds

## BFS NS instance: NS_B1_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.5074045, 0.6909623, -2.9769249, 2.4475727, -2.9549773, 3.6678872
1: -0.4277183, 0.5497788, -2.4266717, 2.3839996, -2.8117180, 2.9764504
2: -0.5299577, 0.8344898, -3.2678394, 2.5754032, -3.1053610, 4.1023293
3: -0.3322609, 0.6800675, -3.5165386, 2.2800894, -2.6123502, 4.1966062
4: -0.5501353, 0.5512276, -3.3751776, 2.5786648, -3.1288002, 3.9264052
5: -0.4921862, 0.6308103, -2.8459420, 2.4794927, -2.9716787, 3.4767523
6: -0.4962270, 0.5746090, -2.7872198, 2.7389328, -3.2351599, 3.3618288
7: -0.5240679, 0.5326537, -2.9803951, 2.8915491, -3.4156170, 3.5130486
8: -0.4439952, 2.4763653, -4.6400814, 2.9658952, -3.4098904, 7.1164465
9: -0.6510586, 0.7057608, -2.5466626, 2.8276806, -3.4787393, 3.2524233

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_B1_A1_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0376586, upper bound: 10.1734810
time: 2.15 seconds

## Relational analysis of NS_B1_A1_B2_B2_A2_A1_B2

### Relational analysis result of NS_B1_A1_B2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0622790, upper bound: 10.2504968
time: 1.99 seconds

## BFS NS instance: NS_B1_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.7751960, 0.9109775, -3.1455102, 2.5830231, -3.3582191, 4.0564876
1: -0.6399021, 0.7774657, -2.5921359, 2.5032635, -3.1431656, 3.3696017
2: -0.7005208, 1.1096113, -3.4707236, 2.6889167, -3.3894374, 4.5803347
3: -0.6364986, 0.8716450, -3.7364061, 2.3778443, -3.0143428, 4.6080513
4: -0.8213144, 0.7928160, -3.5628841, 2.7192323, -3.5405467, 4.3557000
5: -0.7148739, 0.8631164, -3.0203927, 2.6042755, -3.3191495, 3.8835092
6: -0.7542198, 0.7908704, -2.9458575, 2.8923306, -3.6465504, 3.7367280
7: -0.7744479, 0.8044263, -3.1527910, 3.0635197, -3.8379676, 3.9572172
8: -0.9703314, 2.5448091, -4.9034309, 3.0335124, -4.0038438, 7.4482403
9: -0.7994223, 0.9467521, -2.6989129, 2.9779568, -3.7773790, 3.6456649

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of NS_B1_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_B1_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
time: 1.97 seconds

## Relational analysis of NS_B1_A1_B2_B2_A2_A2_B2

### Relational analysis result of NS_B1_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
time: 1.85 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -2.7363057, 2.2545002, -2.0864091, 1.7473655, -4.4836712, 4.3409090
1: -2.2167532, 2.2141788, -1.6771207, 1.7404528, -3.9572060, 3.8912995
2: -2.9689834, 2.4109247, -2.1508434, 1.9831524, -4.9521360, 4.5617681
3: -3.2206788, 2.1656651, -2.3221915, 1.7517555, -4.9724340, 4.4878569
4: -3.1071539, 2.3792009, -2.3517411, 1.8255392, -4.9326930, 4.7309418
5: -2.5967045, 2.3054461, -1.9146250, 1.8191981, -4.4159026, 4.2200713
6: -2.5704150, 2.5238826, -1.9687158, 1.9262135, -4.4966288, 4.4925985
7: -2.7337871, 2.6511846, -2.0530674, 1.9914991, -4.7252860, 4.7042522
8: -4.2635489, 2.8647478, -3.2051523, 2.6655407, -6.9290895, 6.0699000
9: -2.3323796, 2.6197484, -1.7572045, 2.0403781, -4.3727579, 4.3769531

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7437333, upper bound: 10.8248509
time: 1.54 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7437333, upper bound: 10.8248509
time: 1.88 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -2.6454153, 2.1824780, -3.9652963, 3.2037225, -5.8491378, 6.1477742
1: -2.1429219, 2.1493957, -3.1844819, 3.0798383, -5.2227602, 5.3338776
2: -2.8565660, 2.3515742, -4.4558759, 3.1884928, -6.0450587, 6.8074503
3: -3.1002579, 2.1116214, -4.7941189, 2.8164973, -5.9167552, 6.9057403
4: -3.0032370, 2.3015044, -4.5120831, 3.3789968, -6.3822336, 6.8135872
5: -2.5019758, 2.2386448, -3.8975408, 3.1722844, -5.6742601, 6.1361856
6: -2.4838526, 2.4419913, -3.6814361, 3.6183853, -6.1022377, 6.1234274
7: -2.6405120, 2.5593629, -3.9739728, 3.9042940, -6.5448060, 6.5333357
8: -4.1189699, 2.8235602, -6.1799049, 3.1695035, -7.2884731, 9.0034657
9: -2.2543931, 2.5391769, -3.4655647, 3.6580400, -5.9124331, 6.0047417

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 210

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7437333, upper bound: 10.8248509
time: 1.50 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7437333, upper bound: 10.8248509
time: 2.17 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -2.7363057, 2.2545002, -2.6216493, 2.1685495, -4.9048553, 4.8761492
1: -2.2167532, 2.2141788, -2.1214333, 2.1263318, -4.3430853, 4.3356123
2: -2.9689834, 2.4109247, -2.8280301, 2.3455245, -5.3145080, 5.2389545
3: -3.2206788, 2.1656651, -3.0502334, 2.0777054, -5.2983842, 5.2158985
4: -3.1071539, 2.3792009, -2.9714189, 2.2762167, -5.3833704, 5.3506198
5: -2.5967045, 2.3054461, -2.4745641, 2.2170124, -4.8137169, 4.7800102
6: -2.5704150, 2.5238826, -2.4582560, 2.4178386, -4.9882536, 4.9821386
7: -2.7337871, 2.6511846, -2.6119673, 2.5289385, -5.2627258, 5.2631521
8: -4.2635489, 2.8647478, -4.0811272, 2.8310831, -7.0946321, 6.9458752
9: -2.3323796, 2.6197484, -2.2319717, 2.5154493, -4.8478289, 4.8517199

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_B1_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_B1_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7405626, upper bound: 10.8241305
time: 2.84 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_B1_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7405626, upper bound: 10.8241305
time: 3.53 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -2.6454153, 2.1824780, -4.5447011, 3.7446413, -6.3900566, 6.7271791
1: -2.1429219, 2.1493957, -3.9430509, 3.5196748, -5.6625967, 6.0924463
2: -2.8565660, 2.3515742, -5.1541281, 3.6556392, -6.5122051, 7.5057020
3: -3.1002579, 2.1116214, -5.5572681, 3.1561601, -6.2564182, 7.6688895
4: -3.0032370, 2.3015044, -5.3435917, 3.8611503, -6.8643875, 7.6450958
5: -2.5019758, 2.2386448, -4.4700565, 3.6953530, -6.1973286, 6.7087011
6: -2.4838526, 2.4419913, -4.2532415, 4.1894855, -6.6733379, 6.6952329
7: -2.6405120, 2.5593629, -4.5926523, 4.5045033, -7.1450152, 7.1520152
8: -4.1189699, 2.8235602, -7.0700865, 3.4871852, -7.6061554, 9.8936462
9: -2.2543931, 2.5391769, -3.9923797, 4.2410870, -6.4954801, 6.5315566

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 210

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_B1_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7405626, upper bound: 10.8241305
time: 2.73 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7405626, upper bound: 10.8241305
time: 2.76 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3.3887010, 2.7953892, -2.0864091, 1.7473655, -5.1360664, 4.8817983
1: -2.8221588, 2.6914012, -1.6771207, 1.7404528, -4.5626116, 4.3685217
2: -3.7658198, 2.8672252, -2.1508434, 1.9831524, -5.7489719, 5.0180683
3: -4.0849423, 2.5461948, -2.3221915, 1.7517555, -5.8366976, 4.8683863
4: -3.8986850, 2.9229469, -2.3517411, 1.8255392, -5.7242241, 5.2746878
5: -3.2715576, 2.8071017, -1.9146250, 1.8191981, -5.0907555, 4.7217269
6: -3.1856747, 3.1244874, -1.9687158, 1.9262135, -5.1118879, 5.0932031
7: -3.4010520, 3.3124642, -2.0530674, 1.9914991, -5.3925509, 5.3655319
8: -5.2717543, 3.1114228, -3.2051523, 2.6655407, -7.9372950, 6.3165751
9: -2.9311011, 3.2082434, -1.7572045, 2.0403781, -4.9714794, 4.9654479

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_B1_A2_B2_A2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7165048, upper bound: 10.8194169
time: 2.35 seconds

## Relational analysis of NS_B1_A2_B2_A2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7165048, upper bound: 10.8194169
time: 2.00 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3.2664058, 2.6922913, -3.9652963, 3.2037225, -6.4701281, 6.6575875
1: -2.7048697, 2.6021519, -3.1844819, 3.0798383, -5.7847080, 5.7866335
2: -3.6165600, 2.7831094, -4.4558759, 3.1884928, -6.8050528, 7.2389851
3: -3.9219251, 2.4774985, -4.7941189, 2.8164973, -6.7384224, 7.2716174
4: -3.7491093, 2.8199146, -4.5120831, 3.3789968, -7.1281061, 7.3319979
5: -3.1462083, 2.7126486, -3.8975408, 3.1722844, -6.3184929, 6.6101894
6: -3.0685334, 3.0121164, -3.6814361, 3.6183853, -6.6869187, 6.6935525
7: -3.2757683, 3.1889238, -3.9739728, 3.9042940, -7.1800623, 7.1628966
8: -5.0847373, 3.0466874, -6.1799049, 3.1695035, -8.2542410, 9.2265921
9: -2.8184290, 3.0967598, -3.4655647, 3.6580400, -6.4764690, 6.5623245

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_B1_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7165048, upper bound: 10.8194169
time: 1.82 seconds

## Relational analysis of NS_B1_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7165048, upper bound: 10.8194169
time: 1.74 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.6939831, 2.2269907, -2.9788461, 2.4508176, -5.1448007, 5.2058368
1: -2.1839449, 2.1959758, -2.4266717, 2.3865745, -4.5705194, 4.6226473
2: -2.9306002, 2.3943145, -3.2692595, 2.5769458, -5.5075459, 5.6635742
3: -3.1695759, 2.1611509, -3.5190670, 2.2804809, -5.4500570, 5.6802177
4: -3.0560048, 2.3529515, -3.3914018, 2.5794566, -5.6354613, 5.7443533
5: -2.5512300, 2.2823532, -2.8459420, 2.4848289, -5.0360589, 5.1282949
6: -2.5304413, 2.4930289, -2.7872198, 2.7421010, -5.2725420, 5.2802486
7: -2.6923130, 2.6092439, -2.9810622, 2.8922243, -5.5845375, 5.5903063
8: -4.2033253, 2.8699119, -4.6415520, 2.9659755, -7.1693010, 7.5114641
9: -2.2948246, 2.5904210, -2.5480881, 2.8314054, -5.1262302, 5.1385088

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of NS_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089027, upper bound: 10.8377565
time: 1.92 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089027, upper bound: 10.8377565
time: 2.13 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.3046653, 2.7202380, -3.1544130, 2.5962420, -5.9009075, 5.8746510
1: -2.7419751, 2.6300216, -2.5960569, 2.5133207, -5.2552958, 5.2260785
2: -3.6662707, 2.8073261, -3.4799361, 2.6971793, -6.3634501, 6.2872620
3: -3.9681041, 2.4987812, -3.7510986, 2.3799064, -6.3480105, 6.2498798
4: -3.7968426, 2.8562317, -3.6047583, 2.7253668, -6.5222092, 6.4609900
5: -3.1837120, 2.7455583, -3.0259979, 2.6204419, -5.8041539, 5.7715559
6: -3.1028121, 3.0477324, -2.9543693, 2.9037123, -6.0065241, 6.0021019
7: -3.3189619, 3.2260485, -3.1597438, 3.0702691, -6.3892307, 6.3857923
8: -5.1462812, 3.0732729, -4.9129362, 3.0368178, -8.1830988, 7.9862089
9: -2.8488634, 3.1318095, -2.7081151, 2.9910734, -5.8399367, 5.8399248

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of NS_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089027, upper bound: 10.8377565
time: 2.94 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089027, upper bound: 10.8377565
time: 2.67 seconds

## BFS NS instance: NS_B2_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1.3913254, 1.4312205, -4.9549704, 4.1017799, -5.4931054, 6.3861909
1: -1.1094677, 1.3188944, -4.3657265, 3.7865992, -4.8960667, 5.6846209
2: -1.0246638, 1.8029537, -5.6395326, 3.9404986, -4.9651623, 7.4424863
3: -1.2707716, 1.3157338, -5.9149270, 3.2190866, -4.4898582, 7.2306609
4: -1.3310376, 1.3543497, -5.8374100, 4.1625233, -5.4935608, 7.1917596
5: -1.2354146, 1.3309577, -4.8901324, 3.9499094, -5.1853242, 6.2210903
6: -1.2740637, 1.2999444, -4.5576596, 4.5601029, -5.8341665, 5.8576040
7: -1.3373322, 1.3483528, -4.9786892, 4.9098926, -6.2472248, 6.3270421
8: -2.0393753, 2.6154494, -7.7505288, 3.7638307, -5.8032060, 10.3659782
9: -1.2048825, 1.4856789, -4.3454690, 4.6057243, -5.8106070, 5.8311481

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_A1

### Relational analysis result of NS_B2_A1_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0709990, upper bound: 10.1621792
time: 1.80 seconds

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0594909, upper bound: 10.1601029
time: 1.72 seconds

## BFS NS instance: NS_B2_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1.3735383, 1.4958017, -4.9549704, 4.1017799, -5.4753180, 6.4507723
1: -1.0707138, 1.2921267, -4.3657265, 3.7865992, -4.8573132, 5.6578531
2: -1.0111969, 1.7995125, -5.6395326, 3.9404986, -4.9516954, 7.4390450
3: -1.2348689, 1.1703675, -5.9149270, 3.2190866, -4.4539557, 7.0852947
4: -1.2999120, 1.3199115, -5.8374100, 4.1625233, -5.4624352, 7.1573215
5: -1.2062819, 1.3044503, -4.8901324, 3.9499094, -5.1561913, 6.1945829
6: -1.2401099, 1.3011887, -4.5576596, 4.5601029, -5.8002129, 5.8588486
7: -1.3049834, 1.3214926, -4.9786892, 4.9098926, -6.2148762, 6.3001819
8: -2.0619540, 2.7839291, -7.7505288, 3.7638307, -5.8257847, 10.5344582
9: -1.2014073, 1.4798396, -4.3454690, 4.6057243, -5.8071318, 5.8253088

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 8

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_A1

### Relational analysis result of NS_B2_A1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2335801, upper bound: 10.3256926
time: 4.58 seconds

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_A2

### Relational analysis result of NS_B2_A1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2333507, upper bound: 10.3255758
time: 2.34 seconds

## BFS NS instance: NS_B2_A2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -2.6253507, 2.1759059, -0.5788733, 0.8379920, -3.4633427, 2.7547793
1: -2.1241226, 2.1263881, -0.4614120, 0.5883223, -2.7124450, 2.5878000
2: -2.8364377, 2.3422711, -0.5702302, 0.9336587, -3.7700963, 2.9125013
3: -3.0352557, 2.0426555, -0.3879681, 0.6145781, -3.6498337, 2.4306235
4: -2.9802535, 2.2795506, -0.6119021, 0.5963204, -3.5765738, 2.8914528
5: -2.4814379, 2.2155066, -0.5487446, 0.6746964, -3.1561344, 2.7642512
6: -2.4540823, 2.4186141, -0.5479448, 0.6558449, -3.1099272, 2.9665589
7: -2.6194806, 2.5327244, -0.5752618, 0.6063741, -3.2258549, 3.1079862
8: -4.0915298, 2.8620713, -0.6299902, 2.6799631, -6.7714930, 3.4920616
9: -2.2348576, 2.5152371, -0.7079160, 0.7868070, -3.0216646, 3.2231531

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0561016, upper bound: 10.0025910
time: 1.87 seconds

## Relational analysis of NS_B2_A2_B1_A2_B2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1323346, upper bound: 10.0310663
time: 4.04 seconds

## BFS NS instance: NS_B2_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -2.7938778, 2.3074470, -1.0271806, 1.1306820, -3.9245598, 3.3346276
1: -2.2593508, 2.2461207, -0.8043850, 0.9205126, -3.1798635, 3.0505056
2: -3.0396714, 2.4514558, -0.9256492, 1.2789692, -4.3186407, 3.3771050
3: -3.2554679, 2.1404681, -0.9210963, 0.8776923, -4.1331601, 3.0615644
4: -3.1693320, 2.4201384, -1.0821899, 0.9693201, -4.1386518, 3.5023284
5: -2.6564789, 2.3376031, -0.9176158, 1.0192811, -3.6757600, 3.2552190
6: -2.6124969, 2.5692196, -0.9606857, 1.0086813, -3.6211782, 3.5299053
7: -2.7894058, 2.7010469, -0.9800209, 1.0265069, -3.8159127, 3.6810679
8: -4.3565540, 2.9241207, -1.4324156, 2.7442827, -7.1008368, 4.3565364
9: -2.3802817, 2.6619055, -0.9635715, 1.1464343, -3.5267160, 3.6254771

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of NS_B2_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_B2_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732645
time: 1.80 seconds

## Relational analysis of NS_B2_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_B2_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732645
time: 1.77 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.6392698, 0.8131597, -2.3092976, 1.9777462, -2.6170161, 3.1224573
1: -0.5298603, 0.6500444, -1.8281341, 1.8306162, -2.3604765, 2.4781785
2: -0.5986441, 0.9056597, -2.3984396, 2.0817401, -2.6803842, 3.3040993
3: -0.4830235, 0.7594818, -2.4742599, 1.6303940, -2.1134174, 3.2337418
4: -0.6987318, 0.6765992, -2.6137707, 1.9717950, -2.6705267, 3.2903700
5: -0.6188836, 0.7365116, -2.1452017, 1.9236534, -2.5425370, 2.8817134
6: -0.6180427, 0.6820966, -2.1177378, 2.1100988, -2.7281415, 2.7998343
7: -0.6620458, 0.6918727, -2.2741990, 2.1956124, -2.8576584, 2.9660716
8: -0.6562687, 2.4699070, -3.5781770, 2.8763702, -3.5326390, 6.0480843
9: -0.7209373, 0.8145124, -1.9619541, 2.1918983, -2.9128356, 2.7764664

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A2_B2_A1_B1_A1_A1

### Relational analysis result of NS_B2_A2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3672249, upper bound: 10.4218298
time: 5.45 seconds

## Relational analysis of NS_B2_A2_B2_A1_B1_A1_A2

### Relational analysis result of NS_B2_A2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4746780, upper bound: 10.4845519
time: 1.38 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.8846253, 1.6380050, -2.4087510, 2.0535975, -3.9382229, 4.0467558
1: -1.5069959, 1.5671189, -1.9104186, 1.9096422, -3.4166381, 3.4775376
2: -1.8808497, 1.7878538, -2.5288973, 2.1645215, -4.0453711, 4.3167510
3: -1.9989846, 1.5264359, -2.6030090, 1.7097957, -3.7087803, 4.1294451
4: -2.1296091, 1.6565862, -2.7233844, 2.0555012, -4.1851101, 4.3799706
5: -1.7358725, 1.6461581, -2.2485957, 2.0024128, -3.7382853, 3.8947539
6: -1.7722783, 1.7378355, -2.2118261, 2.2015903, -3.9738686, 3.9496617
7: -1.8489643, 1.7921112, -2.3746166, 2.2913647, -4.1403289, 4.1667280
8: -2.8809667, 2.6778705, -3.7465079, 2.9007311, -5.7816978, 6.4243784
9: -1.5846231, 1.8489166, -2.0480351, 2.2837820, -3.8684051, 3.8969517

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6809952, upper bound: 10.6708482
time: 1.86 seconds

## Relational analysis of NS_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6809952, upper bound: 10.6882749
time: 1.82 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.6431252, 1.4448916, -3.5079126, 2.9257092, -4.5688343, 4.9528041
1: -1.3085009, 1.4053481, -2.9253297, 2.7330401, -4.0415411, 4.3306780
2: -1.6060641, 1.6910219, -3.8850436, 2.9461668, -4.5522308, 5.5760655
3: -1.6693033, 1.4129220, -4.0428090, 2.3763571, -4.0456605, 5.4557309
4: -1.8251343, 1.4642169, -4.0253229, 2.9752846, -4.8004189, 5.4895401
5: -1.4977378, 1.4831169, -3.3810313, 2.8503277, -4.3480654, 4.8641481
6: -1.5561957, 1.5195878, -3.2484798, 3.2130675, -4.7692633, 4.7680674
7: -1.5940492, 1.5620537, -3.5012679, 3.4093316, -5.0033808, 5.0633216
8: -2.4609997, 2.6312926, -5.4903646, 3.2237663, -5.6847658, 8.1216574
9: -1.4003731, 1.6439345, -3.0161247, 3.2832279, -4.6836009, 4.6600590

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5424432, upper bound: 10.5230275
time: 1.52 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5424432, upper bound: 10.5230275
time: 1.83 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.1150050, 1.8090740, -3.0008266, 2.5179090, -4.6329141, 4.8099008
1: -1.6867589, 1.7307937, -2.4306097, 2.3577838, -4.0445428, 4.1614037
2: -2.1820443, 1.9902848, -3.2619903, 2.5883999, -4.7704439, 5.2522750
3: -2.2697067, 1.6134641, -3.3612628, 2.0585356, -4.3282423, 4.9747267
4: -2.3968334, 1.8299966, -3.3984454, 2.5462871, -4.9431205, 5.2284422
5: -1.9391505, 1.8129700, -2.8579953, 2.4469910, -4.3861413, 4.6709652
6: -1.9654528, 1.9492610, -2.7585716, 2.7428308, -4.7082834, 4.7078323
7: -2.0830050, 2.0139339, -2.9716303, 2.8879442, -4.9709492, 4.9855642
8: -3.2736886, 2.7218976, -4.6943007, 3.0593066, -6.3329954, 7.4161983
9: -1.7937435, 2.0420928, -2.5628576, 2.8122284, -4.6059718, 4.6049504

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4849272, upper bound: 10.4847548
time: 2.04 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4771871, upper bound: 10.4547445
time: 2.28 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.0751853, 1.0967149, -2.3092976, 1.9777462, -3.0529315, 3.4060125
1: -0.8583297, 0.9635460, -1.8281341, 1.8306162, -2.6889460, 2.7916801
2: -0.9879087, 1.2264627, -2.3984396, 2.0817401, -3.0696487, 3.6249022
3: -1.0183266, 1.0077517, -2.4742599, 1.6303940, -2.6487207, 3.4820116
4: -1.1698031, 1.0312706, -2.6137707, 1.9717950, -3.1415982, 3.6450415
5: -0.9902401, 1.0573615, -2.1452017, 1.9236534, -2.9138935, 3.2025633
6: -1.0233204, 1.0277264, -2.1177378, 2.1100988, -3.1334193, 3.1454642
7: -1.0620767, 1.0799332, -2.2741990, 2.1956124, -3.2576890, 3.3541322
8: -1.4658086, 2.5551400, -3.5781770, 2.8763702, -4.3421788, 6.1333170
9: -0.9930356, 1.1746047, -1.9619541, 2.1918983, -3.1849339, 3.1365588

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6423200, upper bound: 10.6527132
time: 1.99 seconds

## Relational analysis of NS_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6423200, upper bound: 10.6621108
time: 1.45 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.4444761, 2.0627060, -2.4087510, 2.0535975, -4.4980736, 4.4714570
1: -1.9664549, 1.9654901, -1.9104186, 1.9096422, -3.8760972, 3.8759089
2: -2.5739875, 2.1607051, -2.5288973, 2.1645215, -4.7385092, 4.6896024
3: -2.7505031, 1.8462446, -2.6030090, 1.7097957, -4.4602985, 4.4492536
4: -2.7809849, 2.1120973, -2.7233844, 2.0555012, -4.8364859, 4.8354816
5: -2.3018308, 2.0580812, -2.2485957, 2.0024128, -4.3042436, 4.3066769
6: -2.2700760, 2.2457500, -2.2118261, 2.2015903, -4.4716663, 4.4575758
7: -2.4300852, 2.3461809, -2.3746166, 2.2913647, -4.7214499, 4.7207975
8: -3.8007679, 2.8192089, -3.7465079, 2.9007311, -6.7014990, 6.5657167
9: -2.0740814, 2.3409700, -2.0480351, 2.2837820, -4.3578634, 4.3890052

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6530227, upper bound: 10.6542993
time: 1.58 seconds

## Relational analysis of NS_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6530227, upper bound: 10.6697925
time: 1.44 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -2.6253507, 2.1759059, -2.5549037, 2.1720152, -4.7973661, 4.7308097
1: -2.1241226, 2.1263881, -2.0284047, 2.0479758, -4.1720982, 4.1547928
2: -2.8364377, 2.3422711, -2.7235873, 2.2976298, -5.1340675, 5.0658584
3: -3.0352557, 2.0426555, -2.7950768, 1.8349419, -4.8701973, 4.8377323
4: -2.9802535, 2.2795506, -2.8733675, 2.1830091, -5.1632624, 5.1529179
5: -2.4814379, 2.2155066, -2.3954849, 2.1234653, -4.6049032, 4.6109915
6: -2.4540823, 2.4186141, -2.3499155, 2.3430147, -4.7970972, 4.7685299
7: -2.6194806, 2.5327244, -2.5142112, 2.4262133, -5.0456939, 5.0469356
8: -4.0915298, 2.8620713, -3.9993112, 2.9485774, -7.0401073, 6.8613825
9: -2.2348576, 2.5152371, -2.1735249, 2.4238501, -4.6587076, 4.6887617

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of NS_B2_A2_B2_A2_B2_B1_A1

### Relational analysis result of NS_B2_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8002297, upper bound: 10.8002301
time: 1.90 seconds

## Relational analysis of NS_B2_A2_B2_A2_B2_B1_A2

### Relational analysis result of NS_B2_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8002297, upper bound: 10.8002301
time: 2.05 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -2.7938778, 2.3074470, -3.2605886, 2.7238779, -5.5177555, 5.5680356
1: -2.2593508, 2.2461207, -2.6867039, 2.5550773, -4.8144283, 4.9328246
2: -3.0396714, 2.4514558, -3.5879626, 2.7753851, -5.8150568, 6.0394182
3: -3.2554679, 2.1404681, -3.7249711, 2.2399964, -5.4954643, 5.8654394
4: -3.1693320, 2.4201384, -3.7234251, 2.7710719, -5.9404039, 6.1435633
5: -2.6564789, 2.3376031, -3.1282284, 2.6581070, -5.3145857, 5.4658318
6: -2.6124969, 2.5692196, -3.0139804, 2.9855278, -5.5980244, 5.5832000
7: -2.7894058, 2.7010469, -3.2476325, 3.1591821, -5.9485879, 5.9486794
8: -4.3565540, 2.9241207, -5.1026778, 3.1279545, -7.4845085, 8.0267982
9: -2.3802817, 2.6619055, -2.7931716, 3.0583656, -5.4386473, 5.4550772

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of NS_B2_A2_B2_A2_B2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8002297, upper bound: 10.8002301
time: 1.65 seconds

## Relational analysis of NS_B2_A2_B2_A2_B2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8002297, upper bound: 10.8002301
time: 1.36 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.97 seconds
NS_B1_A1_B2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 8, lower bound: -9.8056850, upper bound: 10.0209070
NS_B1_A1_B2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 8, lower bound: -9.8940397, upper bound: 10.0921307
NS_B1_A1_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.0376586, upper bound: 10.1734810
NS_B1_A1_B2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.0622790, upper bound: 10.2504968
NS_B1_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
NS_B1_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
NS_B1_A2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.7437333, upper bound: 10.8248509
NS_B1_A2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.7437333, upper bound: 10.8248509
NS_B1_A2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.7437333, upper bound: 10.8248509
NS_B1_A2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.7437333, upper bound: 10.8248509
NS_B1_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.7405626, upper bound: 10.8241305
NS_B1_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.7405626, upper bound: 10.8241305
NS_B1_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.7405626, upper bound: 10.8241305
NS_B1_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.7405626, upper bound: 10.8241305
NS_B1_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.7165048, upper bound: 10.8194169
NS_B1_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.7165048, upper bound: 10.8194169
NS_B1_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.7165048, upper bound: 10.8194169
NS_B1_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.7165048, upper bound: 10.8194169
NS_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.8089027, upper bound: 10.8377565
NS_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.8089027, upper bound: 10.8377565
NS_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.8089027, upper bound: 10.8377565
NS_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.8089027, upper bound: 10.8377565
NS_B2_A1_A2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.0709990, upper bound: 10.1621792
NS_B2_A1_A2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.0594909, upper bound: 10.1601029
NS_B2_A1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.2335801, upper bound: 10.3256926
NS_B2_A1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.2333507, upper bound: 10.3255758
NS_B2_A2_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.0561016, upper bound: 10.0025910
NS_B2_A2_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.1323346, upper bound: 10.0310663
NS_B2_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732645
NS_B2_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732645
NS_B2_A2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.3672249, upper bound: 10.4218298
NS_B2_A2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.4746780, upper bound: 10.4845519
NS_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.6809952, upper bound: 10.6708482
NS_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.6809952, upper bound: 10.6882749
NS_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.5424432, upper bound: 10.5230275
NS_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.5424432, upper bound: 10.5230275
NS_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.4849272, upper bound: 10.4847548
NS_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.4771871, upper bound: 10.4547445
NS_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.6423200, upper bound: 10.6527132
NS_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.6423200, upper bound: 10.6621108
NS_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.6530227, upper bound: 10.6542993
NS_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.6530227, upper bound: 10.6697925
NS_B2_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.8002297, upper bound: 10.8002301
NS_B2_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.8002297, upper bound: 10.8002301
NS_B2_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.8002297, upper bound: 10.8002301
NS_B2_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 8, lower bound: -10.8002297, upper bound: 10.8002301

## BFS NS instance: NS_B1_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.7751960, 0.9109775, -2.3940406, 1.9925494, -2.7677455, 3.3050179
1: -0.6399021, 0.7774657, -1.9364059, 1.9731816, -2.6130836, 2.7138715
2: -0.7005208, 1.1096113, -2.5541759, 2.1950855, -2.8956063, 3.6637871
3: -0.6364986, 0.8716450, -2.7481613, 1.9515707, -2.5880694, 3.6198063
4: -0.8213144, 0.7928160, -2.7095983, 2.0934634, -2.9147778, 3.5024142
5: -0.7148739, 0.8631164, -2.2354717, 2.0557802, -2.7706542, 3.0985880
6: -0.7542198, 0.7908704, -2.2491481, 2.2147632, -2.9689829, 3.0400186
7: -0.7744479, 0.8044263, -2.3797359, 2.3019433, -3.0763912, 3.1841621
8: -0.9703314, 2.5448091, -3.7210393, 2.7887297, -3.7590611, 6.2658482
9: -0.7994223, 0.9467521, -2.0305181, 2.3209343, -3.1203566, 2.9772701

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_B1_A1_B2_B2_A2_A2_B1_B1

### Relational analysis result of NS_B1_A1_B2_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1325133, upper bound: 10.3064374
time: 2.16 seconds

## Relational analysis of NS_B1_A1_B2_B2_A2_A2_B1_B2

### Relational analysis result of NS_B1_A1_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.1340588, upper bound: 10.3131702
time: 2.05 seconds

## BFS NS instance: NS_B1_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.7751960, 0.9109775, -2.9803145, 2.4494658, -3.2246618, 3.8912921
1: -0.6399021, 0.7774657, -2.4300337, 2.3852596, -3.0251617, 3.2074995
2: -0.7005208, 1.1096113, -3.2709346, 2.5770509, -3.2775717, 4.3805456
3: -0.6364986, 0.8716450, -3.5215473, 2.2811761, -2.9176745, 4.3931923
4: -0.8213144, 0.7928160, -3.3787968, 2.5815613, -3.4028757, 4.1716127
5: -0.7148739, 0.8631164, -2.8494654, 2.4816475, -3.1965213, 3.7125816
6: -0.7542198, 0.7908704, -2.7907593, 2.7414312, -3.4956510, 3.5816298
7: -0.7744479, 0.8044263, -2.9840608, 2.8945692, -3.6690171, 3.7884870
8: -0.9703314, 2.5448091, -4.6442337, 2.9687595, -3.9390910, 7.1890430
9: -0.7994223, 0.9467521, -2.5493019, 2.8303485, -3.6297708, 3.4960539

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_B1_A1_B2_B2_A2_A2_B2_B1

### Relational analysis result of NS_B1_A1_B2_B2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1325133, upper bound: 10.3064373
time: 2.37 seconds

## Relational analysis of NS_B1_A1_B2_B2_A2_A2_B2_B2

### Relational analysis result of NS_B1_A1_B2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.1340588, upper bound: 10.3131702
time: 2.90 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -2.3642178, 1.9606442, -2.0864091, 1.7473655, -4.1115832, 4.0470533
1: -1.9131807, 1.9470960, -1.6771207, 1.7404528, -3.6536336, 3.6242166
2: -2.5082746, 2.1683440, -2.1508434, 1.9831524, -4.4914269, 4.3191872
3: -2.7259469, 1.9498641, -2.3221915, 1.7517555, -4.4777021, 4.2720556
4: -2.6792746, 2.0637350, -2.3517411, 1.8255392, -4.5048137, 4.4154758
5: -2.2033539, 2.0326715, -1.9146250, 1.8191981, -4.0225520, 3.9472966
6: -2.2274861, 2.1860967, -1.9687158, 1.9262135, -4.1536999, 4.1548123
7: -2.3497040, 2.2750111, -2.0530674, 1.9914991, -4.3412032, 4.3280783
8: -3.6666729, 2.7284186, -3.2051523, 2.6655407, -6.3322134, 5.9335709
9: -2.0081916, 2.2922745, -1.7572045, 2.0403781, -4.0485697, 4.0494790

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A1_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7420899, upper bound: 10.8246129
time: 2.40 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A1_B2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7420899, upper bound: 10.8246129
time: 2.58 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -4.2474637, 3.4408956, -2.0864091, 1.7473655, -5.9948292, 5.5273046
1: -3.4191189, 3.3035302, -1.6771207, 1.7404528, -5.1595716, 4.9806509
2: -4.8097744, 3.3790739, -2.1508434, 1.9831524, -6.7929268, 5.5299172
3: -5.2172513, 3.0136786, -2.3221915, 1.7517555, -6.9690065, 5.3358698
4: -4.8375115, 3.6209795, -2.3517411, 1.8255392, -6.6630507, 5.9727206
5: -4.1944313, 3.3898363, -1.9146250, 1.8191981, -6.0136294, 5.3044615
6: -3.9555531, 3.8816004, -1.9687158, 1.9262135, -5.8817663, 5.8503160
7: -4.2676134, 4.2037611, -2.0530674, 1.9914991, -6.2591124, 6.2568283
8: -6.6273251, 3.3069999, -3.2051523, 2.6655407, -9.2928658, 6.5121522
9: -3.7322872, 3.9134307, -1.7572045, 2.0403781, -5.7726650, 5.6706352

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7420899, upper bound: 10.8246129
time: 2.30 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7420899, upper bound: 10.8246129
time: 8.45 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -2.3642178, 1.9606442, -3.9652963, 3.2037225, -5.5679402, 5.9259405
1: -1.9131807, 1.9470960, -3.1844819, 3.0798383, -4.9930191, 5.1315780
2: -2.5082746, 2.1683440, -4.4558759, 3.1884928, -5.6967673, 6.6242199
3: -2.7259469, 1.9498641, -4.7941189, 2.8164973, -5.5424442, 6.7439833
4: -2.6792746, 2.0637350, -4.5120831, 3.3789968, -6.0582714, 6.5758181
5: -2.2033539, 2.0326715, -3.8975408, 3.1722844, -5.3756380, 5.9302120
6: -2.2274861, 2.1860967, -3.6814361, 3.6183853, -5.8458714, 5.8675327
7: -2.3497040, 2.2750111, -3.9739728, 3.9042940, -6.2539978, 6.2489839
8: -3.6666729, 2.7284186, -6.1799049, 3.1695035, -6.8361764, 8.9083233
9: -2.0081916, 2.2922745, -3.4655647, 3.6580400, -5.6662316, 5.7578392

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A1_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7436711, upper bound: 10.8248293
time: 5.03 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A1_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7416852, upper bound: 10.8244364
time: 2.75 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -4.2459078, 3.4283829, -3.9652963, 3.2037225, -7.4496303, 7.3936791
1: -3.4183152, 3.2806730, -3.1844819, 3.0798383, -6.4981537, 6.4651546
2: -4.7894831, 3.3733983, -4.4558759, 3.1884928, -7.9779758, 7.8292742
3: -5.1876254, 3.0136786, -4.7941189, 2.8164973, -8.0041227, 7.8077974
4: -4.8345566, 3.6203635, -4.5120831, 3.3789968, -8.2135534, 8.1324463
5: -4.1941614, 3.3856401, -3.8975408, 3.1722844, -7.3664455, 7.2831812
6: -3.9470117, 3.8763933, -3.6814361, 3.6183853, -7.5653973, 7.5578294
7: -4.2587910, 4.1966529, -3.9739728, 3.9042940, -8.1630850, 8.1706257
8: -6.6228976, 3.2989497, -6.1799049, 3.1695035, -9.7924013, 9.4788551
9: -3.7245111, 3.9041564, -3.4655647, 3.6580400, -7.3825512, 7.3697214

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7416852, upper bound: 10.8244364
time: 2.45 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7416852, upper bound: 10.8244364
time: 2.06 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 6.33 + 596.90 = 603.23 seconds
