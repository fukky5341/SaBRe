## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 7.125826784999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.5828104, 2.9839077, -3.5828104, 2.9839077, -6.5667181, 6.5667181)
1: (-2.8148921, 2.6596637, -2.8148921, 2.6596637, -5.4745560, 5.4745560)
2: (-3.6384213, 2.7152596, -3.6384213, 2.7152596, -6.3536806, 6.3536806)
3: (-4.0147090, 2.3897834, -4.0147090, 2.3897834, -6.4044924, 6.4044924)
4: (-3.9739902, 2.8952332, -3.9739902, 2.8952332, -6.8692236, 6.8692236)
5: (-3.4501100, 2.9820681, -3.4501100, 2.9820681, -6.4321771, 6.4321775)
6: (-3.2314649, 3.2592940, -3.2314649, 3.2592940, -6.4907575, 6.4907589)
7: (-3.3493385, 3.3119178, -3.3493385, 3.3119178, -6.6612563, 6.6612563)
8: (-5.2216048, 3.1262531, -5.2216048, 3.1262531, -8.3478584, 8.3478584)
9: (-3.0501204, 3.2086091, -3.0501204, 3.2086091, -6.2587285, 6.2587290)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.14 + 4.11 = 6.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -7.5008703, upper bound: 7.5008703

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4269333, upper bound: 7.3545148
time: 1.82 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3444414, upper bound: 7.3444414
time: 2.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.30 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.30
Output dim: 8, lower bound: -7.4269333, upper bound: 7.3545148
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.30
Output dim: 8, lower bound: -7.3444414, upper bound: 7.3444414

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.5529716, 2.9620543, -3.5828104, 2.9839077, -6.5368795, 6.5448642
1: -2.7893021, 2.6380491, -2.8148921, 2.6596637, -5.4489660, 5.4529409
2: -3.6050932, 2.6947055, -3.6384213, 2.7152596, -6.3203521, 6.3331261
3: -3.9779975, 2.3711686, -4.0147090, 2.3897834, -6.3677807, 6.3858771
4: -3.9394760, 2.8718376, -3.9739902, 2.8952332, -6.8347092, 6.8458266
5: -3.4222827, 2.9585900, -3.4501100, 2.9820681, -6.4043489, 6.4086995
6: -3.2040727, 3.2336571, -3.2314649, 3.2592940, -6.4633660, 6.4651217
7: -3.3211319, 3.2843447, -3.3493385, 3.3119178, -6.6330500, 6.6336827
8: -5.1787386, 3.1085513, -5.2216048, 3.1262531, -8.3049917, 8.3301563
9: -3.0237226, 3.1826100, -3.0501204, 3.2086091, -6.2323308, 6.2327299

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2835654, upper bound: 7.2589940
time: 2.50 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2554141, upper bound: 7.1757538
time: 2.63 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -5.8006988, 4.6680269, -3.5109420, 2.9310701, -8.7317667, 8.1789684
1: -4.8977828, 4.2458553, -2.7529690, 2.6077373, -7.5055199, 6.9988232
2: -6.2657723, 4.2132750, -3.5576231, 2.6658623, -8.9316330, 7.7708979
3: -6.8137674, 3.5712230, -3.9260814, 2.3460608, -9.1598263, 7.4973040
4: -6.7003613, 4.6733770, -3.8903964, 2.8388824, -9.5392418, 8.5637732
5: -5.6618509, 4.6575503, -3.3826187, 2.9258444, -8.5876942, 8.0401688
6: -5.2675271, 5.2719884, -3.1656690, 3.1972022, -8.4647293, 8.4376574
7: -5.5929146, 5.5810022, -3.2811649, 3.2450604, -8.8379745, 8.8621674
8: -8.5298328, 4.0685792, -5.1178570, 3.0845337, -11.6143665, 9.1864338
9: -5.0602059, 5.1589456, -2.9864793, 3.1459491, -8.2061548, 8.1454248

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3098770, upper bound: 7.3202419
time: 2.64 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3085133, upper bound: 7.3085133
time: 1.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 13.04 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 13.04
Output dim: 8, lower bound: -7.2835654, upper bound: 7.2589940
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.04
Output dim: 8, lower bound: -7.2554141, upper bound: 7.1757538
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 13.04
Output dim: 8, lower bound: -7.3098770, upper bound: 7.3202419
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 13.04
Output dim: 8, lower bound: -7.3085133, upper bound: 7.3085133

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.2931023, 2.7681992, -2.8547325, 2.4479661, -5.7410684, 5.6229310
1: -2.5658643, 2.4571924, -2.2315476, 2.1572013, -4.7230654, 4.6887398
2: -3.3165047, 2.5133100, -2.8304994, 2.2163696, -5.5328741, 5.3438082
3: -3.6678216, 2.2250195, -3.1418092, 1.9828181, -5.6506395, 5.3668289
4: -3.6418881, 2.6692061, -3.1763701, 2.3268499, -5.9687381, 5.8455763
5: -3.1760099, 2.7594233, -2.7575424, 2.4309549, -5.6069651, 5.5169659
6: -2.9686117, 3.0022674, -2.5725212, 2.6197433, -5.5883551, 5.5747881
7: -3.0767140, 3.0433037, -2.6639574, 2.6430039, -5.7197161, 5.7072611
8: -4.8144903, 2.9697952, -4.2042208, 2.7654109, -7.5799012, 7.1740160
9: -2.7955413, 2.9628592, -2.4123344, 2.6090717, -5.4046125, 5.3751926

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2835654, upper bound: 7.2579699
time: 2.59 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2835654, upper bound: 7.2589940
time: 2.38 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2.9638004, 2.5286689, -2.8608909, 2.4457185, -5.4095192, 5.3895597
1: -2.3111873, 2.2303827, -2.2478142, 2.1728764, -4.4840636, 4.4781971
2: -2.9500241, 2.2908826, -2.8642750, 2.2111318, -5.1611557, 5.1551576
3: -3.2712731, 2.0417123, -3.1822400, 2.0001709, -5.2714429, 5.2239523
4: -3.2918267, 2.4117920, -3.2008183, 2.3376427, -5.6294694, 5.6126099
5: -2.8639073, 2.5107121, -2.7720361, 2.4272275, -5.2911348, 5.2827473
6: -2.6704819, 2.7130072, -2.5831912, 2.6110456, -5.2815275, 5.2961984
7: -2.7670741, 2.7423382, -2.6928244, 2.6709912, -5.4380646, 5.4351625
8: -4.3538699, 2.8121150, -4.2412667, 2.7263196, -7.0801897, 7.0533819
9: -2.5061746, 2.6954811, -2.4240832, 2.6289573, -5.1351318, 5.1195641

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2552885, upper bound: 7.1757035
time: 2.71 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2554141, upper bound: 7.1757538
time: 1.84 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -5.6450057, 4.5597153, -3.0171719, 2.5862060, -8.2312107, 7.5768871
1: -4.7663832, 4.1323795, -2.3468368, 2.2515969, -7.0179796, 6.4792151
2: -6.0987654, 4.1061058, -3.0253465, 2.3241379, -8.4229031, 7.1314521
3: -6.6223006, 3.4724526, -3.3065751, 2.0317142, -8.6540146, 6.7790270
4: -6.5252352, 4.5509953, -3.3461013, 2.4505763, -8.9758110, 7.8970966
5: -5.5190191, 4.5425196, -2.9294968, 2.5612807, -8.0802994, 7.4720149
6: -5.1269779, 5.1401854, -2.7180424, 2.7769666, -7.9039445, 7.8582277
7: -5.4466424, 5.4405751, -2.8196547, 2.7953241, -8.2419643, 8.2602291
8: -8.3152437, 3.9556370, -4.4345851, 2.7782006, -11.0934439, 8.3902225
9: -4.9220896, 5.0277147, -2.5516150, 2.7304692, -7.6525583, 7.5793295

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3085133, upper bound: 7.3085133
time: 1.44 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3085133, upper bound: 7.3085133
time: 1.40 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -5.5543861, 4.4948440, -4.1070547, 3.4137321, -8.9681187, 8.6018982
1: -4.6882830, 4.0697832, -3.3419437, 3.0358233, -7.7241049, 7.4117246
2: -5.9992132, 4.0418749, -4.2979722, 3.0666509, -9.0658646, 8.3398476
3: -6.5170097, 3.4213347, -4.6975441, 2.6602118, -9.1772194, 8.1188774
4: -6.4205875, 4.4802475, -4.6480298, 3.3223155, -9.7429008, 9.1282768
5: -5.4344101, 4.4746447, -4.0168867, 3.4019449, -8.8363543, 8.4915295
6: -5.0451765, 5.0635881, -3.7163515, 3.7482123, -8.7933865, 8.7799377
7: -5.3602934, 5.3567929, -3.9005413, 3.8847075, -9.2449970, 9.2573338
8: -8.1861343, 3.8946300, -6.0444608, 3.2177668, -11.4039011, 9.9390907
9: -4.8429499, 4.9504528, -3.5295911, 3.6890361, -8.5319862, 8.4800434

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3085133, upper bound: 7.3085133
time: 1.54 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3085133, upper bound: 7.3085133
time: 1.41 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.06 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 8, lower bound: -7.2835654, upper bound: 7.2579699
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 8, lower bound: -7.2835654, upper bound: 7.2589940
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 8, lower bound: -7.2552885, upper bound: 7.1757035
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 8, lower bound: -7.2554141, upper bound: 7.1757538
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 8, lower bound: -7.3085133, upper bound: 7.3085133
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 8, lower bound: -7.3085133, upper bound: 7.3085133
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 8, lower bound: -7.3085133, upper bound: 7.3085133
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 8, lower bound: -7.3085133, upper bound: 7.3085133

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.7219315, 2.3567083, -2.7504370, 2.3692255, -5.0911570, 5.1071453
1: -2.1240854, 2.0531235, -2.1526754, 2.0832372, -4.2073226, 4.2057981
2: -2.6577251, 2.1383047, -2.7115455, 2.1455407, -4.8032656, 4.8498502
3: -2.9541640, 1.9065573, -3.0100415, 1.9211680, -4.8753319, 4.9165983
4: -3.0132437, 2.2170224, -3.0620401, 2.2432203, -5.2564640, 5.2790623
5: -2.6212232, 2.3482461, -2.6530623, 2.3544765, -4.9756994, 5.0013080
6: -2.4439001, 2.5061328, -2.4756815, 2.5277057, -4.9716058, 4.9818144
7: -2.5216587, 2.4926958, -2.5635641, 2.5428386, -5.0644970, 5.0562592
8: -3.9927490, 2.7405241, -4.0519505, 2.7149961, -6.7077451, 6.7924743
9: -2.2850385, 2.4872949, -2.3192272, 2.5222077, -4.8072462, 4.8065224

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2107396, upper bound: 7.2028803
time: 2.34 seconds

## Relational analysis of NS_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2047510, upper bound: 7.1963811
time: 2.45 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.1622124, 2.6705606, -2.8115859, 2.4154282, -5.5776405, 5.4821463
1: -2.4504180, 2.3650978, -2.1987441, 2.1264822, -4.5769005, 4.5638418
2: -3.1680679, 2.4234757, -2.7809839, 2.1868212, -5.3548889, 5.2044597
3: -3.5057492, 2.1490386, -3.0871377, 1.9573872, -5.4631357, 5.2361765
4: -3.4987183, 2.5659642, -3.1288774, 2.2920260, -5.7907443, 5.6948414
5: -3.0491812, 2.6607971, -2.7140889, 2.3991427, -5.4483237, 5.3748860
6: -2.8489997, 2.8873405, -2.5318646, 2.5817165, -5.4307165, 5.4192052
7: -2.9509735, 2.9174836, -2.6215825, 2.6012168, -5.5521903, 5.5390654
8: -4.6230612, 2.9092326, -4.1414089, 2.7468846, -7.3699455, 7.0506415
9: -2.6801248, 2.8511415, -2.3736219, 2.5729668, -5.2530918, 5.2247624

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2828898, upper bound: 7.2589940
time: 2.48 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2828898, upper bound: 7.2589940
time: 2.83 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.4337707, 2.1413038, -2.7545090, 2.3666208, -4.8003912, 4.8958130
1: -1.9088175, 1.8545805, -2.1672473, 2.0978634, -4.0066810, 4.0218277
2: -2.3323710, 1.9411957, -2.7436168, 2.1399682, -4.4723392, 4.6848125
3: -2.5949135, 1.7437909, -3.0476165, 1.9376664, -4.5325799, 4.7914076
4: -2.6998148, 1.9881349, -3.0842333, 2.2530918, -4.9529066, 5.0723681
5: -2.3336744, 2.1357520, -2.6652284, 2.3504806, -4.6841550, 4.8009806
6: -2.1874442, 2.2463484, -2.4871340, 2.5169663, -4.7044106, 4.7334824
7: -2.2490087, 2.2231212, -2.5906367, 2.5682499, -4.8172588, 4.8137579
8: -3.5893912, 2.6134009, -4.0841818, 2.6786113, -6.2680025, 6.6975827
9: -2.0273361, 2.2548683, -2.3295145, 2.5407419, -4.5680780, 4.5843830

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2483881, upper bound: 7.1712967
time: 2.51 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2483881, upper bound: 7.1757035
time: 2.63 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.8385441, 2.4359479, -2.8177018, 2.4138579, -5.2524023, 5.2536497
1: -2.2176027, 2.1418858, -2.2150931, 2.1423924, -4.3599949, 4.3569789
2: -2.8072693, 2.2063527, -2.8151834, 2.1823788, -4.9896474, 5.0215359
3: -3.1139109, 1.9689691, -3.1275473, 1.9750013, -5.0889120, 5.0965166
4: -3.1557667, 2.3117750, -3.1534641, 2.3032870, -5.4590540, 5.4652390
5: -2.7400119, 2.4180028, -2.7285652, 2.3962164, -5.1362286, 5.1465683
6: -2.5544493, 2.6036253, -2.5443571, 2.5729463, -5.1273947, 5.1479826
7: -2.6454775, 2.6228478, -2.6512642, 2.6290448, -5.2745218, 5.2741117
8: -4.1725688, 2.7582772, -4.1778727, 2.7088709, -6.8814397, 6.9361496
9: -2.3948174, 2.5917783, -2.3856359, 2.5933270, -4.9881444, 4.9774141

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2483881, upper bound: 7.1712967
time: 3.09 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2483881, upper bound: 7.1757538
time: 2.09 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.3096004, 4.3238683, -3.0171719, 2.5862060, -7.8958054, 7.3410387
1: -4.4800377, 3.8883533, -2.3468368, 2.2515969, -6.7316346, 6.2351894
2: -5.7393017, 3.8751357, -3.0253465, 2.3241379, -8.0634384, 6.9004822
3: -6.2053514, 3.2578862, -3.3065751, 2.0317142, -8.2370653, 6.5644612
4: -6.1460490, 4.2859893, -3.3461013, 2.4505763, -8.5966244, 7.6320906
5: -5.2092619, 4.2920647, -2.9294968, 2.5612807, -7.7705426, 7.2215610
6: -4.8265057, 4.8543034, -2.7180424, 2.7769666, -7.6034713, 7.5723457
7: -5.1306157, 5.1346846, -2.8196547, 2.7953241, -7.9259391, 7.9543386
8: -7.8509283, 3.7139161, -4.4345851, 2.7782006, -10.6291294, 8.1485014
9: -4.6242189, 4.7433448, -2.5516150, 2.7304692, -7.3546867, 7.2949600

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3086951, upper bound: 7.3186372
time: 1.79 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3098770, upper bound: 7.3202419
time: 1.93 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.4007897, 5.1564326, -3.0171719, 2.5862060, -8.9869957, 8.1736031
1: -5.4876442, 4.6735187, -2.3468368, 2.2515969, -7.7392397, 7.0203552
2: -7.0108008, 4.6211100, -3.0253465, 2.3241379, -9.3349361, 7.6464553
3: -7.5927162, 3.8863859, -3.3065751, 2.0317142, -9.6244307, 7.1929607
4: -7.4635201, 5.1583891, -3.3461013, 2.4505763, -9.9140968, 8.5044899
5: -6.2976418, 5.1377463, -2.9294968, 2.5612807, -8.8589230, 8.0672417
6: -5.8194895, 5.8240886, -2.7180424, 2.7769666, -8.5964546, 8.5421314
7: -6.2188525, 6.2253466, -2.8196547, 2.7953241, -9.0141735, 9.0450010
8: -9.4621868, 4.3126020, -4.4345851, 2.7782006, -12.2403870, 8.7471867
9: -5.6068130, 5.7055073, -2.5516150, 2.7304692, -8.3372822, 8.2571220

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3086951, upper bound: 7.3186372
time: 1.54 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3098770, upper bound: 7.3202419
time: 1.73 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.3096004, 4.3238683, -4.1070547, 3.4137321, -8.7233324, 8.4309216
1: -4.4800377, 3.8883533, -3.3419437, 3.0358233, -7.5158606, 7.2302966
2: -5.7393017, 3.8751357, -4.2979722, 3.0666509, -8.8059521, 8.1731081
3: -6.2053514, 3.2578862, -4.6975441, 2.6602118, -8.8655624, 7.9554296
4: -6.1460490, 4.2859893, -4.6480298, 3.3223155, -9.4683609, 8.9340191
5: -5.2092619, 4.2920647, -4.0168867, 3.4019449, -8.6112051, 8.3089514
6: -4.8265057, 4.8543034, -3.7163515, 3.7482123, -8.5747166, 8.5706549
7: -5.1306157, 5.1346846, -3.9005413, 3.8847075, -9.0153198, 9.0352259
8: -7.8509283, 3.7139161, -6.0444608, 3.2177668, -11.0686951, 9.7583771
9: -4.6242189, 4.7433448, -3.5295911, 3.6890361, -8.3132544, 8.2729359

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3048044, upper bound: 7.3022394
time: 1.61 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3085133, upper bound: 7.3085133
time: 1.86 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.4007897, 5.1564326, -4.1070547, 3.4137321, -9.8145199, 9.2634869
1: -5.4876442, 4.6735187, -3.3419437, 3.0358233, -8.5234661, 8.0154629
2: -7.0108008, 4.6211100, -4.2979722, 3.0666509, -10.0774498, 8.9190826
3: -7.5927162, 3.8863859, -4.6975441, 2.6602118, -10.2529278, 8.5839291
4: -7.4635201, 5.1583891, -4.6480298, 3.3223155, -10.7858353, 9.8064194
5: -6.2976418, 5.1377463, -4.0168867, 3.4019449, -9.6995840, 9.1546307
6: -5.8194895, 5.8240886, -3.7163515, 3.7482123, -9.5677013, 9.5404377
7: -6.2188525, 6.2253466, -3.9005413, 3.8847075, -10.1035557, 10.1258879
8: -9.4621868, 4.3126020, -6.0444608, 3.2177668, -12.6799536, 10.3570633
9: -5.6068130, 5.7055073, -3.5295911, 3.6890361, -9.2958488, 9.2350979

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3048044, upper bound: 7.3022394
time: 1.48 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3085133, upper bound: 7.3085133
time: 1.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.44 seconds
NS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.2107396, upper bound: 7.2028803
NS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.2047510, upper bound: 7.1963811
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.2828898, upper bound: 7.2589940
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.2828898, upper bound: 7.2589940
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.2483881, upper bound: 7.1712967
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.2483881, upper bound: 7.1757035
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.2483881, upper bound: 7.1712967
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.2483881, upper bound: 7.1757538
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.3086951, upper bound: 7.3186372
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.3098770, upper bound: 7.3202419
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.3086951, upper bound: 7.3186372
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.3098770, upper bound: 7.3202419
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.3048044, upper bound: 7.3022394
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.3085133, upper bound: 7.3085133
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.3048044, upper bound: 7.3022394
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.44
Output dim: 8, lower bound: -7.3085133, upper bound: 7.3085133

## BFS NS instance: NS_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -2.3055713, 2.0605240, -2.6150708, 2.2738774, -4.5794487, 4.6755948
1: -1.8019303, 1.7505115, -2.0512083, 1.9850520, -3.7869823, 3.8017197
2: -2.1977253, 1.8487592, -2.5648389, 2.0518746, -4.2495999, 4.4135981
3: -2.4104636, 1.6264927, -2.8343980, 1.8334587, -4.2439222, 4.4608908
4: -2.5583079, 1.8761915, -2.9141300, 2.1347744, -4.6930823, 4.7903214
5: -2.2124777, 2.0542164, -2.5232136, 2.2581480, -4.4706259, 4.5774298
6: -2.0717440, 2.1459587, -2.3520961, 2.4117000, -4.4834442, 4.4980545
7: -2.1233807, 2.1044996, -2.4374211, 2.4172146, -4.5405951, 4.5419207
8: -3.4188333, 2.4998162, -3.8665731, 2.6370780, -6.0559111, 6.3663893
9: -1.9116898, 2.1472139, -2.1987362, 2.4111848, -4.3228745, 4.3459501

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2047510, upper bound: 7.1963811
time: 2.44 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2047510, upper bound: 7.1963811
time: 2.52 seconds

## BFS NS instance: NS_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -3.2868748, 2.7854164, -2.5345995, 2.2180457, -5.5049205, 5.3200159
1: -2.5665326, 2.4588008, -1.9910984, 1.9312606, -4.4977932, 4.4498992
2: -3.3565958, 2.4915578, -2.4768434, 1.9967453, -5.3533411, 4.9684010
3: -3.6899405, 2.2039950, -2.7384462, 1.7882735, -5.4782133, 4.9424410
4: -3.6644044, 2.6737216, -2.8262753, 2.0717950, -5.7361994, 5.4999971
5: -3.2313879, 2.7651143, -2.4447782, 2.2026100, -5.4339981, 5.2098923
6: -2.9626002, 3.0128984, -2.2804449, 2.3446465, -5.3072467, 5.2933435
7: -3.0943379, 3.0857790, -2.3620410, 2.3419547, -5.4362926, 5.4478197
8: -4.8522034, 2.8116128, -3.7547112, 2.5980246, -7.4502277, 6.5663242
9: -2.8059642, 2.9633863, -2.1286447, 2.3463812, -5.1523447, 5.0920310

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2047510, upper bound: 7.1963811
time: 3.37 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2047510, upper bound: 7.1963811
time: 1.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.1622124, 2.6705606, -2.3145220, 2.0519395, -5.2141519, 4.9850826
1: -2.4504180, 2.3650978, -1.8153800, 1.7736697, -4.2240877, 4.1804771
2: -3.1680679, 2.4234757, -2.1970344, 1.8573515, -5.0254192, 4.6205101
3: -3.5057492, 2.1490386, -2.4439411, 1.6776204, -5.1833696, 4.5929794
4: -3.4987183, 2.5659642, -2.5654178, 1.8939211, -5.3926392, 5.1313820
5: -3.0491812, 2.6607971, -2.2091937, 2.0524440, -5.1016254, 4.8699908
6: -2.8489997, 2.8873405, -2.0853977, 2.1405027, -4.9895024, 4.9727383
7: -2.9509735, 2.9174836, -2.1320248, 2.1099949, -5.0609684, 5.0495081
8: -4.6230612, 2.9092326, -3.4180045, 2.5699184, -7.1929798, 6.3272371
9: -2.6801248, 2.8511415, -1.9262421, 2.1566160, -4.8367405, 4.7773838

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1965682, upper bound: 7.2033096
time: 3.13 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1913183, upper bound: 7.1985269
time: 1.95 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.1622124, 2.6705606, -2.7314901, 2.3554926, -5.5177050, 5.4020510
1: -2.4504180, 2.3650978, -2.1377695, 2.0696387, -4.5200567, 4.5028672
2: -3.1680679, 2.4234757, -2.6888995, 2.1329861, -5.3010530, 5.1123753
3: -3.5057492, 2.1490386, -2.9853725, 1.9103801, -5.4161291, 5.1344109
4: -3.4987183, 2.5659642, -3.0405328, 2.2276356, -5.7263536, 5.6064963
5: -3.0491812, 2.6607971, -2.6330485, 2.3409162, -5.3900976, 5.2938457
6: -2.8489997, 2.8873405, -2.4583659, 2.5110326, -5.3600321, 5.3457055
7: -2.9509735, 2.9174836, -2.5446169, 2.5233455, -5.4743190, 5.4621000
8: -4.6230612, 2.9092326, -4.0250511, 2.7132573, -7.3363180, 6.9342823
9: -2.6801248, 2.8511415, -2.3018477, 2.5065036, -5.1866283, 5.1529884

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1965682, upper bound: 7.2117383
time: 2.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1913183, upper bound: 7.2083973
time: 2.26 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.4337707, 2.1413038, -2.3439102, 2.0649438, -4.4987144, 4.4852142
1: -1.9088175, 1.8545805, -1.8463837, 1.8067434, -3.7155609, 3.7009642
2: -2.3323710, 1.9411957, -2.2558057, 1.8671194, -4.1994905, 4.1970015
3: -2.5949135, 1.7437909, -2.5099747, 1.7066946, -4.3016081, 4.2537656
4: -2.6998148, 1.9881349, -2.6164832, 1.9223967, -4.6222115, 4.6046181
5: -2.3336744, 2.1357520, -2.2431769, 2.0654135, -4.3990879, 4.3789291
6: -2.1874442, 2.2463484, -2.1214657, 2.1502473, -4.3376913, 4.3678141
7: -2.2490087, 2.2231212, -2.1798139, 2.1579819, -4.4069905, 4.4029350
8: -3.5893912, 2.6134009, -3.4772680, 2.5501652, -6.1395564, 6.0906687
9: -2.0273361, 2.2548683, -1.9582283, 2.1946645, -4.2220006, 4.2130966

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0318580, upper bound: 7.0408628
time: 2.13 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2436976, upper bound: 7.1637791
time: 2.59 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.4337707, 2.1413038, -2.7374320, 2.3543844, -4.7881551, 4.8787355
1: -1.9088175, 1.8545805, -2.1539211, 2.0857599, -3.9945774, 4.0085015
2: -2.3323710, 1.9411957, -2.7235663, 2.1286333, -4.4610043, 4.6647620
3: -2.5949135, 1.7437909, -3.0255303, 1.9279659, -4.5228796, 4.7693214
4: -2.6998148, 1.9881349, -3.0650120, 2.2391584, -4.9389734, 5.0531468
5: -2.3336744, 2.1357520, -2.6472960, 2.3383069, -4.6719813, 4.7830477
6: -2.1874442, 2.2463484, -2.4717474, 2.5019064, -4.6893506, 4.7180958
7: -2.2490087, 2.2231212, -2.5736339, 2.5508950, -4.7999039, 4.7967548
8: -3.5893912, 2.6134009, -4.0598955, 2.6769092, -6.2663002, 6.6732965
9: -2.0273361, 2.2548683, -2.3140345, 2.5267193, -4.5540552, 4.5689030

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0438471, upper bound: 6.9826152
time: 3.01 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0595135, upper bound: 6.9849739
time: 2.37 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.8385441, 2.4359479, -2.3439102, 2.0649438, -4.9034882, 4.7798581
1: -2.2176027, 2.1418858, -1.8463837, 1.8067434, -4.0243464, 3.9882693
2: -2.8072693, 2.2063527, -2.2558057, 1.8671194, -4.6743889, 4.4621582
3: -3.1139109, 1.9689691, -2.5099747, 1.7066946, -4.8206043, 4.4789438
4: -3.1557667, 2.3117750, -2.6164832, 1.9223967, -5.0781631, 4.9282579
5: -2.7400119, 2.4180028, -2.2431769, 2.0654135, -4.8054256, 4.6611795
6: -2.5544493, 2.6036253, -2.1214657, 2.1502473, -4.7046967, 4.7250910
7: -2.6454775, 2.6228478, -2.1798139, 2.1579819, -4.8034582, 4.8026619
8: -4.1725688, 2.7582772, -3.4772680, 2.5501652, -6.7227340, 6.2355452
9: -2.3948174, 2.5917783, -1.9582283, 2.1946645, -4.5894818, 4.5500069

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0323627, upper bound: 7.0469442
time: 2.32 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2436976, upper bound: 7.1637791
time: 3.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.8385441, 2.4359479, -2.7374320, 2.3543844, -5.1929283, 5.1733794
1: -2.2176027, 2.1418858, -2.1539211, 2.0857599, -4.3033628, 4.2958069
2: -2.8072693, 2.2063527, -2.7235663, 2.1286333, -4.9359026, 4.9299192
3: -3.1139109, 1.9689691, -3.0255303, 1.9279659, -5.0418768, 4.9944992
4: -3.1557667, 2.3117750, -3.0650120, 2.2391584, -5.3949251, 5.3767872
5: -2.7400119, 2.4180028, -2.6472960, 2.3383069, -5.0783186, 5.0652990
6: -2.5544493, 2.6036253, -2.4717474, 2.5019064, -5.0563555, 5.0753727
7: -2.6454775, 2.6228478, -2.5736339, 2.5508950, -5.1963720, 5.1964817
8: -4.1725688, 2.7582772, -4.0598955, 2.6769092, -6.8494778, 6.8181725
9: -2.3948174, 2.5917783, -2.3140345, 2.5267193, -4.9215364, 4.9058127

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0323627, upper bound: 7.0648828
time: 3.08 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2436976, upper bound: 7.1681330
time: 2.91 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.1925683, 4.2341099, -2.4936459, 2.2026157, -7.3951840, 6.7277555
1: -4.3759136, 3.8055546, -1.9455515, 1.8812459, -6.2571592, 5.7511063
2: -5.6075459, 3.7936649, -2.4131410, 1.9792565, -7.5868020, 6.2068052
3: -6.0614653, 3.1895742, -2.6319950, 1.7331817, -7.7946467, 5.8215694
4: -6.0098147, 4.1937981, -2.7664330, 2.0275593, -8.0373745, 6.9602308
5: -5.0958977, 4.2011256, -2.4050875, 2.1891775, -7.2850742, 6.6062117
6: -4.7197046, 4.7500539, -2.2383149, 2.3169465, -7.0366507, 6.9883690
7: -5.0176678, 5.0222163, -2.3056664, 2.2789893, -7.2966571, 7.3278828
8: -7.6798239, 3.6376371, -3.6850562, 2.5756152, -10.2554388, 7.3226933
9: -4.5206327, 4.6413174, -2.0784688, 2.2955813, -6.8162112, 6.7197862

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of NS_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1642431, upper bound: 7.2013829
time: 2.45 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1578091, upper bound: 7.1559706
time: 2.25 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.2617331, 4.2873306, -2.8909206, 2.4934096, -7.7551427, 7.1782513
1: -4.4370565, 3.8543835, -2.2524562, 2.1633966, -6.6004529, 6.1068392
2: -5.6850638, 3.8417356, -2.8811345, 2.2397556, -7.9248195, 6.7228699
3: -6.1460547, 3.2300837, -3.1477535, 1.9586062, -8.1046610, 6.3778372
4: -6.0898395, 4.2481604, -3.2104633, 2.3498411, -8.4396801, 7.4586234
5: -5.1626506, 4.2549510, -2.8050935, 2.4679120, -7.6305628, 7.0600443
6: -4.7828941, 4.8117170, -2.6018286, 2.6678078, -7.4507017, 7.4135456
7: -5.0840645, 5.0883131, -2.6981626, 2.6756606, -7.7597251, 7.7864757
8: -7.7808719, 3.6840587, -4.2536578, 2.7228184, -10.5036907, 7.9377165
9: -4.5817051, 4.7015448, -2.4404695, 2.6264906, -7.2081957, 7.1420135

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of NS_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1651020, upper bound: 7.2017796
time: 1.77 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1604281, upper bound: 7.1604281
time: 1.80 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.2773366, 5.0622759, -2.4936459, 2.2026157, -8.4799519, 7.5559216
1: -5.3780499, 4.5861974, -1.9455515, 1.8812459, -7.2592945, 6.5317488
2: -6.8718777, 4.5357409, -2.4131410, 1.9792565, -8.8511333, 6.9488821
3: -7.4412889, 3.8145015, -2.6319950, 1.7331817, -9.1744709, 6.4464960
4: -7.3198109, 5.0613794, -2.7664330, 2.0275593, -9.3473692, 7.8278122
5: -6.1781759, 5.0423212, -2.4050875, 2.1891775, -8.3673506, 7.4474068
6: -5.7073035, 5.7147217, -2.2383149, 2.3169465, -8.0242500, 7.9530363
7: -6.0998297, 6.1070032, -2.3056664, 2.2789893, -8.3788185, 8.4126701
8: -9.2822695, 4.2323446, -3.6850562, 2.5756152, -11.8578844, 7.9174008
9: -5.4976454, 5.5983081, -2.0784688, 2.2955813, -7.7932258, 7.6767745

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of NS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3034958, upper bound: 7.3133440
time: 1.93 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3034958, upper bound: 7.3186372
time: 2.00 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.3512197, 5.1187954, -2.8909206, 2.4934096, -8.8446293, 8.0097160
1: -5.4431944, 4.6383667, -2.2524562, 2.1633966, -7.6065907, 6.8908229
2: -6.9546781, 4.5867739, -2.8811345, 2.2397556, -9.1944332, 7.4679084
3: -7.5315852, 3.8576717, -3.1477535, 1.9586062, -9.4901915, 7.0054255
4: -7.4053507, 5.1192951, -3.2104633, 2.3498411, -9.7551918, 8.3297577
5: -6.2493839, 5.0995646, -2.8050935, 2.4679120, -8.7172956, 7.9046559
6: -5.7744522, 5.7802372, -2.6018286, 2.6678078, -8.4422598, 8.3820658
7: -6.1707406, 6.1774416, -2.6981626, 2.6756606, -8.8464003, 8.8756046
8: -9.3898916, 4.2816248, -4.2536578, 2.7228184, -12.1127100, 8.5352821
9: -5.5628414, 5.6623812, -2.4404695, 2.6264906, -8.1893320, 8.1028500

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of NS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3034958, upper bound: 7.3133440
time: 1.89 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3034958, upper bound: 7.3202419
time: 1.87 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.1925683, 4.2341099, -3.4975319, 2.9521396, -8.1447077, 7.7316408
1: -4.3759136, 3.8055546, -2.7871504, 2.6010604, -6.9769740, 6.5927043
2: -5.6075459, 3.7936649, -3.5873106, 2.6496458, -8.2571917, 7.3809748
3: -6.0614653, 3.1895742, -3.9341323, 2.3170400, -8.3785057, 7.1237059
4: -6.0098147, 4.1937981, -3.9164402, 2.8370843, -8.8468990, 8.1102381
5: -5.0958977, 4.2011256, -3.4210641, 2.9406433, -8.0365410, 7.6221886
6: -4.7197046, 4.7500539, -3.1605904, 3.2020230, -7.9217253, 7.9106445
7: -5.0176678, 5.0222163, -3.2987692, 3.2788868, -8.2965536, 8.3209848
8: -7.6798239, 3.6376371, -5.1391382, 2.9198439, -10.5996675, 8.7767754
9: -4.5206327, 4.6413174, -2.9821150, 3.1516030, -7.6722345, 7.6234312

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3133440, upper bound: 7.3034958
time: 1.99 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3133440, upper bound: 7.3034958
time: 2.02 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.2617331, 4.2873306, -3.9733930, 3.3118274, -8.5735607, 8.2607231
1: -4.4370565, 3.8543835, -3.2224376, 2.9409449, -7.3780012, 7.0768204
2: -5.6850638, 3.8417356, -4.1459713, 2.9734416, -8.6585054, 7.9877071
3: -6.1460547, 3.2300837, -4.5316215, 2.5823066, -8.7283592, 7.7617049
4: -6.0898395, 4.2481604, -4.4910970, 3.2168202, -9.3066578, 8.7392569
5: -5.1626506, 4.2549510, -3.8869901, 3.2989569, -8.4616051, 8.1419411
6: -4.7828941, 4.8117170, -3.5943100, 3.6291411, -8.4120331, 8.4060259
7: -5.0840645, 5.0883131, -3.7712269, 3.7551961, -8.8392591, 8.8595390
8: -7.7808719, 3.6840587, -5.8484354, 3.1430221, -10.9238930, 9.5324936
9: -4.5817051, 4.7015448, -3.4112234, 3.5720494, -8.1537542, 8.1127682

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3186372, upper bound: 7.3086951
time: 1.64 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3186372, upper bound: 7.3098770
time: 1.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6.2773366, 5.0622759, -3.4975319, 2.9521396, -9.2294741, 8.5598078
1: -5.3780499, 4.5861974, -2.7871504, 2.6010604, -7.9791102, 7.3733468
2: -6.8718777, 4.5357409, -3.5873106, 2.6496458, -9.5215235, 8.1230516
3: -7.4412889, 3.8145015, -3.9341323, 2.3170400, -9.7583294, 7.7486339
4: -7.3198109, 5.0613794, -3.9164402, 2.8370843, -10.1568947, 8.9778194
5: -6.1781759, 5.0423212, -3.4210641, 2.9406433, -9.1188192, 8.4633846
6: -5.7073035, 5.7147217, -3.1605904, 3.2020230, -8.9093256, 8.8753109
7: -6.0998297, 6.1070032, -3.2987692, 3.2788868, -9.3787165, 9.4057722
8: -9.2822695, 4.2323446, -5.1391382, 2.9198439, -12.2021132, 9.3714819
9: -5.4976454, 5.5983081, -2.9821150, 3.1516030, -8.6492481, 8.5804224

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of NS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3019449, upper bound: 7.3019449
time: 1.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3019449, upper bound: 7.3022394
time: 1.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.3512197, 5.1187954, -3.9733930, 3.3118274, -9.6630468, 9.0921879
1: -5.4431944, 4.6383667, -3.2224376, 2.9409449, -8.3841391, 7.8608046
2: -6.9546781, 4.5867739, -4.1459713, 2.9734416, -9.9281178, 8.7327452
3: -7.5315852, 3.8576717, -4.5316215, 2.5823066, -10.1138878, 8.3892918
4: -7.4053507, 5.1192951, -4.4910970, 3.2168202, -10.6221695, 9.6103916
5: -6.2493839, 5.0995646, -3.8869901, 3.2989569, -9.5483408, 8.9865532
6: -5.7744522, 5.7802372, -3.5943100, 3.6291411, -9.4035931, 9.3745470
7: -6.1707406, 6.1774416, -3.7712269, 3.7551961, -9.9259367, 9.9486675
8: -9.3898916, 4.2816248, -5.8484354, 3.1430221, -12.5329132, 10.1300592
9: -5.5628414, 5.6623812, -3.4112234, 3.5720494, -9.1348896, 9.0736046

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3022394, upper bound: 7.3048044
time: 2.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3022394, upper bound: 7.3085133
time: 1.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.43 seconds
NS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.2047510, upper bound: 7.1963811
NS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.2047510, upper bound: 7.1963811
NS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.2047510, upper bound: 7.1963811
NS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.2047510, upper bound: 7.1963811
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.1965682, upper bound: 7.2033096
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.1913183, upper bound: 7.1985269
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.1965682, upper bound: 7.2117383
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.1913183, upper bound: 7.2083973
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.0318580, upper bound: 7.0408628
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.2436976, upper bound: 7.1637791
NS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.0438471, upper bound: 6.9826152
NS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.0595135, upper bound: 6.9849739
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.0323627, upper bound: 7.0469442
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.2436976, upper bound: 7.1637791
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.0323627, upper bound: 7.0648828
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.2436976, upper bound: 7.1681330
NS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.1642431, upper bound: 7.2013829
NS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.1578091, upper bound: 7.1559706
NS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.1651020, upper bound: 7.2017796
NS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.1604281, upper bound: 7.1604281
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.3034958, upper bound: 7.3133440
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.3034958, upper bound: 7.3186372
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.3034958, upper bound: 7.3133440
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.3034958, upper bound: 7.3202419
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.3133440, upper bound: 7.3034958
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.3133440, upper bound: 7.3034958
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.3186372, upper bound: 7.3086951
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.3186372, upper bound: 7.3098770
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.3019449, upper bound: 7.3019449
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.3019449, upper bound: 7.3022394
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.3022394, upper bound: 7.3048044
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 8, lower bound: -7.3022394, upper bound: 7.3085133

## BFS NS instance: NS_A1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -2.3055713, 2.0605240, -2.3339863, 2.0734396, -4.3790112, 4.3945103
1: -1.8019303, 1.7505115, -1.8310617, 1.7808008, -3.5827312, 3.5815732
2: -2.1977253, 1.8487592, -2.2572627, 1.8541918, -4.0519171, 4.1060219
3: -2.4104636, 1.6264927, -2.4643946, 1.6434097, -4.0538731, 4.0908871
4: -2.5583079, 1.8761915, -2.6049139, 1.9055356, -4.4638433, 4.4811053
5: -2.2124777, 2.0542164, -2.2447882, 2.0612876, -4.2737656, 4.2990046
6: -2.0717440, 2.1459587, -2.0994964, 2.1670604, -4.2388043, 4.2454548
7: -2.1233807, 2.1044996, -2.1687660, 2.1518314, -4.2752123, 4.2732658
8: -3.4188333, 2.4998162, -3.4750214, 2.4872022, -5.9060354, 5.9748373
9: -1.9116898, 2.1472139, -1.9480103, 2.1784582, -4.0901480, 4.0952244

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1958135, upper bound: 7.1985859
time: 2.40 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1958135, upper bound: 7.2028803
time: 2.14 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -2.3055713, 2.0605240, -3.3722138, 2.8297391, -5.1353102, 5.4327378
1: -1.8019303, 1.7505115, -2.6391277, 2.5220814, -4.3240118, 4.3896394
2: -2.1977253, 1.8487592, -3.4752326, 2.5334864, -4.7312117, 5.3239918
3: -2.4104636, 1.6264927, -3.8196301, 2.2448902, -4.6553535, 5.4461231
4: -2.5583079, 1.8761915, -3.7738051, 2.7425776, -5.3008852, 5.6499968
5: -2.2124777, 2.0542164, -3.3229754, 2.8067884, -5.0192661, 5.3771915
6: -2.0717440, 2.1459587, -3.0382118, 3.0795100, -5.1512537, 5.1841707
7: -2.1233807, 2.1044996, -3.1957357, 3.1899695, -5.3133502, 5.3002353
8: -3.4188333, 2.4998162, -5.0029240, 2.8092127, -6.2280459, 7.5027399
9: -1.9116898, 2.1472139, -2.8898199, 3.0418286, -4.9535184, 5.0370340

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1958135, upper bound: 7.1985859
time: 1.95 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1958135, upper bound: 7.2028803
time: 2.27 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -3.2868748, 2.7854164, -2.3339863, 2.0734396, -5.3603144, 5.1194029
1: -2.5665326, 2.4588008, -1.8310617, 1.7808008, -4.3473334, 4.2898626
2: -3.3565958, 2.4915578, -2.2572627, 1.8541918, -5.2107863, 4.7488203
3: -3.6899405, 2.2039950, -2.4643946, 1.6434097, -5.3333497, 4.6683893
4: -3.6644044, 2.6737216, -2.6049139, 1.9055356, -5.5699401, 5.2786350
5: -3.2313879, 2.7651143, -2.2447882, 2.0612876, -5.2926755, 5.0099025
6: -2.9626002, 3.0128984, -2.0994964, 2.1670604, -5.1296606, 5.1123948
7: -3.0943379, 3.0857790, -2.1687660, 2.1518314, -5.2461691, 5.2545452
8: -4.8522034, 2.8116128, -3.4750214, 2.4872022, -7.3394055, 6.2866344
9: -2.8059642, 2.9633863, -1.9480103, 2.1784582, -4.9844222, 4.9113965

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1886204, upper bound: 7.1916945
time: 2.40 seconds

## Relational analysis of NS_A1_B1_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1886204, upper bound: 7.1963811
time: 2.36 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3.2868748, 2.7854164, -3.3722138, 2.8297391, -6.1166139, 6.1576295
1: -2.5665326, 2.4588008, -2.6391277, 2.5220814, -5.0886140, 5.0979285
2: -3.3565958, 2.4915578, -3.4752326, 2.5334864, -5.8900819, 5.9667902
3: -3.6899405, 2.2039950, -3.8196301, 2.2448902, -5.9348307, 6.0236254
4: -3.6644044, 2.6737216, -3.7738051, 2.7425776, -6.4069819, 6.4475269
5: -3.2313879, 2.7651143, -3.3229754, 2.8067884, -6.0381765, 6.0880899
6: -2.9626002, 3.0128984, -3.0382118, 3.0795100, -6.0421104, 6.0511103
7: -3.0943379, 3.0857790, -3.1957357, 3.1899695, -6.2843065, 6.2815146
8: -4.8522034, 2.8116128, -5.0029240, 2.8092127, -7.6614161, 7.8145370
9: -2.8059642, 2.9633863, -2.8898199, 3.0418286, -5.8477926, 5.8532062

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1886204, upper bound: 7.1916945
time: 2.21 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1886204, upper bound: 7.1963811
time: 1.96 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.6988387, 2.3486423, -2.1938157, 1.9685777, -4.6674166, 4.5424581
1: -2.1092844, 2.0314989, -1.7208656, 1.6855414, -3.7948258, 3.7523642
2: -2.6665545, 2.1057489, -2.0675869, 1.7731042, -4.4396586, 4.1733360
3: -2.9113708, 1.8511525, -2.2873056, 1.5951610, -4.5065308, 4.1384583
4: -3.0033400, 2.1959286, -2.4315181, 1.7964840, -4.7998238, 4.6274452
5: -2.6151505, 2.3255954, -2.0872364, 1.9700019, -4.5851526, 4.4128304
6: -2.4252775, 2.4960108, -1.9803491, 2.0354667, -4.4607439, 4.4763598
7: -2.5157926, 2.4960999, -2.0178065, 1.9969417, -4.5127344, 4.5139065
8: -3.9896395, 2.6372719, -3.2465832, 2.5036917, -6.4933310, 5.8838549
9: -2.2704482, 2.4724805, -1.8207898, 2.0592189, -4.3296671, 4.2932701

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1913183, upper bound: 7.1985269
time: 3.45 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1913183, upper bound: 7.1985269
time: 2.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.7755179, 3.1316326, -2.1233175, 1.9209089, -5.6964269, 5.2549500
1: -2.9261055, 2.7992928, -1.6671858, 1.6389523, -4.5650578, 4.4664783
2: -3.9281001, 2.8059430, -1.9933735, 1.7246183, -5.6527185, 4.7993164
3: -4.2963915, 2.4680707, -2.2099075, 1.5563445, -5.8527360, 4.6779785
4: -4.1863484, 3.0611196, -2.3523908, 1.7424084, -5.9287567, 5.4135103
5: -3.6989820, 3.1250224, -2.0147188, 1.9236145, -5.6225967, 5.1397409
6: -3.4167094, 3.4270949, -1.9195052, 1.9778197, -5.3945284, 5.3466001
7: -3.5828748, 3.5508084, -1.9521804, 1.9311125, -5.5139875, 5.5029888
8: -5.5720172, 3.0211506, -3.1432834, 2.4766278, -8.0486450, 6.1644330
9: -3.2370172, 3.3693666, -1.7607933, 2.0055897, -5.2426062, 5.1301599

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1913183, upper bound: 7.1985269
time: 2.43 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1913183, upper bound: 7.1985269
time: 2.43 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.6988387, 2.3486423, -2.5959044, 2.2603514, -4.9591904, 4.9445467
1: -2.1092844, 2.0314989, -2.0361910, 1.9714665, -4.0807509, 4.0676899
2: -2.6665545, 2.1057489, -2.5422602, 2.0392170, -4.7057714, 4.6480093
3: -2.9113708, 1.8511525, -2.8094640, 1.8226153, -4.7339859, 4.6606164
4: -3.0033400, 2.1959286, -2.8924341, 2.1192265, -5.1225662, 5.0883627
5: -2.6151505, 2.3255954, -2.5030723, 2.2446613, -4.8598118, 4.8286667
6: -2.4252775, 2.4960108, -2.3347418, 2.3949869, -4.8202643, 4.8307524
7: -2.5157926, 2.4960999, -2.4183011, 2.3976240, -4.9134169, 4.9144011
8: -3.9896395, 2.6372719, -3.8396368, 2.6354439, -6.6250834, 6.4769087
9: -2.2704482, 2.4724805, -2.1813002, 2.3953800, -4.6658282, 4.6537809

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2143260, upper bound: 7.2083973
time: 2.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2143260, upper bound: 7.2083973
time: 3.26 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.7755179, 3.1316326, -2.5151522, 2.2042217, -5.9797397, 5.6467848
1: -2.9261055, 2.7992928, -1.9757640, 1.9174087, -4.8435144, 4.7750564
2: -3.9281001, 2.8059430, -2.4537902, 1.9837815, -5.9118814, 5.2597332
3: -4.2963915, 2.4680707, -2.7130229, 1.7771941, -6.0735855, 5.1810937
4: -4.1863484, 3.0611196, -2.8040860, 2.0558906, -6.2422390, 5.8652053
5: -3.6989820, 3.1250224, -2.4241910, 2.1889787, -5.8879604, 5.5492134
6: -3.4167094, 3.4270949, -2.2627277, 2.3275468, -5.7442555, 5.6898227
7: -3.5828748, 3.5508084, -2.3425074, 2.3219981, -5.9048724, 5.8933158
8: -5.5720172, 3.0211506, -3.7271709, 2.5960546, -8.1680717, 6.7483201
9: -3.2370172, 3.3693666, -2.1108565, 2.3302715, -5.5672884, 5.4802232

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2143260, upper bound: 7.2083973
time: 2.90 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2143260, upper bound: 7.2083973
time: 2.93 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.9024465, 1.7700586, -2.1238558, 1.9064958, -3.8089423, 3.8939145
1: -1.4949057, 1.4759188, -1.6720152, 1.6473919, -3.1422977, 3.1479340
2: -1.7560129, 1.5811390, -2.0068617, 1.7210194, -3.4770322, 3.5880008
3: -1.9603344, 1.3942609, -2.2268865, 1.5575963, -3.5179307, 3.6211474
4: -2.1022921, 1.5727082, -2.3679626, 1.7467335, -3.8490257, 3.9406710
5: -1.7814882, 1.7654946, -2.0122271, 1.9072813, -3.6887693, 3.7777216
6: -1.7168510, 1.7934461, -1.9230633, 1.9578702, -3.6747212, 3.7165093
7: -1.7426380, 1.7311926, -1.9670651, 1.9479605, -3.6905985, 3.6982577
8: -2.8160057, 2.3982489, -3.1529250, 2.4644735, -5.2804794, 5.5511742
9: -1.5838097, 1.8354518, -1.7674838, 2.0107994, -3.5946093, 3.6029358

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0834198, upper bound: 7.0116267
time: 1.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0515299, upper bound: 6.9569679
time: 2.24 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.2550602, 2.0122917, -2.1238558, 1.9064958, -4.1615562, 4.1361475
1: -1.7659352, 1.7238919, -1.6720152, 1.6473919, -3.4133272, 3.3959069
2: -2.1480794, 1.8068094, -2.0068617, 1.7210194, -3.8690987, 3.8136711
3: -2.3742266, 1.5893635, -2.2268865, 1.5575963, -3.9318228, 3.8162498
4: -2.5075209, 1.8427389, -2.3679626, 1.7467335, -4.2542543, 4.2107015
5: -2.1511729, 1.9933684, -2.0122271, 1.9072813, -4.0584540, 4.0055952
6: -2.0202258, 2.0983982, -1.9230633, 1.9578702, -3.9780960, 4.0214615
7: -2.0846517, 2.0690181, -1.9670651, 1.9479605, -4.0326123, 4.0360832
8: -3.3434200, 2.4811532, -3.1529250, 2.4644735, -5.8078938, 5.6340780
9: -1.8774936, 2.1054540, -1.7674838, 2.0107994, -3.8882930, 3.8729377

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0834198, upper bound: 7.0116267
time: 2.38 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0515299, upper bound: 6.9569679
time: 1.92 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.2550602, 2.0122917, -2.4925771, 2.1744037, -4.4294639, 4.5048685
1: -1.7659352, 1.7238919, -1.9630566, 1.9098904, -3.6758256, 3.6869483
2: -2.1480794, 1.8068094, -2.4470844, 1.9606330, -4.1087122, 4.2538939
3: -2.3742266, 1.5893635, -2.7036209, 1.7669770, -4.1412034, 4.2929845
4: -2.5075209, 1.8427389, -2.7936969, 2.0421112, -4.5496321, 4.6364355
5: -2.1511729, 1.9933684, -2.3996136, 2.1577797, -4.3089523, 4.3929820
6: -2.0202258, 2.0983982, -2.2467024, 2.2876971, -4.3079228, 4.3451004
7: -2.0846517, 2.0690181, -2.3370554, 2.3154080, -4.4000597, 4.4060736
8: -3.3434200, 2.4811532, -3.7079742, 2.5618923, -5.9053125, 6.1891274
9: -1.8774936, 2.1054540, -2.0939751, 2.3191071, -4.1966009, 4.1994290

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0170240, upper bound: 7.0061321
time: 3.59 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0872996, upper bound: 7.0147708
time: 2.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0761733, upper bound: 6.9779834
time: 2.70 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -4.9709196, 4.0656776, -1.8810791, 1.7597046, -6.7306242, 5.9467568
1: -4.1808815, 3.6518722, -1.4797509, 1.4667102, -5.6475916, 5.1316233
2: -5.3623700, 3.6359379, -1.7470591, 1.5666103, -6.9289804, 5.3829970
3: -5.7977481, 3.0625196, -1.9428984, 1.3827407, -7.1804876, 5.0054178
4: -5.7564340, 4.0193253, -2.0856109, 1.5552424, -7.3116765, 6.1049361
5: -4.8828650, 4.0289574, -1.7686630, 1.7659680, -6.6488328, 5.7976198
6: -4.5182519, 4.5535302, -1.7072289, 1.7779808, -6.2962322, 6.2607594
7: -4.8069839, 4.8141961, -1.7281321, 1.7206849, -6.5276690, 6.5423284
8: -7.3662410, 3.4899068, -2.7993045, 2.3687267, -9.7349682, 6.2892103
9: -4.3254170, 4.4528060, -1.5698304, 1.8285049, -6.1539221, 6.0226364

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1605714, upper bound: 7.1941883
time: 1.99 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1605714, upper bound: 7.2013829
time: 1.79 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -4.6386890, 3.8099437, -1.9266338, 1.7761003, -6.4147892, 5.7365775
1: -3.8857448, 3.4189830, -1.5207123, 1.5081422, -5.3938870, 4.9396954
2: -4.9894800, 3.4018755, -1.8072164, 1.5938438, -6.5833225, 5.2090921
3: -5.3959212, 2.8743234, -2.0085402, 1.4199548, -6.8158760, 4.8828635
4: -5.3722048, 3.7576547, -2.1517434, 1.5920531, -6.9642577, 5.9093981
5: -4.5616684, 3.7675390, -1.8170180, 1.7821736, -6.3438420, 5.5845571
6: -4.2131147, 4.2543712, -1.7494082, 1.7948322, -6.0079470, 6.0037794
7: -4.4883981, 4.4962759, -1.7862614, 1.7764553, -6.2648535, 6.2825375
8: -6.8930898, 3.2785091, -2.8711312, 2.3690071, -9.2620964, 6.1496401
9: -4.0289750, 4.1682491, -1.6070476, 1.8670114, -5.8959846, 5.7752967

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1557849, upper bound: 7.1557849
time: 1.52 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1557849, upper bound: 7.1559706
time: 1.60 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -5.0384731, 4.1178918, -2.2561102, 2.0173340, -7.0558071, 6.3740020
1: -4.2408786, 3.6994820, -1.7691944, 1.7258837, -5.9667625, 5.4686766
2: -5.4381156, 3.6829307, -2.1653864, 1.8028239, -7.2409396, 5.8483171
3: -5.8806171, 3.1019609, -2.3762965, 1.5938337, -7.4744492, 5.4782572
4: -5.8346400, 4.0723772, -2.5161018, 1.8419435, -7.6765833, 6.5884790
5: -4.9481287, 4.0818129, -2.1625948, 2.0075972, -6.9557257, 6.2444077
6: -4.5800767, 4.6138840, -2.0306623, 2.0999935, -6.6800699, 6.6445465
7: -4.8718576, 4.8789377, -2.0909822, 2.0780690, -6.9499259, 6.9699202
8: -7.4650517, 3.5346675, -3.3599031, 2.4584184, -9.9234695, 6.8945704
9: -4.3852596, 4.5117893, -1.8803803, 2.1151242, -6.5003824, 6.3921695

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1605714, upper bound: 7.1941883
time: 1.87 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1605714, upper bound: 7.2017796
time: 1.63 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -4.7068343, 3.8626425, -2.2856123, 2.0248687, -6.7317028, 6.1482549
1: -3.9463158, 3.4671285, -1.7987845, 1.7570596, -5.7033753, 5.2659130
2: -5.0660119, 3.4492805, -2.2160516, 1.8198063, -6.8858161, 5.6653318
3: -5.4796653, 2.9139326, -2.4241672, 1.6237713, -7.1034365, 5.3380995
4: -5.4511223, 3.8111792, -2.5639575, 1.8715580, -7.3226805, 6.3751364
5: -4.6275053, 3.8209214, -2.1941428, 2.0170102, -6.6445155, 6.0150633
6: -4.2755589, 4.3152246, -2.0615964, 2.1033173, -6.3788762, 6.3768210
7: -4.5538239, 4.5616999, -2.1377542, 2.1232777, -6.6771016, 6.6994543
8: -6.9928141, 3.3236303, -3.4114733, 2.4488919, -9.4417057, 6.7351031
9: -4.0894060, 4.2277308, -1.9120117, 2.1449604, -6.2343664, 6.1397424

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1557849, upper bound: 7.1578091
time: 1.45 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1557849, upper bound: 7.1604281
time: 1.43 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.7972994, 4.7027397, -2.4936459, 2.2026157, -7.9999151, 7.1963854
1: -4.9425526, 4.2433925, -1.9455515, 1.8812459, -6.8237963, 6.1889439
2: -6.3175087, 4.2106214, -2.4131410, 1.9792565, -8.2967634, 6.6237621
3: -6.8445168, 3.5439091, -2.6319950, 1.7331817, -8.5776987, 6.1759043
4: -6.7437863, 4.6822171, -2.7664330, 2.0275593, -8.7713451, 7.4486504
5: -5.7104568, 4.6763554, -2.4050875, 2.1891775, -7.8996344, 7.0814428
6: -5.2698193, 5.2900910, -2.2383149, 2.3169465, -7.5867658, 7.5284061
7: -5.6256223, 5.6311369, -2.3056664, 2.2789893, -7.9046116, 7.9368033
8: -8.5763617, 3.9589789, -3.6850562, 2.5756152, -11.1519766, 7.6440349
9: -5.0670795, 5.1780701, -2.0784688, 2.2955813, -7.3626590, 7.2565370

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1497750, upper bound: 7.1578298
time: 1.56 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1480647, upper bound: 7.1579087
time: 1.43 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.2587528, 5.0486021, -2.4936459, 2.2026157, -8.4613686, 7.5422478
1: -5.3602028, 4.5727620, -1.9455515, 1.8812459, -7.2414474, 6.5183134
2: -6.8498821, 4.5226798, -2.4131410, 1.9792565, -8.8291368, 6.9358206
3: -7.4174433, 3.8041203, -2.6319950, 1.7331817, -9.1506252, 6.4361143
4: -7.2967587, 5.0463190, -2.7664330, 2.0275593, -9.3243179, 7.8127518
5: -6.1593552, 5.0283585, -2.4050875, 2.1891775, -8.3485327, 7.4334450
6: -5.6903877, 5.6984243, -2.2383149, 2.3169465, -8.0073338, 7.9367390
7: -6.0808654, 6.0879784, -2.3056664, 2.2789893, -8.3598547, 8.3936443
8: -9.2549353, 4.2240658, -3.6850562, 2.5756152, -11.8305502, 7.9091210
9: -5.4808083, 5.5818958, -2.0784688, 2.2955813, -7.7763896, 7.6603627

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 117

### Candidate
type: A, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A2_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1497750, upper bound: 7.1625912
time: 1.67 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1480647, upper bound: 7.1630606
time: 1.83 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.7972994, 4.7027397, -2.8909206, 2.4934096, -8.2907085, 7.5936604
1: -4.9425526, 4.2433925, -2.2524562, 2.1633966, -7.1059494, 6.4958487
2: -6.3175087, 4.2106214, -2.8811345, 2.2397556, -8.5572634, 7.0917559
3: -6.8445168, 3.5439091, -3.1477535, 1.9586062, -8.8031235, 6.6916618
4: -6.7437863, 4.6822171, -3.2104633, 2.3498411, -9.0936270, 7.8926802
5: -5.7104568, 4.6763554, -2.8050935, 2.4679120, -8.1783686, 7.4814477
6: -5.2698193, 5.2900910, -2.6018286, 2.6678078, -7.9376268, 7.8919196
7: -5.6256223, 5.6311369, -2.6981626, 2.6756606, -8.3012829, 8.3292980
8: -8.5763617, 3.9589789, -4.2536578, 2.7228184, -11.2991800, 8.2126369
9: -5.0670795, 5.1780701, -2.4404695, 2.6264906, -7.6935701, 7.6185379

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1497750, upper bound: 7.1578298
time: 1.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1480647, upper bound: 7.1579125
time: 1.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.2587528, 5.0486021, -2.8909206, 2.4934096, -8.7521629, 7.9395218
1: -5.3602028, 4.5727620, -2.2524562, 2.1633966, -7.5235972, 6.8252177
2: -6.8498821, 4.5226798, -2.8811345, 2.2397556, -9.0896378, 7.4038143
3: -7.4174433, 3.8041203, -3.1477535, 1.9586062, -9.3760481, 6.9518728
4: -7.2967587, 5.0463190, -3.2104633, 2.3498411, -9.6465998, 8.2567825
5: -6.1593552, 5.0283585, -2.8050935, 2.4679120, -8.6272669, 7.8334489
6: -5.6903877, 5.6984243, -2.6018286, 2.6678078, -8.3581944, 8.3002529
7: -6.0808654, 6.0879784, -2.6981626, 2.6756606, -8.7565260, 8.7861404
8: -9.2549353, 4.2240658, -4.2536578, 2.7228184, -11.9777536, 8.4777241
9: -5.4808083, 5.5818958, -2.4404695, 2.6264906, -8.1072989, 8.0223637

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1497750, upper bound: 7.1656168
time: 1.87 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1480647, upper bound: 7.1668530
time: 1.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.7488866, 3.9029794, -3.4975319, 2.9521396, -7.7010260, 7.4005089
1: -3.9749842, 3.4879909, -2.7871504, 2.6010604, -6.5760446, 6.2751412
2: -5.0968695, 3.4925261, -3.5873106, 2.6496458, -7.7465153, 7.0798349
3: -5.5075827, 2.9379201, -3.9341323, 2.3170400, -7.8246226, 6.8720522
4: -5.4785924, 3.8428841, -3.9164402, 2.8370843, -8.3156767, 7.7593241
5: -4.6664882, 3.8627701, -3.4210641, 2.9406433, -7.6071315, 7.2838345
6: -4.3136492, 4.3562274, -3.1605904, 3.2020230, -7.5156698, 7.5168180
7: -4.5802288, 4.5832319, -3.2987692, 3.2788868, -7.8591146, 7.8820009
8: -7.0289254, 3.3792624, -5.1391382, 2.9198439, -9.9487696, 8.5184002
9: -4.1214943, 4.2522373, -2.9821150, 3.1516030, -7.2730961, 7.2343512

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 117

### Candidate
type: A, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1578298, upper bound: 7.1497750
time: 2.29 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1579087, upper bound: 7.1480647
time: 2.05 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.1732364, 4.2196846, -3.4975319, 2.9521396, -8.1253757, 7.7172165
1: -4.3572798, 3.7915714, -2.7871504, 2.6010604, -6.9583392, 6.5787206
2: -5.5846834, 3.7799127, -3.5873106, 2.6496458, -8.2343292, 7.3672233
3: -6.0367036, 3.1785607, -3.9341323, 2.3170400, -8.3537436, 7.1126928
4: -5.9859338, 4.1780605, -3.9164402, 2.8370843, -8.8230181, 8.0945005
5: -5.0762482, 4.1863823, -3.4210641, 2.9406433, -8.0168915, 7.6074467
6: -4.7020140, 4.7330303, -3.1605904, 3.2020230, -7.9040346, 7.8936205
7: -4.9979749, 5.0023522, -3.2987692, 3.2788868, -8.2768612, 8.3011217
8: -7.6512012, 3.6289577, -5.1391382, 2.9198439, -10.5710440, 8.7680950
9: -4.5031366, 4.6241751, -2.9821150, 3.1516030, -7.6547394, 7.6062889

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 117

### Candidate
type: A, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1578298, upper bound: 7.1498052
time: 2.21 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1579087, upper bound: 7.1480647
time: 1.84 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.7488866, 3.9029794, -3.9733930, 3.3118274, -8.0607128, 7.8763709
1: -3.9749842, 3.4879909, -3.2224376, 2.9409449, -6.9159288, 6.7104282
2: -5.0968695, 3.4925261, -4.1459713, 2.9734416, -8.0703106, 7.6384974
3: -5.5075827, 2.9379201, -4.5316215, 2.5823066, -8.0898876, 7.4695411
4: -5.4785924, 3.8428841, -4.4910970, 3.2168202, -8.6954117, 8.3339806
5: -4.6664882, 3.8627701, -3.8869901, 3.2989569, -7.9654431, 7.7497602
6: -4.3136492, 4.3562274, -3.5943100, 3.6291411, -7.9427872, 7.9505367
7: -4.5802288, 4.5832319, -3.7712269, 3.7551961, -8.3354235, 8.3544588
8: -7.0289254, 3.3792624, -5.8484354, 3.1430221, -10.1719475, 9.2276974
9: -4.1214943, 4.2522373, -3.4112234, 3.5720494, -7.6935430, 7.6634593

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1578298, upper bound: 7.1535936
time: 1.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1579087, upper bound: 7.1559606
time: 1.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.1732364, 4.2196846, -3.9733930, 3.3118274, -8.4850636, 8.1930771
1: -4.3572798, 3.7915714, -3.2224376, 2.9409449, -7.2982244, 7.0140076
2: -5.5846834, 3.7799127, -4.1459713, 2.9734416, -8.5581236, 7.9258842
3: -6.0367036, 3.1785607, -4.5316215, 2.5823066, -8.6190071, 7.7101808
4: -5.9859338, 4.1780605, -4.4910970, 3.2168202, -9.2027512, 8.6691570
5: -5.0762482, 4.1863823, -3.8869901, 3.2989569, -8.3752050, 8.0733719
6: -4.7020140, 4.7330303, -3.5943100, 3.6291411, -8.3311539, 8.3273401
7: -4.9979749, 5.0023522, -3.7712269, 3.7551961, -8.7531710, 8.7735786
8: -7.6512012, 3.6289577, -5.8484354, 3.1430221, -10.7942228, 9.4773922
9: -4.5031366, 4.6241751, -3.4112234, 3.5720494, -8.0751858, 8.0353985

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1578298, upper bound: 7.1555991
time: 1.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1579087, upper bound: 7.1573674
time: 1.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.7972994, 4.7027397, -3.4975319, 2.9521396, -8.7494373, 8.2002716
1: -4.9425526, 4.2433925, -2.7871504, 2.6010604, -7.5436130, 7.0305429
2: -6.3175087, 4.2106214, -3.5873106, 2.6496458, -8.9671545, 7.7979317
3: -6.8445168, 3.5439091, -3.9341323, 2.3170400, -9.1615562, 7.4780416
4: -6.7437863, 4.6822171, -3.9164402, 2.8370843, -9.5808706, 8.5986567
5: -5.7104568, 4.6763554, -3.4210641, 2.9406433, -8.6511002, 8.0974197
6: -5.2698193, 5.2900910, -3.1605904, 3.2020230, -8.4718409, 8.4506807
7: -5.6256223, 5.6311369, -3.2987692, 3.2788868, -8.9045076, 8.9299059
8: -8.5763617, 3.9589789, -5.1391382, 2.9198439, -11.4962044, 9.0981159
9: -5.0670795, 5.1780701, -2.9821150, 3.1516030, -8.2186813, 8.1601830

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 35

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 117

### Candidate
type: A, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A2_B1_A1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1477016, upper bound: 7.1460263
time: 1.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1464169, upper bound: 7.1464170
time: 1.45 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.2587528, 5.0486021, -3.4975319, 2.9521396, -9.2108908, 8.5461330
1: -5.3602028, 4.5727620, -2.7871504, 2.6010604, -7.9612632, 7.3599114
2: -6.8498821, 4.5226798, -3.5873106, 2.6496458, -9.4995279, 8.1099901
3: -7.4174433, 3.8041203, -3.9341323, 2.3170400, -9.7344828, 7.7382517
4: -7.2967587, 5.0463190, -3.9164402, 2.8370843, -10.1338425, 8.9627590
5: -6.1593552, 5.0283585, -3.4210641, 2.9406433, -9.0999985, 8.4494228
6: -5.6903877, 5.6984243, -3.1605904, 3.2020230, -8.8924074, 8.8590136
7: -6.0808654, 6.0879784, -3.2987692, 3.2788868, -9.3597517, 9.3867474
8: -9.2549353, 4.2240658, -5.1391382, 2.9198439, -12.1747789, 9.3632030
9: -5.4808083, 5.5818958, -2.9821150, 3.1516030, -8.6324110, 8.5640097

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 117

### Candidate
type: A, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A2_B1_A2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1477016, upper bound: 7.1461668
time: 1.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1464169, upper bound: 7.1467538
time: 1.47 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.7972994, 4.7027397, -3.9733930, 3.3118274, -9.1091261, 8.6761322
1: -4.9425526, 4.2433925, -3.2224376, 2.9409449, -7.8834972, 7.4658298
2: -6.3175087, 4.2106214, -4.1459713, 2.9734416, -9.2909479, 8.3565922
3: -6.8445168, 3.5439091, -4.5316215, 2.5823066, -9.4268236, 8.0755310
4: -6.7437863, 4.6822171, -4.4910970, 3.2168202, -9.9606037, 9.1733141
5: -5.7104568, 4.6763554, -3.8869901, 3.2989569, -9.0094137, 8.5633450
6: -5.2698193, 5.2900910, -3.5943100, 3.6291411, -8.8989601, 8.8844013
7: -5.6256223, 5.6311369, -3.7712269, 3.7551961, -9.3808165, 9.4023628
8: -8.5763617, 3.9589789, -5.8484354, 3.1430221, -11.7193832, 9.8074131
9: -5.0670795, 5.1780701, -3.4112234, 3.5720494, -8.6391277, 8.5892925

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1477016, upper bound: 7.1502754
time: 1.93 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1464169, upper bound: 7.1506348
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.2587528, 5.0486021, -3.9733930, 3.3118274, -9.5705795, 9.0219955
1: -5.3602028, 4.5727620, -3.2224376, 2.9409449, -8.3011475, 7.7951994
2: -6.8498821, 4.5226798, -4.1459713, 2.9734416, -9.8233213, 8.6686516
3: -7.4174433, 3.8041203, -4.5316215, 2.5823066, -9.9997482, 8.3357410
4: -7.2967587, 5.0463190, -4.4910970, 3.2168202, -10.5135784, 9.5374155
5: -6.1593552, 5.0283585, -3.8869901, 3.2989569, -9.4583111, 8.9153481
6: -5.6903877, 5.6984243, -3.5943100, 3.6291411, -9.3195267, 9.2927332
7: -6.0808654, 6.0879784, -3.7712269, 3.7551961, -9.8360605, 9.8592052
8: -9.2549353, 4.2240658, -5.8484354, 3.1430221, -12.3979568, 10.0725002
9: -5.4808083, 5.5818958, -3.4112234, 3.5720494, -9.0528574, 8.9931183

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 117

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1477016, upper bound: 7.1549210
time: 1.69 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1464169, upper bound: 7.1561105
time: 1.60 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 18.81 seconds
NS_A1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1958135, upper bound: 7.1985859
NS_A1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1958135, upper bound: 7.2028803
NS_A1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1958135, upper bound: 7.1985859
NS_A1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1958135, upper bound: 7.2028803
NS_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1886204, upper bound: 7.1916945
NS_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1886204, upper bound: 7.1963811
NS_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1886204, upper bound: 7.1916945
NS_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1886204, upper bound: 7.1963811
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1913183, upper bound: 7.1985269
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1913183, upper bound: 7.1985269
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1913183, upper bound: 7.1985269
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1913183, upper bound: 7.1985269
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.2143260, upper bound: 7.2083973
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.2143260, upper bound: 7.2083973
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.2143260, upper bound: 7.2083973
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.2143260, upper bound: 7.2083973
NS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.0834198, upper bound: 7.0116267
NS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.0515299, upper bound: 6.9569679
NS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.0834198, upper bound: 7.0116267
NS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.0515299, upper bound: 6.9569679
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.0872996, upper bound: 7.0147708
NS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.0761733, upper bound: 6.9779834
NS_A2_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1605714, upper bound: 7.1941883
NS_A2_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1605714, upper bound: 7.2013829
NS_A2_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1557849, upper bound: 7.1557849
NS_A2_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1557849, upper bound: 7.1559706
NS_A2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1605714, upper bound: 7.1941883
NS_A2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1605714, upper bound: 7.2017796
NS_A2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1557849, upper bound: 7.1578091
NS_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1557849, upper bound: 7.1604281
NS_A2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1497750, upper bound: 7.1578298
NS_A2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1480647, upper bound: 7.1579087
NS_A2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1497750, upper bound: 7.1625912
NS_A2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1480647, upper bound: 7.1630606
NS_A2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1497750, upper bound: 7.1578298
NS_A2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1480647, upper bound: 7.1579125
NS_A2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1497750, upper bound: 7.1656168
NS_A2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1480647, upper bound: 7.1668530
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1578298, upper bound: 7.1497750
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1579087, upper bound: 7.1480647
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1578298, upper bound: 7.1498052
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1579087, upper bound: 7.1480647
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1578298, upper bound: 7.1535936
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1579087, upper bound: 7.1559606
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1578298, upper bound: 7.1555991
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1579087, upper bound: 7.1573674
NS_A2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1477016, upper bound: 7.1460263
NS_A2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1464169, upper bound: 7.1464170
NS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1477016, upper bound: 7.1461668
NS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1464169, upper bound: 7.1467538
NS_A2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1477016, upper bound: 7.1502754
NS_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1464169, upper bound: 7.1506348
NS_A2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1477016, upper bound: 7.1549210
NS_A2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 18.81
Output dim: 8, lower bound: -7.1464169, upper bound: 7.1561105

## BFS NS instance: NS_A1_B1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -2.3055713, 2.0605240, -1.9362557, 1.7973945, -4.1029658, 3.9967797
1: -1.8019303, 1.7505115, -1.5222006, 1.5045407, -3.3064709, 3.2727122
2: -2.1977253, 1.8487592, -1.8055546, 1.6020460, -3.7997713, 3.6543138
3: -2.4104636, 1.6264927, -2.0040545, 1.4176931, -3.8281567, 3.6305473
4: -2.5583079, 1.8761915, -2.1476226, 1.5963902, -4.1546984, 4.0238142
5: -2.2124777, 2.0542164, -1.8257143, 1.8034595, -4.0159373, 3.8799307
6: -2.0717440, 2.1459587, -1.7562027, 1.8233088, -3.8950529, 3.9021614
7: -2.1233807, 2.1044996, -1.7789311, 1.7698075, -3.8931880, 3.8834307
8: -3.4188333, 2.4998162, -2.8819098, 2.3903224, -5.8091555, 5.3817263
9: -1.9116898, 2.1472139, -1.6123421, 1.8702300, -3.7819197, 3.7595561

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A1_B1_A1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0898932, upper bound: 7.1110439
time: 2.07 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2793159, upper bound: 7.2543882
time: 2.72 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -2.3055713, 2.0605240, -2.3139479, 2.0591354, -4.3647070, 4.3744717
1: -1.8019303, 1.7505115, -1.8151355, 1.7664415, -3.5683718, 3.5656471
2: -2.1977253, 1.8487592, -2.2330651, 1.8407602, -4.0384855, 4.0818243
3: -2.4104636, 1.6264927, -2.4408383, 1.6317768, -4.0422401, 4.0673313
4: -2.5583079, 1.8761915, -2.5816936, 1.8891046, -4.4474125, 4.4578853
5: -2.2124777, 2.0542164, -2.2231500, 2.0477293, -4.2602072, 4.2773666
6: -2.0717440, 2.1459587, -2.0821438, 2.1493886, -4.2211323, 4.2281027
7: -2.1233807, 2.1044996, -2.1482289, 2.1320219, -4.2554026, 4.2527285
8: -3.4188333, 2.4998162, -3.4460275, 2.4855568, -5.9043903, 5.9458437
9: -1.9116898, 2.1472139, -1.9307464, 2.1623175, -4.0740070, 4.0779605

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A1_B1_A1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1174497, upper bound: 7.0998571
time: 2.86 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1277809, upper bound: 7.0987763
time: 2.19 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 6.26 + 598.62 = 604.87 seconds
