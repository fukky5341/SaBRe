## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.009087408


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0032582, 0.0039995, -0.0032582, 0.0039995, -0.0072577, 0.0072577)
1: (0.9874097, 1.0027788, 0.9874097, 1.0027788, -0.0153691, 0.0153691)
2: (-0.0097837, 0.0025387, -0.0097837, 0.0025387, -0.0123224, 0.0123224)
3: (-0.0013881, 0.0076918, -0.0013881, 0.0076918, -0.0090799, 0.0090799)
4: (-0.0035895, 0.0133376, -0.0035895, 0.0133376, -0.0169270, 0.0169270)
5: (-0.0043585, 0.0128399, -0.0043585, 0.0128399, -0.0171984, 0.0171984)
6: (-0.0092749, 0.0078574, -0.0092749, 0.0078574, -0.0171322, 0.0171322)
7: (-0.0107944, -0.0034377, -0.0107944, -0.0034377, -0.0073566, 0.0073566)
8: (-0.0050840, 0.0086255, -0.0050840, 0.0086255, -0.0137094, 0.0137094)
9: (-0.0078495, 0.0026556, -0.0078495, 0.0026556, -0.0105051, 0.0105051)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.83 + 3.12 = 4.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
time: 2.03 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091791
time: 2.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.29 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.29
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.29
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091791

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0032731, 0.0036677, -0.0032545, 0.0039566, -0.0072297, 0.0069222
1: 0.9881123, 1.0028105, 0.9875004, 1.0027711, -0.0146588, 0.0153100
2: -0.0096431, 0.0013926, -0.0097369, 0.0023906, -0.0120337, 0.0111295
3: -0.0014068, 0.0072766, -0.0013836, 0.0076381, -0.0090449, 0.0086602
4: -0.0028356, 0.0121941, -0.0034724, 0.0131898, -0.0160253, 0.0156665
5: -0.0043939, 0.0120536, -0.0043499, 0.0127383, -0.0171321, 0.0164035
6: -0.0083762, 0.0078901, -0.0091587, 0.0078494, -0.0162256, 0.0170487
7: -0.0104580, -0.0034226, -0.0107509, -0.0034414, -0.0070166, 0.0073283
8: -0.0035781, 0.0086275, -0.0048893, 0.0086250, -0.0122031, 0.0135168
9: -0.0073692, 0.0026772, -0.0077874, 0.0026504, -0.0100196, 0.0104647

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
time: 1.95 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
time: 1.98 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0032433, 0.0038140, -0.0032582, 0.0039995, -0.0072428, 0.0070721
1: 0.9878025, 1.0027474, 0.9874097, 1.0027788, -0.0149763, 0.0153378
2: -0.0096282, 0.0018978, -0.0097837, 0.0025387, -0.0121669, 0.0116815
3: -0.0013695, 0.0074596, -0.0013881, 0.0076918, -0.0090612, 0.0088477
4: -0.0030829, 0.0126982, -0.0035895, 0.0133376, -0.0164205, 0.0162876
5: -0.0043232, 0.0124002, -0.0043585, 0.0128399, -0.0171631, 0.0167587
6: -0.0087723, 0.0078247, -0.0092749, 0.0078574, -0.0166297, 0.0170996
7: -0.0106063, -0.0034528, -0.0107944, -0.0034377, -0.0071686, 0.0073416
8: -0.0042419, 0.0086235, -0.0050840, 0.0086255, -0.0128674, 0.0137074
9: -0.0075810, 0.0026341, -0.0078495, 0.0026556, -0.0102366, 0.0104836

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
time: 2.22 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
time: 2.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 6.26 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.26
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.26
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.26
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.26
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032731, 0.0036677, -0.0032731, 0.0036677, -0.0069408, 0.0069408
1: 0.9881123, 1.0028105, 0.9881123, 1.0028105, -0.0146981, 0.0146981
2: -0.0096431, 0.0013926, -0.0096431, 0.0013926, -0.0110357, 0.0110357
3: -0.0014068, 0.0072766, -0.0014068, 0.0072766, -0.0086834, 0.0086834
4: -0.0028356, 0.0121941, -0.0028356, 0.0121941, -0.0150297, 0.0150297
5: -0.0043939, 0.0120536, -0.0043939, 0.0120536, -0.0164474, 0.0164474
6: -0.0083762, 0.0078901, -0.0083762, 0.0078901, -0.0162662, 0.0162662
7: -0.0104580, -0.0034226, -0.0104580, -0.0034226, -0.0070354, 0.0070354
8: -0.0035781, 0.0086275, -0.0035781, 0.0086275, -0.0122055, 0.0122055
9: -0.0073692, 0.0026772, -0.0073692, 0.0026772, -0.0100464, 0.0100464

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091740
time: 1.91 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.18 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032731, 0.0036677, -0.0032433, 0.0038140, -0.0070870, 0.0069110
1: 0.9881123, 1.0028105, 0.9878025, 1.0027474, -0.0146351, 0.0150080
2: -0.0096431, 0.0013926, -0.0096282, 0.0018978, -0.0115409, 0.0110208
3: -0.0014068, 0.0072766, -0.0013695, 0.0074596, -0.0088664, 0.0086461
4: -0.0028356, 0.0121941, -0.0030829, 0.0126982, -0.0155337, 0.0152770
5: -0.0043939, 0.0120536, -0.0043232, 0.0124002, -0.0167941, 0.0163768
6: -0.0083762, 0.0078901, -0.0087723, 0.0078247, -0.0162009, 0.0166624
7: -0.0104580, -0.0034226, -0.0106063, -0.0034528, -0.0070052, 0.0071837
8: -0.0035781, 0.0086275, -0.0042419, 0.0086235, -0.0122016, 0.0128694
9: -0.0073692, 0.0026772, -0.0075810, 0.0026341, -0.0100033, 0.0102582

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091741, upper bound: 0.0091360
time: 2.21 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.24 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0032433, 0.0038140, -0.0032731, 0.0036677, -0.0069110, 0.0070870
1: 0.9878025, 1.0027474, 0.9881123, 1.0028105, -0.0150080, 0.0146351
2: -0.0096282, 0.0018978, -0.0096431, 0.0013926, -0.0110208, 0.0115409
3: -0.0013695, 0.0074596, -0.0014068, 0.0072766, -0.0086461, 0.0088664
4: -0.0030829, 0.0126982, -0.0028356, 0.0121941, -0.0152770, 0.0155337
5: -0.0043232, 0.0124002, -0.0043939, 0.0120536, -0.0163768, 0.0167941
6: -0.0087723, 0.0078247, -0.0083762, 0.0078901, -0.0166624, 0.0162009
7: -0.0106063, -0.0034528, -0.0104580, -0.0034226, -0.0071837, 0.0070052
8: -0.0042419, 0.0086235, -0.0035781, 0.0086275, -0.0128694, 0.0122016
9: -0.0075810, 0.0026341, -0.0073692, 0.0026772, -0.0102582, 0.0100033

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091738
time: 2.64 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.41 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0032433, 0.0038140, -0.0032433, 0.0038140, -0.0070572, 0.0070572
1: 0.9878025, 1.0027474, 0.9878025, 1.0027474, -0.0149449, 0.0149449
2: -0.0096282, 0.0018978, -0.0096282, 0.0018978, -0.0115260, 0.0115260
3: -0.0013695, 0.0074596, -0.0013695, 0.0074596, -0.0088291, 0.0088291
4: -0.0030829, 0.0126982, -0.0030829, 0.0126982, -0.0157811, 0.0157811
5: -0.0043232, 0.0124002, -0.0043232, 0.0124002, -0.0167234, 0.0167234
6: -0.0087723, 0.0078247, -0.0087723, 0.0078247, -0.0165970, 0.0165970
7: -0.0106063, -0.0034528, -0.0106063, -0.0034528, -0.0071535, 0.0071535
8: -0.0042419, 0.0086235, -0.0042419, 0.0086235, -0.0128654, 0.0128654
9: -0.0075810, 0.0026341, -0.0075810, 0.0026341, -0.0102150, 0.0102150

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091787, upper bound: 0.0091360
time: 2.76 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.82 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091740
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 1, lower bound: -0.0091741, upper bound: 0.0091360
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091738
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 1, lower bound: -0.0091787, upper bound: 0.0091360
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0032581, 0.0035306, -0.0032731, 0.0036677, -0.0069258, 0.0068036
1: 0.9884027, 1.0027788, 0.9881123, 1.0028105, -0.0144078, 0.0146664
2: -0.0096356, 0.0009190, -0.0096431, 0.0013926, -0.0110282, 0.0105621
3: -0.0013880, 0.0071051, -0.0014068, 0.0072766, -0.0086646, 0.0085119
4: -0.0028111, 0.0117216, -0.0028356, 0.0121941, -0.0150052, 0.0145571
5: -0.0043583, 0.0117286, -0.0043939, 0.0120536, -0.0164119, 0.0161225
6: -0.0080048, 0.0078572, -0.0083762, 0.0078901, -0.0158948, 0.0162333
7: -0.0103190, -0.0034378, -0.0104580, -0.0034226, -0.0068964, 0.0070202
8: -0.0029558, 0.0086255, -0.0035781, 0.0086275, -0.0115833, 0.0122035
9: -0.0071708, 0.0026555, -0.0073692, 0.0026772, -0.0098480, 0.0100247

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.20 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.26 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0033526, 0.0034700, -0.0032702, 0.0036288, -0.0069814, 0.0067403
1: 0.9885309, 1.0029787, 0.9881946, 1.0028044, -0.0142735, 0.0147840
2: -0.0096828, 0.0007099, -0.0096417, 0.0012583, -0.0109411, 0.0103515
3: -0.0015062, 0.0070293, -0.0014032, 0.0072280, -0.0087342, 0.0084326
4: -0.0029652, 0.0115129, -0.0028309, 0.0120601, -0.0150253, 0.0143438
5: -0.0045822, 0.0115852, -0.0043871, 0.0119615, -0.0165437, 0.0159722
6: -0.0078408, 0.0080642, -0.0082709, 0.0078838, -0.0157246, 0.0163351
7: -0.0102577, -0.0033420, -0.0104186, -0.0034255, -0.0068322, 0.0070766
8: -0.0026810, 0.0086381, -0.0034017, 0.0086271, -0.0113081, 0.0120397
9: -0.0070831, 0.0027922, -0.0073130, 0.0026731, -0.0097562, 0.0101052

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
time: 2.53 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.90 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0032731, 0.0036677, -0.0032282, 0.0036785, -0.0069516, 0.0068958
1: 0.9881123, 1.0028105, 0.9880894, 1.0027153, -0.0146030, 0.0147210
2: -0.0096431, 0.0013926, -0.0096206, 0.0014299, -0.0110730, 0.0110132
3: -0.0014068, 0.0072766, -0.0013506, 0.0072901, -0.0086969, 0.0086272
4: -0.0028356, 0.0121941, -0.0027623, 0.0122313, -0.0150669, 0.0149564
5: -0.0043939, 0.0120536, -0.0042874, 0.0120792, -0.0164730, 0.0163410
6: -0.0083762, 0.0078901, -0.0084054, 0.0077916, -0.0161678, 0.0162955
7: -0.0104580, -0.0034226, -0.0104690, -0.0034682, -0.0069899, 0.0070464
8: -0.0035781, 0.0086275, -0.0036271, 0.0086215, -0.0121995, 0.0122546
9: -0.0073692, 0.0026772, -0.0073849, 0.0026122, -0.0099814, 0.0100621

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091257, upper bound: 0.0090987
time: 2.35 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091215, upper bound: 0.0090842
time: 4.32 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0032702, 0.0036288, -0.0033256, 0.0036174, -0.0068876, 0.0069544
1: 0.9881946, 1.0028044, 0.9882188, 1.0029217, -0.0147271, 0.0145856
2: -0.0096417, 0.0012583, -0.0096693, 0.0012189, -0.0108606, 0.0109276
3: -0.0014032, 0.0072280, -0.0014725, 0.0072137, -0.0086169, 0.0087004
4: -0.0028309, 0.0120601, -0.0029211, 0.0120208, -0.0148517, 0.0149813
5: -0.0043871, 0.0119615, -0.0045182, 0.0119344, -0.0163215, 0.0164797
6: -0.0082709, 0.0078838, -0.0082400, 0.0080051, -0.0162759, 0.0161238
7: -0.0104186, -0.0034255, -0.0104070, -0.0033694, -0.0070492, 0.0069816
8: -0.0034017, 0.0086271, -0.0033499, 0.0086345, -0.0120361, 0.0119769
9: -0.0073130, 0.0026731, -0.0072964, 0.0027532, -0.0100661, 0.0099695

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
time: 2.40 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.74 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0032282, 0.0036785, -0.0032731, 0.0036677, -0.0068958, 0.0069516
1: 0.9880894, 1.0027153, 0.9881123, 1.0028105, -0.0147210, 0.0146030
2: -0.0096206, 0.0014299, -0.0096431, 0.0013926, -0.0110132, 0.0110730
3: -0.0013506, 0.0072901, -0.0014068, 0.0072766, -0.0086272, 0.0086969
4: -0.0027623, 0.0122313, -0.0028356, 0.0121941, -0.0149564, 0.0150669
5: -0.0042874, 0.0120792, -0.0043939, 0.0120536, -0.0163410, 0.0164730
6: -0.0084054, 0.0077916, -0.0083762, 0.0078901, -0.0162955, 0.0161678
7: -0.0104690, -0.0034682, -0.0104580, -0.0034226, -0.0070464, 0.0069899
8: -0.0036271, 0.0086215, -0.0035781, 0.0086275, -0.0122546, 0.0121995
9: -0.0073849, 0.0026122, -0.0073692, 0.0026772, -0.0100621, 0.0099814

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091257
time: 2.50 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091210
time: 2.53 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0033256, 0.0036174, -0.0032702, 0.0036288, -0.0069544, 0.0068876
1: 0.9882188, 1.0029217, 0.9881946, 1.0028044, -0.0145856, 0.0147271
2: -0.0096693, 0.0012189, -0.0096417, 0.0012583, -0.0109276, 0.0108606
3: -0.0014725, 0.0072137, -0.0014032, 0.0072280, -0.0087004, 0.0086169
4: -0.0029211, 0.0120208, -0.0028309, 0.0120601, -0.0149813, 0.0148517
5: -0.0045182, 0.0119344, -0.0043871, 0.0119615, -0.0164797, 0.0163215
6: -0.0082400, 0.0080051, -0.0082709, 0.0078838, -0.0161238, 0.0162759
7: -0.0104070, -0.0033694, -0.0104186, -0.0034255, -0.0069816, 0.0070492
8: -0.0033499, 0.0086345, -0.0034017, 0.0086271, -0.0119769, 0.0120361
9: -0.0072964, 0.0027532, -0.0073130, 0.0026731, -0.0099695, 0.0100661

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090842
time: 2.44 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.40 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0032433, 0.0038140, -0.0032282, 0.0036785, -0.0069218, 0.0070421
1: 0.9878025, 1.0027474, 0.9880894, 1.0027153, -0.0149128, 0.0146580
2: -0.0096282, 0.0018978, -0.0096206, 0.0014299, -0.0110581, 0.0115185
3: -0.0013695, 0.0074596, -0.0013506, 0.0072901, -0.0086596, 0.0088102
4: -0.0030829, 0.0126982, -0.0027623, 0.0122313, -0.0153142, 0.0154605
5: -0.0043232, 0.0124002, -0.0042874, 0.0120792, -0.0164024, 0.0166876
6: -0.0087723, 0.0078247, -0.0084054, 0.0077916, -0.0165639, 0.0162301
7: -0.0106063, -0.0034528, -0.0104690, -0.0034682, -0.0071381, 0.0070162
8: -0.0042419, 0.0086235, -0.0036271, 0.0086215, -0.0128634, 0.0122506
9: -0.0075810, 0.0026341, -0.0073849, 0.0026122, -0.0101931, 0.0100189

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.68 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.69 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0032404, 0.0037745, -0.0033256, 0.0036174, -0.0068578, 0.0071001
1: 0.9878860, 1.0027412, 0.9882188, 1.0029217, -0.0150357, 0.0145224
2: -0.0096267, 0.0017616, -0.0096693, 0.0012189, -0.0108456, 0.0114310
3: -0.0013659, 0.0074103, -0.0014725, 0.0072137, -0.0085796, 0.0088827
4: -0.0029753, 0.0125623, -0.0029211, 0.0120208, -0.0149960, 0.0154834
5: -0.0043164, 0.0123068, -0.0045182, 0.0119344, -0.0162508, 0.0168250
6: -0.0086655, 0.0078184, -0.0082400, 0.0080051, -0.0166706, 0.0160583
7: -0.0105663, -0.0034557, -0.0104070, -0.0033694, -0.0071969, 0.0069513
8: -0.0040629, 0.0086231, -0.0033499, 0.0086345, -0.0126974, 0.0119729
9: -0.0075239, 0.0026299, -0.0072964, 0.0027532, -0.0102770, 0.0099263

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090842
time: 2.52 seconds

## Relational analysis of NS_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.81 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
NS_A1_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0091257, upper bound: 0.0090987
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0091215, upper bound: 0.0090842
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
NS_A1_B2_B2_A2, status: Status.VERIFIED, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091257
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091210
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090842
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090842
NS_A2_B2_B2_B2, status: Status.VERIFIED, split count: 4, time: 6.81
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032581, 0.0035306, -0.0032581, 0.0035306, -0.0067886, 0.0067886
1: 0.9884027, 1.0027788, 0.9884027, 1.0027788, -0.0143761, 0.0143761
2: -0.0096356, 0.0009190, -0.0096356, 0.0009190, -0.0105545, 0.0105545
3: -0.0013880, 0.0071051, -0.0013880, 0.0071051, -0.0084931, 0.0084931
4: -0.0028111, 0.0117216, -0.0028111, 0.0117216, -0.0145327, 0.0145327
5: -0.0043583, 0.0117286, -0.0043583, 0.0117286, -0.0160869, 0.0160869
6: -0.0080048, 0.0078572, -0.0080048, 0.0078572, -0.0158620, 0.0158620
7: -0.0103190, -0.0034378, -0.0103190, -0.0034378, -0.0068812, 0.0068812
8: -0.0029558, 0.0086255, -0.0029558, 0.0086255, -0.0115812, 0.0115812
9: -0.0071708, 0.0026555, -0.0071708, 0.0026555, -0.0098263, 0.0098263

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091601
time: 2.25 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091605
time: 2.26 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032581, 0.0035306, -0.0033526, 0.0034700, -0.0067281, 0.0068831
1: 0.9884027, 1.0027788, 0.9885309, 1.0029787, -0.0145760, 0.0142479
2: -0.0096356, 0.0009190, -0.0096828, 0.0007099, -0.0103454, 0.0106018
3: -0.0013880, 0.0071051, -0.0015062, 0.0070293, -0.0084173, 0.0086113
4: -0.0028111, 0.0117216, -0.0029652, 0.0115129, -0.0143240, 0.0146867
5: -0.0043583, 0.0117286, -0.0045822, 0.0115852, -0.0159434, 0.0163108
6: -0.0080048, 0.0078572, -0.0078408, 0.0080642, -0.0160690, 0.0156980
7: -0.0103190, -0.0034378, -0.0102577, -0.0033420, -0.0069770, 0.0068198
8: -0.0029558, 0.0086255, -0.0026810, 0.0086381, -0.0115939, 0.0113065
9: -0.0071708, 0.0026555, -0.0070831, 0.0027922, -0.0099630, 0.0097386

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091257
time: 2.47 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091215
time: 2.62 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0032692, 0.0034214, -0.0032702, 0.0036288, -0.0068980, 0.0066916
1: 0.9886339, 1.0028023, 0.9881946, 1.0028044, -0.0141705, 0.0146076
2: -0.0096411, 0.0005418, -0.0096417, 0.0012583, -0.0108994, 0.0101835
3: -0.0014019, 0.0069685, -0.0014032, 0.0072280, -0.0086299, 0.0083717
4: -0.0028292, 0.0113452, -0.0028309, 0.0120601, -0.0148893, 0.0141762
5: -0.0043846, 0.0114699, -0.0043871, 0.0119615, -0.0163461, 0.0158570
6: -0.0077090, 0.0078815, -0.0082709, 0.0078838, -0.0155928, 0.0161524
7: -0.0102083, -0.0034265, -0.0104186, -0.0034255, -0.0067829, 0.0069921
8: -0.0024602, 0.0086269, -0.0034017, 0.0086271, -0.0110873, 0.0120286
9: -0.0070127, 0.0026716, -0.0073130, 0.0026731, -0.0096858, 0.0099845

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090953
time: 2.52 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090958
time: 2.68 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0031897, 0.0036169, -0.0032282, 0.0036785, -0.0068681, 0.0068450
1: 0.9882199, 1.0026338, 0.9880894, 1.0027153, -0.0144955, 0.0145444
2: -0.0096014, 0.0012171, -0.0096206, 0.0014299, -0.0110313, 0.0108378
3: -0.0013024, 0.0072131, -0.0013506, 0.0072901, -0.0085926, 0.0085637
4: -0.0026995, 0.0120190, -0.0027623, 0.0122313, -0.0149308, 0.0147814
5: -0.0041962, 0.0119332, -0.0042874, 0.0120792, -0.0162753, 0.0162206
6: -0.0082386, 0.0077072, -0.0084054, 0.0077916, -0.0160302, 0.0161127
7: -0.0104065, -0.0035072, -0.0104690, -0.0034682, -0.0069384, 0.0069618
8: -0.0033476, 0.0086163, -0.0036271, 0.0086215, -0.0119690, 0.0122434
9: -0.0072957, 0.0025565, -0.0073849, 0.0026122, -0.0099079, 0.0099413

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091108, upper bound: 0.0090987
time: 2.26 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091108, upper bound: 0.0090987
time: 2.50 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0032041, 0.0037961, -0.0032169, 0.0036720, -0.0068761, 0.0070130
1: 0.9878403, 1.0026646, 0.9881031, 1.0026915, -0.0148512, 0.0145615
2: -0.0096086, 0.0018363, -0.0096150, 0.0014075, -0.0110161, 0.0114512
3: -0.0013205, 0.0074373, -0.0013365, 0.0072820, -0.0086025, 0.0087738
4: -0.0030343, 0.0126367, -0.0027439, 0.0122090, -0.0152432, 0.0153807
5: -0.0042304, 0.0123580, -0.0042607, 0.0120638, -0.0162942, 0.0166187
6: -0.0087241, 0.0077389, -0.0083879, 0.0077669, -0.0164910, 0.0161268
7: -0.0105882, -0.0034925, -0.0104624, -0.0034796, -0.0071087, 0.0069699
8: -0.0041610, 0.0086183, -0.0035977, 0.0086200, -0.0127810, 0.0122159
9: -0.0075552, 0.0025774, -0.0073755, 0.0025959, -0.0101510, 0.0099529

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091053, upper bound: 0.0090840
time: 2.63 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091053, upper bound: 0.0090840
time: 2.72 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0031868, 0.0035780, -0.0033256, 0.0036174, -0.0068042, 0.0069036
1: 0.9883022, 1.0026277, 0.9882188, 1.0029217, -0.0146195, 0.0144089
2: -0.0095999, 0.0010828, -0.0096693, 0.0012189, -0.0108188, 0.0107521
3: -0.0012988, 0.0071644, -0.0014725, 0.0072137, -0.0085125, 0.0086369
4: -0.0026949, 0.0118850, -0.0029211, 0.0120208, -0.0147156, 0.0148061
5: -0.0041893, 0.0118410, -0.0045182, 0.0119344, -0.0161238, 0.0163593
6: -0.0081332, 0.0077010, -0.0082400, 0.0080051, -0.0161383, 0.0159409
7: -0.0103671, -0.0035101, -0.0104070, -0.0033694, -0.0069977, 0.0068970
8: -0.0031710, 0.0086159, -0.0033499, 0.0086345, -0.0118055, 0.0119658
9: -0.0072394, 0.0025523, -0.0072964, 0.0027532, -0.0099926, 0.0098487

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090981
time: 2.57 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090839, upper bound: 0.0090981
time: 2.89 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032282, 0.0036785, -0.0031897, 0.0036169, -0.0068450, 0.0068681
1: 0.9880894, 1.0027153, 0.9882199, 1.0026338, -0.0145444, 0.0144955
2: -0.0096206, 0.0014299, -0.0096014, 0.0012171, -0.0108378, 0.0110313
3: -0.0013506, 0.0072901, -0.0013024, 0.0072131, -0.0085637, 0.0085926
4: -0.0027623, 0.0122313, -0.0026995, 0.0120190, -0.0147814, 0.0149308
5: -0.0042874, 0.0120792, -0.0041962, 0.0119332, -0.0162206, 0.0162753
6: -0.0084054, 0.0077916, -0.0082386, 0.0077072, -0.0161127, 0.0160302
7: -0.0104690, -0.0034682, -0.0104065, -0.0035072, -0.0069618, 0.0069384
8: -0.0036271, 0.0086215, -0.0033476, 0.0086163, -0.0122434, 0.0119690
9: -0.0073849, 0.0026122, -0.0072957, 0.0025565, -0.0099413, 0.0099079

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091107
time: 2.09 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091104
time: 2.59 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032169, 0.0036720, -0.0032041, 0.0037961, -0.0070130, 0.0068761
1: 0.9881031, 1.0026915, 0.9878403, 1.0026646, -0.0145615, 0.0148512
2: -0.0096150, 0.0014075, -0.0096086, 0.0018363, -0.0114512, 0.0110161
3: -0.0013365, 0.0072820, -0.0013205, 0.0074373, -0.0087738, 0.0086025
4: -0.0027439, 0.0122090, -0.0030343, 0.0126367, -0.0153807, 0.0152432
5: -0.0042607, 0.0120638, -0.0042304, 0.0123580, -0.0166187, 0.0162942
6: -0.0083879, 0.0077669, -0.0087241, 0.0077389, -0.0161268, 0.0164910
7: -0.0104624, -0.0034796, -0.0105882, -0.0034925, -0.0069699, 0.0071087
8: -0.0035977, 0.0086200, -0.0041610, 0.0086183, -0.0122159, 0.0127810
9: -0.0073755, 0.0025959, -0.0075552, 0.0025774, -0.0099529, 0.0101510

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091053
time: 2.43 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091053
time: 2.36 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0033256, 0.0036174, -0.0031868, 0.0035780, -0.0069036, 0.0068042
1: 0.9882188, 1.0029217, 0.9883022, 1.0026277, -0.0144089, 0.0146195
2: -0.0096693, 0.0012189, -0.0095999, 0.0010828, -0.0107521, 0.0108188
3: -0.0014725, 0.0072137, -0.0012988, 0.0071644, -0.0086369, 0.0085125
4: -0.0029211, 0.0120208, -0.0026949, 0.0118850, -0.0148061, 0.0147156
5: -0.0045182, 0.0119344, -0.0041893, 0.0118410, -0.0163593, 0.0161238
6: -0.0082400, 0.0080051, -0.0081332, 0.0077010, -0.0159409, 0.0161383
7: -0.0104070, -0.0033694, -0.0103671, -0.0035101, -0.0068970, 0.0069977
8: -0.0033499, 0.0086345, -0.0031710, 0.0086159, -0.0119658, 0.0118055
9: -0.0072964, 0.0027532, -0.0072394, 0.0025523, -0.0098487, 0.0099926

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090982, upper bound: 0.0090842
time: 3.22 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090983, upper bound: 0.0090839
time: 2.42 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0032282, 0.0036785, -0.0032282, 0.0036785, -0.0069066, 0.0069066
1: 0.9880894, 1.0027153, 0.9880894, 1.0027153, -0.0146259, 0.0146259
2: -0.0096206, 0.0014299, -0.0096206, 0.0014299, -0.0110505, 0.0110505
3: -0.0013506, 0.0072901, -0.0013506, 0.0072901, -0.0086407, 0.0086407
4: -0.0027623, 0.0122313, -0.0027623, 0.0122313, -0.0149936, 0.0149936
5: -0.0042874, 0.0120792, -0.0042874, 0.0120792, -0.0163666, 0.0163666
6: -0.0084054, 0.0077916, -0.0084054, 0.0077916, -0.0161970, 0.0161970
7: -0.0104690, -0.0034682, -0.0104690, -0.0034682, -0.0070008, 0.0070008
8: -0.0036271, 0.0086215, -0.0036271, 0.0086215, -0.0122485, 0.0122485
9: -0.0073849, 0.0026122, -0.0073849, 0.0026122, -0.0099970, 0.0099970

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091303, upper bound: 0.0090987
time: 2.98 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091284, upper bound: 0.0090842
time: 2.36 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0033256, 0.0036174, -0.0032282, 0.0036785, -0.0070041, 0.0068456
1: 0.9882188, 1.0029217, 0.9880894, 1.0027153, -0.0144966, 0.0148323
2: -0.0096693, 0.0012189, -0.0096206, 0.0014299, -0.0110993, 0.0108395
3: -0.0014725, 0.0072137, -0.0013506, 0.0072901, -0.0087626, 0.0085643
4: -0.0029211, 0.0120208, -0.0027623, 0.0122313, -0.0151525, 0.0147831
5: -0.0045182, 0.0119344, -0.0042874, 0.0120792, -0.0165974, 0.0162218
6: -0.0082400, 0.0080051, -0.0084054, 0.0077916, -0.0160316, 0.0164105
7: -0.0104070, -0.0033694, -0.0104690, -0.0034682, -0.0069389, 0.0070996
8: -0.0033499, 0.0086345, -0.0036271, 0.0086215, -0.0119713, 0.0122616
9: -0.0072964, 0.0027532, -0.0073849, 0.0026122, -0.0099086, 0.0101380

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091303, upper bound: 0.0090987
time: 2.93 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091284, upper bound: 0.0090842
time: 2.58 seconds

## BFS NS instance: NS_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0032404, 0.0037745, -0.0032423, 0.0035633, -0.0068037, 0.0070168
1: 0.9878860, 1.0027412, 0.9883333, 1.0027454, -0.0148594, 0.0144079
2: -0.0096267, 0.0017616, -0.0096277, 0.0010321, -0.0106589, 0.0113893
3: -0.0013659, 0.0074103, -0.0013683, 0.0071461, -0.0085120, 0.0087786
4: -0.0029753, 0.0125623, -0.0027854, 0.0118345, -0.0148098, 0.0153477
5: -0.0043164, 0.0123068, -0.0043210, 0.0118063, -0.0161227, 0.0166277
6: -0.0086655, 0.0078184, -0.0080935, 0.0078227, -0.0164882, 0.0159119
7: -0.0105663, -0.0034557, -0.0103522, -0.0034538, -0.0071125, 0.0068965
8: -0.0040629, 0.0086231, -0.0031045, 0.0086234, -0.0126863, 0.0117276
9: -0.0075239, 0.0026299, -0.0072182, 0.0026327, -0.0101566, 0.0098481

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090955, upper bound: 0.0090842
time: 2.84 seconds

## Relational analysis of NS_A2_B2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090842
time: 3.06 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.81 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091601
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091605
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091257
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091215
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090953
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090958
NS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0091108, upper bound: 0.0090987
NS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0091108, upper bound: 0.0090987
NS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0091053, upper bound: 0.0090840
NS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0091053, upper bound: 0.0090840
NS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090981
NS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0090839, upper bound: 0.0090981
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091107
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091104
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091053
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091053
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0090982, upper bound: 0.0090842
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0090983, upper bound: 0.0090839
NS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0091303, upper bound: 0.0090987
NS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0091284, upper bound: 0.0090842
NS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0091303, upper bound: 0.0090987
NS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0091284, upper bound: 0.0090842
NS_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0090955, upper bound: 0.0090842
NS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090842

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0032310, 0.0033685, -0.0032579, 0.0035294, -0.0067604, 0.0066264
1: 0.9887458, 1.0027213, 0.9884051, 1.0027783, -0.0140325, 0.0143162
2: -0.0096220, 0.0003593, -0.0096355, 0.0009151, -0.0105371, 0.0099948
3: -0.0013541, 0.0069024, -0.0013878, 0.0071037, -0.0084578, 0.0082901
4: -0.0027669, 0.0111632, -0.0028108, 0.0117177, -0.0144846, 0.0139739
5: -0.0042941, 0.0113447, -0.0043578, 0.0117260, -0.0160200, 0.0157025
6: -0.0075659, 0.0077978, -0.0080017, 0.0078568, -0.0154227, 0.0157995
7: -0.0101548, -0.0034653, -0.0103179, -0.0034380, -0.0067168, 0.0068526
8: -0.0022204, 0.0086218, -0.0029507, 0.0086254, -0.0108458, 0.0115725
9: -0.0069362, 0.0026162, -0.0071691, 0.0026552, -0.0095914, 0.0097854

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091413
time: 2.43 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091316
time: 2.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0032682, 0.0033685, -0.0032559, 0.0035120, -0.0067802, 0.0066244
1: 0.9887459, 1.0028001, 0.9884421, 1.0027740, -0.0140281, 0.0143580
2: -0.0096406, 0.0003593, -0.0096345, 0.0008547, -0.0104954, 0.0099937
3: -0.0014007, 0.0069024, -0.0013853, 0.0070818, -0.0084825, 0.0082876
4: -0.0028276, 0.0111631, -0.0028075, 0.0116575, -0.0144851, 0.0139706
5: -0.0043823, 0.0113447, -0.0043531, 0.0116846, -0.0160669, 0.0156978
6: -0.0075659, 0.0078794, -0.0079544, 0.0078524, -0.0154183, 0.0158338
7: -0.0101548, -0.0034276, -0.0103002, -0.0034400, -0.0067148, 0.0068726
8: -0.0022204, 0.0086268, -0.0028714, 0.0086252, -0.0108456, 0.0114982
9: -0.0069362, 0.0026701, -0.0071438, 0.0026523, -0.0095885, 0.0098140

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091399
time: 2.19 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091307
time: 2.75 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0032581, 0.0035306, -0.0032692, 0.0034214, -0.0066794, 0.0067997
1: 0.9884027, 1.0027788, 0.9886339, 1.0028023, -0.0143996, 0.0141449
2: -0.0096356, 0.0009190, -0.0096411, 0.0005418, -0.0101774, 0.0105601
3: -0.0013880, 0.0071051, -0.0014019, 0.0069685, -0.0083565, 0.0085070
4: -0.0028111, 0.0117216, -0.0028292, 0.0113452, -0.0141563, 0.0145508
5: -0.0043583, 0.0117286, -0.0043846, 0.0114699, -0.0158282, 0.0161133
6: -0.0080048, 0.0078572, -0.0077090, 0.0078815, -0.0158863, 0.0155662
7: -0.0103190, -0.0034378, -0.0102083, -0.0034265, -0.0068925, 0.0067705
8: -0.0029558, 0.0086255, -0.0024602, 0.0086269, -0.0115827, 0.0110857
9: -0.0071708, 0.0026555, -0.0070127, 0.0026716, -0.0098423, 0.0096682

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091103
time: 2.41 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091108
time: 2.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0032470, 0.0035249, -0.0032828, 0.0036063, -0.0068534, 0.0068077
1: 0.9884147, 1.0027554, 0.9882423, 1.0028310, -0.0144163, 0.0145131
2: -0.0096301, 0.0008993, -0.0096480, 0.0011807, -0.0108107, 0.0105473
3: -0.0013742, 0.0070980, -0.0014189, 0.0071999, -0.0085740, 0.0085169
4: -0.0027931, 0.0117020, -0.0028514, 0.0119826, -0.0147757, 0.0145534
5: -0.0043321, 0.0117152, -0.0044168, 0.0119082, -0.0162403, 0.0161320
6: -0.0079894, 0.0078330, -0.0082100, 0.0079113, -0.0159007, 0.0160430
7: -0.0103133, -0.0034490, -0.0103958, -0.0034128, -0.0069005, 0.0069468
8: -0.0029300, 0.0086240, -0.0032996, 0.0086288, -0.0115588, 0.0119236
9: -0.0071625, 0.0026395, -0.0072804, 0.0026913, -0.0098538, 0.0099199

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A1_B1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091047
time: 2.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091053
time: 2.25 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032679, 0.0034111, -0.0032303, 0.0034675, -0.0067355, 0.0066414
1: 0.9886556, 1.0027995, 0.9885361, 1.0027198, -0.0140642, 0.0142634
2: -0.0096405, 0.0005064, -0.0096217, 0.0007013, -0.0103418, 0.0101280
3: -0.0014004, 0.0069556, -0.0013532, 0.0070262, -0.0084266, 0.0083089
4: -0.0028271, 0.0113099, -0.0027658, 0.0115044, -0.0143315, 0.0140757
5: -0.0043816, 0.0114456, -0.0042925, 0.0115793, -0.0159609, 0.0157381
6: -0.0076813, 0.0078788, -0.0078341, 0.0077963, -0.0154776, 0.0157129
7: -0.0101979, -0.0034278, -0.0102552, -0.0034660, -0.0067320, 0.0068273
8: -0.0024137, 0.0086268, -0.0026698, 0.0086217, -0.0110354, 0.0112966
9: -0.0069979, 0.0026697, -0.0070795, 0.0026153, -0.0096131, 0.0097493

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A1_B1_A2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090817, upper bound: 0.0090817
time: 2.76 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090817, upper bound: 0.0090824
time: 2.59 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032684, 0.0034122, -0.0032559, 0.0034823, -0.0067507, 0.0066682
1: 0.9886532, 1.0028005, 0.9885049, 1.0027741, -0.0141209, 0.0142956
2: -0.0096407, 0.0005102, -0.0096345, 0.0007523, -0.0103930, 0.0101447
3: -0.0014009, 0.0069570, -0.0013854, 0.0070447, -0.0084456, 0.0083424
4: -0.0028279, 0.0113137, -0.0028076, 0.0115552, -0.0143831, 0.0141213
5: -0.0043826, 0.0114482, -0.0043532, 0.0116143, -0.0159969, 0.0158014
6: -0.0076843, 0.0078797, -0.0078741, 0.0078525, -0.0155367, 0.0157538
7: -0.0101991, -0.0034274, -0.0102701, -0.0034400, -0.0067591, 0.0068427
8: -0.0024187, 0.0086268, -0.0027368, 0.0086252, -0.0110439, 0.0113636
9: -0.0069995, 0.0026704, -0.0071009, 0.0026524, -0.0096519, 0.0097713

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A1_B1_A2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090820, upper bound: 0.0090824
time: 2.50 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090817, upper bound: 0.0090828
time: 2.55 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0031895, 0.0036158, -0.0032013, 0.0035188, -0.0067083, 0.0068170
1: 0.9882222, 1.0026335, 0.9884275, 1.0026584, -0.0144362, 0.0142059
2: -0.0096013, 0.0012133, -0.0096072, 0.0008785, -0.0104798, 0.0108204
3: -0.0013022, 0.0072117, -0.0013169, 0.0070904, -0.0083926, 0.0085286
4: -0.0026992, 0.0120152, -0.0027185, 0.0116812, -0.0143804, 0.0147336
5: -0.0041957, 0.0119306, -0.0042237, 0.0117009, -0.0158966, 0.0161542
6: -0.0082355, 0.0077068, -0.0079730, 0.0077327, -0.0159682, 0.0156799
7: -0.0104054, -0.0035074, -0.0103071, -0.0034954, -0.0069100, 0.0067998
8: -0.0033425, 0.0086163, -0.0029026, 0.0086179, -0.0119603, 0.0115189
9: -0.0072941, 0.0025562, -0.0071538, 0.0025732, -0.0098673, 0.0097100

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090986
time: 2.50 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090987
time: 2.99 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0031874, 0.0035983, -0.0032359, 0.0035217, -0.0067091, 0.0068342
1: 0.9882592, 1.0026292, 0.9884215, 1.0027317, -0.0144725, 0.0142077
2: -0.0096002, 0.0011530, -0.0096245, 0.0008883, -0.0104886, 0.0107774
3: -0.0012996, 0.0071898, -0.0013603, 0.0070940, -0.0083936, 0.0085501
4: -0.0026959, 0.0119550, -0.0027749, 0.0116910, -0.0143869, 0.0147299
5: -0.0041909, 0.0118892, -0.0043057, 0.0117076, -0.0158986, 0.0161949
6: -0.0081883, 0.0077024, -0.0079808, 0.0078086, -0.0159968, 0.0156832
7: -0.0103877, -0.0035094, -0.0103100, -0.0034603, -0.0069274, 0.0068006
8: -0.0032632, 0.0086160, -0.0029155, 0.0086225, -0.0118857, 0.0115316
9: -0.0072688, 0.0025532, -0.0071579, 0.0026234, -0.0098922, 0.0097112

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_B1_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090986
time: 4.72 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090987
time: 2.25 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0032039, 0.0037950, -0.0031899, 0.0035123, -0.0067162, 0.0069849
1: 0.9878427, 1.0026641, 0.9884413, 1.0026343, -0.0147916, 0.0142227
2: -0.0096085, 0.0018323, -0.0096015, 0.0008558, -0.0104643, 0.0114338
3: -0.0013203, 0.0074359, -0.0013027, 0.0070822, -0.0084025, 0.0087386
4: -0.0030311, 0.0126328, -0.0026999, 0.0116586, -0.0146897, 0.0153328
5: -0.0042300, 0.0123553, -0.0041967, 0.0116853, -0.0159153, 0.0165520
6: -0.0087210, 0.0077385, -0.0079553, 0.0077078, -0.0164287, 0.0156938
7: -0.0105871, -0.0034927, -0.0103005, -0.0035069, -0.0070801, 0.0068078
8: -0.0041558, 0.0086182, -0.0028729, 0.0086164, -0.0127722, 0.0114911
9: -0.0075535, 0.0025771, -0.0071443, 0.0025568, -0.0101103, 0.0097214

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091053, upper bound: 0.0090840
time: 3.60 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091053, upper bound: 0.0090840
time: 2.54 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0032019, 0.0037779, -0.0032244, 0.0035151, -0.0067170, 0.0070024
1: 0.9878789, 1.0026598, 0.9884354, 1.0027075, -0.0148286, 0.0142244
2: -0.0096075, 0.0017734, -0.0096188, 0.0008656, -0.0104731, 0.0113921
3: -0.0013177, 0.0074146, -0.0013459, 0.0070857, -0.0084035, 0.0087605
4: -0.0029846, 0.0125740, -0.0027563, 0.0116683, -0.0146529, 0.0153303
5: -0.0042252, 0.0123148, -0.0042786, 0.0116920, -0.0159172, 0.0165935
6: -0.0086748, 0.0077341, -0.0079629, 0.0077835, -0.0164583, 0.0156970
7: -0.0105698, -0.0034947, -0.0103034, -0.0034719, -0.0070979, 0.0068086
8: -0.0040784, 0.0086180, -0.0028856, 0.0086210, -0.0126994, 0.0115036
9: -0.0075288, 0.0025742, -0.0071484, 0.0026068, -0.0101356, 0.0097226

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091053, upper bound: 0.0090840
time: 3.71 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091053, upper bound: 0.0090840
time: 2.65 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0031866, 0.0035769, -0.0032982, 0.0034578, -0.0066444, 0.0068751
1: 0.9883046, 1.0026274, 0.9885568, 1.0028638, -0.0145592, 0.0140706
2: -0.0095998, 0.0010789, -0.0096557, 0.0006676, -0.0102674, 0.0107346
3: -0.0012986, 0.0071630, -0.0014382, 0.0070140, -0.0083126, 0.0086012
4: -0.0026945, 0.0118811, -0.0028766, 0.0114708, -0.0141653, 0.0147577
5: -0.0041889, 0.0118384, -0.0044534, 0.0115562, -0.0157451, 0.0162918
6: -0.0081302, 0.0077005, -0.0078077, 0.0079452, -0.0160753, 0.0155082
7: -0.0103660, -0.0035103, -0.0102453, -0.0033971, -0.0069689, 0.0067350
8: -0.0031659, 0.0086159, -0.0026255, 0.0086308, -0.0117967, 0.0112415
9: -0.0072378, 0.0025520, -0.0070654, 0.0027136, -0.0099514, 0.0096174

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090817, upper bound: 0.0090914
time: 2.69 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090817, upper bound: 0.0090914
time: 3.00 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0031846, 0.0035594, -0.0033381, 0.0034684, -0.0066530, 0.0068976
1: 0.9883415, 1.0026231, 0.9885343, 1.0029483, -0.0146068, 0.0140887
2: -0.0095988, 0.0010186, -0.0096756, 0.0007043, -0.0103031, 0.0106942
3: -0.0012960, 0.0071412, -0.0014882, 0.0070273, -0.0083234, 0.0086293
4: -0.0026912, 0.0118210, -0.0029416, 0.0115074, -0.0141987, 0.0147626
5: -0.0041841, 0.0117970, -0.0045480, 0.0115814, -0.0157655, 0.0163450
6: -0.0080829, 0.0076961, -0.0078365, 0.0080326, -0.0161155, 0.0155326
7: -0.0103483, -0.0035123, -0.0102560, -0.0033567, -0.0069916, 0.0067437
8: -0.0030867, 0.0086156, -0.0026738, 0.0086361, -0.0117228, 0.0112894
9: -0.0072125, 0.0025491, -0.0070808, 0.0027714, -0.0099839, 0.0096299

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090811, upper bound: 0.0090915
time: 2.63 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090817, upper bound: 0.0090918
time: 4.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0032013, 0.0035188, -0.0031895, 0.0036158, -0.0068170, 0.0067083
1: 0.9884275, 1.0026584, 0.9882222, 1.0026335, -0.0142059, 0.0144362
2: -0.0096072, 0.0008785, -0.0096013, 0.0012133, -0.0108204, 0.0104798
3: -0.0013169, 0.0070904, -0.0013022, 0.0072117, -0.0085286, 0.0083926
4: -0.0027185, 0.0116812, -0.0026992, 0.0120152, -0.0147336, 0.0143804
5: -0.0042237, 0.0117009, -0.0041957, 0.0119306, -0.0161542, 0.0158966
6: -0.0079730, 0.0077327, -0.0082355, 0.0077068, -0.0156799, 0.0159682
7: -0.0103071, -0.0034954, -0.0104054, -0.0035074, -0.0067998, 0.0069100
8: -0.0029026, 0.0086179, -0.0033425, 0.0086163, -0.0115189, 0.0119603
9: -0.0071538, 0.0025732, -0.0072941, 0.0025562, -0.0097100, 0.0098673

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A1_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090986, upper bound: 0.0090958
time: 2.86 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090958
time: 2.52 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0032359, 0.0035217, -0.0031874, 0.0035983, -0.0068342, 0.0067091
1: 0.9884215, 1.0027317, 0.9882592, 1.0026292, -0.0142077, 0.0144725
2: -0.0096245, 0.0008883, -0.0096002, 0.0011530, -0.0107774, 0.0104886
3: -0.0013603, 0.0070940, -0.0012996, 0.0071898, -0.0085501, 0.0083936
4: -0.0027749, 0.0116910, -0.0026959, 0.0119550, -0.0147299, 0.0143869
5: -0.0043057, 0.0117076, -0.0041909, 0.0118892, -0.0161949, 0.0158986
6: -0.0079808, 0.0078086, -0.0081883, 0.0077024, -0.0156832, 0.0159968
7: -0.0103100, -0.0034603, -0.0103877, -0.0035094, -0.0068006, 0.0069274
8: -0.0029155, 0.0086225, -0.0032632, 0.0086160, -0.0115316, 0.0118857
9: -0.0071579, 0.0026234, -0.0072688, 0.0025532, -0.0097112, 0.0098922

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090953
time: 2.75 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090951
time: 2.45 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0031899, 0.0035123, -0.0032039, 0.0037950, -0.0069849, 0.0067162
1: 0.9884413, 1.0026343, 0.9878427, 1.0026641, -0.0142227, 0.0147916
2: -0.0096015, 0.0008558, -0.0096085, 0.0018323, -0.0114338, 0.0104643
3: -0.0013027, 0.0070822, -0.0013203, 0.0074359, -0.0087386, 0.0084025
4: -0.0026999, 0.0116586, -0.0030311, 0.0126328, -0.0153328, 0.0146897
5: -0.0041967, 0.0116853, -0.0042300, 0.0123553, -0.0165520, 0.0159153
6: -0.0079553, 0.0077078, -0.0087210, 0.0077385, -0.0156938, 0.0164287
7: -0.0103005, -0.0035069, -0.0105871, -0.0034927, -0.0068078, 0.0070801
8: -0.0028729, 0.0086164, -0.0041558, 0.0086182, -0.0114911, 0.0127722
9: -0.0071443, 0.0025568, -0.0075535, 0.0025771, -0.0097214, 0.0101103

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091053
time: 2.59 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091048
time: 2.56 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0032244, 0.0035151, -0.0032019, 0.0037779, -0.0070024, 0.0067170
1: 0.9884354, 1.0027075, 0.9878789, 1.0026598, -0.0142244, 0.0148286
2: -0.0096188, 0.0008656, -0.0096075, 0.0017734, -0.0113921, 0.0104731
3: -0.0013459, 0.0070857, -0.0013177, 0.0074146, -0.0087605, 0.0084035
4: -0.0027563, 0.0116683, -0.0029846, 0.0125740, -0.0153303, 0.0146529
5: -0.0042786, 0.0116920, -0.0042252, 0.0123148, -0.0165935, 0.0159172
6: -0.0079629, 0.0077835, -0.0086748, 0.0077341, -0.0156970, 0.0164583
7: -0.0103034, -0.0034719, -0.0105698, -0.0034947, -0.0068086, 0.0070979
8: -0.0028856, 0.0086210, -0.0040784, 0.0086180, -0.0115036, 0.0126994
9: -0.0071484, 0.0026068, -0.0075288, 0.0025742, -0.0097226, 0.0101356

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091047
time: 2.85 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091047
time: 4.85 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0032982, 0.0034578, -0.0031866, 0.0035769, -0.0068751, 0.0066444
1: 0.9885568, 1.0028638, 0.9883046, 1.0026274, -0.0140706, 0.0145592
2: -0.0096557, 0.0006676, -0.0095998, 0.0010789, -0.0107346, 0.0102674
3: -0.0014382, 0.0070140, -0.0012986, 0.0071630, -0.0086012, 0.0083126
4: -0.0028766, 0.0114708, -0.0026945, 0.0118811, -0.0147577, 0.0141653
5: -0.0044534, 0.0115562, -0.0041889, 0.0118384, -0.0162918, 0.0157451
6: -0.0078077, 0.0079452, -0.0081302, 0.0077005, -0.0155082, 0.0160753
7: -0.0102453, -0.0033971, -0.0103660, -0.0035103, -0.0067350, 0.0069689
8: -0.0026255, 0.0086308, -0.0031659, 0.0086159, -0.0112415, 0.0117967
9: -0.0070654, 0.0027136, -0.0072378, 0.0025520, -0.0096174, 0.0099514

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090917, upper bound: 0.0090817
time: 2.64 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090914, upper bound: 0.0090817
time: 2.70 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0033381, 0.0034684, -0.0031846, 0.0035594, -0.0068976, 0.0066530
1: 0.9885343, 1.0029483, 0.9883415, 1.0026231, -0.0140887, 0.0146068
2: -0.0096756, 0.0007043, -0.0095988, 0.0010186, -0.0106942, 0.0103031
3: -0.0014882, 0.0070273, -0.0012960, 0.0071412, -0.0086293, 0.0083234
4: -0.0029416, 0.0115074, -0.0026912, 0.0118210, -0.0147626, 0.0141987
5: -0.0045480, 0.0115814, -0.0041841, 0.0117970, -0.0163450, 0.0157655
6: -0.0078365, 0.0080326, -0.0080829, 0.0076961, -0.0155326, 0.0161155
7: -0.0102560, -0.0033567, -0.0103483, -0.0035123, -0.0067437, 0.0069916
8: -0.0026738, 0.0086361, -0.0030867, 0.0086156, -0.0112894, 0.0117228
9: -0.0070808, 0.0027714, -0.0072125, 0.0025491, -0.0096299, 0.0099839

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090919, upper bound: 0.0090811
time: 2.59 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090919, upper bound: 0.0090816
time: 2.73 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0031451, 0.0036252, -0.0032282, 0.0036785, -0.0068236, 0.0068533
1: 0.9882023, 1.0025395, 0.9880894, 1.0027153, -0.0145130, 0.0144501
2: -0.0095791, 0.0012458, -0.0096206, 0.0014299, -0.0110090, 0.0108664
3: -0.0012467, 0.0072234, -0.0013506, 0.0072901, -0.0085369, 0.0085740
4: -0.0026269, 0.0120476, -0.0027623, 0.0122313, -0.0148582, 0.0148099
5: -0.0040906, 0.0119528, -0.0042874, 0.0120792, -0.0161698, 0.0162402
6: -0.0082610, 0.0076097, -0.0084054, 0.0077916, -0.0160526, 0.0160151
7: -0.0104149, -0.0035523, -0.0104690, -0.0034682, -0.0069468, 0.0069167
8: -0.0033851, 0.0086104, -0.0036271, 0.0086215, -0.0120066, 0.0122375
9: -0.0073077, 0.0024920, -0.0073849, 0.0026122, -0.0099199, 0.0098769

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091399
time: 2.47 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091399
time: 2.81 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0031564, 0.0037983, -0.0032169, 0.0036720, -0.0068284, 0.0070152
1: 0.9878358, 1.0025636, 0.9881031, 1.0026915, -0.0148557, 0.0144605
2: -0.0095847, 0.0018437, -0.0096150, 0.0014075, -0.0109922, 0.0114587
3: -0.0012609, 0.0074400, -0.0013365, 0.0072820, -0.0085429, 0.0087765
4: -0.0030402, 0.0126442, -0.0027439, 0.0122090, -0.0152491, 0.0153881
5: -0.0041175, 0.0123631, -0.0042607, 0.0120638, -0.0161813, 0.0166238
6: -0.0087299, 0.0076345, -0.0083879, 0.0077669, -0.0164968, 0.0160223
7: -0.0105904, -0.0035408, -0.0104624, -0.0034796, -0.0071108, 0.0069216
8: -0.0041708, 0.0086119, -0.0035977, 0.0086200, -0.0127908, 0.0122096
9: -0.0075583, 0.0025084, -0.0073755, 0.0025959, -0.0101541, 0.0098839

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_B1_A1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091316
time: 2.12 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091316
time: 2.34 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0032423, 0.0035633, -0.0032282, 0.0036785, -0.0069208, 0.0067915
1: 0.9883333, 1.0027454, 0.9880894, 1.0027153, -0.0143820, 0.0146559
2: -0.0096277, 0.0010321, -0.0096206, 0.0014299, -0.0110576, 0.0106528
3: -0.0013683, 0.0071461, -0.0013506, 0.0072901, -0.0086585, 0.0084967
4: -0.0027854, 0.0118345, -0.0027623, 0.0122313, -0.0150167, 0.0145968
5: -0.0043210, 0.0118063, -0.0042874, 0.0120792, -0.0164002, 0.0160937
6: -0.0080935, 0.0078227, -0.0084054, 0.0077916, -0.0158851, 0.0162281
7: -0.0103522, -0.0034538, -0.0104690, -0.0034682, -0.0068841, 0.0070152
8: -0.0031045, 0.0086234, -0.0036271, 0.0086215, -0.0117260, 0.0122504
9: -0.0072182, 0.0026327, -0.0073849, 0.0026122, -0.0098304, 0.0100176

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091208, upper bound: 0.0090987
time: 2.56 seconds

## Relational analysis of NS_A2_B2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091206, upper bound: 0.0090987
time: 2.41 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0032544, 0.0037424, -0.0032169, 0.0036720, -0.0069264, 0.0069593
1: 0.9879540, 1.0027709, 0.9881031, 1.0026915, -0.0147375, 0.0146678
2: -0.0096337, 0.0016508, -0.0096150, 0.0014075, -0.0110412, 0.0112658
3: -0.0013834, 0.0073701, -0.0013365, 0.0072820, -0.0086654, 0.0087066
4: -0.0028877, 0.0124517, -0.0027439, 0.0122090, -0.0150966, 0.0151956
5: -0.0043496, 0.0122307, -0.0042607, 0.0120638, -0.0164134, 0.0164914
6: -0.0085786, 0.0078491, -0.0083879, 0.0077669, -0.0163455, 0.0162369
7: -0.0105338, -0.0034416, -0.0104624, -0.0034796, -0.0070542, 0.0070208
8: -0.0039173, 0.0086250, -0.0035977, 0.0086200, -0.0125373, 0.0122226
9: -0.0074774, 0.0026502, -0.0073755, 0.0025959, -0.0100733, 0.0100256

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091163, upper bound: 0.0090840
time: 2.32 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091162, upper bound: 0.0090840
time: 2.66 seconds

## BFS NS instance: NS_A2_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0032051, 0.0036124, -0.0032411, 0.0035521, -0.0067572, 0.0068535
1: 0.9882294, 1.0026665, 0.9883571, 1.0027428, -0.0145134, 0.0143094
2: -0.0096091, 0.0012016, -0.0096271, 0.0009933, -0.0106024, 0.0108287
3: -0.0013218, 0.0072074, -0.0013668, 0.0071320, -0.0084538, 0.0085742
4: -0.0027247, 0.0120035, -0.0027834, 0.0117957, -0.0145205, 0.0147869
5: -0.0042328, 0.0119225, -0.0043180, 0.0117797, -0.0160125, 0.0162406
6: -0.0082264, 0.0077411, -0.0080631, 0.0078200, -0.0160464, 0.0158042
7: -0.0104020, -0.0034915, -0.0103408, -0.0034550, -0.0069469, 0.0068494
8: -0.0033271, 0.0086184, -0.0030535, 0.0086232, -0.0119503, 0.0116718
9: -0.0072892, 0.0025788, -0.0072019, 0.0026309, -0.0099201, 0.0097808

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090917, upper bound: 0.0090818
time: 3.44 seconds

## Relational analysis of NS_A2_B2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090923, upper bound: 0.0090816
time: 3.97 seconds

## BFS NS instance: NS_A2_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0032261, 0.0036274, -0.0032415, 0.0035536, -0.0067798, 0.0068688
1: 0.9881977, 1.0027111, 0.9883538, 1.0027435, -0.0145458, 0.0143573
2: -0.0096196, 0.0012533, -0.0096273, 0.0009987, -0.0106183, 0.0108806
3: -0.0013481, 0.0072262, -0.0013672, 0.0071339, -0.0084820, 0.0085934
4: -0.0027590, 0.0120551, -0.0027840, 0.0118011, -0.0145601, 0.0148391
5: -0.0042826, 0.0119580, -0.0043190, 0.0117833, -0.0160660, 0.0162770
6: -0.0082669, 0.0077872, -0.0080673, 0.0078208, -0.0160877, 0.0158545
7: -0.0104171, -0.0034702, -0.0103424, -0.0034546, -0.0069625, 0.0068722
8: -0.0033951, 0.0086212, -0.0030605, 0.0086232, -0.0120183, 0.0116817
9: -0.0073109, 0.0026092, -0.0072042, 0.0026315, -0.0099423, 0.0098134

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090919, upper bound: 0.0090822
time: 3.19 seconds

## Relational analysis of NS_A2_B2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090919, upper bound: 0.0090818
time: 2.59 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 7.64 seconds
NS_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091413
NS_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091316
NS_A1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091399
NS_A1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091307
NS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091103
NS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091108
NS_A1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091047
NS_A1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091053
NS_A1_B1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090817, upper bound: 0.0090817
NS_A1_B1_A2_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090817, upper bound: 0.0090824
NS_A1_B1_A2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090820, upper bound: 0.0090824
NS_A1_B1_A2_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090817, upper bound: 0.0090828
NS_A1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090986
NS_A1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090987
NS_A1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090986
NS_A1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090987
NS_A1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091053, upper bound: 0.0090840
NS_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091053, upper bound: 0.0090840
NS_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091053, upper bound: 0.0090840
NS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091053, upper bound: 0.0090840
NS_A1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090817, upper bound: 0.0090914
NS_A1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090817, upper bound: 0.0090914
NS_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090811, upper bound: 0.0090915
NS_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090817, upper bound: 0.0090918
NS_A2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090986, upper bound: 0.0090958
NS_A2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090958
NS_A2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090953
NS_A2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090951
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091053
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091048
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091047
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091047
NS_A2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090917, upper bound: 0.0090817
NS_A2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090914, upper bound: 0.0090817
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090919, upper bound: 0.0090811
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090919, upper bound: 0.0090816
NS_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091399
NS_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091399
NS_A2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091316
NS_A2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091316
NS_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091208, upper bound: 0.0090987
NS_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091206, upper bound: 0.0090987
NS_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091163, upper bound: 0.0090840
NS_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0091162, upper bound: 0.0090840
NS_A2_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090917, upper bound: 0.0090818
NS_A2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090923, upper bound: 0.0090816
NS_A2_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090919, upper bound: 0.0090822
NS_A2_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.64
Output dim: 1, lower bound: -0.0090919, upper bound: 0.0090818

## BFS NS instance: NS_A1_B1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0031470, 0.0033175, -0.0032579, 0.0035294, -0.0066764, 0.0065753
1: 0.9888540, 1.0025436, 0.9884051, 1.0027783, -0.0139243, 0.0141385
2: -0.0095800, 0.0001828, -0.0096355, 0.0009151, -0.0104951, 0.0098183
3: -0.0012490, 0.0068385, -0.0013878, 0.0071037, -0.0083527, 0.0082262
4: -0.0026300, 0.0109871, -0.0028108, 0.0117177, -0.0143476, 0.0137979
5: -0.0040951, 0.0112236, -0.0043578, 0.0117260, -0.0158210, 0.0155815
6: -0.0074276, 0.0076137, -0.0080017, 0.0078568, -0.0152844, 0.0156155
7: -0.0101030, -0.0035504, -0.0103179, -0.0034380, -0.0066650, 0.0067675
8: -0.0019886, 0.0086106, -0.0029507, 0.0086254, -0.0106140, 0.0115613
9: -0.0068623, 0.0024947, -0.0071691, 0.0026552, -0.0095175, 0.0096638

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091352
time: 2.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091357
time: 2.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0031611, 0.0034985, -0.0032469, 0.0035238, -0.0066849, 0.0067454
1: 0.9884706, 1.0025734, 0.9884170, 1.0027549, -0.0142844, 0.0141563
2: -0.0095871, 0.0008083, -0.0096300, 0.0008955, -0.0104825, 0.0104382
3: -0.0012667, 0.0070650, -0.0013740, 0.0070966, -0.0083633, 0.0084389
4: -0.0026530, 0.0116111, -0.0027928, 0.0116981, -0.0143511, 0.0144039
5: -0.0041285, 0.0116527, -0.0043317, 0.0117125, -0.0158411, 0.0159844
6: -0.0079180, 0.0076447, -0.0079863, 0.0078326, -0.0157505, 0.0156310
7: -0.0102865, -0.0035361, -0.0103121, -0.0034492, -0.0068373, 0.0067760
8: -0.0028103, 0.0086125, -0.0029249, 0.0086240, -0.0114343, 0.0115374
9: -0.0071244, 0.0025151, -0.0071609, 0.0026392, -0.0097636, 0.0096760

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091302, upper bound: 0.0091304
time: 3.28 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091303, upper bound: 0.0091305
time: 3.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0031852, 0.0033178, -0.0032559, 0.0035120, -0.0066972, 0.0065737
1: 0.9888532, 1.0026243, 0.9884421, 1.0027740, -0.0139208, 0.0141822
2: -0.0095991, 0.0001840, -0.0096345, 0.0008547, -0.0104538, 0.0098185
3: -0.0012968, 0.0068389, -0.0013853, 0.0070818, -0.0083786, 0.0082242
4: -0.0026922, 0.0109883, -0.0028075, 0.0116575, -0.0143497, 0.0137958
5: -0.0041855, 0.0112244, -0.0043531, 0.0116846, -0.0158701, 0.0155775
6: -0.0074285, 0.0076974, -0.0079544, 0.0078524, -0.0152808, 0.0156518
7: -0.0101033, -0.0035117, -0.0103002, -0.0034400, -0.0066633, 0.0067885
8: -0.0019902, 0.0086157, -0.0028714, 0.0086252, -0.0106153, 0.0114871
9: -0.0068628, 0.0025500, -0.0071438, 0.0026523, -0.0095151, 0.0096938

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091350
time: 2.14 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091354
time: 2.50 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0031969, 0.0035007, -0.0032449, 0.0035063, -0.0067032, 0.0067456
1: 0.9884658, 1.0026492, 0.9884540, 1.0027508, -0.0142849, 0.0141952
2: -0.0096050, 0.0008160, -0.0096290, 0.0008351, -0.0104401, 0.0104449
3: -0.0013115, 0.0070678, -0.0013715, 0.0070747, -0.0083862, 0.0084392
4: -0.0027113, 0.0116188, -0.0027895, 0.0116379, -0.0143492, 0.0144083
5: -0.0042133, 0.0116580, -0.0043269, 0.0116711, -0.0158844, 0.0159849
6: -0.0079240, 0.0077231, -0.0079390, 0.0078282, -0.0157522, 0.0156621
7: -0.0102888, -0.0034998, -0.0102944, -0.0034512, -0.0068376, 0.0067946
8: -0.0028205, 0.0086173, -0.0028456, 0.0086237, -0.0114441, 0.0114629
9: -0.0071276, 0.0025669, -0.0071356, 0.0026363, -0.0097639, 0.0097025

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091299, upper bound: 0.0091301
time: 3.14 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091301, upper bound: 0.0091301
time: 2.45 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0032310, 0.0033685, -0.0032690, 0.0034203, -0.0066512, 0.0066375
1: 0.9887458, 1.0027213, 0.9886363, 1.0028018, -0.0140560, 0.0140850
2: -0.0096220, 0.0003593, -0.0096410, 0.0005379, -0.0101600, 0.0100003
3: -0.0013541, 0.0069024, -0.0014017, 0.0069671, -0.0083212, 0.0083040
4: -0.0027669, 0.0111632, -0.0028289, 0.0113414, -0.0141083, 0.0139921
5: -0.0042941, 0.0113447, -0.0043842, 0.0114673, -0.0157613, 0.0157288
6: -0.0075659, 0.0077978, -0.0077060, 0.0078811, -0.0154470, 0.0155038
7: -0.0101548, -0.0034653, -0.0102072, -0.0034267, -0.0067280, 0.0067419
8: -0.0022204, 0.0086218, -0.0024552, 0.0086269, -0.0108473, 0.0110770
9: -0.0069362, 0.0026162, -0.0070111, 0.0026713, -0.0096075, 0.0096273

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090951, upper bound: 0.0090948
time: 2.39 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090951, upper bound: 0.0090956
time: 2.50 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0032682, 0.0033685, -0.0032670, 0.0034030, -0.0066712, 0.0066356
1: 0.9887459, 1.0028001, 0.9886728, 1.0027976, -0.0140517, 0.0141273
2: -0.0096406, 0.0003593, -0.0096401, 0.0004784, -0.0101190, 0.0099993
3: -0.0014007, 0.0069024, -0.0013992, 0.0069455, -0.0083461, 0.0083015
4: -0.0028276, 0.0111631, -0.0028257, 0.0112820, -0.0141096, 0.0139888
5: -0.0043823, 0.0113447, -0.0043795, 0.0114264, -0.0158087, 0.0157242
6: -0.0075659, 0.0078794, -0.0076593, 0.0078768, -0.0154427, 0.0155387
7: -0.0101548, -0.0034276, -0.0101897, -0.0034287, -0.0067260, 0.0067622
8: -0.0022204, 0.0086268, -0.0023769, 0.0086266, -0.0108470, 0.0110037
9: -0.0069362, 0.0026701, -0.0069861, 0.0026684, -0.0096047, 0.0096563

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090950, upper bound: 0.0090950
time: 2.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090950, upper bound: 0.0090951
time: 2.43 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0032469, 0.0035238, -0.0032552, 0.0034454, -0.0066923, 0.0067790
1: 0.9884170, 1.0027549, 0.9885830, 1.0027727, -0.0143557, 0.0141719
2: -0.0096300, 0.0008955, -0.0096342, 0.0006248, -0.0102548, 0.0105296
3: -0.0013740, 0.0070966, -0.0013844, 0.0069985, -0.0083725, 0.0084810
4: -0.0027928, 0.0116981, -0.0028065, 0.0114281, -0.0142209, 0.0145046
5: -0.0043317, 0.0117125, -0.0043516, 0.0115268, -0.0158585, 0.0160641
6: -0.0079863, 0.0078326, -0.0077741, 0.0078509, -0.0158373, 0.0156067
7: -0.0103121, -0.0034492, -0.0102327, -0.0034407, -0.0068715, 0.0067835
8: -0.0029249, 0.0086240, -0.0025693, 0.0086251, -0.0115500, 0.0111933
9: -0.0071609, 0.0026392, -0.0070475, 0.0026514, -0.0098123, 0.0096867

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 198

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090891
time: 3.57 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090841, upper bound: 0.0090883
time: 2.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0032449, 0.0035063, -0.0032930, 0.0034481, -0.0066930, 0.0067992
1: 0.9884540, 1.0027508, 0.9885772, 1.0028526, -0.0143986, 0.0141735
2: -0.0096290, 0.0008351, -0.0096530, 0.0006343, -0.0102632, 0.0104881
3: -0.0013715, 0.0070747, -0.0014316, 0.0070020, -0.0083734, 0.0085063
4: -0.0027895, 0.0116379, -0.0028680, 0.0114375, -0.0142270, 0.0145059
5: -0.0043269, 0.0116711, -0.0044409, 0.0115333, -0.0158602, 0.0161120
6: -0.0079390, 0.0078282, -0.0077815, 0.0079336, -0.0158726, 0.0156097
7: -0.0102944, -0.0034512, -0.0102355, -0.0034025, -0.0068919, 0.0067843
8: -0.0028456, 0.0086237, -0.0025817, 0.0086301, -0.0114757, 0.0112054
9: -0.0071356, 0.0026363, -0.0070514, 0.0027059, -0.0098416, 0.0096878

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0090896
time: 2.85 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0090894
time: 3.06 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0031882, 0.0036055, -0.0031672, 0.0033549, -0.0065430, 0.0067727
1: 0.9882439, 1.0026306, 0.9887747, 1.0025862, -0.0143423, 0.0138559
2: -0.0096006, 0.0011778, -0.0095901, 0.0003121, -0.0099127, 0.0107680
3: -0.0013005, 0.0071988, -0.0012743, 0.0068853, -0.0081858, 0.0084731
4: -0.0026971, 0.0119798, -0.0026629, 0.0111161, -0.0138132, 0.0146427
5: -0.0041926, 0.0119062, -0.0041429, 0.0113123, -0.0155049, 0.0160491
6: -0.0082078, 0.0077040, -0.0075289, 0.0076579, -0.0158657, 0.0152329
7: -0.0103950, -0.0035087, -0.0101409, -0.0035300, -0.0068650, 0.0066322
8: -0.0032959, 0.0086161, -0.0021584, 0.0086133, -0.0119092, 0.0107746
9: -0.0072792, 0.0025543, -0.0069164, 0.0025239, -0.0098031, 0.0094707

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090987
time: 2.98 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090987
time: 3.13 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0031886, 0.0036067, -0.0031876, 0.0033739, -0.0065625, 0.0067943
1: 0.9882414, 1.0026317, 0.9887344, 1.0026295, -0.0143881, 0.0138973
2: -0.0096008, 0.0011820, -0.0096003, 0.0003779, -0.0099787, 0.0107824
3: -0.0013010, 0.0072004, -0.0012999, 0.0069091, -0.0082101, 0.0085002
4: -0.0026978, 0.0119840, -0.0026962, 0.0111818, -0.0138795, 0.0146802
5: -0.0041936, 0.0119091, -0.0041913, 0.0113575, -0.0155511, 0.0161004
6: -0.0082111, 0.0077049, -0.0075806, 0.0077028, -0.0159139, 0.0152854
7: -0.0103962, -0.0035083, -0.0101603, -0.0035092, -0.0068870, 0.0066520
8: -0.0033014, 0.0086162, -0.0022450, 0.0086160, -0.0119175, 0.0108611
9: -0.0072810, 0.0025549, -0.0069440, 0.0025535, -0.0098345, 0.0094989

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090987
time: 3.13 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090987
time: 2.69 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0031861, 0.0035880, -0.0032038, 0.0033566, -0.0065427, 0.0067918
1: 0.9882810, 1.0026263, 0.9887711, 1.0026637, -0.0143828, 0.0138552
2: -0.0095996, 0.0011175, -0.0096084, 0.0003180, -0.0099176, 0.0107259
3: -0.0012980, 0.0071770, -0.0013201, 0.0068874, -0.0081854, 0.0084971
4: -0.0026938, 0.0119197, -0.0027226, 0.0111220, -0.0138157, 0.0146422
5: -0.0041878, 0.0118649, -0.0042296, 0.0113163, -0.0155042, 0.0160945
6: -0.0081605, 0.0076995, -0.0075336, 0.0077382, -0.0158986, 0.0152331
7: -0.0103773, -0.0035107, -0.0101427, -0.0034929, -0.0068844, 0.0066319
8: -0.0032167, 0.0086158, -0.0021662, 0.0086182, -0.0118349, 0.0107820
9: -0.0072540, 0.0025513, -0.0069189, 0.0025769, -0.0098308, 0.0094703

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090987
time: 2.53 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090986
time: 2.37 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0031865, 0.0035893, -0.0032220, 0.0033702, -0.0065567, 0.0068113
1: 0.9882783, 1.0026273, 0.9887422, 1.0027024, -0.0144240, 0.0138850
2: -0.0095998, 0.0011218, -0.0096175, 0.0003650, -0.0099648, 0.0107393
3: -0.0012985, 0.0071785, -0.0013429, 0.0069044, -0.0082029, 0.0085214
4: -0.0026945, 0.0119239, -0.0027523, 0.0111689, -0.0138634, 0.0146762
5: -0.0041888, 0.0118678, -0.0042728, 0.0113486, -0.0155374, 0.0161406
6: -0.0081638, 0.0077004, -0.0075704, 0.0077781, -0.0159419, 0.0152709
7: -0.0103785, -0.0035103, -0.0101565, -0.0034744, -0.0069042, 0.0066462
8: -0.0032222, 0.0086159, -0.0022280, 0.0086206, -0.0118429, 0.0108439
9: -0.0072557, 0.0025519, -0.0069386, 0.0026033, -0.0098590, 0.0094905

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090987
time: 3.58 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090958, upper bound: 0.0090987
time: 2.89 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0031884, 0.0036593, -0.0031899, 0.0035123, -0.0067007, 0.0068492
1: 0.9881302, 1.0026312, 0.9884413, 1.0026343, -0.0145041, 0.0141898
2: -0.0096007, 0.0013635, -0.0096015, 0.0008558, -0.0104566, 0.0109650
3: -0.0013008, 0.0072661, -0.0013027, 0.0070822, -0.0083831, 0.0085688
4: -0.0026975, 0.0121650, -0.0026999, 0.0116586, -0.0143560, 0.0148650
5: -0.0041932, 0.0120336, -0.0041967, 0.0116853, -0.0158785, 0.0162304
6: -0.0083533, 0.0077045, -0.0079553, 0.0077078, -0.0160611, 0.0156598
7: -0.0104495, -0.0035084, -0.0103005, -0.0035069, -0.0069426, 0.0067921
8: -0.0035398, 0.0086161, -0.0028729, 0.0086164, -0.0121562, 0.0114890
9: -0.0073570, 0.0025546, -0.0071443, 0.0025568, -0.0099138, 0.0096989

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090896, upper bound: 0.0090840
time: 2.22 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090896, upper bound: 0.0090840
time: 2.54 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0032826, 0.0036052, -0.0031899, 0.0035123, -0.0067949, 0.0067951
1: 0.9882447, 1.0028306, 0.9884413, 1.0026343, -0.0143896, 0.0143893
2: -0.0096479, 0.0011768, -0.0096015, 0.0008558, -0.0105037, 0.0107783
3: -0.0014187, 0.0071985, -0.0013027, 0.0070822, -0.0085009, 0.0085012
4: -0.0028511, 0.0119788, -0.0026999, 0.0116586, -0.0145096, 0.0146787
5: -0.0044164, 0.0119055, -0.0041967, 0.0116853, -0.0161017, 0.0161023
6: -0.0082069, 0.0079109, -0.0079553, 0.0077078, -0.0159147, 0.0158662
7: -0.0103947, -0.0034130, -0.0103005, -0.0035069, -0.0068878, 0.0068875
8: -0.0032945, 0.0086287, -0.0028729, 0.0086164, -0.0119109, 0.0115016
9: -0.0072788, 0.0026910, -0.0071443, 0.0025568, -0.0098356, 0.0098353

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090679, upper bound: 0.0089773
time: 2.45 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090699, upper bound: 0.0090494
time: 2.33 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0031863, 0.0036419, -0.0032244, 0.0035151, -0.0067014, 0.0068664
1: 0.9881669, 1.0026267, 0.9884354, 1.0027075, -0.0145406, 0.0141913
2: -0.0095997, 0.0013036, -0.0096188, 0.0008656, -0.0104652, 0.0109223
3: -0.0012982, 0.0072444, -0.0013459, 0.0070857, -0.0083839, 0.0085903
4: -0.0026940, 0.0121053, -0.0027563, 0.0116683, -0.0143623, 0.0148616
5: -0.0041881, 0.0119925, -0.0042786, 0.0116920, -0.0158801, 0.0162712
6: -0.0083064, 0.0076998, -0.0079629, 0.0077835, -0.0160899, 0.0156627
7: -0.0104319, -0.0035106, -0.0103034, -0.0034719, -0.0069600, 0.0067927
8: -0.0034611, 0.0086159, -0.0028856, 0.0086210, -0.0120821, 0.0115015
9: -0.0073319, 0.0025516, -0.0071484, 0.0026068, -0.0099388, 0.0096999

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090896, upper bound: 0.0090840
time: 2.47 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090896, upper bound: 0.0090840
time: 2.66 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0032805, 0.0035881, -0.0032244, 0.0035151, -0.0067956, 0.0068126
1: 0.9882808, 1.0028262, 0.9884354, 1.0027075, -0.0144267, 0.0143908
2: -0.0096468, 0.0011178, -0.0096188, 0.0008656, -0.0105124, 0.0107365
3: -0.0014161, 0.0071771, -0.0013459, 0.0070857, -0.0085018, 0.0085230
4: -0.0028477, 0.0119199, -0.0027563, 0.0116683, -0.0145160, 0.0146762
5: -0.0044115, 0.0118650, -0.0042786, 0.0116920, -0.0161035, 0.0161437
6: -0.0081607, 0.0079063, -0.0079629, 0.0077835, -0.0159442, 0.0158693
7: -0.0103774, -0.0034151, -0.0103034, -0.0034719, -0.0069055, 0.0068883
8: -0.0032170, 0.0086284, -0.0028856, 0.0086210, -0.0118380, 0.0115141
9: -0.0072541, 0.0026880, -0.0071484, 0.0026068, -0.0098609, 0.0098363

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090896, upper bound: 0.0090840
time: 2.49 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090896, upper bound: 0.0090840
time: 2.71 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0031853, 0.0035666, -0.0032635, 0.0032898, -0.0064751, 0.0068301
1: 0.9883264, 1.0026245, 0.9889125, 1.0027902, -0.0144638, 0.0137120
2: -0.0095992, 0.0010435, -0.0096383, 0.0000874, -0.0096866, 0.0106817
3: -0.0012970, 0.0071502, -0.0013948, 0.0068039, -0.0081008, 0.0085449
4: -0.0026924, 0.0118458, -0.0028199, 0.0108919, -0.0135843, 0.0146657
5: -0.0041858, 0.0118141, -0.0043712, 0.0111581, -0.0153439, 0.0161852
6: -0.0081024, 0.0076977, -0.0073527, 0.0078690, -0.0159714, 0.0150504
7: -0.0103556, -0.0035116, -0.0100750, -0.0034323, -0.0069233, 0.0065634
8: -0.0031194, 0.0086157, -0.0018632, 0.0086262, -0.0117455, 0.0104789
9: -0.0072229, 0.0025501, -0.0068223, 0.0026633, -0.0098863, 0.0093724

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090816, upper bound: 0.0090913
time: 2.70 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090816, upper bound: 0.0090916
time: 3.24 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0031857, 0.0035678, -0.0032845, 0.0033083, -0.0064940, 0.0068523
1: 0.9883239, 1.0026255, 0.9888732, 1.0028347, -0.0145108, 0.0137522
2: -0.0095994, 0.0010476, -0.0096488, 0.0001513, -0.0097507, 0.0106964
3: -0.0012975, 0.0071517, -0.0014211, 0.0068270, -0.0081245, 0.0085727
4: -0.0026931, 0.0118499, -0.0028541, 0.0109557, -0.0136488, 0.0147041
5: -0.0041868, 0.0118169, -0.0044208, 0.0112020, -0.0153888, 0.0162378
6: -0.0081057, 0.0076986, -0.0074029, 0.0079150, -0.0160207, 0.0151015
7: -0.0103568, -0.0035112, -0.0100938, -0.0034110, -0.0069457, 0.0065826
8: -0.0031248, 0.0086158, -0.0019472, 0.0086290, -0.0117538, 0.0105630
9: -0.0072247, 0.0025507, -0.0068491, 0.0026937, -0.0099184, 0.0093998

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090817, upper bound: 0.0090911
time: 3.64 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090817, upper bound: 0.0090914
time: 7.02 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0031434, 0.0033983, -0.0033369, 0.0034570, -0.0066004, 0.0067352
1: 0.9886829, 1.0025358, 0.9885585, 1.0029457, -0.0142628, 0.0139773
2: -0.0095782, 0.0004620, -0.0096750, 0.0006647, -0.0102430, 0.0101370
3: -0.0012446, 0.0069396, -0.0014867, 0.0070130, -0.0082575, 0.0084262
4: -0.0026241, 0.0112657, -0.0029397, 0.0114679, -0.0140920, 0.0142053
5: -0.0040866, 0.0114152, -0.0045451, 0.0115542, -0.0156408, 0.0159603
6: -0.0076465, 0.0076059, -0.0078055, 0.0080299, -0.0156764, 0.0154114
7: -0.0101849, -0.0035540, -0.0102444, -0.0033579, -0.0068270, 0.0066904
8: -0.0023554, 0.0086101, -0.0026218, 0.0086360, -0.0109914, 0.0112319
9: -0.0069793, 0.0024895, -0.0070642, 0.0027696, -0.0097489, 0.0095537

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 198

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_B2_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090453, upper bound: 0.0089877
time: 3.40 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090464, upper bound: 0.0090575
time: 3.01 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0031701, 0.0034133, -0.0033373, 0.0034586, -0.0066287, 0.0067505
1: 0.9886510, 1.0025926, 0.9885551, 1.0029464, -0.0142953, 0.0140375
2: -0.0095916, 0.0005138, -0.0096752, 0.0006704, -0.0102620, 0.0101890
3: -0.0012780, 0.0069583, -0.0014871, 0.0070151, -0.0082930, 0.0084454
4: -0.0026677, 0.0113174, -0.0029402, 0.0114736, -0.0141413, 0.0142576
5: -0.0041498, 0.0114507, -0.0045460, 0.0115581, -0.0157080, 0.0159967
6: -0.0076871, 0.0076644, -0.0078099, 0.0080307, -0.0157178, 0.0154743
7: -0.0102001, -0.0035270, -0.0102461, -0.0033575, -0.0068426, 0.0067191
8: -0.0024235, 0.0086137, -0.0026293, 0.0086360, -0.0110595, 0.0112430
9: -0.0070010, 0.0025282, -0.0070666, 0.0027701, -0.0097711, 0.0095948

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090461, upper bound: 0.0089876
time: 3.60 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090470, upper bound: 0.0090573
time: 3.36 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0031672, 0.0033549, -0.0031882, 0.0036055, -0.0067727, 0.0065430
1: 0.9887747, 1.0025862, 0.9882439, 1.0026306, -0.0138559, 0.0143423
2: -0.0095901, 0.0003121, -0.0096006, 0.0011778, -0.0107680, 0.0099127
3: -0.0012743, 0.0068853, -0.0013005, 0.0071988, -0.0084731, 0.0081858
4: -0.0026629, 0.0111161, -0.0026971, 0.0119798, -0.0146427, 0.0138132
5: -0.0041429, 0.0113123, -0.0041926, 0.0119062, -0.0160491, 0.0155049
6: -0.0075289, 0.0076579, -0.0082078, 0.0077040, -0.0152329, 0.0158657
7: -0.0101409, -0.0035300, -0.0103950, -0.0035087, -0.0066322, 0.0068650
8: -0.0021584, 0.0086133, -0.0032959, 0.0086161, -0.0107746, 0.0119092
9: -0.0069164, 0.0025239, -0.0072792, 0.0025543, -0.0094707, 0.0098031

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090956
time: 2.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090953
time: 2.55 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0031876, 0.0033739, -0.0031886, 0.0036067, -0.0067943, 0.0065625
1: 0.9887344, 1.0026295, 0.9882414, 1.0026317, -0.0138973, 0.0143881
2: -0.0096003, 0.0003779, -0.0096008, 0.0011820, -0.0107824, 0.0099787
3: -0.0012999, 0.0069091, -0.0013010, 0.0072004, -0.0085002, 0.0082101
4: -0.0026962, 0.0111818, -0.0026978, 0.0119840, -0.0146802, 0.0138795
5: -0.0041913, 0.0113575, -0.0041936, 0.0119091, -0.0161004, 0.0155511
6: -0.0075806, 0.0077028, -0.0082111, 0.0077049, -0.0152854, 0.0159139
7: -0.0101603, -0.0035092, -0.0103962, -0.0035083, -0.0066520, 0.0068870
8: -0.0022450, 0.0086160, -0.0033014, 0.0086162, -0.0108611, 0.0119175
9: -0.0069440, 0.0025535, -0.0072810, 0.0025549, -0.0094989, 0.0098345

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090957
time: 2.83 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090951
time: 2.59 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0032038, 0.0033566, -0.0031861, 0.0035880, -0.0067918, 0.0065427
1: 0.9887711, 1.0026637, 0.9882810, 1.0026263, -0.0138552, 0.0143828
2: -0.0096084, 0.0003180, -0.0095996, 0.0011175, -0.0107259, 0.0099176
3: -0.0013201, 0.0068874, -0.0012980, 0.0071770, -0.0084971, 0.0081854
4: -0.0027226, 0.0111220, -0.0026938, 0.0119197, -0.0146422, 0.0138157
5: -0.0042296, 0.0113163, -0.0041878, 0.0118649, -0.0160945, 0.0155042
6: -0.0075336, 0.0077382, -0.0081605, 0.0076995, -0.0152331, 0.0158986
7: -0.0101427, -0.0034929, -0.0103773, -0.0035107, -0.0066319, 0.0068844
8: -0.0021662, 0.0086182, -0.0032167, 0.0086158, -0.0107820, 0.0118349
9: -0.0069189, 0.0025769, -0.0072540, 0.0025513, -0.0094703, 0.0098308

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090951
time: 2.55 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090951
time: 2.50 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0032220, 0.0033702, -0.0031865, 0.0035893, -0.0068113, 0.0065567
1: 0.9887422, 1.0027024, 0.9882783, 1.0026273, -0.0138850, 0.0144240
2: -0.0096175, 0.0003650, -0.0095998, 0.0011218, -0.0107393, 0.0099648
3: -0.0013429, 0.0069044, -0.0012985, 0.0071785, -0.0085214, 0.0082029
4: -0.0027523, 0.0111689, -0.0026945, 0.0119239, -0.0146762, 0.0138634
5: -0.0042728, 0.0113486, -0.0041888, 0.0118678, -0.0161406, 0.0155374
6: -0.0075704, 0.0077781, -0.0081638, 0.0077004, -0.0152709, 0.0159419
7: -0.0101565, -0.0034744, -0.0103785, -0.0035103, -0.0066462, 0.0069042
8: -0.0022280, 0.0086206, -0.0032222, 0.0086159, -0.0108439, 0.0118429
9: -0.0069386, 0.0026033, -0.0072557, 0.0025519, -0.0094905, 0.0098590

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090953
time: 2.60 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0090951
time: 2.58 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0031899, 0.0035123, -0.0031884, 0.0036593, -0.0068492, 0.0067007
1: 0.9884413, 1.0026343, 0.9881302, 1.0026312, -0.0141898, 0.0145041
2: -0.0096015, 0.0008558, -0.0096007, 0.0013635, -0.0109650, 0.0104566
3: -0.0013027, 0.0070822, -0.0013008, 0.0072661, -0.0085688, 0.0083831
4: -0.0026999, 0.0116586, -0.0026975, 0.0121650, -0.0148650, 0.0143560
5: -0.0041967, 0.0116853, -0.0041932, 0.0120336, -0.0162304, 0.0158785
6: -0.0079553, 0.0077078, -0.0083533, 0.0077045, -0.0156598, 0.0160611
7: -0.0103005, -0.0035069, -0.0104495, -0.0035084, -0.0067921, 0.0069426
8: -0.0028729, 0.0086164, -0.0035398, 0.0086161, -0.0114890, 0.0121562
9: -0.0071443, 0.0025568, -0.0073570, 0.0025546, -0.0096989, 0.0099138

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0090890
time: 2.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0090890
time: 2.99 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0031899, 0.0035123, -0.0032826, 0.0036052, -0.0067951, 0.0067949
1: 0.9884413, 1.0026343, 0.9882447, 1.0028306, -0.0143893, 0.0143896
2: -0.0096015, 0.0008558, -0.0096479, 0.0011768, -0.0107783, 0.0105037
3: -0.0013027, 0.0070822, -0.0014187, 0.0071985, -0.0085012, 0.0085009
4: -0.0026999, 0.0116586, -0.0028511, 0.0119788, -0.0146787, 0.0145096
5: -0.0041967, 0.0116853, -0.0044164, 0.0119055, -0.0161023, 0.0161017
6: -0.0079553, 0.0077078, -0.0082069, 0.0079109, -0.0158662, 0.0159147
7: -0.0103005, -0.0035069, -0.0103947, -0.0034130, -0.0068875, 0.0068878
8: -0.0028729, 0.0086164, -0.0032945, 0.0086287, -0.0115016, 0.0119109
9: -0.0071443, 0.0025568, -0.0072788, 0.0026910, -0.0098353, 0.0098356

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0089773, upper bound: 0.0090679
time: 2.45 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0089773, upper bound: 0.0090695
time: 3.00 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0032244, 0.0035151, -0.0031863, 0.0036419, -0.0068664, 0.0067014
1: 0.9884354, 1.0027075, 0.9881669, 1.0026267, -0.0141913, 0.0145406
2: -0.0096188, 0.0008656, -0.0095997, 0.0013036, -0.0109223, 0.0104652
3: -0.0013459, 0.0070857, -0.0012982, 0.0072444, -0.0085903, 0.0083839
4: -0.0027563, 0.0116683, -0.0026940, 0.0121053, -0.0148616, 0.0143623
5: -0.0042786, 0.0116920, -0.0041881, 0.0119925, -0.0162712, 0.0158801
6: -0.0079629, 0.0077835, -0.0083064, 0.0076998, -0.0156627, 0.0160899
7: -0.0103034, -0.0034719, -0.0104319, -0.0035106, -0.0067927, 0.0069600
8: -0.0028856, 0.0086210, -0.0034611, 0.0086159, -0.0115015, 0.0120821
9: -0.0071484, 0.0026068, -0.0073319, 0.0025516, -0.0096999, 0.0099388

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0090890
time: 2.94 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0090895
time: 3.53 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0032244, 0.0035151, -0.0032805, 0.0035881, -0.0068126, 0.0067956
1: 0.9884354, 1.0027075, 0.9882808, 1.0028262, -0.0143908, 0.0144267
2: -0.0096188, 0.0008656, -0.0096468, 0.0011178, -0.0107365, 0.0105124
3: -0.0013459, 0.0070857, -0.0014161, 0.0071771, -0.0085230, 0.0085018
4: -0.0027563, 0.0116683, -0.0028477, 0.0119199, -0.0146762, 0.0145160
5: -0.0042786, 0.0116920, -0.0044115, 0.0118650, -0.0161437, 0.0161035
6: -0.0079629, 0.0077835, -0.0081607, 0.0079063, -0.0158693, 0.0159442
7: -0.0103034, -0.0034719, -0.0103774, -0.0034151, -0.0068883, 0.0069055
8: -0.0028856, 0.0086210, -0.0032170, 0.0086284, -0.0115141, 0.0118380
9: -0.0071484, 0.0026068, -0.0072541, 0.0026880, -0.0098363, 0.0098609

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0090894
time: 2.76 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0090888
time: 2.48 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0032635, 0.0032898, -0.0031853, 0.0035666, -0.0068301, 0.0064751
1: 0.9889125, 1.0027902, 0.9883264, 1.0026245, -0.0137120, 0.0144638
2: -0.0096383, 0.0000874, -0.0095992, 0.0010435, -0.0106817, 0.0096866
3: -0.0013948, 0.0068039, -0.0012970, 0.0071502, -0.0085449, 0.0081008
4: -0.0028199, 0.0108919, -0.0026924, 0.0118458, -0.0146657, 0.0135843
5: -0.0043712, 0.0111581, -0.0041858, 0.0118141, -0.0161852, 0.0153439
6: -0.0073527, 0.0078690, -0.0081024, 0.0076977, -0.0150504, 0.0159714
7: -0.0100750, -0.0034323, -0.0103556, -0.0035116, -0.0065634, 0.0069233
8: -0.0018632, 0.0086262, -0.0031194, 0.0086157, -0.0104789, 0.0117455
9: -0.0068223, 0.0026633, -0.0072229, 0.0025501, -0.0093724, 0.0098863

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090917, upper bound: 0.0090817
time: 3.94 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090917, upper bound: 0.0090817
time: 2.47 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0032845, 0.0033083, -0.0031857, 0.0035678, -0.0068523, 0.0064940
1: 0.9888732, 1.0028347, 0.9883239, 1.0026255, -0.0137522, 0.0145108
2: -0.0096488, 0.0001513, -0.0095994, 0.0010476, -0.0106964, 0.0097507
3: -0.0014211, 0.0068270, -0.0012975, 0.0071517, -0.0085727, 0.0081245
4: -0.0028541, 0.0109557, -0.0026931, 0.0118499, -0.0147041, 0.0136488
5: -0.0044208, 0.0112020, -0.0041868, 0.0118169, -0.0162378, 0.0153888
6: -0.0074029, 0.0079150, -0.0081057, 0.0076986, -0.0151015, 0.0160207
7: -0.0100938, -0.0034110, -0.0103568, -0.0035112, -0.0065826, 0.0069457
8: -0.0019472, 0.0086290, -0.0031248, 0.0086158, -0.0105630, 0.0117538
9: -0.0068491, 0.0026937, -0.0072247, 0.0025507, -0.0093998, 0.0099184

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090914, upper bound: 0.0090817
time: 2.76 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090914, upper bound: 0.0090815
time: 6.42 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.95 + 602.23 = 607.18 seconds
