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
execution time: IAR + RelationalAnalysis = 1.81 + 3.11 = 4.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
time: 2.09 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091791
time: 2.06 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.36 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.36
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.36
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

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
time: 1.84 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
time: 1.87 seconds

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

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
time: 2.23 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
time: 2.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 6.05 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.05
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.05
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.05
Output dim: 1, lower bound: -0.0091792, upper bound: 0.0091792
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.05
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

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091740
time: 1.95 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.22 seconds

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

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091784
time: 1.87 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.18 seconds

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

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091738
time: 2.58 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.35 seconds

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

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091740
time: 2.36 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.23 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.52 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.52
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091740
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.52
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.52
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091784
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.52
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.52
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091738
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.52
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.52
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091740
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.52
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

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.29 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.41 seconds

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

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.36 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.60 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0032581, 0.0035306, -0.0032433, 0.0038140, -0.0070720, 0.0067738
1: 0.9884027, 1.0027788, 0.9878025, 1.0027474, -0.0143448, 0.0149763
2: -0.0096356, 0.0009190, -0.0096282, 0.0018978, -0.0115334, 0.0105471
3: -0.0013880, 0.0071051, -0.0013695, 0.0074596, -0.0088476, 0.0084746
4: -0.0028111, 0.0117216, -0.0030829, 0.0126982, -0.0155093, 0.0148045
5: -0.0043583, 0.0117286, -0.0043232, 0.0124002, -0.0167585, 0.0160518
6: -0.0080048, 0.0078572, -0.0087723, 0.0078247, -0.0158295, 0.0166295
7: -0.0103190, -0.0034378, -0.0106063, -0.0034528, -0.0068662, 0.0071685
8: -0.0029558, 0.0086255, -0.0042419, 0.0086235, -0.0115793, 0.0128674
9: -0.0071708, 0.0026555, -0.0075810, 0.0026341, -0.0098048, 0.0102365

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.74 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.82 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0033526, 0.0034700, -0.0032404, 0.0037745, -0.0071271, 0.0067104
1: 0.9885309, 1.0029787, 0.9878860, 1.0027412, -0.0142103, 0.0150927
2: -0.0096828, 0.0007099, -0.0096267, 0.0017616, -0.0114445, 0.0103366
3: -0.0015062, 0.0070293, -0.0013659, 0.0074103, -0.0089165, 0.0083952
4: -0.0029652, 0.0115129, -0.0029753, 0.0125623, -0.0155275, 0.0144882
5: -0.0045822, 0.0115852, -0.0043164, 0.0123068, -0.0168890, 0.0159015
6: -0.0078408, 0.0080642, -0.0086655, 0.0078184, -0.0156592, 0.0167297
7: -0.0102577, -0.0033420, -0.0105663, -0.0034557, -0.0068019, 0.0072243
8: -0.0026810, 0.0086381, -0.0040629, 0.0086231, -0.0113041, 0.0127010
9: -0.0070831, 0.0027922, -0.0075239, 0.0026299, -0.0097130, 0.0103161

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.38 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.37 seconds

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

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 4.98 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.07 seconds

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

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.32 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.37 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0032282, 0.0036785, -0.0032433, 0.0038140, -0.0070421, 0.0069218
1: 0.9880894, 1.0027153, 0.9878025, 1.0027474, -0.0146580, 0.0149128
2: -0.0096206, 0.0014299, -0.0096282, 0.0018978, -0.0115185, 0.0110581
3: -0.0013506, 0.0072901, -0.0013695, 0.0074596, -0.0088102, 0.0086596
4: -0.0027623, 0.0122313, -0.0030829, 0.0126982, -0.0154605, 0.0153142
5: -0.0042874, 0.0120792, -0.0043232, 0.0124002, -0.0166876, 0.0164024
6: -0.0084054, 0.0077916, -0.0087723, 0.0078247, -0.0162301, 0.0165639
7: -0.0104690, -0.0034682, -0.0106063, -0.0034528, -0.0070162, 0.0071381
8: -0.0036271, 0.0086215, -0.0042419, 0.0086235, -0.0122506, 0.0128634
9: -0.0073849, 0.0026122, -0.0075810, 0.0026341, -0.0100189, 0.0101931

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.72 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.68 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0033256, 0.0036174, -0.0032404, 0.0037745, -0.0071001, 0.0068578
1: 0.9882188, 1.0029217, 0.9878860, 1.0027412, -0.0145224, 0.0150357
2: -0.0096693, 0.0012189, -0.0096267, 0.0017616, -0.0114310, 0.0108456
3: -0.0014725, 0.0072137, -0.0013659, 0.0074103, -0.0088827, 0.0085796
4: -0.0029211, 0.0120208, -0.0029753, 0.0125623, -0.0154834, 0.0149960
5: -0.0045182, 0.0119344, -0.0043164, 0.0123068, -0.0168250, 0.0162508
6: -0.0082400, 0.0080051, -0.0086655, 0.0078184, -0.0160583, 0.0166706
7: -0.0104070, -0.0033694, -0.0105663, -0.0034557, -0.0069513, 0.0071969
8: -0.0033499, 0.0086345, -0.0040629, 0.0086231, -0.0119729, 0.0126974
9: -0.0072964, 0.0027532, -0.0075239, 0.0026299, -0.0099263, 0.0102770

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.75 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
time: 2.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 7.56 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091360

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

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091601
time: 2.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091605
time: 2.34 seconds

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

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091600
time: 2.40 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091601
time: 2.66 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0033526, 0.0034700, -0.0032581, 0.0035306, -0.0068831, 0.0067281
1: 0.9885309, 1.0029787, 0.9884027, 1.0027788, -0.0142479, 0.0145760
2: -0.0096828, 0.0007099, -0.0096356, 0.0009190, -0.0106018, 0.0103454
3: -0.0015062, 0.0070293, -0.0013880, 0.0071051, -0.0086113, 0.0084173
4: -0.0029652, 0.0115129, -0.0028111, 0.0117216, -0.0146867, 0.0143240
5: -0.0045822, 0.0115852, -0.0043583, 0.0117286, -0.0163108, 0.0159434
6: -0.0078408, 0.0080642, -0.0080048, 0.0078572, -0.0156980, 0.0160690
7: -0.0102577, -0.0033420, -0.0103190, -0.0034378, -0.0068198, 0.0069770
8: -0.0026810, 0.0086381, -0.0029558, 0.0086255, -0.0113065, 0.0115939
9: -0.0070831, 0.0027922, -0.0071708, 0.0026555, -0.0097386, 0.0099630

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
time: 2.41 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.58 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0033526, 0.0034700, -0.0033526, 0.0034700, -0.0068226, 0.0068226
1: 0.9885309, 1.0029787, 0.9885309, 1.0029787, -0.0144478, 0.0144478
2: -0.0096828, 0.0007099, -0.0096828, 0.0007099, -0.0103927, 0.0103927
3: -0.0015062, 0.0070293, -0.0015062, 0.0070293, -0.0085355, 0.0085355
4: -0.0029652, 0.0115129, -0.0029652, 0.0115129, -0.0144781, 0.0144781
5: -0.0045822, 0.0115852, -0.0045822, 0.0115852, -0.0161674, 0.0161674
6: -0.0078408, 0.0080642, -0.0078408, 0.0080642, -0.0159050, 0.0159050
7: -0.0102577, -0.0033420, -0.0102577, -0.0033420, -0.0069156, 0.0069156
8: -0.0026810, 0.0086381, -0.0026810, 0.0086381, -0.0113191, 0.0113191
9: -0.0070831, 0.0027922, -0.0070831, 0.0027922, -0.0098754, 0.0098754

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
time: 2.56 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.36 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032581, 0.0035306, -0.0032282, 0.0036785, -0.0069366, 0.0067587
1: 0.9884027, 1.0027788, 0.9880894, 1.0027153, -0.0143127, 0.0146893
2: -0.0096356, 0.0009190, -0.0096206, 0.0014299, -0.0110655, 0.0105396
3: -0.0013880, 0.0071051, -0.0013506, 0.0072901, -0.0086781, 0.0084557
4: -0.0028111, 0.0117216, -0.0027623, 0.0122313, -0.0150424, 0.0144839
5: -0.0043583, 0.0117286, -0.0042874, 0.0120792, -0.0164375, 0.0160160
6: -0.0080048, 0.0078572, -0.0084054, 0.0077916, -0.0157964, 0.0162626
7: -0.0103190, -0.0034378, -0.0104690, -0.0034682, -0.0068509, 0.0070312
8: -0.0029558, 0.0086255, -0.0036271, 0.0086215, -0.0115772, 0.0122525
9: -0.0071708, 0.0026555, -0.0073849, 0.0026122, -0.0097829, 0.0100404

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091686
time: 2.16 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091684
time: 2.37 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032581, 0.0035306, -0.0033256, 0.0036174, -0.0068755, 0.0068561
1: 0.9884027, 1.0027788, 0.9882188, 1.0029217, -0.0145190, 0.0145600
2: -0.0096356, 0.0009190, -0.0096693, 0.0012189, -0.0108545, 0.0105883
3: -0.0013880, 0.0071051, -0.0014725, 0.0072137, -0.0086017, 0.0085775
4: -0.0028111, 0.0117216, -0.0029211, 0.0120208, -0.0148319, 0.0146427
5: -0.0043583, 0.0117286, -0.0045182, 0.0119344, -0.0162927, 0.0162469
6: -0.0080048, 0.0078572, -0.0082400, 0.0080051, -0.0160099, 0.0160971
7: -0.0103190, -0.0034378, -0.0104070, -0.0033694, -0.0069496, 0.0069692
8: -0.0029558, 0.0086255, -0.0033499, 0.0086345, -0.0115903, 0.0119753
9: -0.0071708, 0.0026555, -0.0072964, 0.0027532, -0.0099239, 0.0099519

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091689
time: 2.37 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091683
time: 2.38 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0033526, 0.0034700, -0.0032282, 0.0036785, -0.0070310, 0.0066982
1: 0.9885309, 1.0029787, 0.9880894, 1.0027153, -0.0141845, 0.0148892
2: -0.0096828, 0.0007099, -0.0096206, 0.0014299, -0.0111128, 0.0103305
3: -0.0015062, 0.0070293, -0.0013506, 0.0072901, -0.0087963, 0.0083799
4: -0.0029652, 0.0115129, -0.0027623, 0.0122313, -0.0151965, 0.0142752
5: -0.0045822, 0.0115852, -0.0042874, 0.0120792, -0.0166614, 0.0158726
6: -0.0078408, 0.0080642, -0.0084054, 0.0077916, -0.0156324, 0.0164696
7: -0.0102577, -0.0033420, -0.0104690, -0.0034682, -0.0067895, 0.0071269
8: -0.0026810, 0.0086381, -0.0036271, 0.0086215, -0.0113025, 0.0122652
9: -0.0070831, 0.0027922, -0.0073849, 0.0026122, -0.0096953, 0.0101771

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
time: 2.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.26 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0033526, 0.0034700, -0.0033256, 0.0036174, -0.0069700, 0.0067956
1: 0.9885309, 1.0029787, 0.9882188, 1.0029217, -0.0143908, 0.0147599
2: -0.0096828, 0.0007099, -0.0096693, 0.0012189, -0.0109017, 0.0103792
3: -0.0015062, 0.0070293, -0.0014725, 0.0072137, -0.0087199, 0.0085018
4: -0.0029652, 0.0115129, -0.0029211, 0.0120208, -0.0149860, 0.0144341
5: -0.0045822, 0.0115852, -0.0045182, 0.0119344, -0.0165166, 0.0161034
6: -0.0078408, 0.0080642, -0.0082400, 0.0080051, -0.0158459, 0.0163042
7: -0.0102577, -0.0033420, -0.0104070, -0.0033694, -0.0068883, 0.0070650
8: -0.0026810, 0.0086381, -0.0033499, 0.0086345, -0.0113155, 0.0119879
9: -0.0070831, 0.0027922, -0.0072964, 0.0027532, -0.0098363, 0.0100887

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
time: 2.53 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.37 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032282, 0.0036785, -0.0032581, 0.0035306, -0.0067587, 0.0069366
1: 0.9880894, 1.0027153, 0.9884027, 1.0027788, -0.0146893, 0.0143127
2: -0.0096206, 0.0014299, -0.0096356, 0.0009190, -0.0105396, 0.0110655
3: -0.0013506, 0.0072901, -0.0013880, 0.0071051, -0.0084557, 0.0086781
4: -0.0027623, 0.0122313, -0.0028111, 0.0117216, -0.0144839, 0.0150424
5: -0.0042874, 0.0120792, -0.0043583, 0.0117286, -0.0160160, 0.0164375
6: -0.0084054, 0.0077916, -0.0080048, 0.0078572, -0.0162626, 0.0157964
7: -0.0104690, -0.0034682, -0.0103190, -0.0034378, -0.0070312, 0.0068509
8: -0.0036271, 0.0086215, -0.0029558, 0.0086255, -0.0122525, 0.0115772
9: -0.0073849, 0.0026122, -0.0071708, 0.0026555, -0.0100404, 0.0097829

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091268
time: 2.20 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091214
time: 3.43 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032282, 0.0036785, -0.0033526, 0.0034700, -0.0066982, 0.0070310
1: 0.9880894, 1.0027153, 0.9885309, 1.0029787, -0.0148892, 0.0141845
2: -0.0096206, 0.0014299, -0.0096828, 0.0007099, -0.0103305, 0.0111128
3: -0.0013506, 0.0072901, -0.0015062, 0.0070293, -0.0083799, 0.0087963
4: -0.0027623, 0.0122313, -0.0029652, 0.0115129, -0.0142752, 0.0151965
5: -0.0042874, 0.0120792, -0.0045822, 0.0115852, -0.0158726, 0.0166614
6: -0.0084054, 0.0077916, -0.0078408, 0.0080642, -0.0164696, 0.0156324
7: -0.0104690, -0.0034682, -0.0102577, -0.0033420, -0.0071269, 0.0067895
8: -0.0036271, 0.0086215, -0.0026810, 0.0086381, -0.0122652, 0.0113025
9: -0.0073849, 0.0026122, -0.0070831, 0.0027922, -0.0101771, 0.0096953

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091265
time: 4.47 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091211
time: 2.11 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0033256, 0.0036174, -0.0032581, 0.0035306, -0.0068561, 0.0068755
1: 0.9882188, 1.0029217, 0.9884027, 1.0027788, -0.0145600, 0.0145190
2: -0.0096693, 0.0012189, -0.0096356, 0.0009190, -0.0105883, 0.0108545
3: -0.0014725, 0.0072137, -0.0013880, 0.0071051, -0.0085775, 0.0086017
4: -0.0029211, 0.0120208, -0.0028111, 0.0117216, -0.0146427, 0.0148319
5: -0.0045182, 0.0119344, -0.0043583, 0.0117286, -0.0162469, 0.0162927
6: -0.0082400, 0.0080051, -0.0080048, 0.0078572, -0.0160971, 0.0160099
7: -0.0104070, -0.0033694, -0.0103190, -0.0034378, -0.0069692, 0.0069496
8: -0.0033499, 0.0086345, -0.0029558, 0.0086255, -0.0119753, 0.0115903
9: -0.0072964, 0.0027532, -0.0071708, 0.0026555, -0.0099519, 0.0099239

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
time: 2.28 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.61 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0033256, 0.0036174, -0.0033526, 0.0034700, -0.0067956, 0.0069700
1: 0.9882188, 1.0029217, 0.9885309, 1.0029787, -0.0147599, 0.0143908
2: -0.0096693, 0.0012189, -0.0096828, 0.0007099, -0.0103792, 0.0109017
3: -0.0014725, 0.0072137, -0.0015062, 0.0070293, -0.0085018, 0.0087199
4: -0.0029211, 0.0120208, -0.0029652, 0.0115129, -0.0144341, 0.0149860
5: -0.0045182, 0.0119344, -0.0045822, 0.0115852, -0.0161034, 0.0165166
6: -0.0082400, 0.0080051, -0.0078408, 0.0080642, -0.0163042, 0.0158459
7: -0.0104070, -0.0033694, -0.0102577, -0.0033420, -0.0070650, 0.0068883
8: -0.0033499, 0.0086345, -0.0026810, 0.0086381, -0.0119879, 0.0113155
9: -0.0072964, 0.0027532, -0.0070831, 0.0027922, -0.0100887, 0.0098363

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
time: 2.15 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.60 seconds

## BFS NS instance: NS_A2_B2_A1_B1

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

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091268
time: 2.17 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091215
time: 2.22 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032282, 0.0036785, -0.0033256, 0.0036174, -0.0068456, 0.0070041
1: 0.9880894, 1.0027153, 0.9882188, 1.0029217, -0.0148323, 0.0144966
2: -0.0096206, 0.0014299, -0.0096693, 0.0012189, -0.0108395, 0.0110993
3: -0.0013506, 0.0072901, -0.0014725, 0.0072137, -0.0085643, 0.0087626
4: -0.0027623, 0.0122313, -0.0029211, 0.0120208, -0.0147831, 0.0151525
5: -0.0042874, 0.0120792, -0.0045182, 0.0119344, -0.0162218, 0.0165974
6: -0.0084054, 0.0077916, -0.0082400, 0.0080051, -0.0164105, 0.0160316
7: -0.0104690, -0.0034682, -0.0104070, -0.0033694, -0.0070996, 0.0069389
8: -0.0036271, 0.0086215, -0.0033499, 0.0086345, -0.0122616, 0.0119713
9: -0.0073849, 0.0026122, -0.0072964, 0.0027532, -0.0101380, 0.0099086

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091265
time: 1.99 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091215
time: 2.61 seconds

## BFS NS instance: NS_A2_B2_A2_B1

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

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
time: 2.31 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.53 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0033256, 0.0036174, -0.0033256, 0.0036174, -0.0069430, 0.0069430
1: 0.9882188, 1.0029217, 0.9882188, 1.0029217, -0.0147029, 0.0147029
2: -0.0096693, 0.0012189, -0.0096693, 0.0012189, -0.0108882, 0.0108882
3: -0.0014725, 0.0072137, -0.0014725, 0.0072137, -0.0086862, 0.0086862
4: -0.0029211, 0.0120208, -0.0029211, 0.0120208, -0.0149419, 0.0149419
5: -0.0045182, 0.0119344, -0.0045182, 0.0119344, -0.0164526, 0.0164526
6: -0.0082400, 0.0080051, -0.0082400, 0.0080051, -0.0162450, 0.0162450
7: -0.0104070, -0.0033694, -0.0104070, -0.0033694, -0.0070377, 0.0070377
8: -0.0033499, 0.0086345, -0.0033499, 0.0086345, -0.0119843, 0.0119843
9: -0.0072964, 0.0027532, -0.0072964, 0.0027532, -0.0100496, 0.0100496

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
time: 2.08 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.22 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091601
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091605
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091600
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091601
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091686
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091684
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091689
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0091360, upper bound: 0.0091683
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
NS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
NS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091268
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091214
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091265
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091211
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091268
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091215
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091265
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091215
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090987
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.22
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842

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

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091400, upper bound: 0.0091316
time: 2.36 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091316
time: 2.26 seconds

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

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091400, upper bound: 0.0091307
time: 2.88 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091307
time: 2.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0032310, 0.0033685, -0.0033524, 0.0034689, -0.0066999, 0.0067209
1: 0.9887458, 1.0027213, 0.9885333, 1.0029784, -0.0142326, 0.0141881
2: -0.0096220, 0.0003593, -0.0096828, 0.0007060, -0.0103280, 0.0100420
3: -0.0013541, 0.0069024, -0.0015060, 0.0070279, -0.0083820, 0.0084083
4: -0.0027669, 0.0111632, -0.0029649, 0.0115090, -0.0142759, 0.0141280
5: -0.0042941, 0.0113447, -0.0045818, 0.0115825, -0.0158766, 0.0159264
6: -0.0075659, 0.0077978, -0.0078377, 0.0080638, -0.0156297, 0.0156355
7: -0.0101548, -0.0034653, -0.0102565, -0.0033422, -0.0068125, 0.0067912
8: -0.0022204, 0.0086218, -0.0026759, 0.0086380, -0.0108585, 0.0112977
9: -0.0069362, 0.0026162, -0.0070815, 0.0027920, -0.0097282, 0.0096977

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091103
time: 2.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091048
time: 2.15 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0032682, 0.0033685, -0.0033504, 0.0034513, -0.0067196, 0.0067189
1: 0.9887459, 1.0028001, 0.9885704, 1.0029744, -0.0142285, 0.0142297
2: -0.0096406, 0.0003593, -0.0096818, 0.0006453, -0.0102860, 0.0100410
3: -0.0014007, 0.0069024, -0.0015035, 0.0070060, -0.0084066, 0.0084058
4: -0.0028276, 0.0111631, -0.0029617, 0.0114486, -0.0142762, 0.0141248
5: -0.0043823, 0.0113447, -0.0045771, 0.0115409, -0.0159232, 0.0159218
6: -0.0075659, 0.0078794, -0.0077902, 0.0080595, -0.0156254, 0.0156696
7: -0.0101548, -0.0034276, -0.0102387, -0.0033442, -0.0068106, 0.0068112
8: -0.0022204, 0.0086268, -0.0025963, 0.0086378, -0.0108582, 0.0112231
9: -0.0069362, 0.0026701, -0.0070561, 0.0027891, -0.0097253, 0.0097262

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091108
time: 2.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091051
time: 2.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0032692, 0.0034214, -0.0032581, 0.0035306, -0.0067997, 0.0066794
1: 0.9886339, 1.0028023, 0.9884027, 1.0027788, -0.0141449, 0.0143996
2: -0.0096411, 0.0005418, -0.0096356, 0.0009190, -0.0105601, 0.0101774
3: -0.0014019, 0.0069685, -0.0013880, 0.0071051, -0.0085070, 0.0083565
4: -0.0028292, 0.0113452, -0.0028111, 0.0117216, -0.0145508, 0.0141563
5: -0.0043846, 0.0114699, -0.0043583, 0.0117286, -0.0161133, 0.0158282
6: -0.0077090, 0.0078815, -0.0080048, 0.0078572, -0.0155662, 0.0158863
7: -0.0102083, -0.0034265, -0.0103190, -0.0034378, -0.0067705, 0.0068925
8: -0.0024602, 0.0086269, -0.0029558, 0.0086255, -0.0110857, 0.0115827
9: -0.0070127, 0.0026716, -0.0071708, 0.0026555, -0.0096682, 0.0098423

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091108, upper bound: 0.0090987
time: 2.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091108, upper bound: 0.0090986
time: 2.66 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0032692, 0.0034214, -0.0033526, 0.0034700, -0.0067392, 0.0067739
1: 0.9886339, 1.0028023, 0.9885309, 1.0029787, -0.0143448, 0.0142714
2: -0.0096411, 0.0005418, -0.0096828, 0.0007099, -0.0103510, 0.0102246
3: -0.0014019, 0.0069685, -0.0015062, 0.0070293, -0.0084313, 0.0084747
4: -0.0028292, 0.0113452, -0.0029652, 0.0115129, -0.0143421, 0.0143104
5: -0.0043846, 0.0114699, -0.0045822, 0.0115852, -0.0159698, 0.0160521
6: -0.0077090, 0.0078815, -0.0078408, 0.0080642, -0.0157732, 0.0157223
7: -0.0102083, -0.0034265, -0.0102577, -0.0033420, -0.0068663, 0.0068311
8: -0.0024602, 0.0086269, -0.0026810, 0.0086381, -0.0110983, 0.0113080
9: -0.0070127, 0.0026716, -0.0070831, 0.0027922, -0.0098049, 0.0097547

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090841
time: 2.25 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0032310, 0.0033685, -0.0032280, 0.0036774, -0.0069083, 0.0065965
1: 0.9887458, 1.0027213, 0.9880918, 1.0027149, -0.0139691, 0.0146295
2: -0.0096220, 0.0003593, -0.0096205, 0.0014261, -0.0110481, 0.0099798
3: -0.0013541, 0.0069024, -0.0013503, 0.0072888, -0.0086428, 0.0082527
4: -0.0027669, 0.0111632, -0.0027620, 0.0122275, -0.0149944, 0.0139252
5: -0.0042941, 0.0113447, -0.0042870, 0.0120765, -0.0163706, 0.0156316
6: -0.0075659, 0.0077978, -0.0084024, 0.0077912, -0.0153571, 0.0162002
7: -0.0101548, -0.0034653, -0.0104678, -0.0034683, -0.0066864, 0.0070026
8: -0.0022204, 0.0086218, -0.0036221, 0.0086214, -0.0108419, 0.0122439
9: -0.0069362, 0.0026162, -0.0073833, 0.0026119, -0.0095481, 0.0099995

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091399, upper bound: 0.0091316
time: 2.47 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091316
time: 2.49 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0032682, 0.0033685, -0.0032259, 0.0036597, -0.0069279, 0.0065945
1: 0.9887459, 1.0028001, 0.9881293, 1.0027107, -0.0139648, 0.0146708
2: -0.0096406, 0.0003593, -0.0096195, 0.0013650, -0.0110056, 0.0099788
3: -0.0014007, 0.0069024, -0.0013478, 0.0072666, -0.0086673, 0.0082501
4: -0.0028276, 0.0111631, -0.0027587, 0.0121665, -0.0149941, 0.0139218
5: -0.0043823, 0.0113447, -0.0042821, 0.0120346, -0.0164169, 0.0156268
6: -0.0075659, 0.0078794, -0.0083545, 0.0077867, -0.0153526, 0.0162339
7: -0.0101548, -0.0034276, -0.0104499, -0.0034704, -0.0066844, 0.0070224
8: -0.0022204, 0.0086268, -0.0035418, 0.0086212, -0.0108416, 0.0121686
9: -0.0069362, 0.0026701, -0.0073577, 0.0026090, -0.0095452, 0.0100278

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091399, upper bound: 0.0091307
time: 2.40 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091307
time: 2.51 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0032310, 0.0033685, -0.0033254, 0.0036162, -0.0068472, 0.0066939
1: 0.9887458, 1.0027213, 0.9882212, 1.0029212, -0.0141754, 0.0145001
2: -0.0096220, 0.0003593, -0.0096692, 0.0012149, -0.0108369, 0.0100285
3: -0.0013541, 0.0069024, -0.0014722, 0.0072123, -0.0085664, 0.0083746
4: -0.0027669, 0.0111632, -0.0029208, 0.0120168, -0.0147837, 0.0140840
5: -0.0042941, 0.0113447, -0.0045178, 0.0119317, -0.0162257, 0.0158624
6: -0.0075659, 0.0077978, -0.0082369, 0.0080046, -0.0155706, 0.0160346
7: -0.0101548, -0.0034653, -0.0104059, -0.0033696, -0.0067852, 0.0069406
8: -0.0022204, 0.0086218, -0.0033446, 0.0086344, -0.0108549, 0.0119665
9: -0.0069362, 0.0026162, -0.0072948, 0.0027529, -0.0096891, 0.0099110

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091208
time: 2.52 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091159
time: 2.35 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0032682, 0.0033685, -0.0033234, 0.0035993, -0.0068675, 0.0066920
1: 0.9887459, 1.0028001, 0.9882572, 1.0029171, -0.0141712, 0.0145429
2: -0.0096406, 0.0003593, -0.0096683, 0.0011563, -0.0107970, 0.0100275
3: -0.0014007, 0.0069024, -0.0014697, 0.0071911, -0.0085917, 0.0083721
4: -0.0028276, 0.0111631, -0.0029177, 0.0119584, -0.0147860, 0.0140808
5: -0.0043823, 0.0113447, -0.0045131, 0.0118915, -0.0162738, 0.0158578
6: -0.0075659, 0.0078794, -0.0081909, 0.0080003, -0.0155663, 0.0160703
7: -0.0101548, -0.0034276, -0.0103887, -0.0033716, -0.0067832, 0.0069611
8: -0.0022204, 0.0086268, -0.0032677, 0.0086342, -0.0108546, 0.0118945
9: -0.0069362, 0.0026701, -0.0072702, 0.0027501, -0.0096863, 0.0099404

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091206
time: 2.43 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091158
time: 2.35 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0032692, 0.0034214, -0.0032282, 0.0036785, -0.0069477, 0.0066495
1: 0.9886339, 1.0028023, 0.9880894, 1.0027153, -0.0140815, 0.0147128
2: -0.0096411, 0.0005418, -0.0096206, 0.0014299, -0.0110711, 0.0101624
3: -0.0014019, 0.0069685, -0.0013506, 0.0072901, -0.0086921, 0.0083191
4: -0.0028292, 0.0113452, -0.0027623, 0.0122313, -0.0150605, 0.0141076
5: -0.0043846, 0.0114699, -0.0042874, 0.0120792, -0.0164638, 0.0157573
6: -0.0077090, 0.0078815, -0.0084054, 0.0077916, -0.0155006, 0.0162870
7: -0.0102083, -0.0034265, -0.0104690, -0.0034682, -0.0067402, 0.0070424
8: -0.0024602, 0.0086269, -0.0036271, 0.0086215, -0.0110817, 0.0122540
9: -0.0070127, 0.0026716, -0.0073849, 0.0026122, -0.0096249, 0.0100564

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091215, upper bound: 0.0090842
time: 2.99 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091215, upper bound: 0.0090842
time: 2.65 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0032692, 0.0034214, -0.0033256, 0.0036174, -0.0068866, 0.0067469
1: 0.9886339, 1.0028023, 0.9882188, 1.0029217, -0.0142878, 0.0145835
2: -0.0096411, 0.0005418, -0.0096693, 0.0012189, -0.0108600, 0.0102111
3: -0.0014019, 0.0069685, -0.0014725, 0.0072137, -0.0086156, 0.0084409
4: -0.0028292, 0.0113452, -0.0029211, 0.0120208, -0.0148500, 0.0142664
5: -0.0043846, 0.0114699, -0.0045182, 0.0119344, -0.0163190, 0.0159881
6: -0.0077090, 0.0078815, -0.0082400, 0.0080051, -0.0157141, 0.0161215
7: -0.0102083, -0.0034265, -0.0104070, -0.0033694, -0.0068389, 0.0069805
8: -0.0024602, 0.0086269, -0.0033499, 0.0086345, -0.0110947, 0.0119768
9: -0.0070127, 0.0026716, -0.0072964, 0.0027532, -0.0097659, 0.0099680

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.14 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.39 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0031451, 0.0036252, -0.0032581, 0.0035306, -0.0066757, 0.0068832
1: 0.9882023, 1.0025395, 0.9884027, 1.0027788, -0.0145764, 0.0141369
2: -0.0095791, 0.0012458, -0.0096356, 0.0009190, -0.0104980, 0.0108813
3: -0.0012467, 0.0072234, -0.0013880, 0.0071051, -0.0083518, 0.0086114
4: -0.0026269, 0.0120476, -0.0028111, 0.0117216, -0.0143485, 0.0148587
5: -0.0040906, 0.0119528, -0.0043583, 0.0117286, -0.0158193, 0.0163111
6: -0.0082610, 0.0076097, -0.0080048, 0.0078572, -0.0161182, 0.0156145
7: -0.0104149, -0.0035523, -0.0103190, -0.0034378, -0.0069771, 0.0067667
8: -0.0033851, 0.0086104, -0.0029558, 0.0086255, -0.0120106, 0.0115662
9: -0.0073077, 0.0024920, -0.0071708, 0.0026555, -0.0099632, 0.0096628

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091399
time: 2.57 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091399
time: 2.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0031564, 0.0037983, -0.0032470, 0.0035249, -0.0066813, 0.0070453
1: 0.9878358, 1.0025636, 0.9884147, 1.0027554, -0.0149196, 0.0141489
2: -0.0095847, 0.0018437, -0.0096301, 0.0008993, -0.0104841, 0.0114738
3: -0.0012609, 0.0074400, -0.0013742, 0.0070980, -0.0083588, 0.0088142
4: -0.0030402, 0.0126442, -0.0027931, 0.0117020, -0.0147421, 0.0154373
5: -0.0041175, 0.0123631, -0.0043321, 0.0117152, -0.0158327, 0.0166952
6: -0.0087299, 0.0076345, -0.0079894, 0.0078330, -0.0165629, 0.0156239
7: -0.0105904, -0.0035408, -0.0103133, -0.0034490, -0.0071414, 0.0067724
8: -0.0041708, 0.0086119, -0.0029300, 0.0086240, -0.0127948, 0.0115419
9: -0.0075583, 0.0025084, -0.0071625, 0.0026395, -0.0101978, 0.0096709

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091307
time: 2.21 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091307
time: 2.51 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0031451, 0.0036252, -0.0033526, 0.0034700, -0.0066151, 0.0069777
1: 0.9882023, 1.0025395, 0.9885309, 1.0029787, -0.0147763, 0.0140086
2: -0.0095791, 0.0012458, -0.0096828, 0.0007099, -0.0102889, 0.0109286
3: -0.0012467, 0.0072234, -0.0015062, 0.0070293, -0.0082761, 0.0087296
4: -0.0026269, 0.0120476, -0.0029652, 0.0115129, -0.0141398, 0.0150128
5: -0.0040906, 0.0119528, -0.0045822, 0.0115852, -0.0156758, 0.0165350
6: -0.0082610, 0.0076097, -0.0078408, 0.0080642, -0.0163252, 0.0154505
7: -0.0104149, -0.0035523, -0.0102577, -0.0033420, -0.0070729, 0.0067054
8: -0.0033851, 0.0086104, -0.0026810, 0.0086381, -0.0120232, 0.0112914
9: -0.0073077, 0.0024920, -0.0070831, 0.0027922, -0.0100999, 0.0095751

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091215
time: 2.87 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091215
time: 2.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0031564, 0.0037983, -0.0033414, 0.0034642, -0.0066206, 0.0071397
1: 0.9878358, 1.0025636, 0.9885432, 1.0029553, -0.0151196, 0.0140204
2: -0.0095847, 0.0018437, -0.0096773, 0.0006898, -0.0102745, 0.0115210
3: -0.0012609, 0.0074400, -0.0014923, 0.0070221, -0.0082829, 0.0089323
4: -0.0030402, 0.0126442, -0.0029470, 0.0114929, -0.0145330, 0.0155912
5: -0.0041175, 0.0123631, -0.0045558, 0.0115714, -0.0156889, 0.0169189
6: -0.0087299, 0.0076345, -0.0078251, 0.0080398, -0.0167697, 0.0154595
7: -0.0105904, -0.0035408, -0.0102518, -0.0033533, -0.0072371, 0.0067109
8: -0.0041708, 0.0086119, -0.0026546, 0.0086366, -0.0128074, 0.0112665
9: -0.0075583, 0.0025084, -0.0070747, 0.0027761, -0.0103344, 0.0095831

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091047
time: 2.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091053
time: 2.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0032423, 0.0035633, -0.0032581, 0.0035306, -0.0067729, 0.0068214
1: 0.9883333, 1.0027454, 0.9884027, 1.0027788, -0.0144454, 0.0143427
2: -0.0096277, 0.0010321, -0.0096356, 0.0009190, -0.0105467, 0.0106677
3: -0.0013683, 0.0071461, -0.0013880, 0.0071051, -0.0084734, 0.0085341
4: -0.0027854, 0.0118345, -0.0028111, 0.0117216, -0.0145070, 0.0146456
5: -0.0043210, 0.0118063, -0.0043583, 0.0117286, -0.0160496, 0.0161646
6: -0.0080935, 0.0078227, -0.0080048, 0.0078572, -0.0159507, 0.0158275
7: -0.0103522, -0.0034538, -0.0103190, -0.0034378, -0.0069144, 0.0068653
8: -0.0031045, 0.0086234, -0.0029558, 0.0086255, -0.0117300, 0.0115791
9: -0.0072182, 0.0026327, -0.0071708, 0.0026555, -0.0098737, 0.0098035

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091208, upper bound: 0.0090987
time: 3.12 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091206, upper bound: 0.0090987
time: 2.65 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0032423, 0.0035633, -0.0033526, 0.0034700, -0.0067123, 0.0069159
1: 0.9883333, 1.0027454, 0.9885309, 1.0029787, -0.0146453, 0.0142145
2: -0.0096277, 0.0010321, -0.0096828, 0.0007099, -0.0103375, 0.0107150
3: -0.0013683, 0.0071461, -0.0015062, 0.0070293, -0.0083977, 0.0086523
4: -0.0027854, 0.0118345, -0.0029652, 0.0115129, -0.0142983, 0.0147997
5: -0.0043210, 0.0118063, -0.0045822, 0.0115852, -0.0159061, 0.0163885
6: -0.0080935, 0.0078227, -0.0078408, 0.0080642, -0.0161577, 0.0156635
7: -0.0103522, -0.0034538, -0.0102577, -0.0033420, -0.0070102, 0.0068039
8: -0.0031045, 0.0086234, -0.0026810, 0.0086381, -0.0117426, 0.0113044
9: -0.0072182, 0.0026327, -0.0070831, 0.0027922, -0.0100104, 0.0097158

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.22 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.63 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

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

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091316
time: 1.92 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091316
time: 1.83 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

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

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091307
time: 2.30 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091307
time: 2.46 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0031451, 0.0036252, -0.0033256, 0.0036174, -0.0067625, 0.0069507
1: 0.9882023, 1.0025395, 0.9882188, 1.0029217, -0.0147194, 0.0143207
2: -0.0095791, 0.0012458, -0.0096693, 0.0012189, -0.0107980, 0.0109151
3: -0.0012467, 0.0072234, -0.0014725, 0.0072137, -0.0084604, 0.0086959
4: -0.0026269, 0.0120476, -0.0029211, 0.0120208, -0.0146477, 0.0149687
5: -0.0040906, 0.0119528, -0.0045182, 0.0119344, -0.0160250, 0.0164711
6: -0.0082610, 0.0076097, -0.0082400, 0.0080051, -0.0162661, 0.0158497
7: -0.0104149, -0.0035523, -0.0104070, -0.0033694, -0.0070455, 0.0068547
8: -0.0033851, 0.0086104, -0.0033499, 0.0086345, -0.0120196, 0.0119602
9: -0.0073077, 0.0024920, -0.0072964, 0.0027532, -0.0100609, 0.0097884

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091212
time: 2.38 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091214
time: 2.45 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0031564, 0.0037983, -0.0033144, 0.0036107, -0.0067671, 0.0071127
1: 0.9878358, 1.0025636, 0.9882330, 1.0028979, -0.0150621, 0.0143306
2: -0.0095847, 0.0018437, -0.0096637, 0.0011957, -0.0107805, 0.0115075
3: -0.0012609, 0.0074400, -0.0014584, 0.0072053, -0.0084662, 0.0088985
4: -0.0030402, 0.0126442, -0.0029029, 0.0119977, -0.0150379, 0.0155471
5: -0.0041175, 0.0123631, -0.0044917, 0.0119185, -0.0160360, 0.0168547
6: -0.0087299, 0.0076345, -0.0082218, 0.0079805, -0.0167104, 0.0158563
7: -0.0105904, -0.0035408, -0.0104003, -0.0033807, -0.0072097, 0.0068594
8: -0.0041708, 0.0086119, -0.0033195, 0.0086330, -0.0128038, 0.0119313
9: -0.0075583, 0.0025084, -0.0072867, 0.0027369, -0.0102952, 0.0097951

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091048
time: 2.28 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091054
time: 2.13 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

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

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091284, upper bound: 0.0090842
time: 3.19 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091284, upper bound: 0.0090842
time: 2.56 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0032423, 0.0035633, -0.0033256, 0.0036174, -0.0068597, 0.0068889
1: 0.9883333, 1.0027454, 0.9882188, 1.0029217, -0.0145884, 0.0145266
2: -0.0096277, 0.0010321, -0.0096693, 0.0012189, -0.0108466, 0.0107015
3: -0.0013683, 0.0071461, -0.0014725, 0.0072137, -0.0085820, 0.0086185
4: -0.0027854, 0.0118345, -0.0029211, 0.0120208, -0.0148062, 0.0147556
5: -0.0043210, 0.0118063, -0.0045182, 0.0119344, -0.0162554, 0.0163245
6: -0.0080935, 0.0078227, -0.0082400, 0.0080051, -0.0160986, 0.0160626
7: -0.0103522, -0.0034538, -0.0104070, -0.0033694, -0.0069829, 0.0069533
8: -0.0031045, 0.0086234, -0.0033499, 0.0086345, -0.0117390, 0.0119732
9: -0.0072182, 0.0026327, -0.0072964, 0.0027532, -0.0099714, 0.0099291

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
time: 2.40 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.89 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091400, upper bound: 0.0091316
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091316
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091400, upper bound: 0.0091307
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091307
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091103
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091048
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091108
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091051
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091108, upper bound: 0.0090987
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091108, upper bound: 0.0090986
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090841
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091399, upper bound: 0.0091316
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091316
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091399, upper bound: 0.0091307
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091307
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091208
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091159
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090987, upper bound: 0.0091206
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091158
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091215, upper bound: 0.0090842
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091215, upper bound: 0.0090842
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091399
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091399
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091307
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091307
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091215
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091215
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091047
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091053
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091208, upper bound: 0.0090987
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091206, upper bound: 0.0090987
NS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091316
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091316
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091307
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091307
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091212
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091214
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0091048
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091054
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091284, upper bound: 0.0090842
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0091284, upper bound: 0.0090842
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.89
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090842

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032310, 0.0033685, -0.0031742, 0.0034781, -0.0067090, 0.0065427
1: 0.9887458, 1.0027213, 0.9885138, 1.0026010, -0.0138552, 0.0142075
2: -0.0096220, 0.0003593, -0.0095936, 0.0007377, -0.0103597, 0.0099529
3: -0.0013541, 0.0069024, -0.0012831, 0.0070394, -0.0083935, 0.0081854
4: -0.0027669, 0.0111632, -0.0026743, 0.0115407, -0.0143076, 0.0138374
5: -0.0042941, 0.0113447, -0.0041595, 0.0116043, -0.0158983, 0.0155042
6: -0.0075659, 0.0077978, -0.0078626, 0.0076733, -0.0152392, 0.0156604
7: -0.0101548, -0.0034653, -0.0102658, -0.0035229, -0.0066319, 0.0068006
8: -0.0022204, 0.0086218, -0.0027176, 0.0086143, -0.0108347, 0.0113394
9: -0.0069362, 0.0026162, -0.0070948, 0.0025340, -0.0094702, 0.0097110

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091316
time: 2.17 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091316
time: 2.41 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032199, 0.0033628, -0.0031884, 0.0036593, -0.0068791, 0.0065512
1: 0.9887579, 1.0026977, 0.9881302, 1.0026312, -0.0138733, 0.0145675
2: -0.0096165, 0.0003396, -0.0096007, 0.0013635, -0.0109799, 0.0099404
3: -0.0013402, 0.0068952, -0.0013008, 0.0072661, -0.0086063, 0.0081961
4: -0.0027488, 0.0111436, -0.0026975, 0.0121650, -0.0149138, 0.0138410
5: -0.0042677, 0.0113312, -0.0041932, 0.0120336, -0.0163013, 0.0155244
6: -0.0075505, 0.0077734, -0.0083533, 0.0077045, -0.0152550, 0.0161268
7: -0.0101490, -0.0034765, -0.0104495, -0.0035084, -0.0066406, 0.0069729
8: -0.0021946, 0.0086204, -0.0035398, 0.0086161, -0.0108108, 0.0121602
9: -0.0069280, 0.0026002, -0.0073570, 0.0025546, -0.0094826, 0.0099572

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090941, upper bound: 0.0090419
time: 2.26 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090941, upper bound: 0.0090956
time: 2.56 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0032682, 0.0033685, -0.0031721, 0.0034606, -0.0067288, 0.0065407
1: 0.9887459, 1.0028001, 0.9885507, 1.0025967, -0.0138509, 0.0142494
2: -0.0096406, 0.0003593, -0.0095926, 0.0006774, -0.0103180, 0.0099518
3: -0.0014007, 0.0069024, -0.0012805, 0.0070176, -0.0084182, 0.0081828
4: -0.0028276, 0.0111631, -0.0026710, 0.0114805, -0.0143081, 0.0138341
5: -0.0043823, 0.0113447, -0.0041546, 0.0115629, -0.0159452, 0.0154993
6: -0.0075659, 0.0078794, -0.0078154, 0.0076688, -0.0152347, 0.0156947
7: -0.0101548, -0.0034276, -0.0102481, -0.0035249, -0.0066298, 0.0068206
8: -0.0022204, 0.0086268, -0.0026384, 0.0086140, -0.0108344, 0.0112652
9: -0.0069362, 0.0026701, -0.0070695, 0.0025311, -0.0094673, 0.0097396

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091307
time: 2.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091307
time: 11.67 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0032567, 0.0033629, -0.0031863, 0.0036419, -0.0068986, 0.0065491
1: 0.9887578, 1.0027757, 0.9881669, 1.0026267, -0.0138689, 0.0146088
2: -0.0096349, 0.0003397, -0.0095997, 0.0013036, -0.0109385, 0.0099394
3: -0.0013863, 0.0068953, -0.0012982, 0.0072444, -0.0086307, 0.0081935
4: -0.0028089, 0.0111437, -0.0026940, 0.0121053, -0.0149142, 0.0138377
5: -0.0043550, 0.0113313, -0.0041881, 0.0119925, -0.0163476, 0.0155194
6: -0.0075506, 0.0078542, -0.0083064, 0.0076998, -0.0152504, 0.0161606
7: -0.0101490, -0.0034392, -0.0104319, -0.0035106, -0.0066384, 0.0069927
8: -0.0021948, 0.0086253, -0.0034611, 0.0086159, -0.0108106, 0.0120864
9: -0.0069280, 0.0026535, -0.0073319, 0.0025516, -0.0094796, 0.0099855

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090941, upper bound: 0.0090406
time: 2.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090941, upper bound: 0.0090941
time: 2.37 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

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

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091051
time: 2.20 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091047
time: 2.59 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032199, 0.0033628, -0.0032826, 0.0036052, -0.0068251, 0.0066454
1: 0.9887579, 1.0026977, 0.9882447, 1.0028306, -0.0140727, 0.0144531
2: -0.0096165, 0.0003396, -0.0096479, 0.0011768, -0.0107933, 0.0099875
3: -0.0013402, 0.0068952, -0.0014187, 0.0071985, -0.0085386, 0.0083139
4: -0.0027488, 0.0111436, -0.0028511, 0.0119788, -0.0147276, 0.0139946
5: -0.0042677, 0.0113312, -0.0044164, 0.0119055, -0.0161732, 0.0157476
6: -0.0075505, 0.0077734, -0.0082069, 0.0079109, -0.0154614, 0.0159804
7: -0.0101490, -0.0034765, -0.0103947, -0.0034130, -0.0067361, 0.0069182
8: -0.0021946, 0.0086204, -0.0032945, 0.0086287, -0.0108233, 0.0119149
9: -0.0069280, 0.0026002, -0.0072788, 0.0026910, -0.0096190, 0.0098790

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090494, upper bound: 0.0090181
time: 2.40 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090494, upper bound: 0.0090699
time: 2.39 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

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

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091048
time: 2.56 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091048
time: 2.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0032567, 0.0033629, -0.0032805, 0.0035881, -0.0068448, 0.0066434
1: 0.9887578, 1.0027757, 0.9882808, 1.0028262, -0.0140684, 0.0144949
2: -0.0096349, 0.0003397, -0.0096468, 0.0011178, -0.0107527, 0.0099865
3: -0.0013863, 0.0068953, -0.0014161, 0.0071771, -0.0085634, 0.0083113
4: -0.0028089, 0.0111437, -0.0028477, 0.0119199, -0.0147288, 0.0139913
5: -0.0043550, 0.0113313, -0.0044115, 0.0118650, -0.0162201, 0.0157427
6: -0.0075506, 0.0078542, -0.0081607, 0.0079063, -0.0154569, 0.0160148
7: -0.0101490, -0.0034392, -0.0103774, -0.0034151, -0.0067340, 0.0069382
8: -0.0021948, 0.0086253, -0.0032170, 0.0086284, -0.0108232, 0.0118423
9: -0.0069280, 0.0026535, -0.0072541, 0.0026880, -0.0096160, 0.0099076

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090494, upper bound: 0.0090182
time: 2.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090494, upper bound: 0.0090693
time: 2.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032690, 0.0034203, -0.0032310, 0.0033685, -0.0066375, 0.0066512
1: 0.9886363, 1.0028018, 0.9887458, 1.0027213, -0.0140850, 0.0140560
2: -0.0096410, 0.0005379, -0.0096220, 0.0003593, -0.0100003, 0.0101600
3: -0.0014017, 0.0069671, -0.0013541, 0.0069024, -0.0083040, 0.0083212
4: -0.0028289, 0.0113414, -0.0027669, 0.0111632, -0.0139921, 0.0141083
5: -0.0043842, 0.0114673, -0.0042941, 0.0113447, -0.0157288, 0.0157613
6: -0.0077060, 0.0078811, -0.0075659, 0.0077978, -0.0155038, 0.0154470
7: -0.0102072, -0.0034267, -0.0101548, -0.0034653, -0.0067419, 0.0067280
8: -0.0024552, 0.0086269, -0.0022204, 0.0086218, -0.0110770, 0.0108473
9: -0.0070111, 0.0026713, -0.0069362, 0.0026162, -0.0096273, 0.0096075

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091103, upper bound: 0.0090987
time: 2.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091103, upper bound: 0.0090986
time: 3.46 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032670, 0.0034030, -0.0032682, 0.0033685, -0.0066356, 0.0066712
1: 0.9886728, 1.0027976, 0.9887459, 1.0028001, -0.0141273, 0.0140517
2: -0.0096401, 0.0004784, -0.0096406, 0.0003593, -0.0099993, 0.0101190
3: -0.0013992, 0.0069455, -0.0014007, 0.0069024, -0.0083015, 0.0083461
4: -0.0028257, 0.0112820, -0.0028276, 0.0111631, -0.0139888, 0.0141096
5: -0.0043795, 0.0114264, -0.0043823, 0.0113447, -0.0157242, 0.0158087
6: -0.0076593, 0.0078768, -0.0075659, 0.0078794, -0.0155387, 0.0154427
7: -0.0101897, -0.0034287, -0.0101548, -0.0034276, -0.0067622, 0.0067260
8: -0.0023769, 0.0086266, -0.0022204, 0.0086268, -0.0110037, 0.0108470
9: -0.0069861, 0.0026684, -0.0069362, 0.0026701, -0.0096563, 0.0096047

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 131

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090793, upper bound: 0.0090941
time: 4.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090954, upper bound: 0.0090942
time: 4.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032310, 0.0033685, -0.0031449, 0.0036241, -0.0068550, 0.0065135
1: 0.9887458, 1.0027213, 0.9882047, 1.0025392, -0.0137933, 0.0145166
2: -0.0096220, 0.0003593, -0.0095790, 0.0012419, -0.0108639, 0.0099383
3: -0.0013541, 0.0069024, -0.0012465, 0.0072220, -0.0085761, 0.0081488
4: -0.0027669, 0.0111632, -0.0026266, 0.0120437, -0.0148106, 0.0137898
5: -0.0042941, 0.0113447, -0.0040902, 0.0119502, -0.0162443, 0.0154349
6: -0.0075659, 0.0077978, -0.0082580, 0.0076093, -0.0151752, 0.0160558
7: -0.0101548, -0.0034653, -0.0104138, -0.0035525, -0.0066023, 0.0069485
8: -0.0022204, 0.0086218, -0.0033801, 0.0086104, -0.0108308, 0.0120019
9: -0.0069362, 0.0026162, -0.0073061, 0.0024918, -0.0094280, 0.0099223

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091316
time: 2.40 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091316
time: 3.02 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032199, 0.0033628, -0.0031562, 0.0037972, -0.0070170, 0.0065191
1: 0.9887579, 1.0026977, 0.9878381, 1.0025631, -0.0138052, 0.0148596
2: -0.0096165, 0.0003396, -0.0095846, 0.0018398, -0.0114563, 0.0099243
3: -0.0013402, 0.0068952, -0.0012606, 0.0074386, -0.0087788, 0.0081559
4: -0.0027488, 0.0111436, -0.0030371, 0.0126403, -0.0153891, 0.0141806
5: -0.0042677, 0.0113312, -0.0041170, 0.0123604, -0.0166281, 0.0154482
6: -0.0075505, 0.0077734, -0.0087268, 0.0076341, -0.0151846, 0.0165003
7: -0.0101490, -0.0034765, -0.0105893, -0.0035410, -0.0066080, 0.0071127
8: -0.0021946, 0.0086204, -0.0041657, 0.0086119, -0.0108065, 0.0127860
9: -0.0069280, 0.0026002, -0.0075567, 0.0025081, -0.0094361, 0.0101568

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090941, upper bound: 0.0090419
time: 2.45 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090941, upper bound: 0.0090956
time: 2.58 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0032682, 0.0033685, -0.0031429, 0.0036063, -0.0068746, 0.0065114
1: 0.9887459, 1.0028001, 0.9882421, 1.0025349, -0.0137890, 0.0145580
2: -0.0096406, 0.0003593, -0.0095780, 0.0011807, -0.0108214, 0.0099372
3: -0.0014007, 0.0069024, -0.0012440, 0.0071999, -0.0086005, 0.0081463
4: -0.0028276, 0.0111631, -0.0026233, 0.0119827, -0.0148103, 0.0137865
5: -0.0043823, 0.0113447, -0.0040854, 0.0119082, -0.0162905, 0.0154300
6: -0.0075659, 0.0078794, -0.0082100, 0.0076048, -0.0151707, 0.0160894
7: -0.0101548, -0.0034276, -0.0103958, -0.0035545, -0.0066002, 0.0069683
8: -0.0022204, 0.0086268, -0.0032997, 0.0086101, -0.0108305, 0.0119265
9: -0.0069362, 0.0026701, -0.0072805, 0.0024888, -0.0094250, 0.0099506

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091307
time: 2.80 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091307
time: 2.54 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0032567, 0.0033629, -0.0031542, 0.0037799, -0.0070366, 0.0065170
1: 0.9887578, 1.0027757, 0.9878746, 1.0025587, -0.0138009, 0.0149011
2: -0.0096349, 0.0003397, -0.0095836, 0.0017802, -0.0114151, 0.0099234
3: -0.0013863, 0.0068953, -0.0012581, 0.0074170, -0.0088033, 0.0081533
4: -0.0028089, 0.0111437, -0.0029900, 0.0125808, -0.0153897, 0.0141336
5: -0.0043550, 0.0113313, -0.0041122, 0.0123195, -0.0166746, 0.0154434
6: -0.0075506, 0.0078542, -0.0086801, 0.0076296, -0.0151802, 0.0165343
7: -0.0101490, -0.0034392, -0.0105718, -0.0035431, -0.0066060, 0.0071326
8: -0.0021948, 0.0086253, -0.0040874, 0.0086116, -0.0108063, 0.0127126
9: -0.0069280, 0.0026535, -0.0075317, 0.0025051, -0.0094332, 0.0101852

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090941, upper bound: 0.0090406
time: 2.58 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090941, upper bound: 0.0090941
time: 2.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032310, 0.0033685, -0.0032421, 0.0035622, -0.0067932, 0.0066107
1: 0.9887458, 1.0027213, 0.9883357, 1.0027449, -0.0139991, 0.0143856
2: -0.0096220, 0.0003593, -0.0096276, 0.0010282, -0.0106503, 0.0099869
3: -0.0013541, 0.0069024, -0.0013681, 0.0071447, -0.0084987, 0.0082705
4: -0.0027669, 0.0111632, -0.0027851, 0.0118306, -0.0145975, 0.0139483
5: -0.0042941, 0.0113447, -0.0043206, 0.0118036, -0.0160977, 0.0156652
6: -0.0075659, 0.0077978, -0.0080905, 0.0078223, -0.0153882, 0.0158882
7: -0.0101548, -0.0034653, -0.0103511, -0.0034540, -0.0067008, 0.0068858
8: -0.0022204, 0.0086218, -0.0030994, 0.0086233, -0.0108438, 0.0117212
9: -0.0069362, 0.0026162, -0.0072166, 0.0026324, -0.0095686, 0.0098328

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091163
time: 2.41 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091159
time: 2.33 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032199, 0.0033628, -0.0032542, 0.0037413, -0.0069612, 0.0066170
1: 0.9887579, 1.0026977, 0.9879563, 1.0027705, -0.0140126, 0.0147414
2: -0.0096165, 0.0003396, -0.0096336, 0.0016469, -0.0112634, 0.0099733
3: -0.0013402, 0.0068952, -0.0013831, 0.0073687, -0.0087089, 0.0082784
4: -0.0027488, 0.0111436, -0.0028846, 0.0124478, -0.0151966, 0.0140282
5: -0.0042677, 0.0113312, -0.0043491, 0.0122281, -0.0164958, 0.0156803
6: -0.0075505, 0.0077734, -0.0085756, 0.0078487, -0.0153992, 0.0163490
7: -0.0101490, -0.0034765, -0.0105327, -0.0034417, -0.0067073, 0.0070561
8: -0.0021946, 0.0086204, -0.0039122, 0.0086249, -0.0108196, 0.0125326
9: -0.0069280, 0.0026002, -0.0074758, 0.0026499, -0.0095779, 0.0100760

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090494, upper bound: 0.0090284
time: 2.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090494, upper bound: 0.0090801
time: 2.58 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0032682, 0.0033685, -0.0032402, 0.0035455, -0.0068137, 0.0066087
1: 0.9887459, 1.0028001, 0.9883711, 1.0027409, -0.0139950, 0.0144290
2: -0.0096406, 0.0003593, -0.0096266, 0.0009705, -0.0106111, 0.0099859
3: -0.0014007, 0.0069024, -0.0013656, 0.0071237, -0.0085244, 0.0082679
4: -0.0028276, 0.0111631, -0.0027819, 0.0117730, -0.0146006, 0.0139450
5: -0.0043823, 0.0113447, -0.0043159, 0.0117640, -0.0161463, 0.0156606
6: -0.0075659, 0.0078794, -0.0080452, 0.0078180, -0.0153839, 0.0159246
7: -0.0101548, -0.0034276, -0.0103341, -0.0034560, -0.0066988, 0.0069066
8: -0.0022204, 0.0086268, -0.0030235, 0.0086231, -0.0108435, 0.0116503
9: -0.0069362, 0.0026701, -0.0071923, 0.0026296, -0.0095658, 0.0098625

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091158
time: 2.48 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091157
time: 2.46 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0032567, 0.0033629, -0.0032522, 0.0037244, -0.0069810, 0.0066151
1: 0.9887578, 1.0027757, 0.9879923, 1.0027664, -0.0140086, 0.0147834
2: -0.0096349, 0.0003397, -0.0096326, 0.0015883, -0.0112233, 0.0099724
3: -0.0013863, 0.0068953, -0.0013807, 0.0073475, -0.0087339, 0.0082759
4: -0.0028089, 0.0111437, -0.0028383, 0.0123894, -0.0151983, 0.0139820
5: -0.0043550, 0.0113313, -0.0043444, 0.0121879, -0.0165429, 0.0156756
6: -0.0075506, 0.0078542, -0.0085297, 0.0078443, -0.0153949, 0.0163838
7: -0.0101490, -0.0034392, -0.0105155, -0.0034438, -0.0067053, 0.0070763
8: -0.0021948, 0.0086253, -0.0038353, 0.0086247, -0.0108194, 0.0124606
9: -0.0069280, 0.0026535, -0.0074513, 0.0026470, -0.0095750, 0.0101048

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090494, upper bound: 0.0090285
time: 2.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090494, upper bound: 0.0090802
time: 2.64 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032692, 0.0034214, -0.0031451, 0.0036252, -0.0068943, 0.0065665
1: 0.9886339, 1.0028023, 0.9882023, 1.0025395, -0.0139056, 0.0145999
2: -0.0096411, 0.0005418, -0.0095791, 0.0012458, -0.0108869, 0.0101209
3: -0.0014019, 0.0069685, -0.0012467, 0.0072234, -0.0086254, 0.0082152
4: -0.0028292, 0.0113452, -0.0026269, 0.0120476, -0.0148768, 0.0139722
5: -0.0043846, 0.0114699, -0.0040906, 0.0119528, -0.0163374, 0.0155605
6: -0.0077090, 0.0078815, -0.0082610, 0.0076097, -0.0153187, 0.0161426
7: -0.0102083, -0.0034265, -0.0104149, -0.0035523, -0.0066560, 0.0069884
8: -0.0024602, 0.0086269, -0.0033851, 0.0086104, -0.0110706, 0.0120121
9: -0.0070127, 0.0026716, -0.0073077, 0.0024920, -0.0095047, 0.0099793

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091106, upper bound: 0.0090987
time: 2.39 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091108, upper bound: 0.0090987
time: 2.57 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032692, 0.0034214, -0.0031564, 0.0037983, -0.0070675, 0.0065778
1: 0.9886339, 1.0028023, 0.9878358, 1.0025636, -0.0139297, 0.0149665
2: -0.0096411, 0.0005418, -0.0095847, 0.0018437, -0.0114849, 0.0101265
3: -0.0014019, 0.0069685, -0.0012609, 0.0074400, -0.0088420, 0.0082293
4: -0.0028292, 0.0113452, -0.0030402, 0.0126442, -0.0154734, 0.0143854
5: -0.0043846, 0.0114699, -0.0041175, 0.0123631, -0.0167477, 0.0155874
6: -0.0077090, 0.0078815, -0.0087299, 0.0076345, -0.0153435, 0.0166115
7: -0.0102083, -0.0034265, -0.0105904, -0.0035408, -0.0066675, 0.0071639
8: -0.0024602, 0.0086269, -0.0041708, 0.0086119, -0.0110721, 0.0127978
9: -0.0070127, 0.0026716, -0.0075583, 0.0025084, -0.0095211, 0.0102299

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091106, upper bound: 0.0090987
time: 2.28 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091108, upper bound: 0.0090987
time: 2.90 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0031449, 0.0036241, -0.0032310, 0.0033685, -0.0065135, 0.0068550
1: 0.9882047, 1.0025392, 0.9887458, 1.0027213, -0.0145166, 0.0137933
2: -0.0095790, 0.0012419, -0.0096220, 0.0003593, -0.0099383, 0.0108639
3: -0.0012465, 0.0072220, -0.0013541, 0.0069024, -0.0081488, 0.0085761
4: -0.0026266, 0.0120437, -0.0027669, 0.0111632, -0.0137898, 0.0148106
5: -0.0040902, 0.0119502, -0.0042941, 0.0113447, -0.0154349, 0.0162443
6: -0.0082580, 0.0076093, -0.0075659, 0.0077978, -0.0160558, 0.0151752
7: -0.0104138, -0.0035525, -0.0101548, -0.0034653, -0.0069485, 0.0066023
8: -0.0033801, 0.0086104, -0.0022204, 0.0086218, -0.0120019, 0.0108308
9: -0.0073061, 0.0024918, -0.0069362, 0.0026162, -0.0099223, 0.0094280

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091400
time: 2.24 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091399
time: 2.73 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0031429, 0.0036063, -0.0032682, 0.0033685, -0.0065114, 0.0068746
1: 0.9882421, 1.0025349, 0.9887459, 1.0028001, -0.0145580, 0.0137890
2: -0.0095780, 0.0011807, -0.0096406, 0.0003593, -0.0099372, 0.0108214
3: -0.0012440, 0.0071999, -0.0014007, 0.0069024, -0.0081463, 0.0086005
4: -0.0026233, 0.0119827, -0.0028276, 0.0111631, -0.0137865, 0.0148103
5: -0.0040854, 0.0119082, -0.0043823, 0.0113447, -0.0154300, 0.0162905
6: -0.0082100, 0.0076048, -0.0075659, 0.0078794, -0.0160894, 0.0151707
7: -0.0103958, -0.0035545, -0.0101548, -0.0034276, -0.0069683, 0.0066002
8: -0.0032997, 0.0086101, -0.0022204, 0.0086268, -0.0119265, 0.0108305
9: -0.0072805, 0.0024888, -0.0069362, 0.0026701, -0.0099506, 0.0094250

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090941, upper bound: 0.0090438
time: 2.51 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090941, upper bound: 0.0091043
time: 2.42 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0031562, 0.0037972, -0.0032199, 0.0033628, -0.0065191, 0.0070170
1: 0.9878381, 1.0025631, 0.9887579, 1.0026977, -0.0148596, 0.0138052
2: -0.0095846, 0.0018398, -0.0096165, 0.0003396, -0.0099243, 0.0114563
3: -0.0012606, 0.0074386, -0.0013402, 0.0068952, -0.0081559, 0.0087788
4: -0.0030371, 0.0126403, -0.0027488, 0.0111436, -0.0141806, 0.0153891
5: -0.0041170, 0.0123604, -0.0042677, 0.0113312, -0.0154482, 0.0166281
6: -0.0087268, 0.0076341, -0.0075505, 0.0077734, -0.0165003, 0.0151846
7: -0.0105893, -0.0035410, -0.0101490, -0.0034765, -0.0071127, 0.0066080
8: -0.0041657, 0.0086119, -0.0021946, 0.0086204, -0.0127860, 0.0108065
9: -0.0075567, 0.0025081, -0.0069280, 0.0026002, -0.0101568, 0.0094361

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091303
time: 2.39 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091316, upper bound: 0.0091302
time: 2.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0031542, 0.0037799, -0.0032567, 0.0033629, -0.0065170, 0.0070366
1: 0.9878746, 1.0025587, 0.9887578, 1.0027757, -0.0149011, 0.0138009
2: -0.0095836, 0.0017802, -0.0096349, 0.0003397, -0.0099234, 0.0114151
3: -0.0012581, 0.0074170, -0.0013863, 0.0068953, -0.0081533, 0.0088033
4: -0.0029900, 0.0125808, -0.0028089, 0.0111437, -0.0141336, 0.0153897
5: -0.0041122, 0.0123195, -0.0043550, 0.0113313, -0.0154434, 0.0166746
6: -0.0086801, 0.0076296, -0.0075506, 0.0078542, -0.0165343, 0.0151802
7: -0.0105718, -0.0035431, -0.0101490, -0.0034392, -0.0071326, 0.0066060
8: -0.0040874, 0.0086116, -0.0021948, 0.0086253, -0.0127126, 0.0108063
9: -0.0075317, 0.0025051, -0.0069280, 0.0026535, -0.0101852, 0.0094332

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091300
time: 2.80 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091307, upper bound: 0.0091301
time: 3.06 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0031451, 0.0036252, -0.0032692, 0.0034214, -0.0065665, 0.0068943
1: 0.9882023, 1.0025395, 0.9886339, 1.0028023, -0.0145999, 0.0139056
2: -0.0095791, 0.0012458, -0.0096411, 0.0005418, -0.0101209, 0.0108869
3: -0.0012467, 0.0072234, -0.0014019, 0.0069685, -0.0082152, 0.0086254
4: -0.0026269, 0.0120476, -0.0028292, 0.0113452, -0.0139722, 0.0148768
5: -0.0040906, 0.0119528, -0.0043846, 0.0114699, -0.0155605, 0.0163374
6: -0.0082610, 0.0076097, -0.0077090, 0.0078815, -0.0161426, 0.0153187
7: -0.0104149, -0.0035523, -0.0102083, -0.0034265, -0.0069884, 0.0066560
8: -0.0033851, 0.0086104, -0.0024602, 0.0086269, -0.0120121, 0.0110706
9: -0.0073077, 0.0024920, -0.0070127, 0.0026716, -0.0099793, 0.0095047

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091093
time: 2.06 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091093
time: 2.36 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0031451, 0.0036252, -0.0032828, 0.0036063, -0.0067514, 0.0069080
1: 0.9882023, 1.0025395, 0.9882423, 1.0028310, -0.0146286, 0.0142972
2: -0.0095791, 0.0012458, -0.0096480, 0.0011807, -0.0107597, 0.0108937
3: -0.0012467, 0.0072234, -0.0014189, 0.0071999, -0.0084466, 0.0086424
4: -0.0026269, 0.0120476, -0.0028514, 0.0119826, -0.0146096, 0.0148990
5: -0.0040906, 0.0119528, -0.0044168, 0.0119082, -0.0159988, 0.0163697
6: -0.0082610, 0.0076097, -0.0082100, 0.0079113, -0.0161723, 0.0158197
7: -0.0104149, -0.0035523, -0.0103958, -0.0034128, -0.0070022, 0.0068435
8: -0.0033851, 0.0086104, -0.0032996, 0.0086288, -0.0120139, 0.0119100
9: -0.0073077, 0.0024920, -0.0072804, 0.0026913, -0.0099989, 0.0097724

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091093
time: 2.36 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0091093
time: 2.56 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0031562, 0.0037972, -0.0033139, 0.0033032, -0.0064594, 0.0071110
1: 0.9878381, 1.0025631, 0.9888842, 1.0028968, -0.0150587, 0.0136789
2: -0.0095846, 0.0018398, -0.0096635, 0.0001336, -0.0097183, 0.0115033
3: -0.0012606, 0.0074386, -0.0014578, 0.0068206, -0.0080813, 0.0088964
4: -0.0030371, 0.0126403, -0.0029021, 0.0109380, -0.0139751, 0.0155424
5: -0.0041170, 0.0123604, -0.0044905, 0.0111899, -0.0153069, 0.0168509
6: -0.0087268, 0.0076341, -0.0073890, 0.0079794, -0.0167063, 0.0150230
7: -0.0105893, -0.0035410, -0.0100886, -0.0033812, -0.0072080, 0.0065476
8: -0.0041657, 0.0086119, -0.0019239, 0.0086329, -0.0127986, 0.0105358
9: -0.0075567, 0.0025081, -0.0068417, 0.0027362, -0.0102929, 0.0093498

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090889
time: 2.89 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090842, upper bound: 0.0090889
time: 2.87 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0031542, 0.0037799, -0.0033531, 0.0033059, -0.0064601, 0.0071330
1: 0.9878746, 1.0025587, 0.9888785, 1.0029800, -0.0151054, 0.0136802
2: -0.0095836, 0.0017802, -0.0096831, 0.0001429, -0.0097265, 0.0114634
3: -0.0012581, 0.0074170, -0.0015069, 0.0068240, -0.0080820, 0.0089239
4: -0.0029900, 0.0125808, -0.0029661, 0.0109472, -0.0139372, 0.0155469
5: -0.0041122, 0.0123195, -0.0045835, 0.0111962, -0.0153084, 0.0169030
6: -0.0086801, 0.0076296, -0.0073962, 0.0080654, -0.0167455, 0.0150258
7: -0.0105718, -0.0035431, -0.0100913, -0.0033415, -0.0072303, 0.0065482
8: -0.0040874, 0.0086116, -0.0019361, 0.0086381, -0.0127255, 0.0105477
9: -0.0075317, 0.0025051, -0.0068455, 0.0027930, -0.0103247, 0.0093507

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0090896
time: 3.05 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090840, upper bound: 0.0090888
time: 2.93 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032421, 0.0035622, -0.0032310, 0.0033685, -0.0066107, 0.0067932
1: 0.9883357, 1.0027449, 0.9887458, 1.0027213, -0.0143856, 0.0139991
2: -0.0096276, 0.0010282, -0.0096220, 0.0003593, -0.0099869, 0.0106503
3: -0.0013681, 0.0071447, -0.0013541, 0.0069024, -0.0082705, 0.0084987
4: -0.0027851, 0.0118306, -0.0027669, 0.0111632, -0.0139483, 0.0145975
5: -0.0043206, 0.0118036, -0.0042941, 0.0113447, -0.0156652, 0.0160977
6: -0.0080905, 0.0078223, -0.0075659, 0.0077978, -0.0158882, 0.0153882
7: -0.0103511, -0.0034540, -0.0101548, -0.0034653, -0.0068858, 0.0067008
8: -0.0030994, 0.0086233, -0.0022204, 0.0086218, -0.0117212, 0.0108438
9: -0.0072166, 0.0026324, -0.0069362, 0.0026162, -0.0098328, 0.0095686

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091202, upper bound: 0.0090987
time: 6.96 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091202, upper bound: 0.0090987
time: 2.36 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032402, 0.0035455, -0.0032682, 0.0033685, -0.0066087, 0.0068137
1: 0.9883711, 1.0027409, 0.9887459, 1.0028001, -0.0144290, 0.0139950
2: -0.0096266, 0.0009705, -0.0096406, 0.0003593, -0.0099859, 0.0106111
3: -0.0013656, 0.0071237, -0.0014007, 0.0069024, -0.0082679, 0.0085244
4: -0.0027819, 0.0117730, -0.0028276, 0.0111631, -0.0139450, 0.0146006
5: -0.0043159, 0.0117640, -0.0043823, 0.0113447, -0.0156606, 0.0161463
6: -0.0080452, 0.0078180, -0.0075659, 0.0078794, -0.0159246, 0.0153839
7: -0.0103341, -0.0034560, -0.0101548, -0.0034276, -0.0069066, 0.0066988
8: -0.0030235, 0.0086231, -0.0022204, 0.0086268, -0.0116503, 0.0108435
9: -0.0071923, 0.0026296, -0.0069362, 0.0026701, -0.0098625, 0.0095658

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090809, upper bound: 0.0089881
time: 3.09 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0090845, upper bound: 0.0090640
time: 3.28 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0031451, 0.0036252, -0.0031451, 0.0036252, -0.0067703, 0.0067703
1: 0.9882023, 1.0025395, 0.9882023, 1.0025395, -0.0143372, 0.0143372
2: -0.0095791, 0.0012458, -0.0095791, 0.0012458, -0.0108248, 0.0108248
3: -0.0012467, 0.0072234, -0.0012467, 0.0072234, -0.0084702, 0.0084702
4: -0.0026269, 0.0120476, -0.0026269, 0.0120476, -0.0146745, 0.0146745
5: -0.0040906, 0.0119528, -0.0040906, 0.0119528, -0.0160435, 0.0160435
6: -0.0082610, 0.0076097, -0.0082610, 0.0076097, -0.0158707, 0.0158707
7: -0.0104149, -0.0035523, -0.0104149, -0.0035523, -0.0068626, 0.0068626
8: -0.0033851, 0.0086104, -0.0033851, 0.0086104, -0.0119955, 0.0119955
9: -0.0073077, 0.0024920, -0.0073077, 0.0024920, -0.0097997, 0.0097997

Time for backsubstitution: 1.60 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.93 + 595.70 = 600.63 seconds
