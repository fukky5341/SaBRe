## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01217727


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0008509, 0.0102428, 0.0008509, 0.0102428, -0.0093892, 0.0093892)
1: (0.0010717, 0.0027631, 0.0010717, 0.0027631, -0.0016915, 0.0016915)
2: (0.0088459, 0.0144952, 0.0088459, 0.0144952, -0.0056493, 0.0056493)
3: (-0.0055316, -0.0000963, -0.0055316, -0.0000963, -0.0053872, 0.0053872)
4: (-0.0043037, 0.0019513, -0.0043037, 0.0019513, -0.0062550, 0.0062550)
5: (0.0013203, 0.0076106, 0.0013203, 0.0076106, -0.0062903, 0.0062903)
6: (-0.0164160, 0.0078963, -0.0164160, 0.0078963, -0.0243123, 0.0243123)
7: (-0.0133108, 0.0168555, -0.0133108, 0.0168555, -0.0301663, 0.0301663)
8: (0.9797018, 1.0001787, 0.9797018, 1.0001787, -0.0202530, 0.0202530)
9: (-0.0160495, 0.0036146, -0.0160495, 0.0036146, -0.0195672, 0.0195672)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.04 + 2.90 = 4.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0143262, upper bound: 0.0143262

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0133889, upper bound: 0.0135843
time: 1.70 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0135843, upper bound: 0.0135843
time: 1.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.70 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.70
Output dim: 8, lower bound: -0.0133889, upper bound: 0.0135843
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.70
Output dim: 8, lower bound: -0.0135843, upper bound: 0.0135843

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0014439, 0.0101702, 0.0008874, 0.0102388, -0.0084206, 0.0092489
1: 0.0012670, 0.0027616, 0.0010833, 0.0027631, -0.0014775, 0.0016783
2: 0.0088517, 0.0140017, 0.0088462, 0.0144659, -0.0056142, 0.0049385
3: -0.0055256, -0.0004950, -0.0055313, -0.0001203, -0.0053524, 0.0047865
4: -0.0037701, 0.0019448, -0.0042720, 0.0019510, -0.0054742, 0.0062168
5: 0.0015575, 0.0072632, 0.0013337, 0.0075892, -0.0060317, 0.0059296
6: -0.0156374, 0.0065180, -0.0163724, 0.0078115, -0.0234489, 0.0228418
7: -0.0114337, 0.0165224, -0.0131953, 0.0168370, -0.0272320, 0.0297177
8: 0.9810606, 1.0001553, 0.9797847, 1.0001774, -0.0181024, 0.0201317
9: -0.0160283, 0.0020878, -0.0160483, 0.0035228, -0.0194341, 0.0172809

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130230, upper bound: 0.0123541
time: 1.86 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130233, upper bound: 0.0131658
time: 1.93 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0011904, 0.0104652, 0.0009185, 0.0102335, -0.0087851, 0.0095467
1: 0.0012336, 0.0028036, 0.0010971, 0.0027630, -0.0015294, 0.0017065
2: 0.0086912, 0.0141363, 0.0088465, 0.0144348, -0.0057436, 0.0051759
3: -0.0056917, -0.0003511, -0.0055310, -0.0001432, -0.0055485, 0.0050030
4: -0.0039210, 0.0021246, -0.0042387, 0.0019506, -0.0057347, 0.0063633
5: 0.0013735, 0.0074117, 0.0013516, 0.0075710, -0.0061975, 0.0060601
6: -0.0163666, 0.0071072, -0.0163136, 0.0077392, -0.0241058, 0.0234208
7: -0.0122361, 0.0174657, -0.0130968, 0.0168125, -0.0283742, 0.0305625
8: 0.9804941, 1.0008029, 0.9798578, 1.0001762, -0.0189318, 0.0209451
9: -0.0166160, 0.0025923, -0.0160472, 0.0034336, -0.0200497, 0.0180685

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131602, upper bound: 0.0123541
time: 1.70 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131658, upper bound: 0.0131658
time: 1.66 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.62 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 5.62
Output dim: 8, lower bound: -0.0130230, upper bound: 0.0123541
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 5.62
Output dim: 8, lower bound: -0.0130233, upper bound: 0.0131658
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 5.62
Output dim: 8, lower bound: -0.0131602, upper bound: 0.0123541
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 5.62
Output dim: 8, lower bound: -0.0131658, upper bound: 0.0131658

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: 0.0017457, 0.0093610, 0.0008994, 0.0100942, -0.0080033, 0.0084555
1: 0.0013122, 0.0026440, 0.0010912, 0.0027425, -0.0014180, 0.0015528
2: 0.0093019, 0.0138211, 0.0089248, 0.0144497, -0.0051478, 0.0046939
3: -0.0050600, -0.0006732, -0.0054500, -0.0001306, -0.0048900, 0.0045356
4: -0.0035686, 0.0014408, -0.0042550, 0.0018630, -0.0051993, 0.0056958
5: 0.0020176, 0.0070864, 0.0014262, 0.0075822, -0.0055645, 0.0056602
6: -0.0138044, 0.0058164, -0.0160147, 0.0077835, -0.0215879, 0.0218311
7: -0.0104782, 0.0139690, -0.0131572, 0.0163769, -0.0259098, 0.0271262
8: 0.9817365, 0.9983397, 0.9798137, 0.9998604, -0.0171678, 0.0183308
9: -0.0143801, 0.0014465, -0.0157605, 0.0034799, -0.0177896, 0.0163821

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130230, upper bound: 0.0120559
time: 1.88 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130230, upper bound: 0.0123541
time: 1.76 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: 0.0014707, 0.0099953, 0.0008874, 0.0102388, -0.0083913, 0.0086203
1: 0.0012808, 0.0027376, 0.0010833, 0.0027631, -0.0014602, 0.0016371
2: 0.0089438, 0.0139704, 0.0088462, 0.0144659, -0.0052786, 0.0049024
3: -0.0054304, -0.0005162, -0.0055313, -0.0001203, -0.0049894, 0.0047625
4: -0.0037369, 0.0018418, -0.0042720, 0.0019510, -0.0054360, 0.0058291
5: 0.0016875, 0.0072475, 0.0013337, 0.0075892, -0.0058610, 0.0059139
6: -0.0151465, 0.0064558, -0.0163724, 0.0078115, -0.0226084, 0.0227737
7: -0.0113489, 0.0159546, -0.0131953, 0.0168370, -0.0271392, 0.0277494
8: 0.9811242, 0.9997841, 0.9797847, 1.0001774, -0.0180326, 0.0187279
9: -0.0156912, 0.0020014, -0.0160483, 0.0035228, -0.0181497, 0.0171818

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130233, upper bound: 0.0129783
time: 1.74 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130233, upper bound: 0.0131658
time: 1.75 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: 0.0014476, 0.0097179, 0.0009304, 0.0100890, -0.0082577, 0.0087874
1: 0.0012724, 0.0026956, 0.0011050, 0.0027424, -0.0014545, 0.0015906
2: 0.0091044, 0.0139854, 0.0089251, 0.0144187, -0.0053143, 0.0048401
3: -0.0052643, -0.0005028, -0.0054497, -0.0001534, -0.0051109, 0.0046875
4: -0.0037527, 0.0016619, -0.0042218, 0.0018626, -0.0053626, 0.0058837
5: 0.0018162, 0.0072611, 0.0014442, 0.0075640, -0.0057478, 0.0058168
6: -0.0146104, 0.0065094, -0.0159556, 0.0077114, -0.0223218, 0.0223855
7: -0.0114220, 0.0151000, -0.0130590, 0.0163523, -0.0267042, 0.0281590
8: 0.9810714, 0.9991362, 0.9798868, 0.9998592, -0.0177518, 0.0192494
9: -0.0151032, 0.0020508, -0.0157594, 0.0033908, -0.0184940, 0.0169182

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131603, upper bound: 0.0120559
time: 1.89 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131603, upper bound: 0.0120559
time: 1.74 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: 0.0012231, 0.0102449, 0.0009185, 0.0102335, -0.0086617, 0.0092999
1: 0.0012507, 0.0027730, 0.0010971, 0.0027630, -0.0014984, 0.0016758
2: 0.0088084, 0.0140971, 0.0088465, 0.0144348, -0.0056264, 0.0050598
3: -0.0055704, -0.0003774, -0.0055310, -0.0001432, -0.0053874, 0.0049223
4: -0.0038794, 0.0019934, -0.0042387, 0.0019506, -0.0056113, 0.0062321
5: 0.0015329, 0.0073926, 0.0013516, 0.0075710, -0.0060381, 0.0060409
6: -0.0157607, 0.0070312, -0.0163136, 0.0077392, -0.0234998, 0.0233393
7: -0.0121326, 0.0167543, -0.0130968, 0.0168125, -0.0279843, 0.0298511
8: 0.9805726, 1.0003301, 0.9798578, 1.0001762, -0.0186508, 0.0202896
9: -0.0161869, 0.0024848, -0.0160472, 0.0034336, -0.0195547, 0.0177488

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131658, upper bound: 0.0129783
time: 1.69 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131658, upper bound: 0.0129782
time: 1.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.43 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 5.43
Output dim: 8, lower bound: -0.0130230, upper bound: 0.0120559
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 5.43
Output dim: 8, lower bound: -0.0130230, upper bound: 0.0123541
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.43
Output dim: 8, lower bound: -0.0130233, upper bound: 0.0129783
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.43
Output dim: 8, lower bound: -0.0130233, upper bound: 0.0131658
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 5.43
Output dim: 8, lower bound: -0.0131603, upper bound: 0.0120559
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 5.43
Output dim: 8, lower bound: -0.0131603, upper bound: 0.0120559
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.43
Output dim: 8, lower bound: -0.0131658, upper bound: 0.0129783
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.43
Output dim: 8, lower bound: -0.0131658, upper bound: 0.0129782

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0017457, 0.0093610, 0.0014530, 0.0100260, -0.0079140, 0.0075434
1: 0.0013122, 0.0026440, 0.0012727, 0.0027411, -0.0014164, 0.0013555
2: 0.0093019, 0.0138211, 0.0089303, 0.0139895, -0.0044847, 0.0046879
3: -0.0050600, -0.0006732, -0.0054443, -0.0005028, -0.0043228, 0.0045295
4: -0.0035686, 0.0014408, -0.0037572, 0.0018569, -0.0051926, 0.0049671
5: 0.0020176, 0.0070864, 0.0016474, 0.0072579, -0.0052341, 0.0054228
6: -0.0138044, 0.0058164, -0.0152845, 0.0064969, -0.0201119, 0.0208765
7: -0.0104782, 0.0139690, -0.0114049, 0.0160637, -0.0254924, 0.0243313
8: 0.9817365, 0.9983397, 0.9810826, 0.9998385, -0.0171438, 0.0163040
9: -0.0143801, 0.0014465, -0.0157406, 0.0020554, -0.0156336, 0.0163603

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121536, upper bound: 0.0120559
time: 1.72 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121536, upper bound: 0.0120559
time: 1.96 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0017457, 0.0093610, 0.0012018, 0.0103327, -0.0085870, 0.0080807
1: 0.0013122, 0.0026440, 0.0012405, 0.0027849, -0.0014727, 0.0014035
2: 0.0093019, 0.0138211, 0.0087627, 0.0141211, -0.0048192, 0.0050585
3: -0.0050600, -0.0006732, -0.0056177, -0.0003608, -0.0046428, 0.0049231
4: -0.0035686, 0.0014408, -0.0039050, 0.0020445, -0.0056131, 0.0053458
5: 0.0020176, 0.0070864, 0.0014609, 0.0074050, -0.0053874, 0.0056255
6: -0.0138044, 0.0058164, -0.0160279, 0.0070807, -0.0208851, 0.0218443
7: -0.0104782, 0.0139690, -0.0122000, 0.0170407, -0.0275189, 0.0260412
8: 0.9817365, 0.9983397, 0.9805217, 1.0005144, -0.0186788, 0.0175076
9: -0.0143801, 0.0014465, -0.0163542, 0.0025520, -0.0168046, 0.0177537

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121536, upper bound: 0.0123541
time: 1.74 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121536, upper bound: 0.0123541
time: 1.93 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0014707, 0.0099953, 0.0014439, 0.0101702, -0.0083020, 0.0076915
1: 0.0012808, 0.0027376, 0.0012670, 0.0027616, -0.0014587, 0.0013857
2: 0.0089438, 0.0139704, 0.0088517, 0.0140017, -0.0045672, 0.0048964
3: -0.0054304, -0.0005162, -0.0055256, -0.0004950, -0.0044061, 0.0047563
4: -0.0037369, 0.0018418, -0.0037701, 0.0019448, -0.0054293, 0.0050579
5: 0.0016875, 0.0072475, 0.0015575, 0.0072632, -0.0052927, 0.0056534
6: -0.0151465, 0.0064558, -0.0156374, 0.0065180, -0.0203661, 0.0217884
7: -0.0113489, 0.0159546, -0.0114337, 0.0165224, -0.0267207, 0.0247855
8: 0.9811242, 0.9997841, 0.9810606, 1.0001553, -0.0180084, 0.0166501
9: -0.0156912, 0.0020014, -0.0160283, 0.0020878, -0.0159203, 0.0171598

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120604, upper bound: 0.0129773
time: 1.92 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120604, upper bound: 0.0129783
time: 2.21 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0014707, 0.0099953, 0.0011904, 0.0104652, -0.0089583, 0.0083077
1: 0.0012808, 0.0027376, 0.0012336, 0.0028036, -0.0015228, 0.0014896
2: 0.0089438, 0.0139704, 0.0086912, 0.0141363, -0.0049686, 0.0052664
3: -0.0054304, -0.0005162, -0.0056917, -0.0003511, -0.0047765, 0.0051390
4: -0.0037369, 0.0018418, -0.0039210, 0.0021246, -0.0058436, 0.0055003
5: 0.0016875, 0.0072475, 0.0013735, 0.0074117, -0.0056577, 0.0058741
6: -0.0151465, 0.0064558, -0.0163666, 0.0071072, -0.0218134, 0.0228223
7: -0.0113489, 0.0159546, -0.0122361, 0.0174657, -0.0287780, 0.0267391
8: 0.9811242, 0.9997841, 0.9804941, 1.0008029, -0.0195007, 0.0180310
9: -0.0156912, 0.0020014, -0.0166160, 0.0025923, -0.0172771, 0.0185144

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120604, upper bound: 0.0131602
time: 2.00 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120604, upper bound: 0.0131658
time: 2.21 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0014476, 0.0097179, 0.0014530, 0.0100260, -0.0084990, 0.0082649
1: 0.0012724, 0.0026956, 0.0012727, 0.0027411, -0.0014687, 0.0014229
2: 0.0091044, 0.0139854, 0.0089303, 0.0139895, -0.0048851, 0.0050355
3: -0.0052643, -0.0005028, -0.0054443, -0.0005028, -0.0047615, 0.0048786
4: -0.0037527, 0.0016619, -0.0037572, 0.0018569, -0.0055791, 0.0054191
5: 0.0018162, 0.0072611, 0.0016474, 0.0072579, -0.0054417, 0.0056137
6: -0.0146104, 0.0065094, -0.0152845, 0.0064969, -0.0211073, 0.0217939
7: -0.0114220, 0.0151000, -0.0114049, 0.0160637, -0.0273444, 0.0265049
8: 0.9810714, 0.9991362, 0.9810826, 0.9998385, -0.0184587, 0.0180343
9: -0.0151032, 0.0020508, -0.0157406, 0.0020554, -0.0171586, 0.0176069

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124123, upper bound: 0.0120559
time: 1.65 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124123, upper bound: 0.0120559
time: 1.92 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0014476, 0.0097179, 0.0012018, 0.0103327, -0.0082820, 0.0079755
1: 0.0012724, 0.0026956, 0.0012405, 0.0027849, -0.0014690, 0.0014228
2: 0.0091044, 0.0139854, 0.0087627, 0.0141211, -0.0047570, 0.0048955
3: -0.0052643, -0.0005028, -0.0056177, -0.0003608, -0.0045773, 0.0047448
4: -0.0037527, 0.0016619, -0.0039050, 0.0020445, -0.0054246, 0.0052671
5: 0.0018162, 0.0072611, 0.0014609, 0.0074050, -0.0054732, 0.0056243
6: -0.0146104, 0.0065094, -0.0160279, 0.0070807, -0.0210753, 0.0216911
7: -0.0114220, 0.0151000, -0.0122000, 0.0170407, -0.0266474, 0.0256955
8: 0.9810714, 0.9991362, 0.9805217, 1.0005144, -0.0179753, 0.0172832
9: -0.0151032, 0.0020508, -0.0163542, 0.0025520, -0.0165536, 0.0171211

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124123, upper bound: 0.0120559
time: 1.79 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124123, upper bound: 0.0120559
time: 2.16 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0012231, 0.0102449, 0.0014439, 0.0101702, -0.0087935, 0.0084291
1: 0.0012507, 0.0027730, 0.0012670, 0.0027616, -0.0015109, 0.0014952
2: 0.0088084, 0.0140971, 0.0088517, 0.0140017, -0.0049859, 0.0051957
3: -0.0055704, -0.0003774, -0.0055256, -0.0004950, -0.0048392, 0.0050473
4: -0.0038794, 0.0019934, -0.0037701, 0.0019448, -0.0057614, 0.0055267
5: 0.0015329, 0.0073926, 0.0015575, 0.0072632, -0.0056506, 0.0058351
6: -0.0157607, 0.0070312, -0.0156374, 0.0065180, -0.0218467, 0.0226686
7: -0.0121326, 0.0167543, -0.0114337, 0.0165224, -0.0282765, 0.0270850
8: 0.9805726, 1.0003301, 0.9810606, 1.0001553, -0.0191119, 0.0183389
9: -0.0161869, 0.0024848, -0.0160283, 0.0020878, -0.0174532, 0.0182084

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123541, upper bound: 0.0129773
time: 1.94 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123541, upper bound: 0.0129783
time: 1.83 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0012231, 0.0102449, 0.0011904, 0.0104652, -0.0086825, 0.0080879
1: 0.0012507, 0.0027730, 0.0012336, 0.0028036, -0.0015123, 0.0014470
2: 0.0088084, 0.0140971, 0.0086912, 0.0141363, -0.0048200, 0.0051128
3: -0.0055704, -0.0003774, -0.0056917, -0.0003511, -0.0046404, 0.0049771
4: -0.0038794, 0.0019934, -0.0039210, 0.0021246, -0.0056707, 0.0053363
5: 0.0015329, 0.0073926, 0.0013735, 0.0074117, -0.0055115, 0.0058668
6: -0.0157607, 0.0070312, -0.0163666, 0.0071072, -0.0212465, 0.0226436
7: -0.0121326, 0.0167543, -0.0122361, 0.0174657, -0.0279204, 0.0260360
8: 0.9805726, 1.0003301, 0.9804941, 1.0008029, -0.0188644, 0.0175489
9: -0.0161869, 0.0024848, -0.0166160, 0.0025923, -0.0167729, 0.0179428

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123541, upper bound: 0.0129773
time: 1.91 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123541, upper bound: 0.0129783
time: 1.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.89 seconds
NS_A1_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0121536, upper bound: 0.0120559
NS_A1_A1_B1_B2, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0121536, upper bound: 0.0120559
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0121536, upper bound: 0.0123541
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0121536, upper bound: 0.0123541
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0120604, upper bound: 0.0129773
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0120604, upper bound: 0.0129783
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0120604, upper bound: 0.0131602
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0120604, upper bound: 0.0131658
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0124123, upper bound: 0.0120559
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0124123, upper bound: 0.0120559
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0124123, upper bound: 0.0120559
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0124123, upper bound: 0.0120559
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0123541, upper bound: 0.0129773
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0123541, upper bound: 0.0129783
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0123541, upper bound: 0.0129773
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.89
Output dim: 8, lower bound: -0.0123541, upper bound: 0.0129783

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0017457, 0.0093610, 0.0014476, 0.0097179, -0.0079721, 0.0078686
1: 0.0013122, 0.0026440, 0.0012724, 0.0026956, -0.0013834, 0.0013716
2: 0.0093019, 0.0138211, 0.0091044, 0.0139854, -0.0046820, 0.0047167
3: -0.0050600, -0.0006732, -0.0052643, -0.0005028, -0.0045130, 0.0045910
4: -0.0035686, 0.0014408, -0.0037527, 0.0016619, -0.0052305, 0.0051833
5: 0.0020176, 0.0070864, 0.0018162, 0.0072611, -0.0052434, 0.0052702
6: -0.0138044, 0.0058164, -0.0146104, 0.0065094, -0.0203138, 0.0204269
7: -0.0104782, 0.0139690, -0.0114220, 0.0151000, -0.0255782, 0.0253607
8: 0.9817365, 0.9983397, 0.9810714, 0.9991362, -0.0173997, 0.0170328
9: -0.0143801, 0.0014465, -0.0151032, 0.0020508, -0.0163126, 0.0165497

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109483, upper bound: 0.0113782
time: 1.73 seconds

## Relational analysis of NS_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109565, upper bound: 0.0109194
time: 1.67 seconds

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0017457, 0.0093610, 0.0012231, 0.0102449, -0.0084885, 0.0080167
1: 0.0013122, 0.0026440, 0.0012507, 0.0027730, -0.0014608, 0.0013933
2: 0.0093019, 0.0138211, 0.0088084, 0.0140971, -0.0047624, 0.0050128
3: -0.0050600, -0.0006732, -0.0055704, -0.0003774, -0.0045991, 0.0048708
4: -0.0035686, 0.0014408, -0.0038794, 0.0019934, -0.0055619, 0.0052762
5: 0.0020176, 0.0070864, 0.0015329, 0.0073926, -0.0053749, 0.0055535
6: -0.0138044, 0.0058164, -0.0157607, 0.0070312, -0.0208356, 0.0215771
7: -0.0104782, 0.0139690, -0.0121326, 0.0167543, -0.0272324, 0.0258294
8: 0.9817365, 0.9983397, 0.9805726, 1.0003301, -0.0184748, 0.0173643
9: -0.0143801, 0.0014465, -0.0161869, 0.0024848, -0.0166220, 0.0175685

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109483, upper bound: 0.0113782
time: 1.60 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109565, upper bound: 0.0109194
time: 1.82 seconds

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0014707, 0.0099953, 0.0017457, 0.0093610, -0.0075252, 0.0079054
1: 0.0012808, 0.0027376, 0.0013122, 0.0026440, -0.0013454, 0.0014170
2: 0.0089438, 0.0139704, 0.0093019, 0.0138211, -0.0046899, 0.0044631
3: -0.0054304, -0.0005162, -0.0050600, -0.0006732, -0.0045316, 0.0043081
4: -0.0037369, 0.0018418, -0.0035686, 0.0014408, -0.0049441, 0.0051949
5: 0.0016875, 0.0072475, 0.0020176, 0.0070864, -0.0053698, 0.0052235
6: -0.0151465, 0.0064558, -0.0138044, 0.0058164, -0.0207117, 0.0200695
7: -0.0113489, 0.0159546, -0.0104782, 0.0139690, -0.0242737, 0.0254428
8: 0.9811242, 0.9997841, 0.9817365, 0.9983397, -0.0162608, 0.0171520
9: -0.0156912, 0.0020014, -0.0143801, 0.0014465, -0.0163677, 0.0155734

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109458, upper bound: 0.0121862
time: 1.82 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110604, upper bound: 0.0119641
time: 1.55 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0014707, 0.0099953, 0.0014707, 0.0099953, -0.0076598, 0.0076598
1: 0.0012808, 0.0027376, 0.0012808, 0.0027376, -0.0013679, 0.0013679
2: 0.0089438, 0.0139704, 0.0089438, 0.0139704, -0.0045299, 0.0045299
3: -0.0054304, -0.0005162, -0.0054304, -0.0005162, -0.0043810, 0.0043810
4: -0.0037369, 0.0018418, -0.0037369, 0.0018418, -0.0050184, 0.0050184
5: 0.0016875, 0.0072475, 0.0016875, 0.0072475, -0.0052742, 0.0052742
6: -0.0151465, 0.0064558, -0.0151465, 0.0064558, -0.0202925, 0.0202925
7: -0.0113489, 0.0159546, -0.0113489, 0.0159546, -0.0246853, 0.0246853
8: 0.9811242, 0.9997841, 0.9811242, 0.9997841, -0.0165752, 0.0165752
9: -0.0156912, 0.0020014, -0.0156912, 0.0020014, -0.0158171, 0.0158171

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0112858, upper bound: 0.0118924
time: 1.85 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110604, upper bound: 0.0119641
time: 1.74 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0014707, 0.0099953, 0.0014476, 0.0097179, -0.0082472, 0.0084905
1: 0.0012808, 0.0027376, 0.0012724, 0.0026956, -0.0014148, 0.0014652
2: 0.0089438, 0.0139704, 0.0091044, 0.0139854, -0.0050376, 0.0048660
3: -0.0054304, -0.0005162, -0.0052643, -0.0005028, -0.0048807, 0.0047480
4: -0.0037369, 0.0018418, -0.0037527, 0.0016619, -0.0053988, 0.0055814
5: 0.0016875, 0.0072475, 0.0018162, 0.0072611, -0.0055736, 0.0054314
6: -0.0151465, 0.0064558, -0.0146104, 0.0065094, -0.0216559, 0.0210662
7: -0.0113489, 0.0159546, -0.0114220, 0.0151000, -0.0264489, 0.0272947
8: 0.9811242, 0.9997841, 0.9810714, 0.9991362, -0.0179911, 0.0184669
9: -0.0156912, 0.0020014, -0.0151032, 0.0020508, -0.0176143, 0.0171045

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108747, upper bound: 0.0122207
time: 1.76 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109105, upper bound: 0.0118084
time: 1.80 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0014707, 0.0099953, 0.0012231, 0.0102449, -0.0083974, 0.0082387
1: 0.0012808, 0.0027376, 0.0012507, 0.0027730, -0.0014773, 0.0014583
2: 0.0089438, 0.0139704, 0.0088084, 0.0140971, -0.0048795, 0.0049486
3: -0.0054304, -0.0005162, -0.0055704, -0.0003774, -0.0047267, 0.0048140
4: -0.0037369, 0.0018418, -0.0038794, 0.0019934, -0.0054872, 0.0054079
5: 0.0016875, 0.0072475, 0.0015329, 0.0073926, -0.0056133, 0.0056321
6: -0.0151465, 0.0064558, -0.0157607, 0.0070312, -0.0216380, 0.0217731
7: -0.0113489, 0.0159546, -0.0121326, 0.0167543, -0.0269848, 0.0265177
8: 0.9811242, 0.9997841, 0.9805726, 1.0003301, -0.0182640, 0.0178759
9: -0.0156912, 0.0020014, -0.0161869, 0.0024848, -0.0170644, 0.0173501

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108747, upper bound: 0.0122229
time: 1.87 seconds

## Relational analysis of NS_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109105, upper bound: 0.0118100
time: 1.92 seconds

## BFS NS instance: NS_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0014476, 0.0097179, 0.0017457, 0.0093610, -0.0078686, 0.0079721
1: 0.0012724, 0.0026956, 0.0013122, 0.0026440, -0.0013716, 0.0013834
2: 0.0091044, 0.0139854, 0.0093019, 0.0138211, -0.0047167, 0.0046820
3: -0.0052643, -0.0005028, -0.0050600, -0.0006732, -0.0045910, 0.0045130
4: -0.0037527, 0.0016619, -0.0035686, 0.0014408, -0.0051833, 0.0052305
5: 0.0018162, 0.0072611, 0.0020176, 0.0070864, -0.0052702, 0.0052434
6: -0.0146104, 0.0065094, -0.0138044, 0.0058164, -0.0204269, 0.0203138
7: -0.0114220, 0.0151000, -0.0104782, 0.0139690, -0.0253607, 0.0255782
8: 0.9810714, 0.9991362, 0.9817365, 0.9983397, -0.0170328, 0.0173997
9: -0.0151032, 0.0020508, -0.0143801, 0.0014465, -0.0165497, 0.0163126

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_A1_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0114148, upper bound: 0.0108747
time: 1.79 seconds

## Relational analysis of NS_A2_A1_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109321, upper bound: 0.0109105
time: 1.86 seconds

## BFS NS instance: NS_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0014476, 0.0097179, 0.0014707, 0.0099953, -0.0084905, 0.0082472
1: 0.0012724, 0.0026956, 0.0012808, 0.0027376, -0.0014652, 0.0014148
2: 0.0091044, 0.0139854, 0.0089438, 0.0139704, -0.0048660, 0.0050376
3: -0.0052643, -0.0005028, -0.0054304, -0.0005162, -0.0047480, 0.0048807
4: -0.0037527, 0.0016619, -0.0037369, 0.0018418, -0.0055814, 0.0053988
5: 0.0018162, 0.0072611, 0.0016875, 0.0072475, -0.0054314, 0.0055736
6: -0.0146104, 0.0065094, -0.0151465, 0.0064558, -0.0210662, 0.0216559
7: -0.0114220, 0.0151000, -0.0113489, 0.0159546, -0.0272947, 0.0264489
8: 0.9810714, 0.9991362, 0.9811242, 0.9997841, -0.0184669, 0.0179911
9: -0.0151032, 0.0020508, -0.0156912, 0.0020014, -0.0171045, 0.0176143

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_A1_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0114148, upper bound: 0.0108747
time: 1.84 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109321, upper bound: 0.0109105
time: 2.07 seconds

## BFS NS instance: NS_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0014476, 0.0097179, 0.0014476, 0.0097179, -0.0076674, 0.0076674
1: 0.0012724, 0.0026956, 0.0012724, 0.0026956, -0.0013802, 0.0013802
2: 0.0091044, 0.0139854, 0.0091044, 0.0139854, -0.0045556, 0.0045556
3: -0.0052643, -0.0005028, -0.0052643, -0.0005028, -0.0043932, 0.0043932
4: -0.0037527, 0.0016619, -0.0037527, 0.0016619, -0.0050440, 0.0050440
5: 0.0018162, 0.0072611, 0.0018162, 0.0072611, -0.0052782, 0.0052782
6: -0.0146104, 0.0065094, -0.0146104, 0.0065094, -0.0203064, 0.0203064
7: -0.0114220, 0.0151000, -0.0114220, 0.0151000, -0.0247064, 0.0247064
8: 0.9810714, 0.9991362, 0.9810714, 0.9991362, -0.0166041, 0.0166041
9: -0.0151032, 0.0020508, -0.0151032, 0.0020508, -0.0158764, 0.0158764

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0114148, upper bound: 0.0108101
time: 1.75 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109313, upper bound: 0.0108290
time: 1.55 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0014476, 0.0097179, 0.0012231, 0.0102449, -0.0082561, 0.0079249
1: 0.0012724, 0.0026956, 0.0012507, 0.0027730, -0.0014668, 0.0014032
2: 0.0091044, 0.0139854, 0.0088084, 0.0140971, -0.0046954, 0.0048872
3: -0.0052643, -0.0005028, -0.0055704, -0.0003774, -0.0045454, 0.0047362
4: -0.0037527, 0.0016619, -0.0038794, 0.0019934, -0.0054154, 0.0052034
5: 0.0018162, 0.0072611, 0.0015329, 0.0073926, -0.0054291, 0.0055633
6: -0.0146104, 0.0065094, -0.0157607, 0.0070312, -0.0209050, 0.0214913
7: -0.0114220, 0.0151000, -0.0121326, 0.0167543, -0.0265456, 0.0255216
8: 0.9810714, 0.9991362, 0.9805726, 1.0003301, -0.0179418, 0.0171812
9: -0.0151032, 0.0020508, -0.0161869, 0.0024848, -0.0164148, 0.0170907

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0114148, upper bound: 0.0108101
time: 1.88 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109313, upper bound: 0.0108290
time: 1.65 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0012231, 0.0102449, 0.0017457, 0.0093610, -0.0080167, 0.0084885
1: 0.0012507, 0.0027730, 0.0013122, 0.0026440, -0.0013933, 0.0014608
2: 0.0088084, 0.0140971, 0.0093019, 0.0138211, -0.0050128, 0.0047624
3: -0.0055704, -0.0003774, -0.0050600, -0.0006732, -0.0048708, 0.0045991
4: -0.0038794, 0.0019934, -0.0035686, 0.0014408, -0.0052762, 0.0055619
5: 0.0015329, 0.0073926, 0.0020176, 0.0070864, -0.0055535, 0.0053749
6: -0.0157607, 0.0070312, -0.0138044, 0.0058164, -0.0215771, 0.0208356
7: -0.0121326, 0.0167543, -0.0104782, 0.0139690, -0.0258294, 0.0272324
8: 0.9805726, 1.0003301, 0.9817365, 0.9983397, -0.0173643, 0.0184748
9: -0.0161869, 0.0024848, -0.0143801, 0.0014465, -0.0175685, 0.0166220

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0113782, upper bound: 0.0118532
time: 1.84 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109194, upper bound: 0.0118796
time: 1.60 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0012231, 0.0102449, 0.0014707, 0.0099953, -0.0082387, 0.0083974
1: 0.0012507, 0.0027730, 0.0012808, 0.0027376, -0.0014583, 0.0014773
2: 0.0088084, 0.0140971, 0.0089438, 0.0139704, -0.0049486, 0.0048795
3: -0.0055704, -0.0003774, -0.0054304, -0.0005162, -0.0048140, 0.0047267
4: -0.0038794, 0.0019934, -0.0037369, 0.0018418, -0.0054079, 0.0054872
5: 0.0015329, 0.0073926, 0.0016875, 0.0072475, -0.0056321, 0.0056133
6: -0.0157607, 0.0070312, -0.0151465, 0.0064558, -0.0217731, 0.0216380
7: -0.0121326, 0.0167543, -0.0113489, 0.0159546, -0.0265177, 0.0269848
8: 0.9805726, 1.0003301, 0.9811242, 0.9997841, -0.0178759, 0.0182640
9: -0.0161869, 0.0024848, -0.0156912, 0.0020014, -0.0173501, 0.0170644

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0113782, upper bound: 0.0118532
time: 1.61 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109194, upper bound: 0.0118796
time: 1.68 seconds

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0012231, 0.0102449, 0.0014476, 0.0097179, -0.0079249, 0.0082561
1: 0.0012507, 0.0027730, 0.0012724, 0.0026956, -0.0014032, 0.0014668
2: 0.0088084, 0.0140971, 0.0091044, 0.0139854, -0.0048872, 0.0046954
3: -0.0055704, -0.0003774, -0.0052643, -0.0005028, -0.0047362, 0.0045454
4: -0.0038794, 0.0019934, -0.0037527, 0.0016619, -0.0052034, 0.0054154
5: 0.0015329, 0.0073926, 0.0018162, 0.0072611, -0.0055633, 0.0054291
6: -0.0157607, 0.0070312, -0.0146104, 0.0065094, -0.0214913, 0.0209050
7: -0.0121326, 0.0167543, -0.0114220, 0.0151000, -0.0255216, 0.0265456
8: 0.9805726, 1.0003301, 0.9810714, 0.9991362, -0.0171812, 0.0179418
9: -0.0161869, 0.0024848, -0.0151032, 0.0020508, -0.0170907, 0.0164148

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_A2_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109328, upper bound: 0.0120881
time: 1.69 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109193, upper bound: 0.0117584
time: 1.60 seconds

## BFS NS instance: NS_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0012231, 0.0102449, 0.0012231, 0.0102449, -0.0080256, 0.0080256
1: 0.0012507, 0.0027730, 0.0012507, 0.0027730, -0.0014209, 0.0014209
2: 0.0088084, 0.0140971, 0.0088084, 0.0140971, -0.0047409, 0.0047409
3: -0.0055704, -0.0003774, -0.0055704, -0.0003774, -0.0045979, 0.0045979
4: -0.0038794, 0.0019934, -0.0038794, 0.0019934, -0.0052539, 0.0052539
5: 0.0015329, 0.0073926, 0.0015329, 0.0073926, -0.0054646, 0.0054646
6: -0.0157607, 0.0070312, -0.0157607, 0.0070312, -0.0210661, 0.0210661
7: -0.0121326, 0.0167543, -0.0121326, 0.0167543, -0.0258296, 0.0258296
8: 0.9805726, 1.0003301, 0.9805726, 1.0003301, -0.0174154, 0.0174154
9: -0.0161869, 0.0024848, -0.0161869, 0.0024848, -0.0165884, 0.0165884

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0113782, upper bound: 0.0117511
time: 1.83 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109193, upper bound: 0.0117585
time: 1.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.57 seconds
NS_A1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109483, upper bound: 0.0113782
NS_A1_A1_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109565, upper bound: 0.0109194
NS_A1_A1_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109483, upper bound: 0.0113782
NS_A1_A1_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109565, upper bound: 0.0109194
NS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109458, upper bound: 0.0121862
NS_A1_A2_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0110604, upper bound: 0.0119641
NS_A1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0112858, upper bound: 0.0118924
NS_A1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0110604, upper bound: 0.0119641
NS_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0108747, upper bound: 0.0122207
NS_A1_A2_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109105, upper bound: 0.0118084
NS_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0108747, upper bound: 0.0122229
NS_A1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109105, upper bound: 0.0118100
NS_A2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0114148, upper bound: 0.0108747
NS_A2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109321, upper bound: 0.0109105
NS_A2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0114148, upper bound: 0.0108747
NS_A2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109321, upper bound: 0.0109105
NS_A2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0114148, upper bound: 0.0108101
NS_A2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109313, upper bound: 0.0108290
NS_A2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0114148, upper bound: 0.0108101
NS_A2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109313, upper bound: 0.0108290
NS_A2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0113782, upper bound: 0.0118532
NS_A2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109194, upper bound: 0.0118796
NS_A2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0113782, upper bound: 0.0118532
NS_A2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109194, upper bound: 0.0118796
NS_A2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109328, upper bound: 0.0120881
NS_A2_A2_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109193, upper bound: 0.0117584
NS_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0113782, upper bound: 0.0117511
NS_A2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.57
Output dim: 8, lower bound: -0.0109193, upper bound: 0.0117585

## BFS NS instance: NS_A1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0014707, 0.0099953, 0.0017814, 0.0090730, -0.0072288, 0.0078799
1: 0.0012808, 0.0027376, 0.0013865, 0.0026109, -0.0013131, 0.0013197
2: 0.0089438, 0.0139704, 0.0094286, 0.0136932, -0.0045518, 0.0043393
3: -0.0054304, -0.0005162, -0.0049289, -0.0007310, -0.0044769, 0.0041801
4: -0.0037369, 0.0018418, -0.0034399, 0.0012989, -0.0048055, 0.0050564
5: 0.0016875, 0.0072475, 0.0023569, 0.0070655, -0.0053549, 0.0048327
6: -0.0151465, 0.0064558, -0.0125955, 0.0057336, -0.0206525, 0.0186924
7: -0.0113489, 0.0159546, -0.0103654, 0.0129637, -0.0232181, 0.0253622
8: 0.9811242, 0.9997841, 0.9818389, 0.9978284, -0.0157614, 0.0170713
9: -0.0156912, 0.0020014, -0.0139161, 0.0011698, -0.0160843, 0.0151202

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_A2_B1_B1_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0100905, upper bound: 0.0112772
time: 1.58 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0100412, upper bound: 0.0112993
time: 1.66 seconds

## BFS NS instance: NS_A1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0014707, 0.0099953, 0.0014830, 0.0094227, -0.0079520, 0.0084608
1: 0.0012808, 0.0027376, 0.0013451, 0.0026614, -0.0013806, 0.0013925
2: 0.0089438, 0.0139704, 0.0092354, 0.0138574, -0.0048918, 0.0047350
3: -0.0054304, -0.0005162, -0.0051288, -0.0005603, -0.0048205, 0.0046125
4: -0.0037369, 0.0018418, -0.0036240, 0.0015152, -0.0052521, 0.0054359
5: 0.0016875, 0.0072475, 0.0021590, 0.0072403, -0.0055528, 0.0050885
6: -0.0151465, 0.0064558, -0.0133921, 0.0064271, -0.0215736, 0.0198479
7: -0.0113489, 0.0159546, -0.0113098, 0.0140708, -0.0254197, 0.0272010
8: 0.9811242, 0.9997841, 0.9811737, 0.9986077, -0.0174614, 0.0183740
9: -0.0156912, 0.0020014, -0.0146235, 0.0017712, -0.0173138, 0.0166249

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_A2_B2_B1_B1_B1

### Relational analysis result of NS_A1_A2_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0100151, upper bound: 0.0112961
time: 2.98 seconds

## Relational analysis of NS_A1_A2_B2_B1_B1_B2

### Relational analysis result of NS_A1_A2_B2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0099765, upper bound: 0.0113704
time: 1.84 seconds

## BFS NS instance: NS_A1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0014707, 0.0099953, 0.0012588, 0.0099533, -0.0080959, 0.0082099
1: 0.0012808, 0.0027376, 0.0013203, 0.0027389, -0.0014431, 0.0013653
2: 0.0089438, 0.0139704, 0.0089386, 0.0139722, -0.0047388, 0.0048176
3: -0.0054304, -0.0005162, -0.0054357, -0.0004343, -0.0046675, 0.0046785
4: -0.0037369, 0.0018418, -0.0037539, 0.0018475, -0.0053405, 0.0052659
5: 0.0016875, 0.0072475, 0.0018635, 0.0073716, -0.0055964, 0.0052616
6: -0.0151465, 0.0064558, -0.0145761, 0.0069482, -0.0215711, 0.0204649
7: -0.0113489, 0.0159546, -0.0120195, 0.0157408, -0.0259257, 0.0264267
8: 0.9811242, 0.9997841, 0.9806746, 0.9998047, -0.0177357, 0.0177854
9: -0.0156912, 0.0020014, -0.0157100, 0.0022109, -0.0167664, 0.0168706

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A1_A2_B2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0101983, upper bound: 0.0115020
time: 1.71 seconds

## Relational analysis of NS_A1_A2_B2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0100649, upper bound: 0.0113708
time: 1.64 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.78 seconds
NS_A1_A2_B1_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 5.78
Output dim: 8, lower bound: -0.0100905, upper bound: 0.0112772
NS_A1_A2_B1_B1_B1_B2, status: Status.VERIFIED, split count: 6, time: 5.78
Output dim: 8, lower bound: -0.0100412, upper bound: 0.0112993
NS_A1_A2_B2_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 5.78
Output dim: 8, lower bound: -0.0100151, upper bound: 0.0112961
NS_A1_A2_B2_B1_B1_B2, status: Status.VERIFIED, split count: 6, time: 5.78
Output dim: 8, lower bound: -0.0099765, upper bound: 0.0113704
NS_A1_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.78
Output dim: 8, lower bound: -0.0101983, upper bound: 0.0115020
NS_A1_A2_B2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.78
Output dim: 8, lower bound: -0.0100649, upper bound: 0.0113708

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.94 + 181.08 = 186.01 seconds
