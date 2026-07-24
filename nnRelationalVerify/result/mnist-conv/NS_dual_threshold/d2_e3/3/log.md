## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.23181255539999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9982100, 0.9982100)
1: (-10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.9129472, 0.9129472)
2: (-8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5460193, 0.5460193)
3: (-3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6636717, 0.6636720)
4: (-10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1278839, 1.1278849)
5: (8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6244779, 0.6244783)
6: (-7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6638887, 0.6638887)
7: (-12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1221027, 1.1221032)
8: (-1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5523930, 0.5523930)
9: (-3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.8008251, 0.8008251)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.79 + 33.23 = 56.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.2320443, upper bound: 0.2320458

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 5848
type: B, layer: 1, pos: 5848
type: B, layer: 1, pos: 831
type: A, layer: 1, pos: 831
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 5861
type: B, layer: 1, pos: 5861
type: A, layer: 1, pos: 4667
type: B, layer: 1, pos: 4667
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 6139

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2320405, upper bound: 0.2297884
time: 4.90 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2320405, upper bound: 0.2320419
time: 3.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.79 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.79
Output dim: 5, lower bound: -0.2320405, upper bound: 0.2297884
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.79
Output dim: 5, lower bound: -0.2320405, upper bound: 0.2320419

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.7592144, -9.2750092, -10.7633934, -9.2697868, -0.9907465, 0.9892783
1: -10.7550287, -9.5178280, -10.7584095, -9.5122404, -0.9060221, 0.9020905
2: -8.5538731, -7.6575727, -8.5544910, -7.6558299, -0.5442705, 0.5425644
3: -3.3049874, -2.1961546, -3.3083432, -2.1913245, -0.6564479, 0.6544375
4: -10.5481157, -9.2224598, -10.5582714, -9.2100334, -1.1087365, 1.1058683
5: 8.1216755, 8.8883686, 8.1150446, 8.8927784, -0.6126966, 0.6156812
6: -7.1350231, -5.8627167, -7.1380944, -5.8591533, -0.6580138, 0.6573329
7: -12.1587162, -10.7157764, -12.1657887, -10.7050056, -1.1075344, 1.1035781
8: -1.9673805, -1.1651254, -1.9680886, -1.1638641, -0.5497482, 0.5496480
9: -3.4041505, -2.5755329, -3.4077473, -2.5724912, -0.7943954, 0.7944157

Time for backsubstitution: 21.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 5848
type: B, layer: 1, pos: 5848
type: A, layer: 1, pos: 831
type: B, layer: 1, pos: 831
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 5861
type: B, layer: 1, pos: 5861
type: A, layer: 1, pos: 4667
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 4667

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 6139

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2297874
time: 3.89 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2297883
time: 4.67 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -10.7666664, -9.2695198, -10.7666693, -9.2695217, -0.9970155, 0.9982076
1: -10.7616873, -9.5120468, -10.7616901, -9.5120478, -0.9057117, 0.9120388
2: -8.5551338, -7.6554103, -8.5551338, -7.6554098, -0.5458090, 0.5451243
3: -3.3115916, -2.1908967, -3.3115938, -2.1908965, -0.6558506, 0.6630290
4: -10.5667286, -9.2098618, -10.5667334, -9.2098627, -1.1193991, 1.1267738
5: 8.1148434, 8.8970280, 8.1148453, 8.8970308, -0.6237483, 0.6141133
6: -7.1409607, -5.8589869, -7.1409636, -5.8589869, -0.6582823, 0.6638868
7: -12.1724844, -10.7043161, -12.1724920, -10.7043152, -1.1061392, 1.1194372
8: -1.9681168, -1.1633472, -1.9681172, -1.1633468, -0.5537996, 0.5506639
9: -3.4103060, -2.5724671, -3.4103072, -2.5724673, -0.8006139, 0.8015656

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 5848
type: A, layer: 1, pos: 5848
type: B, layer: 1, pos: 831
type: A, layer: 1, pos: 831
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 5861
type: A, layer: 1, pos: 5861
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 4667

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6139

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2320420
time: 4.65 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2320408
time: 5.16 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 32.17 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 32.17
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2297874
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 32.17
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2297883
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 32.17
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2320420
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 32.17
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2320408

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -10.7666664, -9.2695198, -10.7592144, -9.2750092, -0.9927583, 0.9909611
1: -10.7616873, -9.5120468, -10.7550287, -9.5178280, -0.9062390, 0.9053326
2: -8.5551338, -7.6554103, -8.5538731, -7.6575727, -0.5429513, 0.5438967
3: -3.3115916, -2.1908967, -3.3049874, -2.1961546, -0.6578493, 0.6563084
4: -10.5667286, -9.2098618, -10.5481157, -9.2224598, -1.1149378, 1.1079869
5: 8.1148434, 8.8970280, 8.1216755, 8.8883686, -0.6150746, 0.6174326
6: -7.1409607, -5.8589869, -7.1350231, -5.8627167, -0.6601524, 0.6581955
7: -12.1724844, -10.7043161, -12.1587162, -10.7157764, -1.1088123, 1.1056409
8: -1.9681168, -1.1633472, -1.9673805, -1.1651254, -0.5482686, 0.5491183
9: -3.4103060, -2.5724671, -3.4041505, -2.5755329, -0.7973523, 0.7939637

Time for backsubstitution: 22.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5848
type: A, layer: 1, pos: 5848
type: A, layer: 1, pos: 831
type: B, layer: 1, pos: 831
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 5861
type: A, layer: 1, pos: 5861
type: A, layer: 1, pos: 4667
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 4667
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 5848

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2297869, upper bound: 0.2320415
time: 3.65 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2297868, upper bound: 0.2320415
time: 3.48 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -10.7666664, -9.2695198, -10.7666664, -9.2695198, -0.9970155, 0.9970155
1: -10.7616873, -9.5120468, -10.7616873, -9.5120468, -0.9057107, 0.9057107
2: -8.5551338, -7.6554103, -8.5551338, -7.6554103, -0.5458086, 0.5458083
3: -3.3115916, -2.1908967, -3.3115916, -2.1908967, -0.6558499, 0.6558498
4: -10.5667286, -9.2098618, -10.5667286, -9.2098618, -1.1193976, 1.1193981
5: 8.1148434, 8.8970280, 8.1148434, 8.8970280, -0.6141129, 0.6141129
6: -7.1409607, -5.8589869, -7.1409607, -5.8589869, -0.6582825, 0.6582823
7: -12.1724844, -10.7043161, -12.1724844, -10.7043161, -1.1061382, 1.1061382
8: -1.9681168, -1.1633472, -1.9681168, -1.1633472, -0.5537989, 0.5537989
9: -3.4103060, -2.5724671, -3.4103060, -2.5724671, -0.8015642, 0.8015642

Time for backsubstitution: 22.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5848
type: B, layer: 1, pos: 5848
type: A, layer: 1, pos: 831
type: B, layer: 1, pos: 831
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 5861
type: B, layer: 1, pos: 5861
type: B, layer: 1, pos: 4667
type: A, layer: 1, pos: 4667
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 5848

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2297868, upper bound: 0.2320406
time: 4.65 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2297868, upper bound: 0.2320404
time: 5.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 32.69 seconds
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 32.69
Output dim: 5, lower bound: -0.2297869, upper bound: 0.2320415
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 32.69
Output dim: 5, lower bound: -0.2297868, upper bound: 0.2320415
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.69
Output dim: 5, lower bound: -0.2297868, upper bound: 0.2320406
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.69
Output dim: 5, lower bound: -0.2297868, upper bound: 0.2320404

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -10.7666283, -9.2696762, -10.7591219, -9.2754126, -0.9917760, 0.9900937
1: -10.7616453, -9.5149498, -10.7549229, -9.5252829, -0.8984213, 0.9021573
2: -8.5551329, -7.6557393, -8.5538740, -7.6584177, -0.5420597, 0.5435498
3: -3.3115072, -2.1913040, -3.3047712, -2.1971989, -0.6567128, 0.6556735
4: -10.5648489, -9.2099152, -10.5432949, -9.2225981, -1.1128817, 1.1029568
5: 8.1148834, 8.8969927, 8.1217775, 8.8882771, -0.6148081, 0.6172209
6: -7.1408997, -5.8600678, -7.1348667, -5.8654909, -0.6568980, 0.6566496
7: -12.1717434, -10.7043915, -12.1568184, -10.7159719, -1.1062937, 1.1019220
8: -1.9680991, -1.1655037, -1.9673386, -1.1706605, -0.5426641, 0.5469000
9: -3.4082465, -2.5724931, -3.3988676, -2.5755744, -0.7952185, 0.7885637

Time for backsubstitution: 22.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 831
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 5848
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 5861
type: B, layer: 1, pos: 5861
type: A, layer: 1, pos: 4667
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 4667

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 831

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2285052, upper bound: 0.2318063
time: 3.41 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2295522, upper bound: 0.2318063
time: 3.47 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -10.7666655, -9.2695198, -10.7610893, -9.2732801, -0.9944015, 0.9937968
1: -10.7616854, -9.5120554, -10.7740068, -9.5171471, -0.9037375, 0.9146576
2: -8.5551338, -7.6554117, -8.5580511, -7.6568604, -0.5433440, 0.5471690
3: -3.3115914, -2.1908972, -3.3100381, -2.1957216, -0.6577675, 0.6616840
4: -10.5667248, -9.2098608, -10.5502176, -9.2101927, -1.1164160, 1.1096177
5: 8.1148443, 8.8970270, 8.1196289, 8.8887873, -0.6160078, 0.6190448
6: -7.1409597, -5.8589902, -7.1428709, -5.8623881, -0.6596560, 0.6646061
7: -12.1724834, -10.7043142, -12.1611691, -10.7087631, -1.1089211, 1.1057792
8: -1.9681156, -1.1633506, -1.9797263, -1.1634855, -0.5482602, 0.5571897
9: -3.4103022, -2.5724666, -3.4054990, -2.5638561, -0.7999070, 0.7935698

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 831
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 4667
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 5861
type: A, layer: 1, pos: 5861
type: A, layer: 1, pos: 5848
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 4667

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 831

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2285051, upper bound: 0.2318063
time: 3.41 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2295522, upper bound: 0.2318063
time: 3.68 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -10.7665777, -9.2699223, -10.7666283, -9.2696762, -0.9961476, 0.9960356
1: -10.7615805, -9.5194979, -10.7616453, -9.5149498, -0.9025340, 0.8978944
2: -8.5551338, -7.6562529, -8.5551329, -7.6557393, -0.5454607, 0.5449164
3: -3.3113742, -2.1919389, -3.3115072, -2.1913040, -0.6552153, 0.6547145
4: -10.5619030, -9.2100000, -10.5648489, -9.2099152, -1.1143689, 1.1173425
5: 8.1149473, 8.8969355, 8.1148834, 8.8969927, -0.6139002, 0.6138468
6: -7.1408033, -5.8617592, -7.1408997, -5.8600678, -0.6567366, 0.6550288
7: -12.1705828, -10.7045088, -12.1717434, -10.7043915, -1.1025143, 1.1038117
8: -1.9680738, -1.1688826, -1.9680991, -1.1655037, -0.5515804, 0.5481944
9: -3.4050207, -2.5725083, -3.4082465, -2.5724931, -0.7961626, 0.7994299

Time for backsubstitution: 22.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 831
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 5848
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 5861
type: A, layer: 1, pos: 5861
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 4667

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 831

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2295522, upper bound: 0.2307597
time: 3.45 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2295522, upper bound: 0.2318068
time: 3.66 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.7685394, -9.2677860, -10.7666655, -9.2695198, -0.9998283, 0.9986668
1: -10.7806644, -9.5113602, -10.7616854, -9.5120554, -0.9187407, 0.9032068
2: -8.5593090, -7.6546926, -8.5551338, -7.6554117, -0.5490816, 0.5461993
3: -3.3166394, -2.1904619, -3.3115914, -2.1908972, -0.6612258, 0.6557671
4: -10.5688009, -9.1975946, -10.5667248, -9.2098608, -1.1210117, 1.1274834
5: 8.1127892, 8.8974428, 8.1148443, 8.8970270, -0.6157436, 0.6150479
6: -7.1488094, -5.8586545, -7.1409597, -5.8589902, -0.6646962, 0.6577897
7: -12.1749191, -10.6973019, -12.1724834, -10.7043142, -1.1062784, 1.1136098
8: -1.9804614, -1.1617117, -1.9681156, -1.1633506, -0.5601056, 0.5537887
9: -3.4116483, -2.5607893, -3.4103022, -2.5724666, -0.8011665, 0.8031898

Time for backsubstitution: 22.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 831
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 5861
type: B, layer: 1, pos: 5861
type: B, layer: 1, pos: 5848
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 4667

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 831

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2295522, upper bound: 0.2307595
time: 4.42 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2295522, upper bound: 0.2318057
time: 4.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.76 seconds
NS_A2_B1_B1_B1, status: Status.VERIFIED, split count: 4, time: 31.76
Output dim: 5, lower bound: -0.2285052, upper bound: 0.2318063
NS_A2_B1_B1_B2, status: Status.VERIFIED, split count: 4, time: 31.76
Output dim: 5, lower bound: -0.2295522, upper bound: 0.2318063
NS_A2_B1_B2_B1, status: Status.VERIFIED, split count: 4, time: 31.76
Output dim: 5, lower bound: -0.2285051, upper bound: 0.2318063
NS_A2_B1_B2_B2, status: Status.VERIFIED, split count: 4, time: 31.76
Output dim: 5, lower bound: -0.2295522, upper bound: 0.2318063
NS_A2_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 31.76
Output dim: 5, lower bound: -0.2295522, upper bound: 0.2307597
NS_A2_B2_A1_A2, status: Status.VERIFIED, split count: 4, time: 31.76
Output dim: 5, lower bound: -0.2295522, upper bound: 0.2318068
NS_A2_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 31.76
Output dim: 5, lower bound: -0.2295522, upper bound: 0.2307595
NS_A2_B2_A2_A2, status: Status.VERIFIED, split count: 4, time: 31.76
Output dim: 5, lower bound: -0.2295522, upper bound: 0.2318057

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 56.02 + 254.48 = 310.51 seconds
