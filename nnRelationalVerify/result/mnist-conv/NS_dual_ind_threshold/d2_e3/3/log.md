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
execution time: IAR + RelationalAnalysis = 23.46 + 34.09 = 57.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.2320443, upper bound: 0.2320458

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 4667
type: A, layer: 1, pos: 5861
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 5848
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6139

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2320405, upper bound: 0.2297884
time: 5.01 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2320405, upper bound: 0.2320419
time: 3.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.86 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.86
Output dim: 5, lower bound: -0.2320405, upper bound: 0.2297884
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.86
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

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 5861
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 5848
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 106

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6139

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2297874
time: 3.86 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2297883
time: 4.72 seconds

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

Time for backsubstitution: 20.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 5861
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 5848
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 106

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6139

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2320420
time: 4.60 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2320408
time: 4.94 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.72 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 30.72
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2297874
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 30.72
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2297883
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.72
Output dim: 5, lower bound: -0.2297870, upper bound: 0.2320420
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.72
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

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 4667
type: A, layer: 1, pos: 5861
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 5848
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 106

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2295509, upper bound: 0.2320388
time: 5.75 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2297864, upper bound: 0.2320398
time: 4.89 seconds

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

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 4667
type: A, layer: 1, pos: 5861
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 5848
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2295509, upper bound: 0.2320393
time: 4.49 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2297864, upper bound: 0.2320403
time: 4.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.21 seconds
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.21
Output dim: 5, lower bound: -0.2295509, upper bound: 0.2320388
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.21
Output dim: 5, lower bound: -0.2297864, upper bound: 0.2320398
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.21
Output dim: 5, lower bound: -0.2295509, upper bound: 0.2320393
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.21
Output dim: 5, lower bound: -0.2297864, upper bound: 0.2320403

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -10.7466946, -9.2715254, -10.7506599, -9.2758389, -0.9725757, 0.9821377
1: -10.7431574, -9.5131092, -10.7470922, -9.5182686, -0.8866696, 0.8961954
2: -8.5543652, -7.6575270, -8.5535574, -7.6584864, -0.5418570, 0.5416679
3: -3.3011675, -2.1915307, -3.3005230, -2.1964185, -0.6469481, 0.6507740
4: -10.5637531, -9.2114029, -10.5468321, -9.2231016, -1.1066637, 1.1002045
5: 8.1156120, 8.8966351, 8.1220093, 8.8882065, -0.6129141, 0.6157255
6: -7.1352472, -5.8592682, -7.1325784, -5.8628330, -0.6542525, 0.6552875
7: -12.1584349, -10.7072258, -12.1526909, -10.7169867, -1.0929165, 1.0951376
8: -1.9676814, -1.1664009, -1.9672012, -1.1664364, -0.5466187, 0.5458691
9: -3.4089687, -2.5755560, -3.4035888, -2.5768559, -0.7931001, 0.7879913

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 5861
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 5848
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 831

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2282691, upper bound: 0.2318036
time: 3.65 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2293161, upper bound: 0.2318036
time: 3.87 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.7678356, -9.2351398, -10.7591734, -9.2750120, -0.9891968, 0.9993987
1: -10.7617836, -9.4785843, -10.7549896, -9.5178299, -0.9028974, 0.9161916
2: -8.5587883, -7.6548519, -8.5538731, -7.6575761, -0.5467017, 0.5437653
3: -3.3125052, -2.1741679, -3.3049653, -2.1961570, -0.6547771, 0.6612999
4: -10.5684681, -9.2046289, -10.5481091, -9.2224627, -1.1125388, 1.1076555
5: 8.1143045, 8.8986731, 8.1216774, 8.8883677, -0.6146936, 0.6198931
6: -7.1414638, -5.8502498, -7.1350060, -5.8627176, -0.6588316, 0.6657748
7: -12.1732264, -10.6800528, -12.1586838, -10.7157822, -1.1068978, 1.1080103
8: -1.9735880, -1.1625404, -1.9673796, -1.1651325, -0.5537705, 0.5490835
9: -3.4166641, -2.5724711, -3.4041471, -2.5755415, -0.8016908, 0.7928452

Time for backsubstitution: 22.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 5861
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 5848
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 106

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 831

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2285047, upper bound: 0.2318059
time: 3.72 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2295517, upper bound: 0.2318059
time: 3.96 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -10.7466946, -9.2715254, -10.7581120, -9.2703505, -0.9768319, 0.9881921
1: -10.7431574, -9.5131092, -10.7537470, -9.5124865, -0.8861423, 0.8965688
2: -8.5543652, -7.6575270, -8.5548162, -7.6563234, -0.5447142, 0.5435770
3: -3.3011675, -2.1915307, -3.3071260, -2.1911590, -0.6449547, 0.6503170
4: -10.5637531, -9.2114029, -10.5654449, -9.2105007, -1.1111250, 1.1123686
5: 8.1156120, 8.8966351, 8.1151781, 8.8968630, -0.6119518, 0.6124086
6: -7.1352472, -5.8592682, -7.1385136, -5.8591027, -0.6523831, 0.6553741
7: -12.1584349, -10.7072258, -12.1664591, -10.7055273, -1.0902967, 1.0971856
8: -1.9676814, -1.1664009, -1.9679365, -1.1646593, -0.5521488, 0.5505507
9: -3.4089687, -2.5755560, -3.4097376, -2.5737894, -0.7973108, 0.7955875

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 5861
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 5848
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 831

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2282691, upper bound: 0.2318041
time: 3.66 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2293161, upper bound: 0.2318041
time: 3.44 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.7678356, -9.2351398, -10.7666245, -9.2695265, -0.9934535, 1.0066810
1: -10.7617836, -9.4785843, -10.7616472, -9.5120478, -0.9023728, 0.9257121
2: -8.5587883, -7.6548519, -8.5551329, -7.6554132, -0.5495591, 0.5456650
3: -3.3125052, -2.1741679, -3.3115687, -2.1908987, -0.6527905, 0.6666553
4: -10.5684681, -9.2046289, -10.5667191, -9.2098637, -1.1169996, 1.1264400
5: 8.1143045, 8.8986731, 8.1148453, 8.8970261, -0.6137257, 0.6165862
6: -7.1414638, -5.8502498, -7.1409464, -5.8589859, -0.6569607, 0.6669981
7: -12.1732264, -10.6800528, -12.1724529, -10.7043200, -1.1041813, 1.1218061
8: -1.9735880, -1.1625404, -1.9681153, -1.1633549, -0.5593011, 0.5537641
9: -3.4166641, -2.5724711, -3.4103022, -2.5724757, -0.8053510, 0.8004456

Time for backsubstitution: 22.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 5861
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 5848
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 831

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2285047, upper bound: 0.2318064
time: 3.60 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2295517, upper bound: 0.2318063
time: 3.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.05 seconds
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 30.05
Output dim: 5, lower bound: -0.2282691, upper bound: 0.2318036
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 30.05
Output dim: 5, lower bound: -0.2293161, upper bound: 0.2318036
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 30.05
Output dim: 5, lower bound: -0.2285047, upper bound: 0.2318059
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 30.05
Output dim: 5, lower bound: -0.2295517, upper bound: 0.2318059
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 30.05
Output dim: 5, lower bound: -0.2282691, upper bound: 0.2318041
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 30.05
Output dim: 5, lower bound: -0.2293161, upper bound: 0.2318041
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 30.05
Output dim: 5, lower bound: -0.2285047, upper bound: 0.2318064
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 30.05
Output dim: 5, lower bound: -0.2295517, upper bound: 0.2318063

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 57.55 + 253.78 = 311.33 seconds
