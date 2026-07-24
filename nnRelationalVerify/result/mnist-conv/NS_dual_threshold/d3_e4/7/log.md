## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.9085582323


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0339298, 2.0339298)
1: (-19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4372330, 2.4372330)
2: (-6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9289122, 1.9289124)
3: (-10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3628707, 2.3628707)
4: (-13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2080922, 2.2080922)
5: (-4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7177835, 1.7177835)
6: (-4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0725527, 2.0725527)
7: (-12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9651766, 2.9651771)
8: (-5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4434562, 1.4434562)
9: (-1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6988459, 2.6988463)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.32 + 34.97 = 59.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.9094677, upper bound: 0.9094670

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9062302, upper bound: 0.9092268
time: 5.46 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094651, upper bound: 0.9094648
time: 5.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.74 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 10.74
Output dim: 0, lower bound: -0.9062302, upper bound: 0.9092268
NS_B2, status: Status.UNKNOWN, split count: 1, time: 10.74
Output dim: 0, lower bound: -0.9094651, upper bound: 0.9094648

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: 7.7583418, 10.2263012, 7.7684436, 10.2174206, -2.0093207, 2.0086422
1: -19.2421150, -15.2778959, -19.2248154, -15.2928419, -2.3981991, 2.3965278
2: -6.5179687, -3.5506084, -6.5115719, -3.5520186, -1.9175253, 1.9118516
3: -10.8063602, -7.7944455, -10.7933283, -7.8039846, -2.3324652, 2.3365135
4: -13.5792742, -10.5961924, -13.5695925, -10.6057529, -2.1838589, 2.1831961
5: -4.6331673, -2.1629701, -4.6272173, -2.1693697, -1.6991425, 1.7008159
6: -4.5127392, -1.9435682, -4.4925213, -1.9654670, -2.0200939, 2.0217962
7: -12.8206997, -8.7985296, -12.8079014, -8.8138399, -2.9324150, 2.9304304
8: -5.4464722, -3.1581054, -5.4340744, -3.1679492, -1.4175701, 1.4148602
9: -1.9062848, 1.0458272, -1.8838611, 1.0316906, -2.6576643, 2.6496611

Time for backsubstitution: 22.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9022344, upper bound: 0.9089407
time: 5.07 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9062216, upper bound: 0.9092188
time: 5.11 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: 7.7540412, 10.2373829, 7.7540426, 10.2373781, -2.0214620, 2.0339231
1: -19.2597122, -15.2714062, -19.2597046, -15.2714100, -2.4372220, 2.4239225
2: -6.5238466, -3.5489054, -6.5238447, -3.5489085, -1.9291382, 1.9289050
3: -10.8192272, -7.7928057, -10.8192215, -7.7928076, -2.3700876, 2.3615274
4: -13.5905027, -10.5921230, -13.5904989, -10.5921268, -2.2073436, 2.1927252
5: -4.6404042, -2.1593943, -4.6404023, -2.1593964, -1.7160425, 1.7138946
6: -4.5149150, -1.9158911, -4.5149150, -1.9159002, -2.0253797, 2.0725412
7: -12.8235617, -8.7824402, -12.8235607, -8.7824450, -2.9439578, 2.9628000
8: -5.4501815, -3.1462460, -5.4501791, -3.1462479, -1.4229407, 1.4434507
9: -1.9316516, 1.0465152, -1.9316463, 1.0465159, -2.6988373, 2.6767364

Time for backsubstitution: 21.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of NS_B2_B1

### Relational analysis result of NS_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054694, upper bound: 0.9091785
time: 5.33 seconds

## Relational analysis of NS_B2_B2

### Relational analysis result of NS_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094565, upper bound: 0.9094567
time: 4.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.92 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 31.92
Output dim: 0, lower bound: -0.9022344, upper bound: 0.9089407
NS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 31.92
Output dim: 0, lower bound: -0.9062216, upper bound: 0.9092188
NS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 31.92
Output dim: 0, lower bound: -0.9054694, upper bound: 0.9091785
NS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 31.92
Output dim: 0, lower bound: -0.9094565, upper bound: 0.9094567

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: 7.7605095, 10.2140265, 7.7782645, 10.1912088, -1.9802647, 1.9862800
1: -19.2389679, -15.2791605, -19.2184887, -15.2969675, -2.3900418, 2.3936734
2: -6.5118265, -3.5512085, -6.4989200, -3.5554147, -1.9078321, 1.8896523
3: -10.8052006, -7.7961988, -10.7896690, -7.8088312, -2.3243384, 2.3288794
4: -13.5696869, -10.5969992, -13.5483284, -10.6108551, -2.1669474, 2.1613374
5: -4.6324425, -2.1666446, -4.6248541, -2.1774473, -1.6887431, 1.6939867
6: -4.5104718, -1.9504185, -4.4857645, -1.9801084, -2.0058079, 2.0059457
7: -12.8084574, -8.7994890, -12.7819271, -8.8199358, -2.9125834, 2.9036021
8: -5.4447842, -3.1635914, -5.4283876, -3.1802025, -1.4035530, 1.3988178
9: -1.8937225, 1.0456927, -1.8567615, 1.0274448, -2.6415319, 2.6222458

Time for backsubstitution: 22.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of NS_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_B1_B1_B1

### Relational analysis result of NS_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9022339, upper bound: 0.9087299
time: 4.83 seconds

## Relational analysis of NS_B1_B1_B2

### Relational analysis result of NS_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9022339, upper bound: 0.9089397
time: 6.67 seconds

## BFS NS instance: NS_B1_B2

### Backsubstitution after applying NS history:
0: 7.7583432, 10.2262993, 7.7684436, 10.2174129, -1.9868279, 2.0086408
1: -19.2421150, -15.2778997, -19.2248116, -15.2928429, -2.3980584, 2.3926082
2: -6.5179672, -3.5506065, -6.5115662, -3.5520165, -1.9175243, 1.9052658
3: -10.8063583, -7.7944460, -10.7933292, -7.8039842, -2.3417983, 2.3343863
4: -13.5792732, -10.5961924, -13.5695877, -10.6057529, -2.1832905, 2.1697469
5: -4.6331673, -2.1629694, -4.6272178, -2.1693707, -1.6984406, 1.7039142
6: -4.5127397, -1.9435685, -4.4925199, -1.9654701, -2.0105572, 2.0211244
7: -12.8206987, -8.7985258, -12.8078957, -8.8138418, -2.9321842, 2.9106069
8: -5.4464712, -3.1581063, -5.4340749, -3.1679511, -1.4077888, 1.4148586
9: -1.9062839, 1.0458274, -1.8838577, 1.0316916, -2.6576653, 2.6454458

Time for backsubstitution: 22.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5736

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of NS_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_B1_B2_B1

### Relational analysis result of NS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9062211, upper bound: 0.9090082
time: 5.37 seconds

## Relational analysis of NS_B1_B2_B2

### Relational analysis result of NS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9062211, upper bound: 0.9092184
time: 5.39 seconds

## BFS NS instance: NS_B2_B1

### Backsubstitution after applying NS history:
0: 7.7561874, 10.2251101, 7.7637677, 10.2111721, -1.9923716, 2.0116658
1: -19.2565613, -15.2726498, -19.2533913, -15.2754498, -2.4291368, 2.4210997
2: -6.5176930, -3.5495090, -6.5111160, -3.5523095, -1.9194317, 1.9066129
3: -10.8180676, -7.7945495, -10.8155165, -7.7976041, -2.3619838, 2.3538389
4: -13.5809021, -10.5929241, -13.5691814, -10.5972080, -2.1903963, 2.1707821
5: -4.6396780, -2.1630447, -4.6380396, -2.1673663, -1.7057829, 1.7070408
6: -4.5126591, -1.9227444, -4.5081911, -1.9305447, -2.0110955, 2.0567122
7: -12.8113213, -8.7834005, -12.7976408, -8.7885475, -2.9241524, 2.9360294
8: -5.4485078, -3.1517344, -5.4445019, -3.1584911, -1.4089355, 1.4274731
9: -1.9190817, 1.0463800, -1.9045296, 1.0422697, -2.6826868, 2.6492395

Time for backsubstitution: 22.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of NS_B2_B1_A1

### Relational analysis result of NS_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9052838, upper bound: 0.9053072
time: 5.31 seconds

## Relational analysis of NS_B2_B1_A2

### Relational analysis result of NS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054668, upper bound: 0.9091755
time: 6.27 seconds

## BFS NS instance: NS_B2_B2

### Backsubstitution after applying NS history:
0: 7.7540412, 10.2373829, 7.7540436, 10.2373724, -1.9989696, 2.0339208
1: -19.2597084, -15.2714071, -19.2597008, -15.2714119, -2.4370718, 2.4200568
2: -6.5238466, -3.5489070, -6.5238414, -3.5489101, -1.9291382, 1.9223180
3: -10.8192263, -7.7928076, -10.8192225, -7.7928100, -2.3794374, 2.3594108
4: -13.5905037, -10.5921240, -13.5904951, -10.5921268, -2.2067738, 2.1792388
5: -4.6404052, -2.1593947, -4.6404009, -2.1593983, -1.7153292, 1.7170415
6: -4.5149155, -1.9158930, -4.5149150, -1.9159026, -2.0158796, 2.0718651
7: -12.8235607, -8.7824411, -12.8235550, -8.7824459, -2.9437203, 2.9430227
8: -5.4501810, -3.1462469, -5.4501781, -3.1462493, -1.4131589, 1.4434495
9: -1.9316521, 1.0465152, -1.9316425, 1.0465150, -2.6988382, 2.6725235

Time for backsubstitution: 22.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of NS_B2_B2_A1

### Relational analysis result of NS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092710, upper bound: 0.9055855
time: 5.06 seconds

## Relational analysis of NS_B2_B2_A2

### Relational analysis result of NS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094539, upper bound: 0.9094538
time: 5.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.21 seconds
NS_B1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 33.21
Output dim: 0, lower bound: -0.9022339, upper bound: 0.9087299
NS_B1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 33.21
Output dim: 0, lower bound: -0.9022339, upper bound: 0.9089397
NS_B1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 33.21
Output dim: 0, lower bound: -0.9062211, upper bound: 0.9090082
NS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 33.21
Output dim: 0, lower bound: -0.9062211, upper bound: 0.9092184
NS_B2_B1_A1, status: Status.VERIFIED, split count: 3, time: 33.21
Output dim: 0, lower bound: -0.9052838, upper bound: 0.9053072
NS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.21
Output dim: 0, lower bound: -0.9054668, upper bound: 0.9091755
NS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 33.21
Output dim: 0, lower bound: -0.9092710, upper bound: 0.9055855
NS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.21
Output dim: 0, lower bound: -0.9094539, upper bound: 0.9094538

## BFS NS instance: NS_B1_B1_B1

### Backsubstitution after applying NS history:
0: 7.7609854, 10.2137823, 7.7806301, 10.1899815, -1.9748116, 1.9795518
1: -19.2368107, -15.2793522, -19.2077332, -15.2979107, -2.3867798, 2.3822956
2: -6.5116215, -3.5531299, -6.4978981, -3.5649991, -1.8980818, 1.8868709
3: -10.8046789, -7.7985229, -10.7870531, -7.8204193, -2.3124647, 2.3242159
4: -13.5683289, -10.5972376, -13.5415659, -10.6120348, -2.1647820, 2.1543827
5: -4.6307125, -2.1672230, -4.6162305, -2.1803133, -1.6840568, 1.6847439
6: -4.5092702, -1.9505336, -4.4798212, -1.9806831, -1.9998460, 1.9962459
7: -12.8079557, -8.8029070, -12.7794075, -8.8369780, -2.8940916, 2.8955250
8: -5.4443598, -3.1638970, -5.4262934, -3.1817365, -1.3980792, 1.3935025
9: -1.8927178, 1.0439754, -1.8517427, 1.0188849, -2.6321392, 2.6163650

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 900

### Candidate
type: A, layer: 1, pos: 900

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of NS_B1_B1_B1_A1

### Relational analysis result of NS_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9022340, upper bound: 0.9063981
time: 5.16 seconds

## Relational analysis of NS_B1_B1_B1_A2

### Relational analysis result of NS_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9022340, upper bound: 0.9087298
time: 4.97 seconds

## BFS NS instance: NS_B1_B1_B2

### Backsubstitution after applying NS history:
0: 7.7605157, 10.2140255, 7.7653246, 10.1997166, -1.9971409, 1.9948649
1: -19.2389603, -15.2791615, -19.2253208, -15.2469654, -2.4230361, 2.3980446
2: -6.5118241, -3.5512173, -6.5393357, -3.5509348, -1.9111996, 1.9312689
3: -10.8051977, -7.7962084, -10.8433599, -7.8034410, -2.3289986, 2.3622003
4: -13.5696831, -10.5970001, -13.5508490, -10.5742693, -2.1906424, 2.1629763
5: -4.6324344, -2.1666496, -4.6317439, -2.1358285, -1.7129469, 1.6991348
6: -4.5104680, -1.9504189, -4.4913349, -1.9598165, -2.0222831, 2.0181487
7: -12.8084564, -8.7995033, -12.8587809, -8.8074350, -2.9278665, 2.9399848
8: -5.4447794, -3.1635933, -5.4382477, -3.1745214, -1.4062035, 1.4137604
9: -1.8937154, 1.0456843, -1.9152761, 1.0279946, -2.6415644, 2.6825843

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 900

### Candidate
type: A, layer: 1, pos: 900

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of NS_B1_B1_B2_A1

### Relational analysis result of NS_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9022340, upper bound: 0.9066083
time: 4.84 seconds

## Relational analysis of NS_B1_B1_B2_A2

### Relational analysis result of NS_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9022340, upper bound: 0.9089398
time: 7.07 seconds

## BFS NS instance: NS_B1_B2_B1

### Backsubstitution after applying NS history:
0: 7.7588153, 10.2260580, 7.7708039, 10.2161932, -1.9813848, 2.0019150
1: -19.2399597, -15.2780933, -19.2140694, -15.2937965, -2.3947830, 2.3812227
2: -6.5177617, -3.5525284, -6.5105448, -3.5616090, -1.9077606, 1.9024925
3: -10.8058319, -7.7967710, -10.7906685, -7.8155804, -2.3299108, 2.3296618
4: -13.5779209, -10.5964317, -13.5628271, -10.6069450, -2.1811066, 2.1627913
5: -4.6314397, -2.1635451, -4.6185923, -2.1722479, -1.6938133, 1.6946702
6: -4.5115380, -1.9436848, -4.4865599, -1.9660449, -2.0045948, 2.0113893
7: -12.8201942, -8.8019457, -12.8053637, -8.8308926, -2.9136724, 2.9025245
8: -5.4460478, -3.1584082, -5.4319715, -3.1694813, -1.4023161, 1.4095407
9: -1.9052811, 1.0441108, -1.8788443, 1.0231292, -2.6482611, 2.6395726

Time for backsubstitution: 22.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5736

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 900

### Candidate
type: B, layer: 1, pos: 900

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of NS_B1_B2_B1_A1

### Relational analysis result of NS_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9062211, upper bound: 0.9066762
time: 5.59 seconds

## Relational analysis of NS_B1_B2_B1_A2

### Relational analysis result of NS_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9062211, upper bound: 0.9090082
time: 5.01 seconds

## BFS NS instance: NS_B1_B2_B2

### Backsubstitution after applying NS history:
0: 7.7583466, 10.2263002, 7.7554870, 10.2259350, -2.0035815, 2.0172114
1: -19.2421074, -15.2778997, -19.2316551, -15.2428751, -2.4308915, 2.3969631
2: -6.5179663, -3.5506170, -6.5520191, -3.5475364, -1.9208822, 1.9458303
3: -10.8063583, -7.7944546, -10.8470058, -7.7986088, -2.3464518, 2.3673120
4: -13.5792694, -10.5961943, -13.5721283, -10.5691833, -2.2080221, 2.1713915
5: -4.6331587, -2.1629715, -4.6340742, -2.1277928, -1.7226820, 1.7090359
6: -4.5127339, -1.9435697, -4.4981132, -1.9451671, -2.0270233, 2.0352681
7: -12.8206949, -8.7985392, -12.8847008, -8.8013763, -2.9474864, 2.9457331
8: -5.4464693, -3.1581097, -5.4439526, -3.1622710, -1.4104395, 1.4315305
9: -1.9062757, 1.0458207, -1.9424295, 1.0322371, -2.6576891, 2.7052879

Time for backsubstitution: 23.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 5736

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 900

### Candidate
type: B, layer: 1, pos: 900

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of NS_B1_B2_B2_A1

### Relational analysis result of NS_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9062211, upper bound: 0.9068867
time: 5.56 seconds

## Relational analysis of NS_B1_B2_B2_A2

### Relational analysis result of NS_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9062211, upper bound: 0.9092184
time: 5.66 seconds

## BFS NS instance: NS_B2_B1_A2

### Backsubstitution after applying NS history:
0: 7.7507591, 10.2255821, 7.7637787, 10.2111645, -1.9976888, 2.0083394
1: -19.2590313, -15.2659512, -19.2533627, -15.2754517, -2.4280367, 2.4276462
2: -6.5205817, -3.5485184, -6.5111151, -3.5524166, -1.9204097, 1.9053512
3: -10.8210840, -7.7849011, -10.8155069, -7.7977200, -2.3620586, 2.3636856
4: -13.5813370, -10.5878687, -13.5689201, -10.5972071, -2.1871319, 2.1727319
5: -4.6400681, -2.1589491, -4.6379337, -2.1673672, -1.7018056, 1.7048764
6: -4.5307016, -1.9217166, -4.5081825, -1.9305625, -2.0286422, 2.0470948
7: -12.8177834, -8.7821712, -12.7976236, -8.7885571, -2.9206514, 2.9309711
8: -5.4621353, -3.1508436, -5.4445028, -3.1585984, -1.4223061, 1.4209166
9: -1.9217982, 1.0568821, -1.9045196, 1.0422387, -2.6824226, 2.6599607

Time for backsubstitution: 23.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_B2_B1_A2_B1

### Relational analysis result of NS_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054663, upper bound: 0.9089648
time: 6.51 seconds

## Relational analysis of NS_B2_B1_A2_B2

### Relational analysis result of NS_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054663, upper bound: 0.9091751
time: 5.01 seconds

## BFS NS instance: NS_B2_B2_A1

### Backsubstitution after applying NS history:
0: 7.7556772, 10.2316847, 7.7550163, 10.2339811, -1.9933524, 2.0271893
1: -19.2533798, -15.2738934, -19.2559319, -15.2728882, -2.4290771, 2.4134393
2: -6.5225010, -3.5524826, -6.5230436, -3.5519905, -1.9258857, 1.9176393
3: -10.8098392, -7.7941284, -10.8136358, -7.7935925, -2.3687811, 2.3520598
4: -13.5849333, -10.5930042, -13.5871792, -10.5926495, -2.2003961, 2.1740141
5: -4.6357183, -2.1609702, -4.6376114, -2.1603355, -1.7088423, 1.7106910
6: -4.5128360, -1.9327724, -4.5136771, -1.9259490, -2.0031457, 2.0533872
7: -12.8226576, -8.7883568, -12.8230190, -8.7859688, -2.9403911, 2.9352102
8: -5.4487209, -3.1583304, -5.4493113, -3.1534381, -1.4042759, 1.4303398
9: -1.9189320, 1.0461745, -1.9240723, 1.0463130, -2.6857738, 2.6645064

Time for backsubstitution: 23.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_B2_B2_A1_B1

### Relational analysis result of NS_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092705, upper bound: 0.9053748
time: 5.19 seconds

## Relational analysis of NS_B2_B2_A1_B2

### Relational analysis result of NS_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092705, upper bound: 0.9055849
time: 5.03 seconds

## BFS NS instance: NS_B2_B2_A2

### Backsubstitution after applying NS history:
0: 7.7486134, 10.2378531, 7.7540531, 10.2373638, -2.0042267, 2.0305896
1: -19.2621784, -15.2647114, -19.2596741, -15.2714148, -2.4359879, 2.4266014
2: -6.5266995, -3.5479176, -6.5238414, -3.5490191, -1.9300728, 1.9210477
3: -10.8222647, -7.7831631, -10.8192120, -7.7929277, -2.3794470, 2.3692570
4: -13.5909395, -10.5870667, -13.5902367, -10.5921288, -2.2035122, 2.1811676
5: -4.6407928, -2.1552992, -4.6402950, -2.1593997, -1.7113476, 1.7148085
6: -4.5329590, -1.9148659, -4.5149045, -1.9159209, -2.0334182, 2.0622544
7: -12.8300133, -8.7812052, -12.8235359, -8.7824516, -2.9402657, 2.9379649
8: -5.4638042, -3.1453609, -5.4501753, -3.1463542, -1.4265294, 1.4369042
9: -1.9343653, 1.0570166, -1.9316335, 1.0464818, -2.6985674, 2.6832423

Time for backsubstitution: 23.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_B2_B2_A2_B1

### Relational analysis result of NS_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094534, upper bound: 0.9092431
time: 5.05 seconds

## Relational analysis of NS_B2_B2_A2_B2

### Relational analysis result of NS_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094534, upper bound: 0.9094532
time: 4.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 33.79 seconds
NS_B1_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 33.79
Output dim: 0, lower bound: -0.9022340, upper bound: 0.9063981
NS_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -0.9022340, upper bound: 0.9087298
NS_B1_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 33.79
Output dim: 0, lower bound: -0.9022340, upper bound: 0.9066083
NS_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -0.9022340, upper bound: 0.9089398
NS_B1_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 33.79
Output dim: 0, lower bound: -0.9062211, upper bound: 0.9066762
NS_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -0.9062211, upper bound: 0.9090082
NS_B1_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 33.79
Output dim: 0, lower bound: -0.9062211, upper bound: 0.9068867
NS_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -0.9062211, upper bound: 0.9092184
NS_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -0.9054663, upper bound: 0.9089648
NS_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -0.9054663, upper bound: 0.9091751
NS_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -0.9092705, upper bound: 0.9053748
NS_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -0.9092705, upper bound: 0.9055849
NS_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -0.9094534, upper bound: 0.9092431
NS_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -0.9094534, upper bound: 0.9094532

## BFS NS instance: NS_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: 7.7566609, 10.2248631, 7.7806301, 10.1899815, -1.9793878, 1.9834876
1: -19.2543983, -15.2728519, -19.2077332, -15.2979107, -2.4042387, 2.3886909
2: -6.5174813, -3.5514321, -6.4978981, -3.5649991, -1.9055824, 1.8896341
3: -10.8175411, -7.7968760, -10.7870531, -7.8204193, -2.3287497, 2.3232360
4: -13.5795412, -10.5931711, -13.5415659, -10.6120348, -2.1756625, 2.1576805
5: -4.6379480, -2.1636262, -4.6162305, -2.1803133, -1.6918063, 1.6869044
6: -4.5114536, -1.9228693, -4.4798212, -1.9806831, -2.0025611, 2.0072927
7: -12.8108158, -8.7868195, -12.7794075, -8.8369780, -2.8940315, 2.9144115
8: -5.4480834, -3.1520395, -5.4262934, -3.1817365, -1.4023182, 1.4002421
9: -1.9180698, 1.0446644, -1.8517427, 1.0188849, -2.6578598, 2.6172347

Time for backsubstitution: 23.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 900

### Candidate
type: B, layer: 1, pos: 900

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of NS_B1_B1_B1_A2_A1

### Relational analysis result of NS_B1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9022340, upper bound: 0.9050218
time: 5.17 seconds

## Relational analysis of NS_B1_B1_B1_A2_A2

### Relational analysis result of NS_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9022340, upper bound: 0.9087298
time: 5.02 seconds

## BFS NS instance: NS_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: 7.7561932, 10.2251043, 7.7653246, 10.1997166, -1.9971581, 1.9988718
1: -19.2565441, -15.2726545, -19.2253208, -15.2469654, -2.4270554, 2.4044456
2: -6.5176864, -3.5495210, -6.5393357, -3.5509348, -1.9187016, 1.9319251
3: -10.8180571, -7.7945604, -10.8433599, -7.8034410, -2.3452806, 2.3548751
4: -13.5808926, -10.5929289, -13.5508490, -10.5742693, -2.1931543, 2.1662779
5: -4.6396675, -2.1630509, -4.6317439, -2.1358285, -1.7162328, 1.7013035
6: -4.5126510, -1.9227527, -4.4913349, -1.9598165, -2.0177498, 2.0215483
7: -12.8113194, -8.7834215, -12.8587809, -8.8074350, -2.9278121, 2.9449496
8: -5.4485054, -3.1517372, -5.4382477, -3.1745214, -1.4104469, 1.4151175
9: -1.9190702, 1.0463738, -1.9152761, 1.0279946, -2.6672955, 2.6721601

Time for backsubstitution: 23.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 900

### Candidate
type: B, layer: 1, pos: 900

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of NS_B1_B1_B2_A2_A1

### Relational analysis result of NS_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9022340, upper bound: 0.9052318
time: 6.63 seconds

## Relational analysis of NS_B1_B1_B2_A2_A2

### Relational analysis result of NS_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9022340, upper bound: 0.9089397
time: 7.41 seconds

## BFS NS instance: NS_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 7.7545152, 10.2371330, 7.7708039, 10.2161932, -1.9859395, 2.0105166
1: -19.2575474, -15.2716112, -19.2140694, -15.2937965, -2.4122391, 2.3876381
2: -6.5236382, -3.5508304, -6.5105448, -3.5616090, -1.9152727, 1.9052615
3: -10.8186951, -7.7951345, -10.7906685, -7.8155804, -2.3462167, 2.3286786
4: -13.5891418, -10.5923710, -13.5628271, -10.6069450, -2.1919904, 2.1661077
5: -4.6386752, -2.1599753, -4.6185923, -2.1722479, -1.7015591, 1.6968565
6: -4.5137091, -1.9160167, -4.4865599, -1.9660449, -2.0073309, 2.0243955
7: -12.8230534, -8.7858582, -12.8053637, -8.8308926, -2.9136086, 2.9213638
8: -5.4497547, -3.1465521, -5.4319715, -3.1694813, -1.4065475, 1.4180241
9: -1.9306412, 1.0448010, -1.8788443, 1.0231292, -2.6740084, 2.6404424

Time for backsubstitution: 23.06 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 59.29 + 552.90 = 612.19 seconds
