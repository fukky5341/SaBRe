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
execution time: IAR + RelationalAnalysis = 22.82 + 34.70 = 57.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.9094677, upper bound: 0.9094670

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092271, upper bound: 0.9062301
time: 4.77 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094654, upper bound: 0.9094644
time: 5.14 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.01 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.01
Output dim: 0, lower bound: -0.9092271, upper bound: 0.9062301
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.01
Output dim: 0, lower bound: -0.9094654, upper bound: 0.9094644

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 7.7684436, 10.2174206, 7.7583418, 10.2263012, -2.0086422, 2.0093207
1: -19.2248154, -15.2928419, -19.2421150, -15.2778959, -2.3965278, 2.3981991
2: -6.5115719, -3.5520186, -6.5179687, -3.5506084, -1.9118519, 1.9175253
3: -10.7933283, -7.8039846, -10.8063602, -7.7944455, -2.3365145, 2.3324652
4: -13.5695925, -10.6057529, -13.5792742, -10.5961924, -2.1831961, 2.1838584
5: -4.6272173, -2.1693697, -4.6331673, -2.1629701, -1.7008162, 1.6991422
6: -4.4925213, -1.9654670, -4.5127392, -1.9435682, -2.0217962, 2.0200939
7: -12.8079014, -8.8138399, -12.8206997, -8.7985296, -2.9304304, 2.9324145
8: -5.4340744, -3.1679492, -5.4464722, -3.1581054, -1.4148600, 1.4175706
9: -1.8838611, 1.0316906, -1.9062848, 1.0458272, -2.6496611, 2.6576648

Time for backsubstitution: 21.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9052313, upper bound: 0.9059436
time: 5.20 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092185, upper bound: 0.9062220
time: 5.01 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 7.7540426, 10.2373781, 7.7540412, 10.2373829, -2.0339236, 2.0214620
1: -19.2597046, -15.2714100, -19.2597122, -15.2714062, -2.4239225, 2.4372215
2: -6.5238447, -3.5489085, -6.5238466, -3.5489054, -1.9289050, 1.9291382
3: -10.8192215, -7.7928076, -10.8192272, -7.7928057, -2.3615270, 2.3700871
4: -13.5904989, -10.5921268, -13.5905027, -10.5921230, -2.1927252, 2.2073436
5: -4.6404023, -2.1593964, -4.6404042, -2.1593943, -1.7138948, 1.7160428
6: -4.5149150, -1.9159002, -4.5149150, -1.9158911, -2.0725412, 2.0253797
7: -12.8235607, -8.7824450, -12.8235617, -8.7824402, -2.9628000, 2.9439583
8: -5.4501791, -3.1462479, -5.4501815, -3.1462460, -1.4434507, 1.4229405
9: -1.9316463, 1.0465159, -1.9316516, 1.0465152, -2.6767359, 2.6988373

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054697, upper bound: 0.9091782
time: 4.93 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094568, upper bound: 0.9094590
time: 5.42 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 32.03 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 32.03
Output dim: 0, lower bound: -0.9052313, upper bound: 0.9059436
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 32.03
Output dim: 0, lower bound: -0.9092185, upper bound: 0.9062220
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 32.03
Output dim: 0, lower bound: -0.9054697, upper bound: 0.9091782
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 32.03
Output dim: 0, lower bound: -0.9094568, upper bound: 0.9094590

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 7.7684426, 10.2174187, 7.7583442, 10.2262964, -1.9861493, 2.0093193
1: -19.2248135, -15.2928410, -19.2421131, -15.2778969, -2.3963823, 2.3942800
2: -6.5115685, -3.5520170, -6.5179648, -3.5506094, -1.9118509, 1.9109395
3: -10.7933283, -7.8039837, -10.8063593, -7.7944455, -2.3459301, 2.3303499
4: -13.5695915, -10.6057529, -13.5792732, -10.5961933, -2.1826286, 2.1703897
5: -4.6272192, -2.1693690, -4.6331673, -2.1629701, -1.7001109, 1.7022266
6: -4.4925189, -1.9654664, -4.5127397, -1.9435723, -2.0122833, 2.0194230
7: -12.8079014, -8.8138409, -12.8206940, -8.7985287, -2.9301996, 2.9126205
8: -5.4340734, -3.1679502, -5.4464712, -3.1581078, -1.4050791, 1.4175692
9: -1.8838601, 1.0316906, -1.9062800, 1.0458272, -2.6496582, 2.6534529

Time for backsubstitution: 21.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9089408, upper bound: 0.9022342
time: 5.17 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9089408, upper bound: 0.9062217
time: 6.22 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 7.7561884, 10.2251043, 7.7637653, 10.2111750, -2.0048909, 1.9990878
1: -19.2565517, -15.2726517, -19.2533970, -15.2754459, -2.4158397, 2.4343972
2: -6.5176902, -3.5495076, -6.5111189, -3.5523109, -1.9191961, 1.9068465
3: -10.8180637, -7.7945485, -10.8155165, -7.7976017, -2.3534236, 2.3624010
4: -13.5808983, -10.5929289, -13.5691881, -10.5972061, -2.1757522, 2.1854148
5: -4.6396770, -2.1630466, -4.6380391, -2.1673636, -1.7035303, 1.7092440
6: -4.5126576, -1.9227533, -4.5081916, -1.9305370, -2.0582571, 2.0095491
7: -12.8113194, -8.7834044, -12.7976437, -8.7885437, -2.9429631, 2.9172101
8: -5.4485083, -3.1517334, -5.4445047, -3.1584902, -1.4294462, 1.4069622
9: -1.9190769, 1.0463793, -1.9045353, 1.0422716, -2.6605864, 2.6713428

Time for backsubstitution: 22.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9052840, upper bound: 0.9053071
time: 5.79 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054671, upper bound: 0.9091752
time: 4.85 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 7.7540417, 10.2373772, 7.7540421, 10.2373753, -2.0114307, 2.0214596
1: -19.2597065, -15.2714109, -19.2597122, -15.2714081, -2.4237728, 2.4333539
2: -6.5238447, -3.5489087, -6.5238447, -3.5489082, -1.9289026, 1.9225526
3: -10.8192205, -7.7928076, -10.8192253, -7.7928071, -2.3708777, 2.3679738
4: -13.5904970, -10.5921268, -13.5905008, -10.5921230, -2.1921577, 2.1938567
5: -4.6404023, -2.1593966, -4.6404028, -2.1593976, -1.7131801, 1.7191887
6: -4.5149145, -1.9159007, -4.5149155, -1.9158955, -2.0630417, 2.0247040
7: -12.8235588, -8.7824450, -12.8235531, -8.7824411, -2.9625640, 2.9241791
8: -5.4501791, -3.1462493, -5.4501801, -3.1462488, -1.4336696, 1.4229381
9: -1.9316473, 1.0465150, -1.9316483, 1.0465150, -2.6767330, 2.6946259

Time for backsubstitution: 23.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9091791, upper bound: 0.9054687
time: 4.62 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9091791, upper bound: 0.9094565
time: 4.86 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 32.76 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.76
Output dim: 0, lower bound: -0.9089408, upper bound: 0.9022342
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.76
Output dim: 0, lower bound: -0.9089408, upper bound: 0.9062217
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 32.76
Output dim: 0, lower bound: -0.9052840, upper bound: 0.9053071
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.76
Output dim: 0, lower bound: -0.9054671, upper bound: 0.9091752
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.76
Output dim: 0, lower bound: -0.9091791, upper bound: 0.9054687
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.76
Output dim: 0, lower bound: -0.9091791, upper bound: 0.9094565

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 7.7782645, 10.1912088, 7.7585568, 10.2262945, -1.9881191, 1.9822254
1: -19.2184887, -15.2969675, -19.2421131, -15.2779980, -2.3948536, 2.3938050
2: -6.4989200, -3.5554147, -6.5179577, -3.5506084, -1.8903232, 1.9139595
3: -10.7896690, -7.8088312, -10.8059406, -7.7944446, -2.3288054, 2.3240576
4: -13.5483284, -10.6108551, -13.5792713, -10.5962706, -2.1614256, 2.1784515
5: -4.6248541, -2.1774473, -4.6331673, -2.1630926, -1.6971207, 1.6886866
6: -4.4857645, -1.9801084, -4.5127363, -1.9435705, -2.0137963, 2.0076537
7: -12.7819271, -8.8199358, -12.8206539, -8.7985296, -2.9050426, 2.9253716
8: -5.4283876, -3.1802025, -5.4464712, -3.1581221, -1.4044185, 1.4055109
9: -1.8567615, 1.0274448, -1.9062781, 1.0457592, -2.6224146, 2.6542468

Time for backsubstitution: 23.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9028996, upper bound: 0.9022340
time: 4.77 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9028996, upper bound: 0.9022339
time: 5.65 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 7.7684436, 10.2174129, 7.7583442, 10.2262964, -1.9861484, 1.9868279
1: -19.2248116, -15.2928429, -19.2421131, -15.2778969, -2.3926072, 2.3942795
2: -6.5115662, -3.5520165, -6.5179648, -3.5506094, -1.9052658, 1.9109385
3: -10.7933292, -7.8039842, -10.8063593, -7.7944455, -2.3459282, 2.3417959
4: -13.5695877, -10.6057529, -13.5792732, -10.5961933, -2.1697459, 2.1703882
5: -4.6272178, -2.1693707, -4.6331673, -2.1629701, -1.7039118, 1.7022240
6: -4.4925199, -1.9654701, -4.5127397, -1.9435723, -2.0122809, 2.0105557
7: -12.8078957, -8.8138418, -12.8206940, -8.7985287, -2.9106064, 2.9126191
8: -5.4340749, -3.1679511, -5.4464712, -3.1581078, -1.4050782, 1.4077880
9: -1.8838577, 1.0316916, -1.9062800, 1.0458272, -2.6454439, 2.6534534

Time for backsubstitution: 22.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9028996, upper bound: 0.9022342
time: 5.34 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9028996, upper bound: 0.9022342
time: 6.24 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 7.7507591, 10.2255783, 7.7637768, 10.2111673, -2.0107312, 1.9961858
1: -19.2590256, -15.2659559, -19.2533703, -15.2754478, -2.4147348, 2.4409423
2: -6.5200462, -3.5485320, -6.5111179, -3.5524154, -1.9211988, 1.9071739
3: -10.8210764, -7.7849083, -10.8155098, -7.7977228, -2.3541675, 2.3722401
4: -13.5813198, -10.5878677, -13.5689230, -10.5972071, -2.1729074, 2.1900716
5: -4.6400518, -2.1589489, -4.6379366, -2.1673667, -1.7033620, 1.7132971
6: -4.5306983, -1.9217249, -4.5081816, -1.9305538, -2.0646272, 1.9999309
7: -12.8177805, -8.7821751, -12.7976255, -8.7885504, -2.9491758, 2.9211841
8: -5.4621334, -3.1508493, -5.4445038, -3.1585970, -1.4364266, 1.4004011
9: -1.9217930, 1.0568793, -1.9045262, 1.0422382, -2.6603193, 2.6820617

Time for backsubstitution: 22.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9015984, upper bound: 0.9089926
time: 5.84 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9015984, upper bound: 0.9091757
time: 4.25 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 7.7637677, 10.2111721, 7.7542558, 10.2373753, -2.0127234, 1.9943695
1: -19.2533913, -15.2754498, -19.2597065, -15.2715025, -2.4222608, 2.4328990
2: -6.5111160, -3.5523095, -6.5238366, -3.5489101, -1.9072824, 1.9255695
3: -10.8155165, -7.7976041, -10.8188076, -7.7928090, -2.3537626, 2.3617072
4: -13.5691814, -10.5972080, -13.5904980, -10.5922012, -2.1708884, 2.2019024
5: -4.6380396, -2.1673663, -4.6404033, -2.1595206, -1.7101870, 1.7057223
6: -4.5081911, -1.9305447, -4.5149121, -1.9158945, -2.0587111, 2.0129247
7: -12.7976408, -8.7885475, -12.8235149, -8.7824402, -2.9374719, 2.9369459
8: -5.4445019, -3.1584911, -5.4501796, -3.1462636, -1.4305577, 1.4108844
9: -1.9045296, 1.0422697, -1.9316483, 1.0464480, -2.6494102, 2.6954250

Time for backsubstitution: 23.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9015984, upper bound: 0.9052832
time: 6.51 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9054667, upper bound: 0.9054661
time: 5.07 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 7.7540436, 10.2373724, 7.7540421, 10.2373753, -2.0114293, 1.9989681
1: -19.2597008, -15.2714119, -19.2597122, -15.2714081, -2.4200554, 2.4333525
2: -6.5238414, -3.5489101, -6.5238447, -3.5489082, -1.9223175, 1.9225516
3: -10.8192225, -7.7928100, -10.8192253, -7.7928071, -2.3708739, 2.3794346
4: -13.5904951, -10.5921268, -13.5905008, -10.5921230, -2.1792383, 2.1938558
5: -4.6404009, -2.1593983, -4.6404028, -2.1593976, -1.7170382, 1.7191868
6: -4.5149150, -1.9159026, -4.5149155, -1.9158955, -2.0630393, 2.0158777
7: -12.8235550, -8.7824459, -12.8235531, -8.7824411, -2.9430213, 2.9241791
8: -5.4501781, -3.1462493, -5.4501801, -3.1462488, -1.4336686, 1.4131577
9: -1.9316425, 1.0465150, -1.9316483, 1.0465150, -2.6725235, 2.6946259

Time for backsubstitution: 23.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9015985, upper bound: 0.9052832
time: 4.94 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9054667, upper bound: 0.9054661
time: 4.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 32.81 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 32.81
Output dim: 0, lower bound: -0.9028996, upper bound: 0.9022340
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 32.81
Output dim: 0, lower bound: -0.9028996, upper bound: 0.9022339
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 32.81
Output dim: 0, lower bound: -0.9028996, upper bound: 0.9022342
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 32.81
Output dim: 0, lower bound: -0.9028996, upper bound: 0.9022342
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.81
Output dim: 0, lower bound: -0.9015984, upper bound: 0.9089926
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.81
Output dim: 0, lower bound: -0.9015984, upper bound: 0.9091757
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 32.81
Output dim: 0, lower bound: -0.9015984, upper bound: 0.9052832
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 32.81
Output dim: 0, lower bound: -0.9054667, upper bound: 0.9054661
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 32.81
Output dim: 0, lower bound: -0.9015985, upper bound: 0.9052832
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 32.81
Output dim: 0, lower bound: -0.9054667, upper bound: 0.9054661

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 7.7507591, 10.2255783, 7.7654247, 10.2054806, -2.0049591, 1.9976492
1: -19.2590256, -15.2659559, -19.2470684, -15.2779446, -2.4153132, 2.4346700
2: -6.5200462, -3.5485320, -6.5097742, -3.5558829, -1.9176331, 1.9061561
3: -10.8210764, -7.7849083, -10.8061275, -7.7989364, -2.3536286, 2.3625250
4: -13.5813198, -10.5878677, -13.5636234, -10.5980911, -2.1745090, 2.1846609
5: -4.6400518, -2.1589489, -4.6333542, -2.1689682, -1.7002544, 1.7079680
6: -4.5306983, -1.9217249, -4.5060997, -1.9474179, -2.0477009, 2.0078773
7: -12.8177805, -8.7821751, -12.7967482, -8.7944603, -2.9418983, 2.9130287
8: -5.4621334, -3.1508493, -5.4430351, -3.1705742, -1.4243848, 1.4060252
9: -1.9217930, 1.0568793, -1.8918247, 1.0419285, -2.6628504, 2.6692395

Time for backsubstitution: 22.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9015984, upper bound: 0.9052843
time: 6.21 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9015984, upper bound: 0.9089923
time: 6.27 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 7.7507591, 10.2255783, 7.7583346, 10.2116489, -2.0025764, 1.9973731
1: -19.2590256, -15.2659559, -19.2558765, -15.2687435, -2.4162040, 2.4347839
2: -6.5200462, -3.5485320, -6.5140505, -3.5513208, -1.9181061, 1.9073477
3: -10.8210764, -7.7849083, -10.8185005, -7.7879534, -2.3627205, 2.3709726
4: -13.5813198, -10.5878677, -13.5696201, -10.5921516, -2.1733985, 2.1823854
5: -4.6400518, -2.1589489, -4.6384296, -2.1632702, -1.7046628, 1.7060227
6: -4.5306983, -1.9217249, -4.5262294, -1.9295030, -2.0504417, 2.0017180
7: -12.8177805, -8.7821751, -12.8041115, -8.7873154, -2.9391479, 2.9228272
8: -5.4621334, -3.1508493, -5.4581289, -3.1575947, -1.4254889, 1.4029858
9: -1.9217930, 1.0568793, -1.9072633, 1.0527732, -2.6668205, 2.6775870

Time for backsubstitution: 22.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9015984, upper bound: 0.9054676
time: 5.17 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9015985, upper bound: 0.9091756
time: 4.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 32.49 seconds
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 32.49
Output dim: 0, lower bound: -0.9015984, upper bound: 0.9052843
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.49
Output dim: 0, lower bound: -0.9015984, upper bound: 0.9089923
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 32.49
Output dim: 0, lower bound: -0.9015984, upper bound: 0.9054676
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.49
Output dim: 0, lower bound: -0.9015985, upper bound: 0.9091756

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 7.7488317, 10.2378445, 7.7654247, 10.2054806, -2.0068340, 1.9991817
1: -19.2621632, -15.2648077, -19.2470684, -15.2779446, -2.4190927, 2.4358296
2: -6.5261483, -3.5479298, -6.5097742, -3.5558829, -1.9237218, 1.9068289
3: -10.8218451, -7.7831678, -10.8061275, -7.7989364, -2.3533826, 2.3624473
4: -13.5909185, -10.5871458, -13.5636234, -10.5980911, -2.1860185, 2.1847553
5: -4.6407747, -2.1554260, -4.6333542, -2.1689682, -1.7001958, 1.7110534
6: -4.5329533, -1.9148768, -4.5060997, -1.9474179, -2.0474758, 2.0147049
7: -12.8299685, -8.7812071, -12.7967482, -8.7944603, -2.9546900, 2.9144797
8: -5.4638042, -3.1453810, -5.4430351, -3.1705742, -1.4245231, 1.4078232
9: -1.9343538, 1.0569463, -1.8918247, 1.0419285, -2.6755810, 2.6694098

Time for backsubstitution: 22.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5814

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9015743, upper bound: 0.9082154
time: 4.92 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9015741, upper bound: 0.9089677
time: 5.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 7.7488317, 10.2378445, 7.7583346, 10.2116489, -2.0045137, 2.0027637
1: -19.2621632, -15.2648077, -19.2558765, -15.2687435, -2.4199824, 2.4359422
2: -6.5261483, -3.5479298, -6.5140505, -3.5513208, -1.9242516, 1.9080181
3: -10.8218451, -7.7831678, -10.8185005, -7.7879534, -2.3624763, 2.3708949
4: -13.5909185, -10.5871458, -13.5696201, -10.5921516, -2.1849084, 2.1824799
5: -4.6407747, -2.1554260, -4.6384296, -2.1632702, -1.7045918, 1.7091086
6: -4.5329533, -1.9148768, -4.5262294, -1.9295030, -2.0522742, 2.0095758
7: -12.8299685, -8.7812071, -12.8041115, -8.7873154, -2.9519386, 2.9242454
8: -5.4638042, -3.1453810, -5.4581289, -3.1575947, -1.4274364, 1.4085963
9: -1.9343538, 1.0569463, -1.9072633, 1.0527732, -2.6795511, 2.6777573

Time for backsubstitution: 22.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 5814

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9018224, upper bound: 0.9083984
time: 5.03 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9018222, upper bound: 0.9091505
time: 7.14 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 34.91 seconds
NS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 34.91
Output dim: 0, lower bound: -0.9015743, upper bound: 0.9082154
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 34.91
Output dim: 0, lower bound: -0.9015741, upper bound: 0.9089677
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 34.91
Output dim: 0, lower bound: -0.9018224, upper bound: 0.9083984
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 34.91
Output dim: 0, lower bound: -0.9018222, upper bound: 0.9091505

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 7.7488384, 10.2378445, 7.7600303, 10.2391434, -2.0096354, 2.0051091
1: -19.2621670, -15.2648125, -19.2761593, -15.2725048, -2.4261594, 2.4527535
2: -6.5261464, -3.5479281, -6.5134559, -3.5363708, -1.9434481, 1.9105899
3: -10.8218412, -7.7831717, -10.8157911, -7.7944107, -2.3638306, 2.3710899
4: -13.5909176, -10.5871487, -13.5976114, -10.5945921, -2.1910706, 2.2032917
5: -4.6407728, -2.1554279, -4.6380868, -2.1543221, -1.7248483, 1.7154317
6: -4.5329514, -1.9148769, -4.5125618, -1.9381930, -2.0519705, 2.0229464
7: -12.8299675, -8.7812119, -12.8065510, -8.7887001, -2.9637952, 2.9228125
8: -5.4638014, -3.1453838, -5.4595928, -3.1665254, -1.4285338, 1.4111198
9: -1.9343529, 1.0569448, -1.9168220, 1.0432253, -2.6861238, 2.6863317

Time for backsubstitution: 22.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5745

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9008313, upper bound: 0.9089626
time: 6.00 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9015692, upper bound: 0.9089622
time: 5.17 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 7.7488384, 10.2378445, 7.7529292, 10.2453089, -2.0136704, 2.0088663
1: -19.2621670, -15.2648125, -19.2849522, -15.2632742, -2.4270754, 2.4613984
2: -6.5261464, -3.5479281, -6.5177450, -3.5318584, -1.9439883, 1.9117846
3: -10.8218412, -7.7831717, -10.8281422, -7.7834320, -2.3729224, 2.3795714
4: -13.5909176, -10.5871487, -13.6036110, -10.5886364, -2.1899371, 2.2091136
5: -4.6407728, -2.1554279, -4.6431637, -2.1486320, -1.7292428, 1.7134910
6: -4.5329514, -1.9148769, -4.5327096, -1.9202673, -2.0608678, 2.0284164
7: -12.8299675, -8.7812119, -12.8139915, -8.7815790, -2.9714279, 2.9325919
8: -5.4638014, -3.1453838, -5.4746833, -3.1535640, -1.4316332, 1.4150648
9: -1.9343529, 1.0569448, -1.9322453, 1.0540683, -2.6915760, 2.7015972

Time for backsubstitution: 23.27 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.53 + 552.98 = 610.51 seconds
