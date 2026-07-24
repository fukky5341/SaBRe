## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.6321674314


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5920982, 1.5920992)
1: (-12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7782269, 1.7782259)
2: (-8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9221311, 1.9221311)
3: (-10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9307184, 1.9307184)
4: (-4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6834741, 1.6834741)
5: (-2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6719646, 1.6719651)
6: (9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.1016169, 1.1016171)
7: (-21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7424874, 1.7424872)
8: (-2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3420033, 1.3420031)
9: (-13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.5054874, 1.5054879)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.03 + 49.61 = 72.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.6334342, upper bound: 0.6334355

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 5846
type: B, layer: 1, pos: 5846
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 6168
type: A, layer: 1, pos: 6168
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 6213
type: B, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334303, upper bound: 0.6287730
time: 7.22 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334303, upper bound: 0.6334314
time: 5.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.75 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.75
Output dim: 6, lower bound: -0.6334303, upper bound: 0.6287730
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.75
Output dim: 6, lower bound: -0.6334303, upper bound: 0.6334314

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -9.2244301, -6.8706088, -9.2274294, -6.8695865, -1.5797558, 1.5820775
1: -12.2128296, -9.8791866, -12.2187195, -9.8789539, -1.7629528, 1.7686672
2: -8.2839775, -6.0824966, -8.2844677, -6.0815310, -1.9141960, 1.9125938
3: -10.4498911, -8.2052660, -10.4571009, -8.2052155, -1.9113288, 1.9190369
4: -4.8052850, -2.7016330, -4.8120165, -2.7010770, -1.6608458, 1.6678734
5: -2.4749694, -0.3344632, -2.4751964, -0.3294178, -1.6640310, 1.6592398
6: 9.4841928, 11.4543991, 9.4837666, 11.4616871, -1.0896842, 1.0827794
7: -21.0080624, -18.1568222, -21.0157127, -18.1559715, -1.7221155, 1.7290082
8: -2.3590121, -0.5157762, -2.3598700, -0.5088048, -1.3273063, 1.3219218
9: -13.2417831, -11.0142956, -13.2513247, -11.0141315, -1.4815292, 1.4908648

Time for backsubstitution: 20.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5846
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 6168
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 6213
type: A, layer: 1, pos: 6213
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 5801

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5846

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6330017, upper bound: 0.6243595
time: 5.01 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334157, upper bound: 0.6287578
time: 7.75 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -9.2333059, -6.8553367, -9.2318945, -6.8680778, -1.5885706, 1.6034713
1: -12.2360306, -9.8608837, -12.2274981, -9.8786077, -1.7776709, 1.7961378
2: -8.2952499, -6.0780258, -8.2851849, -6.0800896, -1.9299073, 1.9233503
3: -10.4745073, -8.1914921, -10.4678583, -8.2051449, -1.9308720, 1.9443245
4: -4.8295035, -2.6798084, -4.8220377, -2.7002630, -1.6814322, 1.7020721
5: -2.4873419, -0.3195072, -2.4755280, -0.3219023, -1.6838598, 1.6677656
6: 9.4551535, 11.4735546, 9.4831438, 11.4725399, -1.1263287, 1.0905356
7: -21.0281334, -18.1333447, -21.0270958, -18.1547108, -1.7310863, 1.7613299
8: -2.3786216, -0.4956751, -2.3611336, -0.4984264, -1.3533473, 1.3322582
9: -13.2677460, -10.9873409, -13.2655478, -11.0138874, -1.4915085, 1.5258331

Time for backsubstitution: 21.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5846
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6168
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 6213
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 5801

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5846

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6330017, upper bound: 0.6290151
time: 26.15 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334157, upper bound: 0.6334160
time: 4.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 51.89 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 51.89
Output dim: 6, lower bound: -0.6330017, upper bound: 0.6243595
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 51.89
Output dim: 6, lower bound: -0.6334157, upper bound: 0.6287578
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 51.89
Output dim: 6, lower bound: -0.6330017, upper bound: 0.6290151
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 51.89
Output dim: 6, lower bound: -0.6334157, upper bound: 0.6334160

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -9.2137489, -6.8998113, -9.2256517, -6.8854003, -1.5543919, 1.5518174
1: -12.2028828, -9.8896036, -12.2157536, -9.8845930, -1.7468963, 1.7553134
2: -8.2481060, -6.0957623, -8.2648563, -6.0832510, -1.8756266, 1.8796868
3: -10.4315386, -8.2519674, -10.4551840, -8.2311182, -1.8648896, 1.8708186
4: -4.7609882, -2.7125511, -4.7884674, -2.7020087, -1.6168413, 1.6286354
5: -2.4658082, -0.3572338, -2.4739110, -0.3418714, -1.6440678, 1.6357422
6: 9.4971180, 11.4282894, 9.4860649, 11.4471855, -1.0627580, 1.0546970
7: -20.9393291, -18.1827564, -20.9775448, -18.1578846, -1.6521835, 1.6588447
8: -2.3337631, -0.5833998, -2.3579907, -0.5462174, -1.2560358, 1.2526903
9: -13.2121429, -11.0230694, -13.2351618, -11.0149841, -1.4517164, 1.4640741

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 6168
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 6213
type: A, layer: 1, pos: 6213
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 5801

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6291920, upper bound: 0.6239314
time: 8.42 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6329994, upper bound: 0.6243575
time: 8.42 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -9.2244310, -6.8706102, -9.2274294, -6.8695893, -1.5797558, 1.5642014
1: -12.2128305, -9.8791885, -12.2187176, -9.8789539, -1.7626429, 1.7586293
2: -8.2839718, -6.0824947, -8.2844648, -6.0815334, -1.8931041, 1.9125924
3: -10.4498882, -8.2052698, -10.4570999, -8.2052174, -1.9098110, 1.8707252
4: -4.8052807, -2.7016344, -4.8120136, -2.7010782, -1.6278267, 1.6651721
5: -2.4749699, -0.3344646, -2.4751966, -0.3294200, -1.6640320, 1.6512618
6: 9.4841938, 11.4543972, 9.4837656, 11.4616852, -1.0896819, 1.0552142
7: -21.0080605, -18.1568260, -21.0157108, -18.1559715, -1.6475968, 1.7191269
8: -2.3590117, -0.5157838, -2.3598685, -0.5088086, -1.3153076, 1.2558126
9: -13.2417793, -11.0142956, -13.2513237, -11.0141315, -1.4570699, 1.4889979

Time for backsubstitution: 21.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 6168
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6296156, upper bound: 0.6283396
time: 4.72 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334134, upper bound: 0.6287556
time: 6.09 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -9.2226162, -6.8845348, -9.2301168, -6.8838930, -1.5632114, 1.5732117
1: -12.2261200, -9.8712969, -12.2245350, -9.8842487, -1.7616663, 1.7827878
2: -8.2593260, -6.0913115, -8.2655697, -6.0818086, -1.8913050, 1.8904467
3: -10.4561291, -8.2382069, -10.4659443, -8.2310486, -1.8844175, 1.8960910
4: -4.7852945, -2.6907125, -4.7984810, -2.7011905, -1.6374898, 1.6520586
5: -2.4781837, -0.3422654, -2.4742448, -0.3343579, -1.6638851, 1.6442766
6: 9.4681559, 11.4474468, 9.4854374, 11.4580383, -1.0906956, 1.0624528
7: -20.9593906, -18.1592884, -20.9889240, -18.1566200, -1.6611567, 1.6787758
8: -2.3533945, -0.5632792, -2.3592615, -0.5358367, -1.2730727, 1.2630386
9: -13.2380924, -10.9961300, -13.2493830, -11.0147381, -1.4616885, 1.4909933

Time for backsubstitution: 22.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6168
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 6213
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 5801

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6291920, upper bound: 0.6285891
time: 5.45 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6329994, upper bound: 0.6290134
time: 6.16 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -9.2333069, -6.8553414, -9.2318935, -6.8680763, -1.5885696, 1.5855942
1: -12.2360306, -9.8608847, -12.2274961, -9.8786087, -1.7773733, 1.7861066
2: -8.2952461, -6.0780258, -8.2851849, -6.0800905, -1.9088135, 1.9233499
3: -10.4745064, -8.1914949, -10.4678593, -8.2051458, -1.9293590, 1.8960328
4: -4.8294992, -2.6798108, -4.8220363, -2.7002630, -1.6485624, 1.6874199
5: -2.4873407, -0.3195088, -2.4755299, -0.3219026, -1.6838579, 1.6597862
6: 9.4551544, 11.4735489, 9.4831438, 11.4725399, -1.1176209, 1.0629711
7: -21.0281296, -18.1333408, -21.0270939, -18.1547127, -1.6565804, 1.7390664
8: -2.3786206, -0.4956818, -2.3611336, -0.4984303, -1.3323407, 1.2661495
9: -13.2677441, -10.9873409, -13.2655478, -11.0138874, -1.4670496, 1.5161378

Time for backsubstitution: 22.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6168
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6296156, upper bound: 0.6329957
time: 6.02 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334134, upper bound: 0.6334130
time: 10.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 38.97 seconds
NS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 38.97
Output dim: 6, lower bound: -0.6291920, upper bound: 0.6239314
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 38.97
Output dim: 6, lower bound: -0.6329994, upper bound: 0.6243575
NS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 38.97
Output dim: 6, lower bound: -0.6296156, upper bound: 0.6283396
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 38.97
Output dim: 6, lower bound: -0.6334134, upper bound: 0.6287556
NS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 38.97
Output dim: 6, lower bound: -0.6291920, upper bound: 0.6285891
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 38.97
Output dim: 6, lower bound: -0.6329994, upper bound: 0.6290134
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 38.97
Output dim: 6, lower bound: -0.6296156, upper bound: 0.6329957
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 38.97
Output dim: 6, lower bound: -0.6334134, upper bound: 0.6334130

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -9.2137470, -6.8998194, -9.2356777, -6.8842316, -1.5554562, 1.5626340
1: -12.2028751, -9.8896046, -12.2173862, -9.8777180, -1.7541828, 1.7562661
2: -8.2480993, -6.0957637, -8.2665968, -6.0740342, -1.8871622, 1.8756509
3: -10.4315271, -8.2519693, -10.4603958, -8.2118578, -1.8798816, 1.8715420
4: -4.7609868, -2.7125628, -4.8134975, -2.7001910, -1.6171627, 1.6388278
5: -2.4658079, -0.3572421, -2.4978790, -0.3403252, -1.6444492, 1.6599445
6: 9.4971189, 11.4282837, 9.4717426, 11.4477835, -1.0556753, 1.0692592
7: -20.9393234, -18.1827583, -20.9795361, -18.1531525, -1.6569386, 1.6586215
8: -2.3337598, -0.5834022, -2.3605108, -0.5453997, -1.2564323, 1.2547858
9: -13.2121401, -11.0230703, -13.2375202, -10.9968576, -1.4697380, 1.4611845

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 6168
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 6213

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6292908, upper bound: 0.6243528
time: 10.01 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6329961, upper bound: 0.6243544
time: 6.06 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -9.2244320, -6.8706188, -9.2374449, -6.8684163, -1.5808234, 1.5750127
1: -12.2128239, -9.8791895, -12.2203588, -9.8720932, -1.7699270, 1.7595849
2: -8.2839642, -6.0824986, -8.2862110, -6.0723276, -1.9046350, 1.9085603
3: -10.4498777, -8.2052717, -10.4623280, -8.1859865, -1.9224305, 1.8714361
4: -4.8052783, -2.7016454, -4.8369961, -2.6992581, -1.6281443, 1.6742258
5: -2.4749677, -0.3344734, -2.4991562, -0.3278735, -1.6644115, 1.6754618
6: 9.4841938, 11.4543934, 9.4694557, 11.4622831, -1.0826020, 1.0697713
7: -21.0080566, -18.1568260, -21.0177078, -18.1512413, -1.6523395, 1.7189043
8: -2.3590083, -0.5157876, -2.3623571, -0.5079899, -1.3157053, 1.2579014
9: -13.2417736, -11.0142975, -13.2536869, -10.9960175, -1.4750504, 1.4861145

Time for backsubstitution: 22.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 6168
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334102, upper bound: 0.6250470
time: 7.42 seconds

## Relational analysis of NS_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334102, upper bound: 0.6287525
time: 6.58 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -9.2226143, -6.8845439, -9.2400818, -6.8827052, -1.5642796, 1.5840421
1: -12.2261133, -9.8712988, -12.2261639, -9.8773766, -1.7689548, 1.7837267
2: -8.2593193, -6.0913115, -8.2673492, -6.0726318, -1.9028487, 1.8863916
3: -10.4561176, -8.2382088, -10.4711361, -8.2117882, -1.8994060, 1.8967986
4: -4.7852917, -2.6907234, -4.8234916, -2.6993318, -1.6377935, 1.6578379
5: -2.4781826, -0.3422734, -2.4982281, -0.3328265, -1.6642551, 1.6684990
6: 9.4681540, 11.4474401, 9.4711189, 11.4586325, -1.0836034, 1.0770111
7: -20.9593868, -18.1592884, -20.9909096, -18.1518822, -1.6659179, 1.6785376
8: -2.3533921, -0.5632825, -2.3617549, -0.5350156, -1.2734857, 1.2650974
9: -13.2380829, -10.9961319, -13.2517080, -10.9966154, -1.4797053, 1.4880915

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6168
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 6213
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 6213

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6329963, upper bound: 0.6253062
time: 6.67 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6329963, upper bound: 0.6290099
time: 6.22 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -9.2311230, -6.8618321, -9.2282934, -6.8789749, -1.5763483, 1.5767899
1: -12.2313204, -9.8621502, -12.2196045, -9.8807592, -1.7709770, 1.7772069
2: -8.2899809, -6.0797815, -8.2763376, -6.0830011, -1.8927741, 1.9042048
3: -10.4576387, -8.1932697, -10.4395924, -8.2081518, -1.9098597, 1.8652191
4: -4.8262544, -2.6947811, -4.8165636, -2.7254114, -1.6217041, 1.6660213
5: -2.4858310, -0.3344834, -2.4729733, -0.3469987, -1.6579447, 1.6431994
6: 9.4559593, 11.4656858, 9.4844990, 11.4593563, -1.1038294, 1.0541632
7: -21.0236130, -18.1347389, -21.0195255, -18.1570473, -1.6491385, 1.7294080
8: -2.3764286, -0.4986291, -2.3574252, -0.5033617, -1.3238878, 1.2580376
9: -13.2569885, -10.9886703, -13.2475042, -11.0161381, -1.4536748, 1.4965630

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6168
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6296124, upper bound: 0.6292892
time: 4.86 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6296124, upper bound: 0.6329922
time: 5.51 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -9.2333021, -6.8553505, -9.2418499, -6.8668909, -1.5896435, 1.5964198
1: -12.2360239, -9.8608847, -12.2291279, -9.8717508, -1.7846594, 1.7870488
2: -8.2952404, -6.0780258, -8.2869692, -6.0709257, -1.9203539, 1.9193001
3: -10.4744949, -8.1914968, -10.4730682, -8.1859140, -1.9419723, 1.8967285
4: -4.8294983, -2.6798205, -4.8470016, -2.6984038, -1.6488662, 1.6932416
5: -2.4873414, -0.3195189, -2.4995034, -0.3203704, -1.6842232, 1.6840053
6: 9.4551582, 11.4735489, 9.4688377, 11.4731312, -1.1105297, 1.0775237
7: -21.0281296, -18.1333447, -21.0290813, -18.1499748, -1.6613302, 1.7388263
8: -2.3786197, -0.4956861, -2.3636012, -0.4976063, -1.3327556, 1.2681942
9: -13.2677374, -10.9873409, -13.2678823, -10.9957743, -1.4850268, 1.5132391

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6168
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 6213
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334102, upper bound: 0.6297043
time: 4.82 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334102, upper bound: 0.6334094
time: 5.20 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 32.07 seconds
NS_A1_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 32.07
Output dim: 6, lower bound: -0.6292908, upper bound: 0.6243528
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 32.07
Output dim: 6, lower bound: -0.6329961, upper bound: 0.6243544
NS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 32.07
Output dim: 6, lower bound: -0.6334102, upper bound: 0.6250470
NS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 32.07
Output dim: 6, lower bound: -0.6334102, upper bound: 0.6287525
NS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 32.07
Output dim: 6, lower bound: -0.6329963, upper bound: 0.6253062
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 32.07
Output dim: 6, lower bound: -0.6329963, upper bound: 0.6290099
NS_A2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 32.07
Output dim: 6, lower bound: -0.6296124, upper bound: 0.6292892
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 32.07
Output dim: 6, lower bound: -0.6296124, upper bound: 0.6329922
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 32.07
Output dim: 6, lower bound: -0.6334102, upper bound: 0.6297043
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 32.07
Output dim: 6, lower bound: -0.6334102, upper bound: 0.6334094

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -9.2137451, -6.8998199, -9.2356749, -6.8842344, -1.5551605, 1.5519462
1: -12.2028713, -9.8896055, -12.2173786, -9.8777180, -1.7541819, 1.7172513
2: -8.2480984, -6.0957632, -8.2665882, -6.0740356, -1.8803926, 1.8285542
3: -10.4315281, -8.2519693, -10.4603958, -8.2118607, -1.8486431, 1.8715358
4: -4.7609859, -2.7125664, -4.8134985, -2.7001987, -1.5949907, 1.6337194
5: -2.4658079, -0.3572452, -2.4978786, -0.3403291, -1.6293259, 1.6588693
6: 9.4971209, 11.4282856, 9.4717426, 11.4477806, -1.0359979, 1.0679156
7: -20.9393215, -18.1827602, -20.9795380, -18.1531525, -1.6569366, 1.6488051
8: -2.3337593, -0.5834026, -2.3605037, -0.5454011, -1.2537785, 1.2438233
9: -13.2121382, -11.0230742, -13.2375126, -10.9968605, -1.4630070, 1.4260907

Time for backsubstitution: 21.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6168
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 6213

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6168

## Relational analysis of NS_A1_A1_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6329843, upper bound: 0.6217198
time: 11.21 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6329946, upper bound: 0.6243509
time: 12.98 seconds

## BFS NS instance: NS_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -9.2103291, -6.8834209, -9.2304840, -6.8731503, -1.5618320, 1.5539989
1: -12.1681538, -9.8910408, -12.1978550, -9.8730431, -1.7210388, 1.7204952
2: -8.2281609, -6.0979419, -8.2577620, -6.0736423, -1.8474913, 1.8581643
3: -10.4295588, -8.2366381, -10.4566536, -8.2019711, -1.8714271, 1.8337688
4: -4.7975187, -2.7260580, -4.8359661, -2.7116501, -1.6064672, 1.6478622
5: -2.4668963, -0.3558311, -2.4977057, -0.3385986, -1.6388378, 1.6503811
6: 9.4974537, 11.4351349, 9.4725294, 11.4524469, -1.0568316, 1.0464361
7: -20.9960442, -18.1670113, -21.0117340, -18.1543884, -1.6365495, 1.6991277
8: -2.3378611, -0.5232725, -2.3519783, -0.5102654, -1.2812760, 1.2300868
9: -13.2018757, -11.0257320, -13.2334080, -10.9968872, -1.4331999, 1.4486585

Time for backsubstitution: 21.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 6168
type: B, layer: 1, pos: 6168
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 6213

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of NS_A1_A2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6288050, upper bound: 0.6250475
time: 7.64 seconds

## Relational analysis of NS_A1_A2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6288050, upper bound: 0.6250466
time: 7.74 seconds

## BFS NS instance: NS_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -9.2244263, -6.8706207, -9.2374449, -6.8684158, -1.5702362, 1.5746984
1: -12.2128143, -9.8791885, -12.2203569, -9.8720932, -1.7309141, 1.7595840
2: -8.2839603, -6.0824966, -8.2862091, -6.0723281, -1.8574305, 1.9051208
3: -10.4498730, -8.2052765, -10.4623270, -8.1859875, -1.9149017, 1.8405156
4: -4.8052778, -2.7016554, -4.8369970, -2.6992605, -1.6280165, 1.6519432
5: -2.4749675, -0.3344771, -2.4991555, -0.3278731, -1.6636620, 1.6603041
6: 9.4841928, 11.4543915, 9.4694557, 11.4622822, -1.0826001, 1.0501037
7: -21.0080547, -18.1568260, -21.0177059, -18.1512432, -1.6428037, 1.7160535
8: -2.3590031, -0.5157881, -2.3623552, -0.5079904, -1.3046477, 1.2578988
9: -13.2417660, -11.0143013, -13.2536850, -10.9960175, -1.4399576, 1.4839854

Time for backsubstitution: 21.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 6168
type: B, layer: 1, pos: 6168
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 6213

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of NS_A1_A2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6288050, upper bound: 0.6287522
time: 7.49 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6288050, upper bound: 0.6287524
time: 6.65 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 72.65 + 536.07 = 608.72 seconds
