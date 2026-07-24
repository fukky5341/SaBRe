## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.133442582


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2867398, 0.2867398)
1: (-14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3827257, 0.3827260)
2: (-4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2954063, 0.2954063)
3: (-16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4403210, 0.4403207)
4: (-1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2216630, 0.2216629)
5: (-6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2256572, 0.2256573)
6: (9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2573330, 0.2573330)
7: (-14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.3041251, 0.3041251)
8: (-4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3534777, 0.3534777)
9: (-11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3350632, 0.3350632)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.71 + 35.42 = 58.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.1361659, upper bound: 0.1361659

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4653
type: A, layer: 1, pos: 529

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4653

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1361622, upper bound: 0.1345626
time: 3.81 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1361622, upper bound: 0.1361616
time: 4.98 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.93 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.93
Output dim: 6, lower bound: -0.1361622, upper bound: 0.1345626
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.93
Output dim: 6, lower bound: -0.1361622, upper bound: 0.1361616

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2.8232565, -2.2948399, -2.8296356, -2.2897878, -0.2752674, 0.2766917
1: -14.9086075, -14.1353188, -14.9129429, -14.1310635, -0.3747425, 0.3715305
2: -4.6650896, -4.2106242, -4.6655660, -4.2092061, -0.2927148, 0.2929220
3: -16.2239666, -15.5242643, -16.2335339, -15.5126858, -0.4207442, 0.4199433
4: -1.7967346, -1.2365634, -1.8006141, -1.2336793, -0.2144288, 0.2154375
5: -6.5634851, -6.1065507, -6.5678339, -6.1028056, -0.2178347, 0.2175111
6: 9.5123577, 10.1050739, 9.5080833, 10.1082163, -0.2500956, 0.2512872
7: -14.1265354, -13.4288635, -14.1303606, -13.4229021, -0.2965260, 0.2941883
8: -4.0723200, -3.3962564, -4.0781803, -3.3920112, -0.3435521, 0.3442748
9: -11.7106304, -10.8459015, -11.7126274, -10.8450565, -0.3308058, 0.3323786

Time for backsubstitution: 19.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4653
type: B, layer: 1, pos: 529

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4653

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345627, upper bound: 0.1345623
time: 4.87 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345627, upper bound: 0.1345627
time: 3.03 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2.8300662, -2.2850871, -2.8300667, -2.2850802, -0.2853105, 0.2768645
1: -14.9170170, -14.1306601, -14.9170246, -14.1306610, -0.3787227, 0.3814464
2: -4.6656599, -4.2078772, -4.6656609, -4.2078729, -0.2927876, 0.2984786
3: -16.2423420, -15.5104809, -16.2423534, -15.5104771, -0.4274375, 0.4379137
4: -1.8037565, -1.2336228, -1.8037624, -1.2336226, -0.2154400, 0.2216612
5: -6.5715957, -6.1027832, -6.5715995, -6.1027842, -0.2185562, 0.2254897
6: 9.5077515, 10.1110086, 9.5077496, 10.1110144, -0.2573316, 0.2512408
7: -14.1340580, -13.4225645, -14.1340618, -13.4225636, -0.2952049, 0.3041224
8: -4.0801945, -3.3881698, -4.0801983, -3.3881640, -0.3534713, 0.3506544
9: -11.7132759, -10.8442421, -11.7132750, -10.8442392, -0.3348999, 0.3336816

Time for backsubstitution: 20.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4653
type: B, layer: 1, pos: 529

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4653

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345627, upper bound: 0.1361619
time: 4.17 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345626, upper bound: 0.1361623
time: 3.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.10 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.10
Output dim: 6, lower bound: -0.1345627, upper bound: 0.1345623
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.10
Output dim: 6, lower bound: -0.1345627, upper bound: 0.1345627
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.10
Output dim: 6, lower bound: -0.1345627, upper bound: 0.1361619
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.10
Output dim: 6, lower bound: -0.1345626, upper bound: 0.1361623

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -2.8232565, -2.2948399, -2.8232565, -2.2948399, -0.2702546, 0.2702546
1: -14.9086075, -14.1353188, -14.9086075, -14.1353188, -0.3690264, 0.3690267
2: -4.6650896, -4.2106242, -4.6650896, -4.2106242, -0.2904668, 0.2904668
3: -16.2239666, -15.5242643, -16.2239666, -15.5242643, -0.4103811, 0.4103808
4: -1.7967346, -1.2365634, -1.7967346, -1.2365634, -0.2115409, 0.2115408
5: -6.5634851, -6.1065507, -6.5634851, -6.1065507, -0.2139384, 0.2139385
6: 9.5123577, 10.1050739, 9.5123577, 10.1050739, -0.2471131, 0.2471132
7: -14.1265354, -13.4288635, -14.1265354, -13.4288635, -0.2903697, 0.2903697
8: -4.0723200, -3.3962564, -4.0723200, -3.3962564, -0.3392622, 0.3392622
9: -11.7106304, -10.8459015, -11.7106304, -10.8459015, -0.3299603, 0.3299603

Time for backsubstitution: 21.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 529

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1343705, upper bound: 0.1345618
time: 3.59 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345622, upper bound: 0.1345617
time: 4.83 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2.8232565, -2.2948399, -2.8300662, -2.2850871, -0.2767155, 0.2754565
1: -14.9086075, -14.1353188, -14.9170170, -14.1306601, -0.3736963, 0.3767703
2: -4.6650896, -4.2106242, -4.6656599, -4.2078772, -0.2914884, 0.2911663
3: -16.2239666, -15.5242643, -16.2423420, -15.5104809, -0.4194758, 0.4211719
4: -1.7967346, -1.2365634, -1.8037565, -1.2336228, -0.2144719, 0.2179059
5: -6.5634851, -6.1065507, -6.5715957, -6.1027832, -0.2176864, 0.2217383
6: 9.5123577, 10.1050739, 9.5077515, 10.1110086, -0.2526230, 0.2514381
7: -14.1265354, -13.4288635, -14.1340580, -13.4225645, -0.2966042, 0.2974483
8: -4.0723200, -3.3962564, -4.0801945, -3.3881698, -0.3473549, 0.3453736
9: -11.7106304, -10.8459015, -11.7132759, -10.8442421, -0.3316188, 0.3332412

Time for backsubstitution: 21.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 529

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1343704, upper bound: 0.1345618
time: 3.56 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345621, upper bound: 0.1345617
time: 3.35 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -2.8300662, -2.2850871, -2.8232565, -2.2948399, -0.2754564, 0.2767155
1: -14.9170170, -14.1306601, -14.9086075, -14.1353188, -0.3767703, 0.3736961
2: -4.6656599, -4.2078772, -4.6650896, -4.2106242, -0.2911663, 0.2914884
3: -16.2423420, -15.5104809, -16.2239666, -15.5242643, -0.4211719, 0.4194760
4: -1.8037565, -1.2336228, -1.7967346, -1.2365634, -0.2179059, 0.2144719
5: -6.5715957, -6.1027832, -6.5634851, -6.1065507, -0.2217382, 0.2176863
6: 9.5077515, 10.1110086, 9.5123577, 10.1050739, -0.2514380, 0.2526230
7: -14.1340580, -13.4225645, -14.1265354, -13.4288635, -0.2974483, 0.2966042
8: -4.0801945, -3.3881698, -4.0723200, -3.3962564, -0.3453736, 0.3473551
9: -11.7132759, -10.8442421, -11.7106304, -10.8459015, -0.3332412, 0.3316188

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 529

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1343698, upper bound: 0.1361607
time: 4.46 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345615, upper bound: 0.1361602
time: 4.36 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -2.8300662, -2.2850871, -2.8300662, -2.2850871, -0.2768638, 0.2768638
1: -14.9170170, -14.1306601, -14.9170170, -14.1306601, -0.3787208, 0.3787208
2: -4.6656599, -4.2078772, -4.6656599, -4.2078772, -0.2984760, 0.2984760
3: -16.2423420, -15.5104809, -16.2423420, -15.5104809, -0.4274359, 0.4274356
4: -1.8037565, -1.2336228, -1.8037565, -1.2336228, -0.2154398, 0.2154399
5: -6.5715957, -6.1027832, -6.5715957, -6.1027832, -0.2185559, 0.2185562
6: 9.5077515, 10.1110086, 9.5077515, 10.1110086, -0.2512406, 0.2512406
7: -14.1340580, -13.4225645, -14.1340580, -13.4225645, -0.2952051, 0.2952051
8: -4.0801945, -3.3881698, -4.0801945, -3.3881698, -0.3506525, 0.3506525
9: -11.7132759, -10.8442421, -11.7132759, -10.8442421, -0.3336799, 0.3336799

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 529

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1343698, upper bound: 0.1361614
time: 3.40 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345615, upper bound: 0.1361613
time: 3.21 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.32 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.32
Output dim: 6, lower bound: -0.1343705, upper bound: 0.1345618
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.32
Output dim: 6, lower bound: -0.1345622, upper bound: 0.1345617
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.32
Output dim: 6, lower bound: -0.1343704, upper bound: 0.1345618
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.32
Output dim: 6, lower bound: -0.1345621, upper bound: 0.1345617
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.32
Output dim: 6, lower bound: -0.1343698, upper bound: 0.1361607
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.32
Output dim: 6, lower bound: -0.1345615, upper bound: 0.1361602
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.32
Output dim: 6, lower bound: -0.1343698, upper bound: 0.1361614
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.32
Output dim: 6, lower bound: -0.1345615, upper bound: 0.1361613

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.8223438, -2.2948422, -2.8232565, -2.2948399, -0.2693195, 0.2702532
1: -14.9085712, -14.1369381, -14.9086075, -14.1353188, -0.3690035, 0.3674066
2: -4.6650891, -4.2112875, -4.6650896, -4.2106242, -0.2903721, 0.2897508
3: -16.2239590, -15.5259171, -16.2239666, -15.5242643, -0.4102294, 0.4086537
4: -1.7967310, -1.2375360, -1.7967346, -1.2365634, -0.2115392, 0.2105677
5: -6.5634494, -6.1071596, -6.5634851, -6.1065507, -0.2138463, 0.2133292
6: 9.5126038, 10.1050730, 9.5123577, 10.1050739, -0.2468375, 0.2471108
7: -14.1258936, -13.4288712, -14.1265354, -13.4288635, -0.2897234, 0.2903590
8: -4.0720501, -3.3962929, -4.0723200, -3.3962564, -0.3389163, 0.3390570
9: -11.7104759, -10.8459015, -11.7106304, -10.8459015, -0.3297930, 0.3299537

Time for backsubstitution: 21.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 529

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1343705, upper bound: 0.1343705
time: 5.52 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1343705, upper bound: 0.1345623
time: 3.64 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.8235590, -2.2892089, -2.8232541, -2.2948403, -0.2707465, 0.2738626
1: -14.9209118, -14.1352949, -14.9086037, -14.1353207, -0.3761027, 0.3697217
2: -4.6697941, -4.2106061, -4.6650896, -4.2106266, -0.2948694, 0.2914944
3: -16.2357216, -15.5241251, -16.2239666, -15.5242672, -0.4161408, 0.4121656
4: -1.8041315, -1.2365417, -1.7967334, -1.2365646, -0.2147424, 0.2119815
5: -6.5679245, -6.1065531, -6.5634842, -6.1065521, -0.2182257, 0.2141953
6: 9.5120640, 10.1057901, 9.5123577, 10.1050758, -0.2472566, 0.2478182
7: -14.1265459, -13.4239931, -14.1265335, -13.4288673, -0.2907026, 0.2952428
8: -4.0724397, -3.3948822, -4.0723181, -3.3962572, -0.3404191, 0.3404660
9: -11.7107296, -10.8447151, -11.7106285, -10.8459015, -0.3303084, 0.3311405

Time for backsubstitution: 21.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 529

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345623, upper bound: 0.1343700
time: 4.53 seconds

## Relational analysis of NS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.8223438, -2.2948422, -2.8300662, -2.2850871, -0.2757788, 0.2754550
1: -14.9085712, -14.1369381, -14.9170170, -14.1306601, -0.3736730, 0.3751502
2: -4.6650891, -4.2112875, -4.6656599, -4.2078772, -0.2913940, 0.2904503
3: -16.2239590, -15.5259171, -16.2423420, -15.5104809, -0.4193234, 0.4194441
4: -1.7967310, -1.2375360, -1.8037565, -1.2336228, -0.2144703, 0.2169323
5: -6.5634494, -6.1071596, -6.5715957, -6.1027832, -0.2175939, 0.2211289
6: 9.5126038, 10.1050730, 9.5077515, 10.1110086, -0.2523475, 0.2514358
7: -14.1258936, -13.4288712, -14.1340580, -13.4225645, -0.2959576, 0.2974375
8: -4.0720501, -3.3962929, -4.0801945, -3.3881698, -0.3470092, 0.3451686
9: -11.7104759, -10.8459015, -11.7132759, -10.8442421, -0.3314514, 0.3332345

Time for backsubstitution: 21.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 529

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1359688, upper bound: 0.1343695
time: 5.52 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1359688, upper bound: 0.1345617
time: 4.73 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.8235590, -2.2892089, -2.8300631, -2.2850869, -0.2768875, 0.2755072
1: -14.9209118, -14.1352949, -14.9170189, -14.1306639, -0.3775153, 0.3774657
2: -4.6697941, -4.2106061, -4.6656599, -4.2078786, -0.2954693, 0.2921939
3: -16.2357216, -15.5241251, -16.2423401, -15.5104847, -0.4193382, 0.4214394
4: -1.8041315, -1.2365417, -1.8037574, -1.2336249, -0.2151346, 0.2179906
5: -6.5679245, -6.1065531, -6.5715957, -6.1027856, -0.2200481, 0.2219952
6: 9.5120640, 10.1057901, 9.5077505, 10.1110096, -0.2527721, 0.2519684
7: -14.1265459, -13.4239931, -14.1340542, -13.4225655, -0.2969370, 0.2974548
8: -4.0724397, -3.3948822, -4.0801935, -3.3881683, -0.3483596, 0.3465774
9: -11.7107296, -10.8447151, -11.7132750, -10.8442421, -0.3319666, 0.3344216

Time for backsubstitution: 21.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 529

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1361605, upper bound: 0.1343699
time: 3.44 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1361605, upper bound: 0.1345617
time: 3.56 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.8291516, -2.2850904, -2.8232565, -2.2948399, -0.2745202, 0.2767141
1: -14.9169779, -14.1322823, -14.9086075, -14.1353188, -0.3767445, 0.3720763
2: -4.6656599, -4.2085428, -4.6650896, -4.2106242, -0.2910719, 0.2907686
3: -16.2423325, -15.5121326, -16.2239666, -15.5242643, -0.4210196, 0.4177480
4: -1.8037550, -1.2345946, -1.7967346, -1.2365634, -0.2179038, 0.2134985
5: -6.5715585, -6.1033940, -6.5634851, -6.1065507, -0.2216471, 0.2170768
6: 9.5079994, 10.1110106, 9.5123577, 10.1050739, -0.2511613, 0.2526208
7: -14.1334162, -13.4225721, -14.1265354, -13.4288635, -0.2968013, 0.2965930
8: -4.0799246, -3.3882055, -4.0723200, -3.3962564, -0.3450284, 0.3471508
9: -11.7131224, -10.8442421, -11.7106304, -10.8459015, -0.3330739, 0.3316116

Time for backsubstitution: 21.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 529

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1343698, upper bound: 0.1359685
time: 5.54 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1343698, upper bound: 0.1361605
time: 3.53 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.8303578, -2.2794540, -2.8232541, -2.2948403, -0.2756287, 0.2767649
1: -14.9293127, -14.1306381, -14.9086037, -14.1353207, -0.3789957, 0.3743894
2: -4.6703649, -4.2078485, -4.6650896, -4.2106266, -0.2953174, 0.2925127
3: -16.2540970, -15.5103264, -16.2239666, -15.5242672, -0.4210339, 0.4197626
4: -1.8111565, -1.2336020, -1.7967334, -1.2365646, -0.2180009, 0.2149131
5: -6.5760417, -6.1027865, -6.5634842, -6.1065521, -0.2225502, 0.2179431
6: 9.5074682, 10.1117277, 9.5123577, 10.1050758, -0.2515781, 0.2526724
7: -14.1340694, -13.4176931, -14.1265335, -13.4288673, -0.2974776, 0.2976462
8: -4.0803242, -3.3867862, -4.0723181, -3.3962572, -0.3465238, 0.3481379
9: -11.7133713, -10.8430557, -11.7106285, -10.8459015, -0.3335850, 0.3327990

Time for backsubstitution: 21.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 529

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1359686
time: 5.35 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1361603
time: 4.29 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.8291516, -2.2850904, -2.8300662, -2.2850871, -0.2759283, 0.2768624
1: -14.9169779, -14.1322823, -14.9170170, -14.1306601, -0.3786988, 0.3771002
2: -4.6656599, -4.2085428, -4.6656599, -4.2078772, -0.2983813, 0.2977602
3: -16.2423325, -15.5121326, -16.2423420, -15.5104809, -0.4272842, 0.4257102
4: -1.8037550, -1.2345946, -1.8037565, -1.2336228, -0.2154380, 0.2144665
5: -6.5715585, -6.1033940, -6.5715957, -6.1027832, -0.2184641, 0.2179466
6: 9.5079994, 10.1110106, 9.5077515, 10.1110086, -0.2509638, 0.2512381
7: -14.1334162, -13.4225721, -14.1340580, -13.4225645, -0.2945580, 0.2951939
8: -4.0799246, -3.3882055, -4.0801945, -3.3881698, -0.3503077, 0.3504486
9: -11.7131224, -10.8442421, -11.7132759, -10.8442421, -0.3335128, 0.3336730

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 529

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1343698, upper bound: 0.1359696
time: 3.92 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1343698, upper bound: 0.1361608
time: 4.87 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.8303578, -2.2794540, -2.8300631, -2.2850869, -0.2773488, 0.2791523
1: -14.9293127, -14.1306381, -14.9170189, -14.1306639, -0.3841298, 0.3794162
2: -4.6703649, -4.2078485, -4.6656599, -4.2078786, -0.3029532, 0.2995026
3: -16.2540970, -15.5103264, -16.2423401, -15.5104847, -0.4296856, 0.4292178
4: -1.8111565, -1.2336020, -1.8037574, -1.2336249, -0.2184521, 0.2158810
5: -6.5760417, -6.1027865, -6.5715957, -6.1027856, -0.2228426, 0.2188129
6: 9.5074682, 10.1117277, 9.5077505, 10.1110096, -0.2513806, 0.2519460
7: -14.1340694, -13.4176931, -14.1340542, -13.4225655, -0.2955370, 0.3000774
8: -4.0803242, -3.3867862, -4.0801935, -3.3881683, -0.3518038, 0.3518653
9: -11.7133713, -10.8430557, -11.7132750, -10.8442421, -0.3340304, 0.3348598

Time for backsubstitution: 21.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 529

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1359690
time: 5.15 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1361610
time: 4.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.69 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 6, lower bound: -0.1343705, upper bound: 0.1343705
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 6, lower bound: -0.1343705, upper bound: 0.1345623
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 6, lower bound: -0.1359688, upper bound: 0.1343695
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 6, lower bound: -0.1359688, upper bound: 0.1345617
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 6, lower bound: -0.1361605, upper bound: 0.1343699
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 6, lower bound: -0.1361605, upper bound: 0.1345617
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 6, lower bound: -0.1343698, upper bound: 0.1359685
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 6, lower bound: -0.1343698, upper bound: 0.1361605
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1359686
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1361603
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 6, lower bound: -0.1343698, upper bound: 0.1359696
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 6, lower bound: -0.1343698, upper bound: 0.1361608
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1359690
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1361610

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.8223438, -2.2948422, -2.8223438, -2.2948422, -0.2693179, 0.2693181
1: -14.9085712, -14.1369381, -14.9085712, -14.1369381, -0.3673837, 0.3673840
2: -4.6650891, -4.2112875, -4.6650891, -4.2112875, -0.2896562, 0.2896562
3: -16.2239590, -15.5259171, -16.2239590, -15.5259171, -0.4085021, 0.4085021
4: -1.7967310, -1.2375360, -1.7967310, -1.2375360, -0.2105662, 0.2105660
5: -6.5634494, -6.1071596, -6.5634494, -6.1071596, -0.2132368, 0.2132369
6: 9.5126038, 10.1050730, 9.5126038, 10.1050730, -0.2468352, 0.2468352
7: -14.1258936, -13.4288712, -14.1258936, -13.4288712, -0.2897127, 0.2897127
8: -4.0720501, -3.3962929, -4.0720501, -3.3962929, -0.3387113, 0.3387113
9: -11.7104759, -10.8459015, -11.7104759, -10.8459015, -0.3297863, 0.3297863

Time for backsubstitution: 20.81 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 1986
type: A, layer: 3, pos: 1495
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 401
type: A, layer: 3, pos: 956
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1504
type: A, layer: 3, pos: 2138
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 1828
type: A, layer: 3, pos: 1082

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1319789, upper bound: 0.1302697
time: 3.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1319789, upper bound: 0.1319791
time: 4.29 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.8223438, -2.2948422, -2.8235590, -2.2892089, -0.2729266, 0.2704225
1: -14.9085712, -14.1369381, -14.9209118, -14.1352949, -0.3690267, 0.3744805
2: -4.6650891, -4.2112875, -4.6697941, -4.2106061, -0.2903161, 0.2941537
3: -16.2239590, -15.5259171, -16.2357216, -15.5241251, -0.4102468, 0.4144135
4: -1.7967310, -1.2375360, -1.8041315, -1.2365417, -0.2115586, 0.2137688
5: -6.5634494, -6.1071596, -6.5679245, -6.1065531, -0.2138445, 0.2176178
6: 9.5126038, 10.1050730, 9.5120640, 10.1057901, -0.2475436, 0.2472161
7: -14.1258936, -13.4288712, -14.1265459, -13.4239931, -0.2945974, 0.2903714
8: -4.0720501, -3.3962929, -4.0724397, -3.3948822, -0.3401210, 0.3390796
9: -11.7104759, -10.8459015, -11.7107296, -10.8447151, -0.3309741, 0.3301070

Time for backsubstitution: 21.74 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 1986
type: A, layer: 3, pos: 1495
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 401
type: A, layer: 3, pos: 956
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1504
type: A, layer: 3, pos: 2138
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 1828
type: A, layer: 3, pos: 1082

Time for candidate selection: 0.44 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1319789, upper bound: 0.1304612
time: 4.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1319789, upper bound: 0.1321705
time: 4.57 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.8223438, -2.2948422, -2.8291516, -2.2850904, -0.2757776, 0.2745187
1: -14.9085712, -14.1369381, -14.9169779, -14.1322823, -0.3720536, 0.3751245
2: -4.6650891, -4.2112875, -4.6656599, -4.2085428, -0.2906740, 0.2903559
3: -16.2239590, -15.5259171, -16.2423325, -15.5121326, -0.4175954, 0.4192917
4: -1.7967310, -1.2375360, -1.8037550, -1.2345946, -0.2134970, 0.2169302
5: -6.5634494, -6.1071596, -6.5715585, -6.1033940, -0.2169846, 0.2210379
6: 9.5126038, 10.1050730, 9.5079994, 10.1110106, -0.2523453, 0.2511590
7: -14.1258936, -13.4288712, -14.1334162, -13.4225721, -0.2959464, 0.2967906
8: -4.0720501, -3.3962929, -4.0799246, -3.3882055, -0.3468049, 0.3448234
9: -11.7104759, -10.8459015, -11.7131224, -10.8442421, -0.3314445, 0.3330672

Time for backsubstitution: 21.04 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 1986
type: A, layer: 3, pos: 1495
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 401
type: A, layer: 3, pos: 956
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1504
type: A, layer: 3, pos: 2138
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 1828
type: A, layer: 3, pos: 1082

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1334282, upper bound: 0.1302695
time: 3.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1334282, upper bound: 0.1319781
time: 5.84 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.8223438, -2.2948422, -2.8303578, -2.2794540, -0.2758286, 0.2756269
1: -14.9085712, -14.1369381, -14.9293127, -14.1306381, -0.3736944, 0.3773737
2: -4.6650891, -4.2112875, -4.6703649, -4.2078485, -0.2913587, 0.2946014
3: -16.2239590, -15.5259171, -16.2540970, -15.5103264, -0.4193792, 0.4193065
4: -1.7967310, -1.2375360, -1.8111565, -1.2336020, -0.2144902, 0.2170273
5: -6.5634494, -6.1071596, -6.5760417, -6.1027865, -0.2175925, 0.2219403
6: 9.5126038, 10.1050730, 9.5074682, 10.1117277, -0.2523973, 0.2515376
7: -14.1258936, -13.4288712, -14.1340694, -13.4176931, -0.2969993, 0.2974579
8: -4.0720501, -3.3962929, -4.0803242, -3.3867862, -0.3477921, 0.3452108
9: -11.7104759, -10.8459015, -11.7133713, -10.8430557, -0.3326325, 0.3333797

Time for backsubstitution: 21.64 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.14 + 544.96 = 603.10 seconds
