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
execution time: IAR + RelationalAnalysis = 21.43 + 35.03 = 56.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.1361659, upper bound: 0.1361659

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4653
type: A, layer: 1, pos: 4653
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4653

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345626, upper bound: 0.1361618
time: 4.88 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1361620, upper bound: 0.1361623
time: 4.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.49 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 9.49
Output dim: 6, lower bound: -0.1345626, upper bound: 0.1361618
NS_B2, status: Status.UNKNOWN, split count: 1, time: 9.49
Output dim: 6, lower bound: -0.1361620, upper bound: 0.1361623

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -2.8296356, -2.2897878, -2.8232565, -2.2948399, -0.2766917, 0.2752676
1: -14.9129429, -14.1310635, -14.9086075, -14.1353188, -0.3715308, 0.3747427
2: -4.6655660, -4.2092061, -4.6650896, -4.2106242, -0.2929223, 0.2927148
3: -16.2335339, -15.5126858, -16.2239666, -15.5242643, -0.4199436, 0.4207442
4: -1.8006141, -1.2336793, -1.7967346, -1.2365634, -0.2154374, 0.2144287
5: -6.5678339, -6.1028056, -6.5634851, -6.1065507, -0.2175113, 0.2178346
6: 9.5080833, 10.1082163, 9.5123577, 10.1050739, -0.2512873, 0.2500956
7: -14.1303606, -13.4229021, -14.1265354, -13.4288635, -0.2941880, 0.2965260
8: -4.0781803, -3.3920112, -4.0723200, -3.3962564, -0.3442748, 0.3435521
9: -11.7126274, -10.8450565, -11.7106304, -10.8459015, -0.3323786, 0.3308058

Time for backsubstitution: 19.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4653
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4653

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345627, upper bound: 0.1345626
time: 4.09 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345627, upper bound: 0.1361623
time: 4.94 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -2.8300667, -2.2850802, -2.8300662, -2.2850871, -0.2768643, 0.2853105
1: -14.9170246, -14.1306610, -14.9170170, -14.1306601, -0.3814464, 0.3787227
2: -4.6656609, -4.2078729, -4.6656599, -4.2078772, -0.2984786, 0.2927876
3: -16.2423534, -15.5104771, -16.2423420, -15.5104809, -0.4379137, 0.4274375
4: -1.8037624, -1.2336226, -1.8037565, -1.2336228, -0.2216612, 0.2154400
5: -6.5715995, -6.1027842, -6.5715957, -6.1027832, -0.2254899, 0.2185560
6: 9.5077496, 10.1110144, 9.5077515, 10.1110086, -0.2512408, 0.2573316
7: -14.1340618, -13.4225636, -14.1340580, -13.4225645, -0.3041224, 0.2952049
8: -4.0801983, -3.3881640, -4.0801945, -3.3881698, -0.3506544, 0.3534713
9: -11.7132750, -10.8442392, -11.7132759, -10.8442421, -0.3336816, 0.3348999

Time for backsubstitution: 20.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 4653
type: A, layer: 1, pos: 529

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of NS_B2_B1

### Relational analysis result of NS_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1361610, upper bound: 0.1359695
time: 3.62 seconds

## Relational analysis of NS_B2_B2

### Relational analysis result of NS_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1361611, upper bound: 0.1361612
time: 3.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.67 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 27.67
Output dim: 6, lower bound: -0.1345627, upper bound: 0.1345626
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 27.67
Output dim: 6, lower bound: -0.1345627, upper bound: 0.1361623
NS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 27.67
Output dim: 6, lower bound: -0.1361610, upper bound: 0.1359695
NS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 27.67
Output dim: 6, lower bound: -0.1361611, upper bound: 0.1361612

## BFS NS instance: NS_B1_A1

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

Time for backsubstitution: 20.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 529

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1343699, upper bound: 0.1345624
time: 4.63 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1345623
time: 3.96 seconds

## BFS NS instance: NS_B1_A2

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

Time for backsubstitution: 20.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 529

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345617, upper bound: 0.1359694
time: 5.30 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1361610
time: 5.30 seconds

## BFS NS instance: NS_B2_B1

### Backsubstitution after applying NS history:
0: -2.8300667, -2.2850802, -2.8291516, -2.2850904, -0.2768631, 0.2843742
1: -14.9170246, -14.1306610, -14.9169779, -14.1322823, -0.3798265, 0.3787005
2: -4.6656609, -4.2078729, -4.6656599, -4.2085428, -0.2977629, 0.2926931
3: -16.2423534, -15.5104771, -16.2423325, -15.5121326, -0.4361858, 0.4272861
4: -1.8037624, -1.2336226, -1.8037550, -1.2345946, -0.2206877, 0.2154381
5: -6.5715995, -6.1027842, -6.5715585, -6.1033940, -0.2248803, 0.2184639
6: 9.5077496, 10.1110144, 9.5079994, 10.1110106, -0.2512383, 0.2570548
7: -14.1340618, -13.4225636, -14.1334162, -13.4225721, -0.3041112, 0.2945578
8: -4.0801983, -3.3881640, -4.0799246, -3.3882055, -0.3504505, 0.3531260
9: -11.7132750, -10.8442392, -11.7131224, -10.8442421, -0.3336747, 0.3347328

Time for backsubstitution: 20.89 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 1986
type: B, layer: 3, pos: 1986
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 1495
type: B, layer: 3, pos: 1495
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: A, layer: 3, pos: 956
type: B, layer: 3, pos: 956
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1504
type: A, layer: 3, pos: 1504
type: B, layer: 3, pos: 2138
type: A, layer: 3, pos: 2138
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 1828
type: A, layer: 3, pos: 1828
type: A, layer: 3, pos: 1082
type: B, layer: 3, pos: 1082

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 766

## Relational analysis of NS_B2_B1_B1

### Relational analysis result of NS_B2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1322608, upper bound: 0.1334285
time: 8.12 seconds

## Relational analysis of NS_B2_B1_B2

### Relational analysis result of NS_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1336205, upper bound: 0.1334286
time: 5.02 seconds

## BFS NS instance: NS_B2_B2

### Backsubstitution after applying NS history:
0: -2.8300648, -2.2850795, -2.8303578, -2.2794540, -0.2798777, 0.2858018
1: -14.9170227, -14.1306648, -14.9293127, -14.1306381, -0.3821402, 0.3841302
2: -4.6656609, -4.2078762, -4.6703649, -4.2078485, -0.2995050, 0.2968011
3: -16.2423515, -15.5104790, -16.2540970, -15.5103264, -0.4388199, 0.4315753
4: -1.8037620, -1.2336242, -1.8111565, -1.2336020, -0.2221024, 0.2184997
5: -6.5715995, -6.1027856, -6.5760417, -6.1027865, -0.2257466, 0.2228427
6: 9.5077505, 10.1110144, 9.5074682, 10.1117277, -0.2519460, 0.2574718
7: -14.1340609, -13.4225636, -14.1340694, -13.4176931, -0.3051775, 0.2955372
8: -4.0801954, -3.3881621, -4.0803242, -3.3867862, -0.3518670, 0.3546219
9: -11.7132759, -10.8442392, -11.7133713, -10.8430557, -0.3348618, 0.3352437

Time for backsubstitution: 20.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 1986
type: B, layer: 3, pos: 1986
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1495
type: B, layer: 3, pos: 1495
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: A, layer: 3, pos: 956
type: B, layer: 3, pos: 956
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1504
type: A, layer: 3, pos: 1504
type: B, layer: 3, pos: 2138
type: A, layer: 3, pos: 2138
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 1828
type: A, layer: 3, pos: 1828
type: A, layer: 3, pos: 1082
type: B, layer: 3, pos: 1082

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 766

## Relational analysis of NS_B2_B2_B1

### Relational analysis result of NS_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1322606, upper bound: 0.1336202
time: 5.67 seconds

## Relational analysis of NS_B2_B2_B2

### Relational analysis result of NS_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1336204, upper bound: 0.1336202
time: 4.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.86 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 31.86
Output dim: 6, lower bound: -0.1343699, upper bound: 0.1345624
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 31.86
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1345623
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 31.86
Output dim: 6, lower bound: -0.1345617, upper bound: 0.1359694
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 31.86
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1361610
NS_B2_B1_B1, status: Status.VERIFIED, split count: 3, time: 31.86
Output dim: 6, lower bound: -0.1322608, upper bound: 0.1334285
NS_B2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 31.86
Output dim: 6, lower bound: -0.1336205, upper bound: 0.1334286
NS_B2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 31.86
Output dim: 6, lower bound: -0.1322606, upper bound: 0.1336202
NS_B2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 31.86
Output dim: 6, lower bound: -0.1336204, upper bound: 0.1336202

## BFS NS instance: NS_B1_A1_A1

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

Time for backsubstitution: 20.78 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 1986
type: B, layer: 3, pos: 1986
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 1495
type: A, layer: 3, pos: 1495
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: B, layer: 3, pos: 956
type: A, layer: 3, pos: 956
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1504
type: B, layer: 3, pos: 1504
type: B, layer: 3, pos: 2138
type: A, layer: 3, pos: 2138
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 1828
type: A, layer: 3, pos: 1828
type: A, layer: 3, pos: 1082
type: B, layer: 3, pos: 1082

Time for candidate selection: 0.40 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_B1_A1_A1_A1

### Relational analysis result of NS_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1319789, upper bound: 0.1304612
time: 3.99 seconds

## Relational analysis of NS_B1_A1_A1_A2

### Relational analysis result of NS_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1319789, upper bound: 0.1321705
time: 3.84 seconds

## BFS NS instance: NS_B1_A1_A2

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

Time for backsubstitution: 20.90 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 1986
type: B, layer: 3, pos: 1986
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1495
type: B, layer: 3, pos: 1495
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: B, layer: 3, pos: 956
type: A, layer: 3, pos: 956
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1504
type: B, layer: 3, pos: 1504
type: B, layer: 3, pos: 2138
type: A, layer: 3, pos: 2138
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 1828
type: A, layer: 3, pos: 1828
type: A, layer: 3, pos: 1082
type: B, layer: 3, pos: 1082

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_B1_A1_A2_A1

### Relational analysis result of NS_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1321704, upper bound: 0.1304611
time: 3.11 seconds

## Relational analysis of NS_B1_A1_A2_A2

### Relational analysis result of NS_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1321704, upper bound: 0.1321705
time: 3.42 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.8300662, -2.2850871, -2.8223438, -2.2948422, -0.2754550, 0.2757789
1: -14.9170170, -14.1306601, -14.9085712, -14.1369381, -0.3751500, 0.3736732
2: -4.6656599, -4.2078772, -4.6650891, -4.2112875, -0.2904503, 0.2913940
3: -16.2423420, -15.5104809, -16.2239590, -15.5259171, -0.4194441, 0.4193234
4: -1.8037565, -1.2336228, -1.7967310, -1.2375360, -0.2169322, 0.2144704
5: -6.5715957, -6.1027832, -6.5634494, -6.1071596, -0.2211289, 0.2175940
6: 9.5077515, 10.1110086, 9.5126038, 10.1050730, -0.2514358, 0.2523475
7: -14.1340580, -13.4225645, -14.1258936, -13.4288712, -0.2974375, 0.2959576
8: -4.0801945, -3.3881698, -4.0720501, -3.3962929, -0.3451686, 0.3470092
9: -11.7132759, -10.8442421, -11.7104759, -10.8459015, -0.3332345, 0.3314517

Time for backsubstitution: 20.87 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 1986
type: B, layer: 3, pos: 1986
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1495
type: B, layer: 3, pos: 1495
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: A, layer: 3, pos: 956
type: B, layer: 3, pos: 956
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1504
type: A, layer: 3, pos: 1504
type: A, layer: 3, pos: 2138
type: B, layer: 3, pos: 2138
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 1828
type: A, layer: 3, pos: 1828
type: A, layer: 3, pos: 1082
type: B, layer: 3, pos: 1082

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1321698, upper bound: 0.1320683
time: 4.15 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1321698, upper bound: 0.1334279
time: 5.68 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.8300631, -2.2850869, -2.8235590, -2.2892089, -0.2755072, 0.2768874
1: -14.9170189, -14.1306639, -14.9209118, -14.1352949, -0.3774655, 0.3775153
2: -4.6656599, -4.2078786, -4.6697941, -4.2106061, -0.2921939, 0.2954693
3: -16.2423401, -15.5104847, -16.2357216, -15.5241251, -0.4214392, 0.4193380
4: -1.8037574, -1.2336249, -1.8041315, -1.2365417, -0.2179906, 0.2151346
5: -6.5715957, -6.1027856, -6.5679245, -6.1065531, -0.2219951, 0.2200481
6: 9.5077505, 10.1110096, 9.5120640, 10.1057901, -0.2519684, 0.2527721
7: -14.1340542, -13.4225655, -14.1265459, -13.4239931, -0.2974548, 0.2969370
8: -4.0801935, -3.3881683, -4.0724397, -3.3948822, -0.3465774, 0.3483596
9: -11.7132750, -10.8442421, -11.7107296, -10.8447151, -0.3344216, 0.3319666

Time for backsubstitution: 20.89 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 1986
type: B, layer: 3, pos: 1986
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1495
type: B, layer: 3, pos: 1495
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: B, layer: 3, pos: 956
type: A, layer: 3, pos: 956
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 1504
type: A, layer: 3, pos: 1504
type: A, layer: 3, pos: 2138
type: B, layer: 3, pos: 2138
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 1828
type: A, layer: 3, pos: 1828
type: A, layer: 3, pos: 1082
type: B, layer: 3, pos: 1082

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1321697, upper bound: 0.1322594
time: 4.92 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1321697, upper bound: 0.1336200
time: 4.65 seconds

## BFS NS instance: NS_B2_B1_B2

### Backsubstitution after applying NS history:
0: -2.8248367, -2.2850819, -2.8195474, -2.2655315, -0.3049045, 0.2723882
1: -14.9169655, -14.1407175, -14.9448700, -14.1508026, -0.3538606, 0.4138732
2: -4.6633172, -4.2079306, -4.6607680, -4.1998668, -0.3052211, 0.2849112
3: -16.2423515, -15.5190296, -16.2636585, -15.5310879, -0.4293590, 0.4271841
4: -1.7974701, -1.2336278, -1.7928026, -1.2077031, -0.2587377, 0.2027140
5: -6.5694509, -6.1027842, -6.5670004, -6.0939460, -0.2340953, 0.2126318
6: 9.5077839, 10.1074076, 9.4969158, 10.1043224, -0.2407302, 0.2732674
7: -14.1340599, -13.4328775, -14.1858530, -13.4436722, -0.2724401, 0.3808084
8: -4.0801311, -3.3891685, -4.0822415, -3.3905263, -0.3454556, 0.3533669
9: -11.7112818, -10.8442402, -11.7084932, -10.8255310, -0.3545270, 0.3265378

Time for backsubstitution: 20.93 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1986
type: B, layer: 3, pos: 1986
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 1495
type: B, layer: 3, pos: 1495
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: B, layer: 3, pos: 956
type: A, layer: 3, pos: 956
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1504
type: A, layer: 3, pos: 1504
type: B, layer: 3, pos: 2138
type: A, layer: 3, pos: 2138
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 1828
type: A, layer: 3, pos: 1828
type: A, layer: 3, pos: 1082
type: B, layer: 3, pos: 1082

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 1986

## Relational analysis of NS_B2_B1_B2_A1

### Relational analysis result of NS_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1321233, upper bound: 0.1310896
time: 3.55 seconds

## Relational analysis of NS_B2_B1_B2_A2

### Relational analysis result of NS_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1320187, upper bound: 0.1318273
time: 3.73 seconds

## BFS NS instance: NS_B2_B2_B1

### Backsubstitution after applying NS history:
0: -2.8249564, -2.2850790, -2.8173332, -2.2794545, -0.2733527, 0.2697077
1: -14.9170017, -14.1378603, -14.9292641, -14.1489830, -0.3529742, 0.3725457
2: -4.6635542, -4.2078953, -4.6654844, -4.2078986, -0.2953455, 0.2879467
3: -16.2423515, -15.5180597, -16.2540970, -15.5254021, -0.4291906, 0.4274454
4: -1.7970157, -1.2336261, -1.7942185, -1.2336068, -0.2137263, 0.1997308
5: -6.5691152, -6.1027865, -6.5698118, -6.1027870, -0.2231634, 0.2158878
6: 9.5077639, 10.1086369, 9.5075035, 10.1058702, -0.2417516, 0.2536628
7: -14.1340618, -13.4358730, -14.1340694, -13.4509649, -0.2582932, 0.2763472
8: -4.0801744, -3.3888097, -4.0802670, -3.3884373, -0.3472683, 0.3518443
9: -11.7086153, -10.8442411, -11.7014885, -10.8430557, -0.3302855, 0.3229036

Time for backsubstitution: 20.91 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1986
type: B, layer: 3, pos: 1986
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 1495
type: B, layer: 3, pos: 1495
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: A, layer: 3, pos: 956
type: B, layer: 3, pos: 956
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1504
type: A, layer: 3, pos: 1504
type: B, layer: 3, pos: 2138
type: A, layer: 3, pos: 2138
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 1828
type: A, layer: 3, pos: 1828
type: A, layer: 3, pos: 1082
type: B, layer: 3, pos: 1082

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 1986

## Relational analysis of NS_B2_B2_B1_A1

### Relational analysis result of NS_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1307711, upper bound: 0.1312796
time: 3.43 seconds

## Relational analysis of NS_B2_B2_B1_A2

### Relational analysis result of NS_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1306589, upper bound: 0.1320187
time: 4.22 seconds

## BFS NS instance: NS_B2_B2_B2

### Backsubstitution after applying NS history:
0: -2.8248334, -2.2850819, -2.8207650, -2.2598953, -0.3081169, 0.2737844
1: -14.9169674, -14.1407204, -14.9571991, -14.1491585, -0.3561754, 0.4213326
2: -4.6633182, -4.2079320, -4.6654711, -4.1991696, -0.3069637, 0.2885938
3: -16.2423515, -15.5190334, -16.2754230, -15.5292816, -0.4319980, 0.4301703
4: -1.7974696, -1.2336276, -1.8002026, -1.2067070, -0.2601495, 0.2058226
5: -6.5694513, -6.1027875, -6.5714788, -6.0933390, -0.2349620, 0.2170081
6: 9.5077848, 10.1074095, 9.4963970, 10.1050415, -0.2414393, 0.2737316
7: -14.1340590, -13.4328756, -14.1865063, -13.4387932, -0.2724570, 0.3817878
8: -4.0801306, -3.3891702, -4.0826445, -3.3891075, -0.3468611, 0.3548608
9: -11.7112808, -10.8442402, -11.7087440, -10.8243446, -0.3557153, 0.3270471

Time for backsubstitution: 20.88 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1986
type: B, layer: 3, pos: 1986
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 1495
type: B, layer: 3, pos: 1495
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: B, layer: 3, pos: 956
type: A, layer: 3, pos: 956
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1504
type: A, layer: 3, pos: 1504
type: B, layer: 3, pos: 2138
type: A, layer: 3, pos: 2138
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 1828
type: A, layer: 3, pos: 1828
type: A, layer: 3, pos: 1082
type: B, layer: 3, pos: 1082

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1986

## Relational analysis of NS_B2_B2_B2_A1

### Relational analysis result of NS_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1321232, upper bound: 0.1312797
time: 3.57 seconds

## Relational analysis of NS_B2_B2_B2_A2

### Relational analysis result of NS_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1320186, upper bound: 0.1320188
time: 3.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.51 seconds
NS_B1_A1_A1_A1, status: Status.VERIFIED, split count: 4, time: 28.51
Output dim: 6, lower bound: -0.1319789, upper bound: 0.1304612
NS_B1_A1_A1_A2, status: Status.VERIFIED, split count: 4, time: 28.51
Output dim: 6, lower bound: -0.1319789, upper bound: 0.1321705
NS_B1_A1_A2_A1, status: Status.VERIFIED, split count: 4, time: 28.51
Output dim: 6, lower bound: -0.1321704, upper bound: 0.1304611
NS_B1_A1_A2_A2, status: Status.VERIFIED, split count: 4, time: 28.51
Output dim: 6, lower bound: -0.1321704, upper bound: 0.1321705
NS_B1_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 28.51
Output dim: 6, lower bound: -0.1321698, upper bound: 0.1320683
NS_B1_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 28.51
Output dim: 6, lower bound: -0.1321698, upper bound: 0.1334279
NS_B1_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 28.51
Output dim: 6, lower bound: -0.1321697, upper bound: 0.1322594
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 6, lower bound: -0.1321697, upper bound: 0.1336200
NS_B2_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 28.51
Output dim: 6, lower bound: -0.1321233, upper bound: 0.1310896
NS_B2_B1_B2_A2, status: Status.VERIFIED, split count: 4, time: 28.51
Output dim: 6, lower bound: -0.1320187, upper bound: 0.1318273
NS_B2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 28.51
Output dim: 6, lower bound: -0.1307711, upper bound: 0.1312796
NS_B2_B2_B1_A2, status: Status.VERIFIED, split count: 4, time: 28.51
Output dim: 6, lower bound: -0.1306589, upper bound: 0.1320187
NS_B2_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 28.51
Output dim: 6, lower bound: -0.1321232, upper bound: 0.1312797
NS_B2_B2_B2_A2, status: Status.VERIFIED, split count: 4, time: 28.51
Output dim: 6, lower bound: -0.1320186, upper bound: 0.1320188

## BFS NS instance: NS_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.8204579, -2.2655296, -2.8179321, -2.2892108, -0.2613610, 0.3033057
1: -14.9449053, -14.1491861, -14.9208708, -14.1460419, -0.4182215, 0.3463376
2: -4.6607685, -4.1992006, -4.6671968, -4.2106495, -0.2837579, 0.3036809
3: -16.2636681, -15.5294399, -16.2357216, -15.5336819, -0.4164445, 0.4097264
4: -1.7928059, -1.2067301, -1.7984245, -1.2365441, -0.2047396, 0.2559370
5: -6.5670376, -6.0933380, -6.5660486, -6.1065540, -0.2158899, 0.2299799
6: 9.4966640, 10.1043234, 9.5120983, 10.1023064, -0.2687683, 0.2419195
7: -14.1864920, -13.4436646, -14.1265430, -13.4353485, -0.3804891, 0.2596307
8: -4.0825095, -3.3904893, -4.0723934, -3.3958657, -0.3469460, 0.3427603
9: -11.7086477, -10.8255291, -11.7085037, -10.8447151, -0.3246987, 0.3532450

Time for backsubstitution: 20.91 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1986
type: A, layer: 3, pos: 1986
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 1495
type: B, layer: 3, pos: 1495
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: A, layer: 3, pos: 956
type: B, layer: 3, pos: 956
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1504
type: A, layer: 3, pos: 1504
type: A, layer: 3, pos: 2138
type: B, layer: 3, pos: 2138
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 1828
type: A, layer: 3, pos: 1828
type: A, layer: 3, pos: 1082
type: B, layer: 3, pos: 1082

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 1986

## Relational analysis of NS_B1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1298306, upper bound: 0.1321234
time: 3.84 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1305681, upper bound: 0.1320179
time: 4.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 29.39 seconds
NS_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 29.39
Output dim: 6, lower bound: -0.1298306, upper bound: 0.1321234
NS_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 29.39
Output dim: 6, lower bound: -0.1305681, upper bound: 0.1320179

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 56.45 + 427.36 = 483.82 seconds
