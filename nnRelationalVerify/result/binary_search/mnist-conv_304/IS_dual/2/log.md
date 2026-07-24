## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.579334386
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.3477392, 4.3477392)
1: (-7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.7961807, 3.7961807)
2: (-10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807)
3: (-12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1542416, 3.1542416)
4: (5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894)
5: (-8.9787197, -5.6989894, -8.9787197, -5.6989894, -3.2797303, 3.2797303)
6: (-12.5030499, -8.9509478, -12.5030499, -8.9509478, -3.3759890, 3.3759892)
7: (-5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9533715)
8: (-1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666)
9: (-6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219)

## BASE Result
execution time: IAR + LP analysis = 15.49 + 34.21 = 49.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -2.5504220, upper bound: 2.5504211


# Binary Search by BASE starts (time budget: 3550.30 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.402289390563965
rel_dist={4: [-1.981378173882315, 1.9813780589154808]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=3.2979745864868164
rel_dist={4: [-1.6115673031162494, 1.6115673021877095]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=3.17470645904541
rel_dist={4: [-1.3305644078822736, 1.3305637963562233]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=3.2363405227661133
rel_dist={4: [-1.4769317174548258, 1.4769330892355903]}

## Binary Search Result
Binary search time: 204.76 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3345.54 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438415, upper bound: 2.0856250
time: 4.80 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0911774, upper bound: 2.0911797
time: 4.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.98 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 9.98
Output dim: 4, lower bound: -2.0438415, upper bound: 2.0856250
IS_B2, status: Status.UNKNOWN, split count: 1, time: 9.98
Output dim: 4, lower bound: -2.0911774, upper bound: 2.0911797

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -13.3082533, -9.0077085, -13.2443447, -9.1210403, -3.8894958, 3.7532387
1: -7.2750816, -3.5116777, -7.1863823, -3.5408008, -3.6059465, 3.5092392
2: -10.0466843, -7.2668419, -10.0132523, -7.2979498, -2.7487345, 2.7464104
3: -12.5522881, -9.4253349, -12.5013924, -9.4653015, -2.9792037, 3.0022180
4: 5.3250570, 8.6787567, 5.4078045, 8.5721302, -3.2470732, 3.2709522
5: -8.9715576, -5.7159948, -8.9391518, -5.7702613, -2.9114065, 2.9185238
6: -12.4917459, -8.9564810, -12.4553852, -8.9797430, -2.7835293, 2.7732391
7: -5.6693201, -2.7628076, -5.5616732, -2.8372822, -2.8320379, 2.7988656
8: -1.2075975, 1.9895597, -1.1583521, 1.9369693, -3.1445668, 3.1479118
9: -6.5773993, -3.8508630, -6.5155468, -3.9069538, -2.6704454, 2.6646838

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438416, upper bound: 2.0438420
time: 4.70 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438436, upper bound: 2.0856251
time: 4.55 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -13.3209705, -8.9732313, -13.3209581, -8.9732561, -4.0634737, 4.0759101
1: -7.3048649, -3.5086842, -7.3048439, -3.5086877, -3.6828370, 3.6606903
2: -10.0570240, -7.2594433, -10.0570183, -7.2594490, -2.7975750, 2.7975750
3: -12.5703182, -9.4160767, -12.5703068, -9.4160843, -3.1258359, 3.0888529
4: 5.3104191, 8.7127085, 5.3104267, 8.7126923, -3.4022732, 3.4022818
5: -8.9787197, -5.6989894, -8.9787130, -5.6990013, -2.9880142, 2.9914398
6: -12.5030499, -8.9509478, -12.5030413, -8.9513454, -2.8489442, 2.8292623
7: -5.7039032, -2.7505317, -5.7038751, -2.7505379, -2.9533653, 2.9533434
8: -1.2158751, 2.0059915, -1.2158689, 2.0059829, -3.2218580, 3.2218604
9: -6.5885048, -3.8328829, -6.5884991, -3.8328958, -2.7556090, 2.7556162

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0856248, upper bound: 2.0438436
time: 4.26 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0856248, upper bound: 2.0438415
time: 5.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.03 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 24.03
Output dim: 4, lower bound: -2.0438416, upper bound: 2.0438420
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 24.03
Output dim: 4, lower bound: -2.0438436, upper bound: 2.0856251
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 24.03
Output dim: 4, lower bound: -2.0856248, upper bound: 2.0438436
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 24.03
Output dim: 4, lower bound: -2.0856248, upper bound: 2.0438415

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -13.2443447, -9.1210403, -13.2443447, -9.1210403, -3.6281986, 3.6281986
1: -7.1863823, -3.5408008, -7.1863823, -3.5408008, -3.4738631, 3.4738626
2: -10.0132523, -7.2979498, -10.0132523, -7.2979498, -2.7153025, 2.7153025
3: -12.5013924, -9.4653015, -12.5013924, -9.4653015, -2.9202967, 2.9202967
4: 5.4078045, 8.5721302, 5.4078045, 8.5721302, -3.1643257, 3.1643257
5: -8.9391518, -5.7702613, -8.9391518, -5.7702613, -2.8614812, 2.8614810
6: -12.4553852, -8.9797430, -12.4553852, -8.9797430, -2.7466111, 2.7466114
7: -5.5616732, -2.8372822, -5.5616732, -2.8372822, -2.7243910, 2.7243910
8: -1.1583521, 1.9369693, -1.1583521, 1.9369693, -3.0953214, 3.0953214
9: -6.5155468, -3.9069538, -6.5155468, -3.9069538, -2.6085930, 2.6085930

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438412, upper bound: 2.0438462
time: 4.57 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438432, upper bound: 2.0438458
time: 5.27 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -13.3209581, -8.9732561, -13.2443447, -9.1210403, -3.8986282, 3.7611413
1: -7.3048439, -3.5086877, -7.1863823, -3.5408008, -3.6243048, 3.5147882
2: -10.0570183, -7.2594490, -10.0132523, -7.2979498, -2.7590685, 2.7538033
3: -12.5703068, -9.4160843, -12.5013924, -9.4653015, -2.9995625, 3.0036714
4: 5.3104267, 8.7126923, 5.4078045, 8.5721302, -3.2617035, 3.3048878
5: -8.9787130, -5.6990013, -8.9391518, -5.7702613, -2.9147463, 2.9276462
6: -12.5030413, -8.9513454, -12.4553852, -8.9797430, -2.7989178, 2.7737927
7: -5.7038751, -2.7505379, -5.5616732, -2.8372822, -2.8383317, 2.8111353
8: -1.2158689, 2.0059829, -1.1583521, 1.9369693, -3.1528382, 3.1643350
9: -6.5884991, -3.8328958, -6.5155468, -3.9069538, -2.6815453, 2.6826510

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438413, upper bound: 2.0856249
time: 4.59 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438432, upper bound: 2.0856246
time: 5.02 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -13.2443447, -9.1210403, -13.3209581, -8.9732561, -3.7611408, 3.8986285
1: -7.1863823, -3.5408008, -7.3048439, -3.5086877, -3.5147877, 3.6243043
2: -10.0132523, -7.2979498, -10.0570183, -7.2594490, -2.7538033, 2.7590685
3: -12.5013924, -9.4653015, -12.5703068, -9.4160843, -3.0036716, 2.9995625
4: 5.4078045, 8.5721302, 5.3104267, 8.7126923, -3.3048878, 3.2617035
5: -8.9391518, -5.7702613, -8.9787130, -5.6990013, -2.9276462, 2.9147458
6: -12.4553852, -8.9797430, -12.5030413, -8.9513454, -2.7737923, 2.7989182
7: -5.5616732, -2.8372822, -5.7038751, -2.7505379, -2.8111353, 2.8383317
8: -1.1583521, 1.9369693, -1.2158689, 2.0059829, -3.1643350, 3.1528382
9: -6.5155468, -3.9069538, -6.5884991, -3.8328958, -2.6826510, 2.6815453

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438416, upper bound: 2.0438416
time: 5.38 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438413, upper bound: 2.0438411
time: 5.01 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -13.3209581, -8.9732561, -13.3209581, -8.9732561, -4.0758753, 4.0758753
1: -7.3048439, -3.5086877, -7.3048439, -3.5086877, -3.6606812, 3.6606803
2: -10.0570183, -7.2594490, -10.0570183, -7.2594490, -2.7975693, 2.7975693
3: -12.5703068, -9.4160843, -12.5703068, -9.4160843, -3.1258192, 3.1258190
4: 5.3104267, 8.7126923, 5.3104267, 8.7126923, -3.4022655, 3.4022655
5: -8.9787130, -5.6990013, -8.9787130, -5.6990013, -2.9880028, 2.9880030
6: -12.5030413, -8.9513454, -12.5030413, -8.9513454, -2.8489347, 2.8489342
7: -5.7038751, -2.7505379, -5.7038751, -2.7505379, -2.9533372, 2.9533372
8: -1.2158689, 2.0059829, -1.2158689, 2.0059829, -3.2218518, 3.2218518
9: -6.5884991, -3.8328958, -6.5884991, -3.8328958, -2.7556033, 2.7556033

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438416, upper bound: 2.0438415
time: 7.39 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438412, upper bound: 2.0911797
time: 5.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.60 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 27.60
Output dim: 4, lower bound: -2.0438412, upper bound: 2.0438462
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 27.60
Output dim: 4, lower bound: -2.0438432, upper bound: 2.0438458
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 27.60
Output dim: 4, lower bound: -2.0438413, upper bound: 2.0856249
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 27.60
Output dim: 4, lower bound: -2.0438432, upper bound: 2.0856246
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 27.60
Output dim: 4, lower bound: -2.0438416, upper bound: 2.0438416
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 27.60
Output dim: 4, lower bound: -2.0438413, upper bound: 2.0438411
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 27.60
Output dim: 4, lower bound: -2.0438416, upper bound: 2.0438415
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 27.60
Output dim: 4, lower bound: -2.0438412, upper bound: 2.0911797

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.2296982, -9.1225748, -3.5379968, 3.5841811
1: -7.1668282, -3.5483451, -7.1834173, -3.5414495, -3.4514871, 3.4598784
2: -9.9397202, -7.3311028, -10.0014143, -7.3008280, -2.6388922, 2.6703115
3: -12.4820814, -9.5253382, -12.4989901, -9.4748440, -2.8791718, 2.8570466
4: 5.4385414, 8.5623474, 5.4118562, 8.5711880, -3.1326466, 3.1504912
5: -8.9218502, -5.8169274, -8.9369545, -5.7777858, -2.8364048, 2.8136606
6: -12.3745804, -8.9950542, -12.4422169, -8.9807224, -2.6671214, 2.7178981
7: -5.5388203, -2.8817828, -5.5599031, -2.8445454, -2.6942749, 2.6781204
8: -1.1019375, 1.9122753, -1.1492989, 1.9347436, -3.0366812, 3.0615742
9: -6.5032806, -3.9169111, -6.5141435, -3.9084992, -2.5947814, 2.5972323

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438458
time: 4.45 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438458
time: 4.56 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -13.2443399, -9.1210403, -13.2443447, -9.1210403, -3.5941086, 3.6267772
1: -7.1863809, -3.5407999, -7.1863823, -3.5408008, -3.4738612, 3.4759188
2: -10.0132484, -7.2979493, -10.0132523, -7.2979498, -2.7152987, 2.7153029
3: -12.5013933, -9.4653034, -12.5013924, -9.4653015, -2.9190469, 2.9073918
4: 5.4078064, 8.5721302, 5.4078045, 8.5721302, -3.1643238, 3.1643257
5: -8.9391508, -5.7702627, -8.9391518, -5.7702613, -2.8614802, 2.8563652
6: -12.4553823, -8.9797430, -12.4553852, -8.9797430, -2.6989236, 2.7466109
7: -5.5616713, -2.8372831, -5.5616732, -2.8372822, -2.7243891, 2.7243900
8: -1.1583488, 1.9369686, -1.1583521, 1.9369693, -3.0953181, 3.0953207
9: -6.5155468, -3.9069538, -6.5155468, -3.9069538, -2.6085930, 2.6085930

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438459
time: 4.68 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438457
time: 4.67 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -13.2316914, -8.9937172, -13.2296982, -9.1225748, -3.8078995, 3.6848588
1: -7.2850342, -3.5163171, -7.1834173, -3.5414495, -3.6020861, 3.5007372
2: -9.9834318, -7.2925563, -10.0014143, -7.3008280, -2.6826038, 2.7088580
3: -12.5505304, -9.4759731, -12.4989901, -9.4748440, -2.9420726, 2.9406257
4: 5.3413177, 8.7029877, 5.4118562, 8.5711880, -3.2298703, 3.2911315
5: -8.9616632, -5.7460165, -8.9369545, -5.7777858, -2.8898273, 2.8794386
6: -12.4220238, -8.9667816, -12.4422169, -8.9807224, -2.7191453, 2.7451124
7: -5.6806841, -2.7950184, -5.5599031, -2.8445454, -2.7913477, 2.7648847
8: -1.1593797, 1.9811001, -1.1492989, 1.9347436, -3.0941234, 3.1303990
9: -6.5759706, -3.8428798, -6.5141435, -3.9084992, -2.6674714, 2.6712637

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0269124, upper bound: 2.0855010
time: 4.85 seconds

## Relational analysis of IS_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438432, upper bound: 2.0856221
time: 4.85 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438432, upper bound: 2.0856222
time: 4.61 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -13.3209572, -8.9732580, -13.2443447, -9.1210403, -3.8642178, 3.7510626
1: -7.3048396, -3.5086896, -7.1863823, -3.5408008, -3.6243038, 3.5168467
2: -10.0570154, -7.2594500, -10.0132523, -7.2979498, -2.7590656, 2.7538023
3: -12.5703049, -9.4160862, -12.5013924, -9.4653015, -2.9931467, 2.9907453
4: 5.3104296, 8.7126913, 5.4078045, 8.5721302, -3.2617006, 3.3048868
5: -8.9787140, -5.6990032, -8.9391518, -5.7702613, -2.9147463, 2.9225299
6: -12.5030384, -8.9513435, -12.4553852, -8.9797430, -2.7508788, 2.7737923
7: -5.7038755, -2.7505393, -5.5616732, -2.8372822, -2.8327956, 2.8111339
8: -1.2158673, 2.0059822, -1.1583521, 1.9369693, -3.1528366, 3.1643343
9: -6.5884991, -3.8328962, -6.5155468, -3.9069538, -2.6815453, 2.6826506

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0269123, upper bound: 2.0855007
time: 4.56 seconds

## Relational analysis of IS_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438412, upper bound: 2.0856220
time: 4.85 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438411, upper bound: 2.0856220
time: 4.80 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.2296982, -9.1225748, -13.2316914, -8.9937172, -3.6848593, 3.8079004
1: -7.1834173, -3.5414495, -7.2850342, -3.5163171, -3.5007372, 3.6020851
2: -10.0014143, -7.3008280, -9.9834318, -7.2925563, -2.7088580, 2.6826038
3: -12.4989901, -9.4748440, -12.5505304, -9.4759731, -2.9406257, 2.9420724
4: 5.4118562, 8.5711880, 5.3413177, 8.7029877, -3.2911315, 3.2298703
5: -8.9369545, -5.7777858, -8.9616632, -5.7460165, -2.8794384, 2.8898273
6: -12.4422169, -8.9807224, -12.4220238, -8.9667816, -2.7451124, 2.7191455
7: -5.5599031, -2.8445454, -5.6806841, -2.7950184, -2.7648847, 2.7913473
8: -1.1492989, 1.9347436, -1.1593797, 1.9811001, -3.1303990, 3.0941234
9: -6.5141435, -3.9084992, -6.5759706, -3.8428798, -2.6712637, 2.6674714

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0855008, upper bound: 2.0269125
time: 4.54 seconds

## Relational analysis of IS_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0856217, upper bound: 2.0438414
time: 4.47 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0856243, upper bound: 2.0438413
time: 4.71 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.2443447, -9.1210403, -13.3209572, -8.9732580, -3.7510629, 3.8642168
1: -7.1863823, -3.5408008, -7.3048396, -3.5086896, -3.5168462, 3.6243038
2: -10.0132523, -7.2979498, -10.0570154, -7.2594500, -2.7538023, 2.7590656
3: -12.5013924, -9.4653015, -12.5703049, -9.4160862, -2.9907455, 2.9931469
4: 5.4078045, 8.5721302, 5.3104296, 8.7126913, -3.3048868, 3.2617006
5: -8.9391518, -5.7702613, -8.9787140, -5.6990032, -2.9225302, 2.9147453
6: -12.4553852, -8.9797430, -12.5030384, -8.9513435, -2.7737923, 2.7508788
7: -5.5616732, -2.8372822, -5.7038755, -2.7505393, -2.8111339, 2.8327956
8: -1.1583521, 1.9369693, -1.2158673, 2.0059822, -3.1643343, 3.1528366
9: -6.5155468, -3.9069538, -6.5884991, -3.8328962, -2.6826506, 2.6815453

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0855003, upper bound: 2.0269125
time: 4.30 seconds

## Relational analysis of IS_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0856217, upper bound: 2.0438413
time: 4.71 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0856217, upper bound: 2.0438414
time: 4.58 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.3063345, -8.9747963, -13.2316914, -8.9937172, -4.0214453, 3.9851618
1: -7.3018589, -3.5093508, -7.2850342, -3.5163171, -3.6467400, 3.6386099
2: -10.0451736, -7.2622795, -9.9834318, -7.2925563, -2.7526174, 2.7211523
3: -12.5678434, -9.4256077, -12.5505304, -9.4759731, -3.0627871, 3.0848236
4: 5.3144875, 8.7117653, 5.3413177, 8.7029877, -3.3885002, 3.3704476
5: -8.9765530, -5.7065763, -8.9616632, -5.7460165, -2.9394865, 2.9627774
6: -12.4898367, -8.9523640, -12.4220238, -8.9667816, -2.8202677, 2.7691717
7: -5.7020702, -2.7577899, -5.6806841, -2.7950184, -2.9070518, 2.9228942
8: -1.2068169, 2.0037317, -1.1593797, 1.9811001, -3.1879170, 3.1631114
9: -6.5870600, -3.8344464, -6.5759706, -3.8428798, -2.7441802, 2.7415242

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B2_A2_B1_B1

### Relational analysis result of IS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0456358, upper bound: 2.0874892
time: 4.72 seconds

## Relational analysis of IS_B2_A2_B1_B2

### Relational analysis result of IS_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0534471, upper bound: 2.0911765
time: 4.66 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.3209581, -8.9732561, -13.3209572, -8.9732580, -4.0744610, 4.0414009
1: -7.3048439, -3.5086877, -7.3048396, -3.5086896, -3.6627374, 3.6606789
2: -10.0570183, -7.2594490, -10.0570154, -7.2594500, -2.7975683, 2.7975664
3: -12.5703068, -9.4160843, -12.5703049, -9.4160862, -3.1133647, 3.1245222
4: 5.3104267, 8.7126923, 5.3104296, 8.7126913, -3.4022646, 3.4022627
5: -8.9787130, -5.6990013, -8.9787140, -5.6990032, -2.9828882, 2.9880035
6: -12.5030413, -8.9513454, -12.5030384, -8.9513435, -2.8489342, 2.8008952
7: -5.7038751, -2.7505379, -5.7038755, -2.7505393, -2.9533358, 2.9533377
8: -1.2158689, 2.0059829, -1.2158673, 2.0059822, -3.2218511, 3.2218502
9: -6.5884991, -3.8328958, -6.5884991, -3.8328962, -2.7556028, 2.7556033

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0459052, upper bound: 2.0329189
time: 4.65 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0534468, upper bound: 2.0911761
time: 4.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.80 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438458
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438458
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438459
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438457
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0438432, upper bound: 2.0856221
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0438432, upper bound: 2.0856222
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0438412, upper bound: 2.0856220
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0438411, upper bound: 2.0856220
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0856217, upper bound: 2.0438414
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0856243, upper bound: 2.0438413
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0856217, upper bound: 2.0438413
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0856217, upper bound: 2.0438414
IS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0456358, upper bound: 2.0874892
IS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0534471, upper bound: 2.0911765
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0459052, upper bound: 2.0329189
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.80
Output dim: 4, lower bound: -2.0534468, upper bound: 2.0911761

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.1549864, -9.1414566, -3.5145407, 3.5145411
1: -7.1668282, -3.5483451, -7.1668282, -3.5483451, -3.4420252, 3.4420257
2: -9.9397202, -7.3311028, -9.9397202, -7.3311028, -2.6086173, 2.6086173
3: -12.4820814, -9.5253382, -12.4820814, -9.5253382, -2.8322740, 2.8322740
4: 5.4385414, 8.5623474, 5.4385414, 8.5623474, -3.1238060, 3.1238060
5: -8.9218502, -5.8169274, -8.9218502, -5.8169274, -2.7979431, 2.7979429
6: -12.3745804, -8.9950542, -12.3745804, -8.9950542, -2.6531992, 2.6531992
7: -5.5388203, -2.8817828, -5.5388203, -2.8817828, -2.6570375, 2.6570375
8: -1.1019375, 1.9122753, -1.1019375, 1.9122753, -3.0142128, 3.0142128
9: -6.5032806, -3.9169111, -6.5032806, -3.9169111, -2.5863695, 2.5863695

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 317
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2333
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 709
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 1845
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 2642
type: B, layer: 3, pos: 2642
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 654
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 2369
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 1396
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 2570
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 611
type: B, layer: 3, pos: 611
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1432
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 2594
type: B, layer: 3, pos: 2594
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850

Time for candidate selection: 12.91 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166381, upper bound: 2.0240563
time: 4.88 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166381, upper bound: 2.0166387
time: 4.53 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.2443371, -9.1210804, -3.5384793, 3.6027722
1: -7.1668282, -3.5483451, -7.1863713, -3.5408142, -3.4522872, 3.4633875
2: -9.9397202, -7.3311028, -10.0132427, -7.2980623, -2.6416578, 2.6821399
3: -12.4820814, -9.5253382, -12.5013895, -9.4654007, -2.8922229, 2.8589125
4: 5.4385414, 8.5623474, 5.4078264, 8.5721016, -3.1335602, 3.1545210
5: -8.9218502, -5.8169274, -8.9391499, -5.7703357, -2.8436804, 2.8156006
6: -12.3745804, -8.9950542, -12.4553404, -8.9797535, -2.6681008, 2.7272015
7: -5.5388203, -2.8817828, -5.5616341, -2.8372846, -2.7015357, 2.6798513
8: -1.1019375, 1.9122753, -1.1583457, 1.9369197, -3.0388572, 3.0706210
9: -6.5032806, -3.9169111, -6.5155330, -3.9069543, -2.5963264, 2.5986218

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 1395
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2488
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 3105
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 1753
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2642
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 654
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 1396
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 2369
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 1384
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 2570
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 611
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 611
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1782
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 2594
type: B, layer: 3, pos: 2314
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 1165
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850

Time for candidate selection: 12.80 seconds

### Candidate
type: B, layer: 3, pos: 1690

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0240557, upper bound: 2.0166390
time: 4.87 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166381, upper bound: 2.0166387
time: 5.01 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -13.2443371, -9.1210804, -13.1549864, -9.1414566, -3.6027718, 3.5384798
1: -7.1863713, -3.5408142, -7.1668282, -3.5483451, -3.4633875, 3.4522867
2: -10.0132427, -7.2980623, -9.9397202, -7.3311028, -2.6821399, 2.6416578
3: -12.5013895, -9.4654007, -12.4820814, -9.5253382, -2.8589125, 2.8922226
4: 5.4078264, 8.5721016, 5.4385414, 8.5623474, -3.1545210, 3.1335602
5: -8.9391499, -5.7703357, -8.9218502, -5.8169274, -2.8156004, 2.8436801
6: -12.4553404, -8.9797535, -12.3745804, -8.9950542, -2.7272010, 2.6681013
7: -5.5616341, -2.8372846, -5.5388203, -2.8817828, -2.6798513, 2.7015357
8: -1.1583457, 1.9369197, -1.1019375, 1.9122753, -3.0706210, 3.0388572
9: -6.5155330, -3.9069543, -6.5032806, -3.9169111, -2.5986218, 2.5963264

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 317
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2333
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2642
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2642
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2369
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 1396
type: B, layer: 3, pos: 1384
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 234
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 611
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 1782
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 2594
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1165
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850

Time for candidate selection: 12.80 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166381, upper bound: 2.0240559
time: 4.73 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166381, upper bound: 2.0166384
time: 4.38 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -13.2443399, -9.1210403, -13.2443399, -9.1210403, -3.5941081, 3.5941081
1: -7.1863809, -3.5407999, -7.1863809, -3.5407999, -3.4759178, 3.4759173
2: -10.0132484, -7.2979493, -10.0132484, -7.2979493, -2.7152991, 2.7152991
3: -12.5013933, -9.4653034, -12.5013933, -9.4653034, -2.9073906, 2.9073908
4: 5.4078064, 8.5721302, 5.4078064, 8.5721302, -3.1643238, 3.1643238
5: -8.9391508, -5.7702627, -8.9391508, -5.7702627, -2.8563643, 2.8563643
6: -12.4553823, -8.9797430, -12.4553823, -8.9797430, -2.6989236, 2.6989236
7: -5.5616713, -2.8372831, -5.5616713, -2.8372831, -2.7243881, 2.7243881
8: -1.1583488, 1.9369686, -1.1583488, 1.9369686, -3.0953174, 3.0953174
9: -6.5155468, -3.9069538, -6.5155468, -3.9069538, -2.6085930, 2.6085930

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1451
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2488
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 1199
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 709
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2642
type: B, layer: 3, pos: 2642
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 2369
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 1396
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 234
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 611
type: B, layer: 3, pos: 611
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 2594
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 2314
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1165
type: B, layer: 3, pos: 1165
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850

Time for candidate selection: 12.83 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166381, upper bound: 2.0240578
time: 4.60 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166381, upper bound: 2.0166401
time: 5.18 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -13.2316914, -8.9937172, -13.1549864, -9.1414566, -3.7843952, 3.6475077
1: -7.2850342, -3.5163171, -7.1668282, -3.5483451, -3.5926142, 3.4828839
2: -9.9834318, -7.2925563, -9.9397202, -7.3311028, -2.6523290, 2.6471639
3: -12.5505304, -9.4759731, -12.4820814, -9.5253382, -2.9116979, 2.9158688
4: 5.3413177, 8.7029877, 5.4385414, 8.5623474, -3.2210298, 3.2644463
5: -8.9616632, -5.7460165, -8.9218502, -5.8169274, -2.8513656, 2.8636463
6: -12.4220238, -8.9667816, -12.3745804, -8.9950542, -2.7052231, 2.6804132
7: -5.6806841, -2.7950184, -5.5388203, -2.8817828, -2.7692287, 2.7438018
8: -1.1593797, 1.9811001, -1.1019375, 1.9122753, -3.0716550, 3.0830376
9: -6.5759706, -3.8428798, -6.5032806, -3.9169111, -2.6590595, 2.6562624

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0269124, upper bound: 2.0855011
time: 5.50 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 1145
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1395
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 2488
type: B, layer: 3, pos: 3105
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 1845
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 2642
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2564
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 654
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 2369
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 2570
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 611
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1782
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 2594
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850

Time for candidate selection: 18.31 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166339, upper bound: 2.0657403
time: 4.44 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166339, upper bound: 2.0585524
time: 5.35 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -13.2316914, -8.9937172, -13.2443371, -9.1210804, -3.8084106, 3.6873353
1: -7.2850342, -3.5163171, -7.1863713, -3.5408142, -3.6009922, 3.5042467
2: -9.9834318, -7.2925563, -10.0132427, -7.2980623, -2.6853695, 2.7206864
3: -12.5505304, -9.4759731, -12.5013895, -9.4654007, -2.9444888, 2.9424779
4: 5.3413177, 8.7029877, 5.4078264, 8.5721016, -3.2307839, 3.2951612
5: -8.9616632, -5.7460165, -8.9391499, -5.7703357, -2.8928230, 2.8781004
6: -12.4220238, -8.9667816, -12.4553404, -8.9797535, -2.7201257, 2.7551584
7: -5.6806841, -2.7950184, -5.5616341, -2.8372846, -2.7936130, 2.7666156
8: -1.1593797, 1.9811001, -1.1583457, 1.9369197, -3.0962994, 3.1394458
9: -6.5759706, -3.8428798, -6.5155330, -3.9069543, -2.6690164, 2.6724012

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0269124, upper bound: 2.0855010
time: 4.67 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 1145
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1395
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1199
type: B, layer: 3, pos: 2488
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 3105
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 1676
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 2642
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 2564
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 654
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1396
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 2369
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 1384
type: B, layer: 3, pos: 1396
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 2570
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 611
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1782
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 2594
type: B, layer: 3, pos: 2314
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 1165
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850

Time for candidate selection: 17.45 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166340, upper bound: 2.0657402
time: 4.45 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166340, upper bound: 2.0585513
time: 4.93 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -13.3209562, -8.9732943, -13.1549864, -9.1414566, -3.8478231, 3.6626420
1: -7.3048363, -3.5087011, -7.1668282, -3.5483451, -3.6129827, 3.4932127
2: -10.0570078, -7.2595644, -9.9397202, -7.3311028, -2.7259049, 2.6801558
3: -12.5703030, -9.4161816, -12.4820814, -9.5253382, -2.9331467, 2.9713597
4: 5.3104515, 8.7126656, 5.4385414, 8.5623474, -3.2518959, 3.2741241
5: -8.9787130, -5.6990790, -8.9218502, -5.8169274, -2.8688664, 2.8914671
6: -12.5030003, -8.9513521, -12.3745804, -8.9950542, -2.7723832, 2.6952834
7: -5.7038364, -2.7505407, -5.5388203, -2.8817828, -2.7891343, 2.7882795
8: -1.2158635, 2.0059309, -1.1019375, 1.9122753, -3.1281388, 3.1078684
9: -6.5884829, -3.8328977, -6.5032806, -3.9169111, -2.6715717, 2.6673679

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_A2_B1_B1

### Relational analysis result of IS_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0269124, upper bound: 2.0855007
time: 4.52 seconds

## Relational analysis of IS_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1704
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1395
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 2488
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 2321
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 1845
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 2642
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2564
type: B, layer: 3, pos: 2642
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 654
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1396
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 2369
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 2369
type: B, layer: 3, pos: 1396
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 234
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 611
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1432
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 2594
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1165
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850

Time for candidate selection: 16.96 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166340, upper bound: 2.0657398
time: 4.53 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166340, upper bound: 2.0585497
time: 5.07 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -13.3209572, -8.9732580, -13.2443399, -9.1210403, -3.8642154, 3.7243257
1: -7.3048396, -3.5086896, -7.1863809, -3.5407999, -3.6243281, 3.5168457
2: -10.0570154, -7.2594500, -10.0132484, -7.2979493, -2.7590661, 2.7537985
3: -12.5703049, -9.4160862, -12.5013933, -9.4653034, -2.9849057, 2.9907439
4: 5.3104296, 8.7126913, 5.4078064, 8.5721302, -3.2617006, 3.3048849
5: -8.9787140, -5.6990032, -8.9391508, -5.7702627, -2.9096313, 2.9199162
6: -12.5030384, -8.9513435, -12.4553823, -8.9797430, -2.7508793, 2.7261052
7: -5.7038755, -2.7505393, -5.5616713, -2.8372831, -2.8140163, 2.8111320
8: -1.2158673, 2.0059822, -1.1583488, 1.9369686, -3.1528358, 3.1643310
9: -6.5884991, -3.8328962, -6.5155468, -3.9069538, -2.6815453, 2.6826506

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_A2_B2_B1

### Relational analysis result of IS_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0269123, upper bound: 2.0855007
time: 4.64 seconds

## Relational analysis of IS_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 1145
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1395
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1199
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 3105
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 1845
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 2642
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2564
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 654
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 2369
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 2570
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 611
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1782
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 2594
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1165
type: B, layer: 3, pos: 1165
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850

Time for candidate selection: 17.41 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166339, upper bound: 2.0657398
time: 4.60 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0166339, upper bound: 2.0585499
time: 4.78 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 41.23 seconds
IS_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166381, upper bound: 2.0240563
IS_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166381, upper bound: 2.0166387
IS_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0240557, upper bound: 2.0166390
IS_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166381, upper bound: 2.0166387
IS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166381, upper bound: 2.0240559
IS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166381, upper bound: 2.0166384
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166381, upper bound: 2.0240578
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166381, upper bound: 2.0166401
IS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166339, upper bound: 2.0657403
IS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166339, upper bound: 2.0585524
IS_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166340, upper bound: 2.0657402
IS_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166340, upper bound: 2.0585513
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166340, upper bound: 2.0657398
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166340, upper bound: 2.0585497
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166339, upper bound: 2.0657398
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 41.23
Output dim: 4, lower bound: -2.0166339, upper bound: 2.0585499
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 41.23
Output dim: 4, lower bound: -2.0856217, upper bound: 2.0438414
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 41.23
Output dim: 4, lower bound: -2.0856243, upper bound: 2.0438413
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 41.23
Output dim: 4, lower bound: -2.0856217, upper bound: 2.0438413
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 41.23
Output dim: 4, lower bound: -2.0856217, upper bound: 2.0438414
IS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 41.23
Output dim: 4, lower bound: -2.0456358, upper bound: 2.0874892
IS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 41.23
Output dim: 4, lower bound: -2.0534471, upper bound: 2.0911765
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 41.23
Output dim: 4, lower bound: -2.0459052, upper bound: 2.0329189
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 41.23
Output dim: 4, lower bound: -2.0534468, upper bound: 2.0911761
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=3.402289390563965
rel_dist={4: [-2.091210306910777, 2.0912106090343423]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 6250
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7298040, upper bound: 1.7040483
time: 5.20 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7405272, upper bound: 1.7405245
time: 4.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.26 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.26
Output dim: 4, lower bound: -1.7298040, upper bound: 1.7040483
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.26
Output dim: 4, lower bound: -1.7405272, upper bound: 1.7405245

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.2443447, -9.1210403, -13.3002338, -9.0312510, -3.3632355, 3.5244317
1: -7.1863823, -3.5408008, -7.2550387, -3.5137615, -3.2848153, 3.3632121
2: -10.0132523, -7.2979498, -10.0396223, -7.2720428, -2.7412095, 2.7416725
3: -12.5013924, -9.4653015, -12.5400057, -9.4318829, -2.7728767, 2.7398574
4: 5.4078045, 8.5721302, 5.3353586, 8.6555586, -3.1717110, 3.1891956
5: -8.9391518, -5.7702613, -8.9665318, -5.7274981, -2.6801910, 2.6819451
6: -12.4553852, -8.9797430, -12.4841042, -8.9603281, -2.4422936, 2.4451993
7: -5.5616732, -2.8372822, -5.6456804, -2.7716033, -2.6445689, 2.5992675
8: -1.1583521, 1.9369693, -1.2017264, 1.9783597, -3.1367118, 3.1386957
9: -6.5155468, -3.9069538, -6.5694513, -3.8631310, -2.5052676, 2.5482779

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040491, upper bound: 1.7040487
time: 6.23 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040511, upper bound: 1.7040490
time: 7.47 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.3209581, -8.9732561, -13.3209705, -8.9732332, -3.7172952, 3.7061563
1: -7.3048439, -3.5086877, -7.3048630, -3.5086875, -3.4278641, 3.4636416
2: -10.0570183, -7.2594490, -10.0570240, -7.2594433, -2.7975750, 2.7975750
3: -12.5703068, -9.4160843, -12.5703182, -9.4160776, -2.8644018, 2.8975284
4: 5.3104267, 8.7126923, 5.3104200, 8.7127085, -3.3595986, 3.3551841
5: -8.9787130, -5.6990013, -8.9787178, -5.6989884, -2.7684274, 2.7591755
6: -12.5030413, -8.9513454, -12.5030479, -8.9509649, -2.5010762, 2.5187085
7: -5.7038751, -2.7505379, -5.7039027, -2.7505319, -2.7803235, 2.8084259
8: -1.2158689, 2.0059829, -1.2158742, 2.0059900, -3.2218590, 3.2218571
9: -6.5884991, -3.8328958, -6.5885057, -3.8328843, -2.6600628, 2.6359539

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7405271, upper bound: 1.7405240
time: 4.96 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7405287, upper bound: 1.7405237
time: 5.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.13 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.13
Output dim: 4, lower bound: -1.7040491, upper bound: 1.7040487
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.13
Output dim: 4, lower bound: -1.7040511, upper bound: 1.7040490
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.13
Output dim: 4, lower bound: -1.7405271, upper bound: 1.7405240
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.13
Output dim: 4, lower bound: -1.7405287, upper bound: 1.7405237

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.2443447, -9.1210403, -13.2443447, -9.1210403, -3.2660685, 3.2660682
1: -7.1863823, -3.5408008, -7.1863823, -3.5408008, -3.2533374, 3.2533379
2: -10.0132523, -7.2979498, -10.0132523, -7.2979498, -2.7153025, 2.7153025
3: -12.5013924, -9.4653015, -12.5013924, -9.4653015, -2.6949673, 2.6949675
4: 5.4078045, 8.5721302, 5.4078045, 8.5721302, -3.1093478, 3.1093473
5: -8.9391518, -5.7702613, -8.9391518, -5.7702613, -2.6367612, 2.6367612
6: -12.4553852, -8.9797430, -12.4553852, -8.9797430, -2.4189901, 2.4189901
7: -5.5616732, -2.8372822, -5.5616732, -2.8372822, -2.5433824, 2.5433829
8: -1.1583521, 1.9369693, -1.1583521, 1.9369693, -3.0953214, 3.0953214
9: -6.5155468, -3.9069538, -6.5155468, -3.9069538, -2.4698105, 2.4698105

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040864, upper bound: 1.7040507
time: 5.10 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040882, upper bound: 1.7040507
time: 5.23 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.2443447, -9.1210403, -13.3175774, -8.9732952, -3.3749557, 3.5402882
1: -7.1863823, -3.5408008, -7.3003254, -3.5087399, -3.2939253, 3.3861177
2: -10.0132523, -7.2979498, -10.0569267, -7.2605362, -2.7527161, 2.7589769
3: -12.5013924, -9.4653015, -12.5701885, -9.4177761, -2.7783513, 2.7610192
4: 5.4078045, 8.5721302, 5.3111296, 8.7126455, -3.1788497, 3.2121058
5: -8.9391518, -5.7702613, -8.9782515, -5.6990557, -2.6923499, 2.6910622
6: -12.4553852, -8.9797430, -12.5023689, -8.9515209, -2.4460073, 2.4702971
7: -5.5616732, -2.8372822, -5.7035522, -2.7524328, -2.6615791, 2.6079900
8: -1.1583521, 1.9369693, -1.2141044, 2.0058858, -3.1642380, 3.1510737
9: -6.5155468, -3.9069538, -6.5864849, -3.8332114, -2.5090504, 2.5557513

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 6250
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040882, upper bound: 1.7040487
time: 6.32 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040484
time: 5.75 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.2948923, -8.9763851, -13.2316990, -8.9936943, -3.6478314, 3.6133771
1: -7.2994795, -3.5100107, -7.2850552, -3.5163157, -3.4111848, 3.4402122
2: -10.0358582, -7.2656765, -9.9834375, -7.2925515, -2.7433066, 2.7177610
3: -12.5658693, -9.4337883, -12.5505409, -9.4759665, -2.7986760, 2.8445575
4: 5.3178964, 8.7107563, 5.3413076, 8.7030020, -3.3412762, 3.3230443
5: -8.9748392, -5.7131019, -8.9616661, -5.7460022, -2.7181726, 2.7271779
6: -12.4792290, -8.9532738, -12.4220304, -8.9664431, -2.4611301, 2.4381068
7: -5.7002420, -2.7634637, -5.6807117, -2.7950120, -2.7322316, 2.7598770
8: -1.1997190, 2.0014510, -1.1593864, 1.9811065, -3.1808255, 3.1608374
9: -6.5857782, -3.8356547, -6.5759768, -3.8428679, -2.6427574, 2.6152344

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040489, upper bound: 1.7298008
time: 4.88 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040511, upper bound: 1.7405269
time: 6.75 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.3209581, -8.9732561, -13.3209667, -8.9732351, -3.7158813, 3.6583123
1: -7.3048439, -3.5086877, -7.3048630, -3.5086856, -3.4297061, 3.4636393
2: -10.0570183, -7.2594490, -10.0570211, -7.2594447, -2.7975736, 2.7975721
3: -12.5703068, -9.4160843, -12.5703163, -9.4160795, -2.8434253, 2.8962314
4: 5.3104267, 8.7126923, 5.3104191, 8.7127075, -3.3596072, 3.3551841
5: -8.9787130, -5.6990013, -8.9787178, -5.6989918, -2.7624092, 2.7591748
6: -12.5030413, -8.9513454, -12.5030460, -8.9509678, -2.5010762, 2.4610620
7: -5.7038751, -2.7505379, -5.7039022, -2.7505326, -2.7540407, 2.8076546
8: -1.2158689, 2.0059829, -1.2158709, 2.0059891, -3.2218580, 3.2218537
9: -6.5884991, -3.8328958, -6.5885043, -3.8328838, -2.6619701, 2.6356726

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298008
time: 5.12 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040485, upper bound: 1.7298014
time: 5.89 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.64 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 25.64
Output dim: 4, lower bound: -1.7040864, upper bound: 1.7040507
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 25.64
Output dim: 4, lower bound: -1.7040882, upper bound: 1.7040507
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.64
Output dim: 4, lower bound: -1.7040882, upper bound: 1.7040487
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.64
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040484
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 25.64
Output dim: 4, lower bound: -1.7040489, upper bound: 1.7298008
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 25.64
Output dim: 4, lower bound: -1.7040511, upper bound: 1.7405269
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 25.64
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298008
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 25.64
Output dim: 4, lower bound: -1.7040485, upper bound: 1.7298014

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -13.2182474, -9.1241589, -13.1549864, -9.1414566, -3.2073402, 3.1738126
1: -7.1810794, -3.5420969, -7.1668282, -3.5483451, -3.2366343, 3.2298760
2: -9.9921074, -7.3042269, -9.9397202, -7.3311028, -2.6610045, 2.6354933
3: -12.4973030, -9.4830341, -12.4820814, -9.5253382, -2.6290026, 2.6418297
4: 5.4152508, 8.5701513, 5.4385414, 8.5623474, -3.0911360, 3.0772386
5: -8.9352074, -5.7842555, -8.9218502, -5.8169274, -2.5871468, 2.6049945
6: -12.4316349, -8.9816055, -12.3745804, -8.9950542, -2.3789916, 2.3386512
7: -5.5581160, -2.8502276, -5.5388203, -2.8817828, -2.4954941, 2.5045271
8: -1.1422050, 1.9324846, -1.1019375, 1.9122753, -3.0544803, 3.0344222
9: -6.5129008, -3.9097018, -6.5032806, -3.9169111, -2.4525108, 2.4492230

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040882
time: 5.00 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040882, upper bound: 1.7040882
time: 5.22 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -13.2443447, -9.1210403, -13.2443399, -9.1210403, -3.2646470, 3.2186110
1: -7.1863823, -3.5408008, -7.1863809, -3.5407999, -3.2551794, 3.2533360
2: -10.0132523, -7.2979498, -10.0132484, -7.2979493, -2.7153029, 2.7152987
3: -12.5013924, -9.4653015, -12.5013933, -9.4653034, -2.6735430, 2.6937177
4: 5.4078045, 8.5721302, 5.4078064, 8.5721302, -3.1093574, 3.1093454
5: -8.9391518, -5.7702613, -8.9391508, -5.7702627, -2.6307430, 2.6367602
6: -12.4553852, -8.9797430, -12.4553823, -8.9797430, -2.4189897, 2.3616946
7: -5.5616732, -2.8372822, -5.5616713, -2.8372831, -2.5175991, 2.5433829
8: -1.1583521, 1.9369693, -1.1583488, 1.9369686, -3.0953207, 3.0953181
9: -6.5155468, -3.9069538, -6.5155468, -3.9069538, -2.4741230, 2.4695034

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040882
time: 4.92 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040862, upper bound: 1.7040882
time: 4.95 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.2915573, -8.9764233, -3.2826896, 3.4711902
1: -7.1668282, -3.5483451, -7.2949600, -3.5100625, -3.2704439, 3.3693986
2: -9.9397202, -7.3311028, -10.0357695, -7.2667637, -2.6729565, 2.7046666
3: -12.4820814, -9.5253382, -12.5657530, -9.4354820, -2.7253242, 2.6953943
4: 5.4385414, 8.5623474, 5.3185954, 8.7107086, -3.1466875, 3.1938219
5: -8.9218502, -5.8169274, -8.9743767, -5.7131543, -2.6471353, 2.6415052
6: -12.3745804, -8.9950542, -12.4785519, -8.9534531, -2.3656855, 2.4272926
7: -5.5388203, -2.8817828, -5.6999216, -2.7653596, -2.6104498, 2.5601957
8: -1.1019375, 1.9122753, -1.1979520, 2.0013564, -3.1032939, 3.1102273
9: -6.5032806, -3.9169111, -6.5837641, -3.8359704, -2.4884915, 2.5384126

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7297986, upper bound: 1.7040480
time: 4.78 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7297986, upper bound: 1.7040480
time: 4.88 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.2443399, -9.1210403, -13.3175774, -8.9732952, -3.3247337, 3.5388680
1: -7.1863809, -3.5407999, -7.3003254, -3.5087399, -3.2939234, 3.3859270
2: -10.0132484, -7.2979493, -10.0569267, -7.2605362, -2.7527122, 2.7589774
3: -12.5013933, -9.4653034, -12.5701885, -9.4177761, -2.7771025, 2.7378330
4: 5.4078064, 8.5721302, 5.3111296, 8.7126455, -3.1780744, 3.2121148
5: -8.9391508, -5.7702627, -8.9782515, -5.6990557, -2.6890559, 2.6850450
6: -12.4553823, -8.9797430, -12.5023689, -8.9515209, -2.3887134, 2.4702976
7: -5.5616713, -2.8372831, -5.7035522, -2.7524328, -2.6581054, 2.5820498
8: -1.1583488, 1.9369686, -1.2141044, 2.0058858, -3.1642346, 3.1510730
9: -6.5155468, -3.9069538, -6.5864849, -3.8332114, -2.5072532, 2.5598116

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7296477, upper bound: 1.6876566
time: 4.66 seconds

## Relational analysis of IS_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7298015, upper bound: 1.7040484
time: 5.05 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7298014, upper bound: 1.7040482
time: 4.80 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -13.2915573, -8.9764233, -13.1549864, -9.1414566, -3.4711905, 3.2826896
1: -7.2949600, -3.5100625, -7.1668282, -3.5483451, -3.3693981, 3.2704434
2: -10.0357695, -7.2667637, -9.9397202, -7.3311028, -2.7046666, 2.6729565
3: -12.5657530, -9.4354820, -12.4820814, -9.5253382, -2.6953943, 2.7253244
4: 5.3185954, 8.7107086, 5.4385414, 8.5623474, -3.1938224, 3.1466880
5: -8.9743767, -5.7131543, -8.9218502, -5.8169274, -2.6415052, 2.6471355
6: -12.4785519, -8.9534531, -12.3745804, -8.9950542, -2.4272923, 2.3656855
7: -5.6999216, -2.7653596, -5.5388203, -2.8817828, -2.5601957, 2.6104500
8: -1.1979520, 2.0013564, -1.1019375, 1.9122753, -3.1102273, 3.1032939
9: -6.5837641, -3.8359704, -6.5032806, -3.9169111, -2.5384126, 2.4884915

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040511, upper bound: 1.7297985
time: 6.70 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040511, upper bound: 1.7297984
time: 6.45 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -13.2948923, -8.9763851, -13.2316914, -8.9937172, -3.6478262, 3.6244857
1: -7.2994795, -3.5100107, -7.2850342, -3.5163171, -3.4111748, 3.4046988
2: -10.0358582, -7.2656765, -9.9834318, -7.2925563, -2.7433019, 2.7177553
3: -12.5658693, -9.4337883, -12.5505304, -9.4759731, -2.8317862, 2.8445415
4: 5.3178964, 8.7107563, 5.3413177, 8.7029877, -3.3368559, 3.3230343
5: -8.9748392, -5.7131019, -8.9616632, -5.7460165, -2.7088523, 2.7271659
6: -12.4792290, -8.9532738, -12.4220238, -8.9667816, -2.4787540, 2.4380968
7: -5.7002420, -2.7634637, -5.6806841, -2.7950184, -2.7322264, 2.7377503
8: -1.1997190, 2.0014510, -1.1593797, 1.9811001, -3.1808190, 3.1608307
9: -6.5857782, -3.8356547, -6.5759706, -3.8428798, -2.6427555, 2.6393287

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040511, upper bound: 1.7405269
time: 6.99 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040490, upper bound: 1.7298009
time: 8.47 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -13.3175774, -8.9732952, -13.2443399, -9.1210403, -3.5388680, 3.3247337
1: -7.3003254, -3.5087399, -7.1863809, -3.5407999, -3.3859267, 3.2939243
2: -10.0569267, -7.2605362, -10.0132484, -7.2979493, -2.7589774, 2.7527122
3: -12.5701885, -9.4177761, -12.5013933, -9.4653034, -2.7378330, 2.7771022
4: 5.3111296, 8.7126455, 5.4078064, 8.5721302, -3.2121143, 3.1780744
5: -8.9782515, -5.6990557, -8.9391508, -5.7702627, -2.6850452, 2.6890562
6: -12.5023689, -8.9515209, -12.4553823, -8.9797430, -2.4702978, 2.3887129
7: -5.7035522, -2.7524328, -5.5616713, -2.8372831, -2.5820498, 2.6581051
8: -1.2141044, 2.0058858, -1.1583488, 1.9369686, -3.1510730, 3.1642346
9: -6.5864849, -3.8332114, -6.5155468, -3.9069538, -2.5598116, 2.5072532

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6876569, upper bound: 1.7296447
time: 4.75 seconds

## Relational analysis of IS_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7297983
time: 5.54 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298008
time: 4.92 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -13.3209581, -8.9732561, -13.3209572, -8.9732580, -3.7158489, 3.6694212
1: -7.3048439, -3.5086877, -7.3048396, -3.5086896, -3.4296975, 3.4278526
2: -10.0570183, -7.2594490, -10.0570154, -7.2594500, -2.7975683, 2.7975664
3: -12.5703068, -9.4160843, -12.5703049, -9.4160862, -2.8765383, 2.8962159
4: 5.3104267, 8.7126923, 5.3104296, 8.7126913, -3.3551846, 3.3551745
5: -8.9787130, -5.6990013, -8.9787140, -5.6990032, -2.7531481, 2.7591650
6: -12.5030413, -8.9513454, -12.5030384, -8.9513435, -2.5186996, 2.4610524
7: -5.7038751, -2.7505379, -5.7038755, -2.7505393, -2.7540350, 2.7803173
8: -1.2158689, 2.0059829, -1.2158673, 2.0059822, -3.2218511, 3.2218502
9: -6.5884991, -3.8328958, -6.5884991, -3.8328962, -2.6619678, 2.6582651

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6876568, upper bound: 1.7296492
time: 5.61 seconds

## Relational analysis of IS_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7405270
time: 5.23 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040485, upper bound: 1.7405267
time: 5.84 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 44.40 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040882
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7040882, upper bound: 1.7040882
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040882
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7040862, upper bound: 1.7040882
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7297986, upper bound: 1.7040480
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7297986, upper bound: 1.7040480
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7298015, upper bound: 1.7040484
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7298014, upper bound: 1.7040482
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7040511, upper bound: 1.7297985
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7040511, upper bound: 1.7297984
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7040511, upper bound: 1.7405269
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7040490, upper bound: 1.7298009
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7297983
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298008
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7405270
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 44.40
Output dim: 4, lower bound: -1.7040485, upper bound: 1.7405267

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.1549864, -9.1414566, -3.1524105, 3.1524107
1: -7.1668282, -3.5483451, -7.1668282, -3.5483451, -3.2215014, 3.2215009
2: -9.9397202, -7.3311028, -9.9397202, -7.3311028, -2.6086173, 2.6086173
3: -12.4820814, -9.5253382, -12.4820814, -9.5253382, -2.6069450, 2.6069450
4: 5.4385414, 8.5623474, 5.4385414, 8.5623474, -3.0685492, 3.0685487
5: -8.9218502, -5.8169274, -8.9218502, -5.8169274, -2.5732231, 2.5732231
6: -12.3745804, -8.9950542, -12.3745804, -8.9950542, -2.3255777, 2.3255780
7: -5.5388203, -2.8817828, -5.5388203, -2.8817828, -2.4744313, 2.4744315
8: -1.1019375, 1.9122753, -1.1019375, 1.9122753, -3.0142128, 3.0142128
9: -6.5032806, -3.9169111, -6.5032806, -3.9169111, -2.4384494, 2.4384499

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 317
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 1753
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 1845
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 2642
type: B, layer: 3, pos: 2642
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2860
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 654
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1396
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 611
type: B, layer: 3, pos: 611
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 2314
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1782
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850

Time for candidate selection: 12.87 seconds

### Candidate
type: B, layer: 3, pos: 1690

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6857529, upper bound: 1.6803444
time: 4.81 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6803878, upper bound: 1.6803872
time: 5.00 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -13.2442789, -9.1219788, -13.1549864, -9.1414566, -3.2262568, 3.1751947
1: -7.1862268, -3.5411444, -7.1668282, -3.5483451, -3.2423468, 3.2310233
2: -10.0130739, -7.3008842, -9.9397202, -7.3311028, -2.6819711, 2.6388359
3: -12.5013351, -9.4677496, -12.4820814, -9.5253382, -2.6330700, 2.6543934
4: 5.4083400, 8.5713844, 5.4385414, 8.5623474, -3.0978422, 3.0787282
5: -8.9391069, -5.7721491, -8.9218502, -5.8169274, -2.5902810, 2.6101370
6: -12.4543362, -8.9800262, -12.3745804, -8.9950542, -2.3878675, 2.3403037
7: -5.5606880, -2.8373220, -5.5388203, -2.8817828, -2.4981749, 2.5093629
8: -1.1582696, 1.9356942, -1.1019375, 1.9122753, -3.0705450, 3.0376318
9: -6.5151930, -3.9069624, -6.5032806, -3.9169111, -2.4548354, 2.4494643

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 317
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 1753
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 1676
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 2642
type: B, layer: 3, pos: 2642
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2369
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2860
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 1396
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 234
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 611
type: A, layer: 3, pos: 611
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2594
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 1782
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850

Time for candidate selection: 12.88 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6803474, upper bound: 1.6857500
time: 5.00 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6803878, upper bound: 1.6803873
time: 5.75 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.2442789, -9.1219788, -3.1751947, 3.2262571
1: -7.1668282, -3.5483451, -7.1862268, -3.5411444, -3.2310238, 3.2423463
2: -9.9397202, -7.3311028, -10.0130739, -7.3008842, -2.6388359, 2.6819711
3: -12.4820814, -9.5253382, -12.5013351, -9.4677496, -2.6543932, 2.6330702
4: 5.4385414, 8.5623474, 5.4083400, 8.5713844, -3.0787282, 3.0978422
5: -8.9218502, -5.8169274, -8.9391069, -5.7721491, -2.6101370, 2.5902820
6: -12.3745804, -8.9950542, -12.4543362, -8.9800262, -2.3403034, 2.3878675
7: -5.5388203, -2.8817828, -5.5606880, -2.8373220, -2.5093627, 2.4981747
8: -1.1019375, 1.9122753, -1.1582696, 1.9356942, -3.0376318, 3.0705450
9: -6.5032806, -3.9169111, -6.5151930, -3.9069624, -2.4494643, 2.4548354

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 1395
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 317
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1451
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 709
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 765
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 2123
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 1384
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 2642
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2369
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 654
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 1396
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 234
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 611
type: B, layer: 3, pos: 611
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 2594
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 2314
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1782
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: B, layer: 3, pos: 1165
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850

Time for candidate selection: 13.18 seconds

### Candidate
type: B, layer: 3, pos: 1690

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6857524, upper bound: 1.6803441
time: 4.71 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6803873, upper bound: 1.6803868
time: 4.61 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -13.2443399, -9.1210403, -13.2443399, -9.1210403, -3.2186103, 3.2186100
1: -7.1863809, -3.5407999, -7.1863809, -3.5407999, -3.2551775, 3.2551775
2: -10.0132484, -7.2979493, -10.0132484, -7.2979493, -2.7152991, 2.7152991
3: -12.5013933, -9.4653034, -12.5013933, -9.4653034, -2.6735420, 2.6735420
4: 5.4078064, 8.5721302, 5.4078064, 8.5721302, -3.1093545, 3.1093550
5: -8.9391508, -5.7702627, -8.9391508, -5.7702627, -2.6307421, 2.6307423
6: -12.4553823, -8.9797430, -12.4553823, -8.9797430, -2.3616948, 2.3616946
7: -5.5616713, -2.8372831, -5.5616713, -2.8372831, -2.5175986, 2.5175986
8: -1.1583488, 1.9369686, -1.1583488, 1.9369686, -3.0953174, 3.0953174
9: -6.5155468, -3.9069538, -6.5155468, -3.9069538, -2.4741225, 2.4741225

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 317
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 765
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 1676
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 2642
type: B, layer: 3, pos: 2642
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 2369
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 654
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1396
type: B, layer: 3, pos: 1396
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 611
type: B, layer: 3, pos: 611
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2594
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: B, layer: 3, pos: 1165
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850

Time for candidate selection: 13.12 seconds

### Candidate
type: B, layer: 3, pos: 1690

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6857524, upper bound: 1.6803439
time: 5.30 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6803873, upper bound: 1.6803872
time: 5.10 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.2284451, -8.9937534, -3.2613235, 3.4260557
1: -7.1668282, -3.5483451, -7.2805090, -3.5163677, -3.2620201, 3.3544555
2: -9.9397202, -7.3311028, -9.9833384, -7.2936392, -2.6460810, 2.6522355
3: -12.4820814, -9.5253382, -12.5504169, -9.4776659, -2.6905494, 2.6736796
4: 5.4385414, 8.5623474, 5.3420167, 8.7029409, -3.1379848, 3.1713448
5: -8.9218502, -5.8169274, -8.9611988, -5.7460699, -2.6284120, 2.6276827
6: -12.3745804, -8.9950542, -12.4213486, -8.9669552, -2.3526316, 2.3765979
7: -5.5388203, -2.8817828, -5.6803694, -2.7969165, -2.5923252, 2.5389001
8: -1.1019375, 1.9122753, -1.1576118, 1.9810028, -3.0829403, 3.0698872
9: -6.5032806, -3.9169111, -6.5739584, -3.8432016, -2.4777613, 2.5242677

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7296477, upper bound: 1.6876568
time: 5.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1704
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1199
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 1451
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 765
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1676
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2642
type: B, layer: 3, pos: 2572
type: B, layer: 3, pos: 2642
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 2860
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 234
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 611
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 1782
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850

Time for candidate selection: 18.14 seconds

### Candidate
type: B, layer: 3, pos: 1690

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7113719, upper bound: 1.6803074
time: 4.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7064126, upper bound: 1.6803503
time: 4.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.3175039, -8.9742393, -3.2752748, 3.4753299
1: -7.1668282, -3.5483451, -7.3001680, -3.5090826, -3.2716122, 3.3742805
2: -9.9397202, -7.3311028, -10.0567513, -7.2634792, -2.6762409, 2.7256484
3: -12.4820814, -9.5253382, -12.5701313, -9.4202452, -2.7337976, 2.6940372
4: 5.4385414, 8.5623474, 5.3116760, 8.7119408, -3.1474485, 3.2004845
5: -8.9218502, -5.8169274, -8.9782104, -5.7009907, -2.6506302, 2.6445847
6: -12.3745804, -8.9950542, -12.5013199, -8.9517994, -2.3673244, 2.4304342
7: -5.5388203, -2.8817828, -5.7025461, -2.7524741, -2.6140332, 2.5572224
8: -1.1019375, 1.9122753, -1.2140174, 2.0045972, -3.1065347, 3.1262927
9: -6.5032806, -3.9169111, -6.5861044, -3.8332214, -2.4882455, 2.5408192

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7296477, upper bound: 1.6876569
time: 4.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 1199
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 1451
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 765
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 2564
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1676
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 2642
type: B, layer: 3, pos: 2572
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 654
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 1396
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 234
type: B, layer: 3, pos: 611
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 2594
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: B, layer: 3, pos: 1165
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850

Time for candidate selection: 17.35 seconds

### Candidate
type: B, layer: 3, pos: 1690

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7113719, upper bound: 1.6803073
time: 4.91 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7064126, upper bound: 1.6803504
time: 4.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.2442789, -9.1219788, -13.2284451, -8.9937534, -3.2909923, 3.4489217
1: -7.1862268, -3.5411444, -7.2805090, -3.5163677, -3.2828655, 3.3620462
2: -10.0130739, -7.3008842, -9.9833384, -7.2936392, -2.7194347, 2.6824541
3: -12.5013351, -9.4677496, -12.5504169, -9.4776659, -2.7166452, 2.6977532
4: 5.4083400, 8.5713844, 5.3420167, 8.7029409, -3.1665654, 3.1814523
5: -8.9391069, -5.7721491, -8.9611988, -5.7460699, -2.6422288, 2.6594205
6: -12.4543362, -8.9800262, -12.4213486, -8.9669552, -2.4154677, 2.3913238
7: -5.5606880, -2.8373220, -5.6803694, -2.7969165, -2.6124592, 2.5587425
8: -1.1582696, 1.9356942, -1.1576118, 1.9810028, -3.1392725, 3.0933061
9: -6.5151930, -3.9069624, -6.5739584, -3.8432016, -2.4925466, 2.5352731

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7296477, upper bound: 1.6876564
time: 5.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1704
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 317
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2488
type: B, layer: 3, pos: 2333
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 765
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 1753
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 1676
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2572
type: B, layer: 3, pos: 2642
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2369
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 611
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1432
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 611
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 1782
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850

Time for candidate selection: 18.46 seconds

### Candidate
type: B, layer: 3, pos: 1690

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7113691, upper bound: 1.6803066
time: 5.15 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7064126, upper bound: 1.6803499
time: 4.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.2443399, -9.1210403, -13.3175745, -8.9732962, -3.3247333, 3.4925091
1: -7.1863809, -3.5407999, -7.3003244, -3.5087399, -3.2957664, 3.3859262
2: -10.0132484, -7.2979493, -10.0569267, -7.2605367, -2.7527118, 2.7589774
3: -12.5013933, -9.4653034, -12.5701885, -9.4177771, -2.7569027, 2.7378318
4: 5.4078064, 8.5721302, 5.3111305, 8.7126436, -3.1774902, 3.2115202
5: -8.9391508, -5.7702627, -8.9782515, -5.6990600, -2.6837149, 2.6850436
6: -12.4553823, -8.9797430, -12.5023670, -8.9515228, -2.3887124, 2.4126511
7: -5.5616713, -2.8372831, -5.7035513, -2.7524340, -2.6347854, 2.5795889
8: -1.1583488, 1.9369686, -1.2141023, 2.0058866, -3.1642354, 3.1510708
9: -6.5155468, -3.9069538, -6.5864844, -3.8332124, -2.5112128, 2.5598111

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7296477, upper bound: 1.6876566
time: 4.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1704
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1199
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 1451
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2488
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2333
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 765
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1676
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 2642
type: B, layer: 3, pos: 2642
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2572
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1396
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 611
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 1782
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 1165
type: A, layer: 3, pos: 1165
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850

Time for candidate selection: 17.72 seconds

### Candidate
type: B, layer: 3, pos: 1690

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7113691, upper bound: 1.6803067
time: 5.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7064119, upper bound: 1.6803498
time: 5.53 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 43.06 seconds
IS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.6857529, upper bound: 1.6803444
IS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.6803878, upper bound: 1.6803872
IS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.6803474, upper bound: 1.6857500
IS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.6803878, upper bound: 1.6803873
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.6857524, upper bound: 1.6803441
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.6803873, upper bound: 1.6803868
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.6857524, upper bound: 1.6803439
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.6803873, upper bound: 1.6803872
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.7113719, upper bound: 1.6803074
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.7064126, upper bound: 1.6803503
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.7113719, upper bound: 1.6803073
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.7064126, upper bound: 1.6803504
IS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.7113691, upper bound: 1.6803066
IS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.7064126, upper bound: 1.6803499
IS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.7113691, upper bound: 1.6803067
IS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 43.06
Output dim: 4, lower bound: -1.7064119, upper bound: 1.6803498
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 43.06
Output dim: 4, lower bound: -1.7040511, upper bound: 1.7297985
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 43.06
Output dim: 4, lower bound: -1.7040511, upper bound: 1.7297984
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 43.06
Output dim: 4, lower bound: -1.7040511, upper bound: 1.7405269
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 43.06
Output dim: 4, lower bound: -1.7040490, upper bound: 1.7298009
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 43.06
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7297983
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 43.06
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298008
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 43.06
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7405270
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 43.06
Output dim: 4, lower bound: -1.7040485, upper bound: 1.7405267
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=3.359607696533203
rel_dist={4: [-1.7405468414083058, 1.7405446936771147]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 6250
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016601
time: 4.76 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115494, upper bound: 1.6115539
time: 5.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.46 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 10.46
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016601
IS_B2, status: Status.UNKNOWN, split count: 1, time: 10.46
Output dim: 4, lower bound: -1.6115494, upper bound: 1.6115539

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -13.2960911, -9.0434008, -13.2443447, -9.1210403, -3.4012184, 3.2316449
1: -7.2447996, -3.5148530, -7.1863823, -3.5408008, -3.2778378, 3.2092566
2: -10.0359745, -7.2747850, -10.0132523, -7.2979498, -2.7380247, 2.7384672
3: -12.5336781, -9.4353437, -12.5013924, -9.4653015, -2.6575127, 2.6954238
4: 5.3407826, 8.6435699, 5.4078045, 8.5721302, -3.1224809, 3.1013339
5: -8.9638901, -5.7333999, -8.9391518, -5.7702613, -2.6041975, 2.5981915
6: -12.4801884, -8.9622803, -12.4553852, -8.9797430, -2.3328352, 2.3313746
7: -5.6334696, -2.7762892, -5.5616732, -2.8372822, -2.5207429, 2.5753510
8: -1.1986203, 1.9725804, -1.1583521, 1.9369693, -3.1355896, 3.1309326
9: -6.5652227, -3.8694601, -6.5155468, -3.9069538, -2.4883509, 2.4450283

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 6250
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823061, upper bound: 1.5823082
time: 5.32 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823083, upper bound: 1.6016601
time: 5.20 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -13.3209705, -8.9732380, -13.3209581, -8.9732561, -3.5866737, 3.5977509
1: -7.3048601, -3.5086854, -7.3048439, -3.5086877, -3.3905711, 3.3502526
2: -10.0570221, -7.2594457, -10.0570183, -7.2594490, -2.7975731, 2.7975726
3: -12.5703163, -9.4160786, -12.5703068, -9.4160843, -2.8214235, 2.7884672
4: 5.3104210, 8.7127037, 5.3104267, 8.7126923, -3.2933292, 3.2979622
5: -8.9787188, -5.6989913, -8.9787130, -5.6990013, -2.6828933, 2.6940877
6: -12.5030479, -8.9510479, -12.5030413, -8.9513454, -2.4086280, 2.3910868
7: -5.7038965, -2.7505322, -5.7038751, -2.7505379, -2.7433290, 2.7138252
8: -1.2158730, 2.0059884, -1.2158689, 2.0059829, -3.2218559, 3.2218573
9: -6.5885034, -3.8328872, -6.5884991, -3.8328958, -2.5779166, 2.6015456

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 495

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115487, upper bound: 1.6115539
time: 5.41 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115489, upper bound: 1.6115534
time: 5.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.74 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 25.74
Output dim: 4, lower bound: -1.5823061, upper bound: 1.5823082
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 25.74
Output dim: 4, lower bound: -1.5823083, upper bound: 1.6016601
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 25.74
Output dim: 4, lower bound: -1.6115487, upper bound: 1.6115539
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 25.74
Output dim: 4, lower bound: -1.6115489, upper bound: 1.6115534

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -13.2443447, -9.1210403, -13.2443447, -9.1210403, -3.1453581, 3.1453581
1: -7.1863823, -3.5408008, -7.1863823, -3.5408008, -3.1798301, 3.1798296
2: -10.0132523, -7.2979498, -10.0132523, -7.2979498, -2.7153025, 2.7153025
3: -12.5013924, -9.4653015, -12.5013924, -9.4653015, -2.6198578, 2.6198578
4: 5.4078045, 8.5721302, 5.4078045, 8.5721302, -3.0476031, 3.0476041
5: -8.9391518, -5.7702613, -8.9391518, -5.7702613, -2.5618548, 2.5618546
6: -12.4553852, -8.9797430, -12.4553852, -8.9797430, -2.3097830, 2.3097830
7: -5.5616732, -2.8372822, -5.5616732, -2.8372822, -2.4739883, 2.4739885
8: -1.1583521, 1.9369693, -1.1583521, 1.9369693, -3.0953214, 3.0953214
9: -6.5155468, -3.9069538, -6.5155468, -3.9069538, -2.4142613, 2.4142609

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823078, upper bound: 1.5823530
time: 5.04 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823078, upper bound: 1.5823526
time: 5.15 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -13.3151913, -8.9740067, -13.2443447, -9.1210403, -3.4196463, 3.2457230
1: -7.2944493, -3.5088406, -7.1863823, -3.5408008, -3.3048553, 3.2199392
2: -10.0566397, -7.2625914, -10.0132523, -7.2979498, -2.7586899, 2.7506609
3: -12.5696917, -9.4197979, -12.5013924, -9.4653015, -2.6809356, 2.7022214
4: 5.3136606, 8.7125721, 5.4078045, 8.5721302, -3.1482019, 3.1098802
5: -8.9769831, -5.6994257, -8.9391518, -5.7702613, -2.6149430, 2.6135578
6: -12.5007944, -8.9518213, -12.4553852, -8.9797430, -2.3590918, 2.3365200
7: -5.7031474, -2.7565615, -5.5616732, -2.8372822, -2.5307374, 2.5936902
8: -1.2118325, 2.0057650, -1.1583521, 1.9369693, -3.1488018, 3.1641171
9: -6.5837078, -3.8341618, -6.5155468, -3.9069538, -2.4971905, 2.4487901

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823061, upper bound: 1.6016596
time: 5.09 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823056, upper bound: 1.6016596
time: 5.22 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -13.2316990, -8.9937000, -13.2881813, -8.9773293, -3.4926672, 3.5227907
1: -7.2850504, -3.5163147, -7.2980933, -3.5104015, -3.3664913, 3.3322020
2: -9.9834356, -7.2925520, -10.0303936, -7.2677031, -2.7157326, 2.7378416
3: -12.5505381, -9.4759684, -12.5648270, -9.4385090, -2.7613811, 2.7211406
4: 5.3413105, 8.7029972, 5.3199091, 8.7101555, -3.2604747, 3.2775838
5: -8.9616661, -5.7460055, -8.9738293, -5.7168784, -2.6470108, 2.6427863
6: -12.4220314, -8.9665146, -12.4730406, -8.9538136, -2.3275290, 2.3445024
7: -5.6807060, -2.7950139, -5.6991549, -2.7667899, -2.6906943, 2.6644163
8: -1.1593831, 1.9811068, -1.1955547, 2.0000935, -3.1594765, 3.1766615
9: -6.5759749, -3.8428702, -6.5850220, -3.8363581, -2.5564380, 2.5830152

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 495

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115486, upper bound: 1.6115520
time: 7.33 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115486, upper bound: 1.6115534
time: 5.41 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -13.3209648, -8.9732399, -13.3209581, -8.9732561, -3.5343747, 3.5963371
1: -7.3048582, -3.5086865, -7.3048439, -3.5086877, -3.3905692, 3.3520231
2: -10.0570202, -7.2594447, -10.0570183, -7.2594490, -2.7975712, 2.7975736
3: -12.5703163, -9.4160795, -12.5703068, -9.4160843, -2.8201261, 2.7646503
4: 5.3104219, 8.7127037, 5.3104267, 8.7126923, -3.2933273, 3.2973986
5: -8.9787169, -5.6989927, -8.9787130, -5.6990013, -2.6828933, 2.6877682
6: -12.5030441, -8.9510489, -12.5030413, -8.9513454, -2.3477793, 2.3910861
7: -5.7038965, -2.7505326, -5.7038751, -2.7505379, -2.7409422, 2.6861830
8: -1.2158701, 2.0059886, -1.2158689, 2.0059829, -3.2218530, 3.2218575
9: -6.5885019, -3.8328876, -6.5884991, -3.8328958, -2.5776367, 2.6026559

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5857451, upper bound: 1.6046486
time: 4.99 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115476, upper bound: 1.6115521
time: 6.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.84 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 4, lower bound: -1.5823078, upper bound: 1.5823530
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 4, lower bound: -1.5823078, upper bound: 1.5823526
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 4, lower bound: -1.5823061, upper bound: 1.6016596
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 4, lower bound: -1.5823056, upper bound: 1.6016596
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 4, lower bound: -1.6115486, upper bound: 1.6115520
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 4, lower bound: -1.6115486, upper bound: 1.6115534
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 4, lower bound: -1.5857451, upper bound: 1.6046486
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 4, lower bound: -1.6115476, upper bound: 1.6115521

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.2115250, -9.1251011, -3.0518804, 3.0779855
1: -7.1668282, -3.5483451, -7.1797142, -3.5424833, -3.1557193, 3.1617208
2: -9.9397202, -7.3311028, -9.9866447, -7.3062539, -2.6334662, 2.6555419
3: -12.4820814, -9.5253382, -12.4963064, -9.4877596, -2.5596256, 2.5522811
4: 5.4385414, 8.5623474, 5.4172583, 8.5695353, -3.0147820, 3.0274529
5: -8.9218502, -5.8169274, -8.9341793, -5.7880020, -2.5262527, 2.5111778
6: -12.3745804, -8.9950542, -12.4254637, -8.9821262, -2.2289438, 2.2631419
7: -5.5388203, -2.8817828, -5.5570507, -2.8535593, -2.4314785, 2.4248126
8: -1.1019375, 1.9122753, -1.1380415, 1.9311433, -3.0330808, 3.0503168
9: -6.5032806, -3.9169111, -6.5121675, -3.9104023, -2.3929181, 2.3957481

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823526, upper bound: 1.5823526
time: 5.05 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823526, upper bound: 1.5823526
time: 4.94 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -13.2443399, -9.1210403, -13.2443447, -9.1210403, -3.0934448, 3.1439369
1: -7.1863809, -3.5407999, -7.1863823, -3.5408008, -3.1798282, 3.1815991
2: -10.0132484, -7.2979493, -10.0132523, -7.2979498, -2.6905255, 2.7153029
3: -12.5013933, -9.4653034, -12.5013924, -9.4653015, -2.6186080, 2.5955932
4: 5.4078064, 8.5721302, 5.4078045, 8.5721302, -3.0476031, 3.0476055
5: -8.9391508, -5.7702627, -8.9391518, -5.7702613, -2.5618539, 2.5555358
6: -12.4553823, -8.9797430, -12.4553852, -8.9797430, -2.2492843, 2.3097825
7: -5.5616713, -2.8372831, -5.5616732, -2.8372822, -2.4739883, 2.4468448
8: -1.1583488, 1.9369686, -1.1583521, 1.9369693, -3.0953181, 3.0953207
9: -6.5155468, -3.9069538, -6.5155468, -3.9069538, -2.4139543, 2.4177799

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823526, upper bound: 1.5823526
time: 5.29 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823526, upper bound: 1.5823526
time: 5.28 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.2824707, -8.9780769, -13.1549864, -9.1414566, -3.3441324, 3.1522751
1: -7.2876921, -3.5105543, -7.1668282, -3.5483451, -3.2867517, 3.1958032
2: -10.0300150, -7.2708406, -9.9397202, -7.3311028, -2.6926894, 2.6688795
3: -12.5642195, -9.4422235, -12.4820814, -9.5253382, -2.6137578, 2.6421225
4: 5.3231411, 8.7100344, 5.4385414, 8.5623474, -3.1277709, 3.0770080
5: -8.9720974, -5.7173042, -8.9218502, -5.8169274, -2.5643377, 2.5655422
6: -12.4707890, -8.9542923, -12.3745804, -8.9950542, -2.3111706, 2.2557011
7: -5.6984339, -2.7728162, -5.5388203, -2.8817828, -2.4816372, 2.5380812
8: -1.1915166, 1.9998784, -1.1019375, 1.9122753, -3.1037920, 3.1018159
9: -6.5802326, -3.8376260, -6.5032806, -3.9169111, -2.4786286, 2.4274843

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823078, upper bound: 1.6016568
time: 5.38 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823078, upper bound: 1.6016568
time: 5.22 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.3151913, -8.9740067, -13.2443399, -9.1210403, -3.4182253, 3.1910317
1: -7.2944493, -3.5088406, -7.1863809, -3.5407999, -3.3045926, 3.2199373
2: -10.0566397, -7.2625914, -10.0132484, -7.2979493, -2.7586904, 2.7396176
3: -12.5696917, -9.4197979, -12.5013933, -9.4653034, -2.6549067, 2.7009721
4: 5.3136606, 8.7125721, 5.4078064, 8.5721302, -3.1475706, 3.1091046
5: -8.9769831, -5.6994257, -8.9391508, -5.7702627, -2.6086264, 2.6102641
6: -12.5007944, -8.9518213, -12.4553823, -8.9797430, -2.3590918, 2.2760222
7: -5.7031474, -2.7565615, -5.5616713, -2.8372831, -2.5034347, 2.5882087
8: -1.2118325, 2.0057650, -1.1583488, 1.9369686, -3.1488011, 3.1641138
9: -6.5837078, -3.8341618, -6.5155468, -3.9069538, -2.5004606, 2.4469929

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5598131, upper bound: 1.6014881
time: 4.81 seconds

## Relational analysis of IS_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823078, upper bound: 1.6016547
time: 7.25 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823078, upper bound: 1.6016546
time: 7.37 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.2316990, -8.9937000, -13.2316914, -8.9937172, -3.4724941, 3.4835715
1: -7.2850504, -3.5163147, -7.2850342, -3.5163171, -3.3587327, 3.3187003
2: -9.9834356, -7.2925520, -9.9834318, -7.2925563, -2.6908793, 2.6908798
3: -12.5505381, -9.4759684, -12.5505304, -9.4759731, -2.7336779, 2.7007203
4: 5.3413105, 8.7029972, 5.3413177, 8.7029877, -3.2525263, 3.2569599
5: -8.9616661, -5.7460055, -8.9616632, -5.7460165, -2.6186080, 2.6300108
6: -12.4220314, -8.9665146, -12.4220238, -8.9667816, -2.3149695, 2.2974269
7: -5.6807060, -2.7950139, -5.6806841, -2.7950184, -2.6738992, 2.6443958
8: -1.1593831, 1.9811068, -1.1593797, 1.9811001, -3.1404831, 3.1404865
9: -6.5759749, -3.8428702, -6.5759706, -3.8428798, -2.5464387, 2.5700579

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 495

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5857451, upper bound: 1.6046490
time: 4.78 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115473, upper bound: 1.6115526
time: 5.36 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.2316990, -8.9937000, -13.3208618, -8.9745779, -3.4948301, 3.5279889
1: -7.2850504, -3.5163147, -7.3046260, -3.5091655, -3.3679895, 3.3390374
2: -9.9834356, -7.2925520, -10.0567713, -7.2635632, -2.7198725, 2.7642193
3: -12.5505381, -9.4759684, -12.5702248, -9.4195328, -2.7737947, 2.7264979
4: 5.3413105, 8.7029972, 5.3111916, 8.7117119, -3.2623496, 3.2853699
5: -8.9616661, -5.7460055, -8.9786568, -5.7017021, -2.6528771, 2.6466663
6: -12.4220314, -8.9665146, -12.5015764, -8.9517298, -2.3295860, 2.3568032
7: -5.6807060, -2.7950139, -5.7024689, -2.7505951, -2.6951952, 2.6678190
8: -1.1593831, 1.9811068, -1.2157483, 2.0041814, -3.1635644, 3.1968551
9: -6.5759749, -3.8428702, -6.5879669, -3.8329082, -2.5572090, 2.5844538

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 495

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5857452, upper bound: 1.6046491
time: 5.10 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115474, upper bound: 1.6115527
time: 6.01 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.3205423, -8.9737034, -13.3198242, -8.9742212, -3.5330830, 3.5940616
1: -7.3046322, -3.5087924, -7.3059473, -3.5089905, -3.3900733, 3.3499050
2: -10.0569420, -7.2597084, -10.0566912, -7.2602029, -2.7967391, 2.7969828
3: -12.5702448, -9.4163342, -12.5745239, -9.4167023, -2.8191371, 2.7686129
4: 5.3109550, 8.7124157, 5.3120399, 8.7121515, -3.2922602, 3.2950664
5: -8.9782944, -5.7001648, -8.9768791, -5.7011046, -2.6803803, 2.6848528
6: -12.5011883, -8.9511881, -12.4998322, -8.9507818, -2.3443141, 2.3877411
7: -5.7036972, -2.7507019, -5.7047634, -2.7509079, -2.7389898, 2.6868591
8: -1.2157342, 2.0058103, -1.2152550, 2.0056150, -3.2213492, 3.2210653
9: -6.5882535, -3.8329306, -6.5877085, -3.8327813, -2.5775504, 2.6015697

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5857452, upper bound: 1.5857454
time: 5.47 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5857452, upper bound: 1.6046488
time: 5.64 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.3209591, -8.9732428, -13.3209553, -8.9732628, -3.5341778, 3.5972221
1: -7.3048582, -3.5090592, -7.3048410, -3.5094151, -3.3922434, 3.3520122
2: -10.0570183, -7.2594471, -10.0570173, -7.2594519, -2.7975664, 2.7975702
3: -12.5703144, -9.4171219, -12.5703039, -9.4181166, -2.8255229, 2.7635112
4: 5.3104239, 8.7127018, 5.3104324, 8.7126904, -3.2930861, 3.2976677
5: -8.9787159, -5.6989999, -8.9787092, -5.6990123, -2.6831141, 2.6877561
6: -12.5030403, -8.9510508, -12.5030365, -8.9513435, -2.3477736, 2.3879747
7: -5.7038956, -2.7508726, -5.7038736, -2.7511988, -2.7395177, 2.6857090
8: -1.2158716, 2.0059857, -1.2158687, 2.0059786, -3.2218502, 3.2218544
9: -6.5885024, -3.8328862, -6.5884981, -3.8328977, -2.5774674, 2.6024935

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6046482, upper bound: 1.5857454
time: 5.16 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6046481, upper bound: 1.6115504
time: 5.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.30 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.5823526, upper bound: 1.5823526
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.5823526, upper bound: 1.5823526
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.5823526, upper bound: 1.5823526
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.5823526, upper bound: 1.5823526
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.5823078, upper bound: 1.6016568
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.5823078, upper bound: 1.6016568
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.5823078, upper bound: 1.6016547
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.5823078, upper bound: 1.6016546
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.5857451, upper bound: 1.6046490
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.6115473, upper bound: 1.6115526
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.5857452, upper bound: 1.6046491
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.6115474, upper bound: 1.6115527
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.5857452, upper bound: 1.5857454
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.5857452, upper bound: 1.6046488
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.6046482, upper bound: 1.5857454
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 4, lower bound: -1.6046481, upper bound: 1.6115504

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.1549864, -9.1414566, -3.0317011, 3.0317006
1: -7.1668282, -3.5483451, -7.1668282, -3.5483451, -3.1479921, 3.1479926
2: -9.9397202, -7.3311028, -9.9397202, -7.3311028, -2.6086173, 2.6086173
3: -12.4820814, -9.5253382, -12.4820814, -9.5253382, -2.5318356, 2.5318353
4: 5.4385414, 8.5623474, 5.4385414, 8.5623474, -3.0068054, 3.0068054
5: -8.9218502, -5.8169274, -8.9218502, -5.8169274, -2.4983168, 2.4983165
6: -12.3745804, -8.9950542, -12.3745804, -8.9950542, -2.2163706, 2.2163708
7: -5.5388203, -2.8817828, -5.5388203, -2.8817828, -2.4050362, 2.4050369
8: -1.1019375, 1.9122753, -1.1019375, 1.9122753, -3.0142128, 3.0142128
9: -6.5032806, -3.9169111, -6.5032806, -3.9169111, -2.3829002, 2.3829002

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 317
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 3105
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 2369
type: B, layer: 3, pos: 2369
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 2642
type: B, layer: 3, pos: 2642
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 234
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 611
type: B, layer: 3, pos: 611
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2594
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1782
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850

Time for candidate selection: 12.88 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5604199, upper bound: 1.5646277
time: 5.42 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5605463, upper bound: 1.5605462
time: 4.63 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.2442513, -9.1223488, -3.0540075, 3.1003737
1: -7.1668282, -3.5483451, -7.1861649, -3.5412786, -3.1572104, 3.1686234
2: -9.9397202, -7.3311028, -10.0130043, -7.3020515, -2.6376686, 2.6704612
3: -12.4820814, -9.5253382, -12.5013123, -9.4687214, -2.5746922, 2.5577481
4: 5.4385414, 8.5623474, 5.4085526, 8.5710869, -3.0166626, 3.0359006
5: -8.9218502, -5.8169274, -8.9390907, -5.7728987, -2.5319946, 2.5151269
6: -12.3745804, -8.9950542, -12.4539185, -8.9801397, -2.2310238, 2.2747283
7: -5.5388203, -2.8817828, -5.5602961, -2.8373384, -2.4374268, 2.4281747
8: -1.1019375, 1.9122753, -1.1582379, 1.9351881, -3.0371256, 3.0705132
9: -6.5032806, -3.9169111, -6.5150528, -3.9069672, -2.3936906, 2.3987699

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 1395
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 1199
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 709
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 2123
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 1384
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 2642
type: B, layer: 3, pos: 1676
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2642
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 2570
type: B, layer: 3, pos: 654
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 1396
type: B, layer: 3, pos: 1396
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 611
type: B, layer: 3, pos: 611
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2594
type: B, layer: 3, pos: 2594
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 1165
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850

Time for candidate selection: 13.25 seconds

### Candidate
type: B, layer: 3, pos: 1690

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5646250, upper bound: 1.5604196
time: 4.79 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5605462, upper bound: 1.5605463
time: 4.58 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -13.2442513, -9.1223488, -13.1549864, -9.1414566, -3.1003737, 3.0540073
1: -7.1861649, -3.5412786, -7.1668282, -3.5483451, -3.1686230, 3.1572104
2: -10.0130043, -7.3020515, -9.9397202, -7.3311028, -2.6704607, 2.6376686
3: -12.5013123, -9.4687214, -12.4820814, -9.5253382, -2.5577483, 2.5746920
4: 5.4085526, 8.5710869, 5.4385414, 8.5623474, -3.0359011, 3.0166631
5: -8.9390907, -5.7728987, -8.9218502, -5.8169274, -2.5151267, 2.5319948
6: -12.4539185, -8.9801397, -12.3745804, -8.9950542, -2.2747283, 2.2310238
7: -5.5602961, -2.8373384, -5.5388203, -2.8817828, -2.4281743, 2.4374270
8: -1.1582379, 1.9351881, -1.1019375, 1.9122753, -3.0705132, 3.0371256
9: -6.5150528, -3.9069672, -6.5032806, -3.9169111, -2.3987699, 2.3936911

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 1704
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1451
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 2369
type: B, layer: 3, pos: 2369
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2642
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 654
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 234
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 611
type: A, layer: 3, pos: 611
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1165
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850

Time for candidate selection: 12.87 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5604199, upper bound: 1.5646272
time: 5.06 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5605463, upper bound: 1.5605458
time: 4.50 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -13.2443399, -9.1210403, -13.2443399, -9.1210403, -3.0934443, 3.0934441
1: -7.1863809, -3.5407999, -7.1863809, -3.5407999, -3.1815977, 3.1815977
2: -10.0132484, -7.2979493, -10.0132484, -7.2979493, -2.6905251, 2.6905253
3: -12.5013933, -9.4653034, -12.5013933, -9.4653034, -2.5955925, 2.5955923
4: 5.4078064, 8.5721302, 5.4078064, 8.5721302, -3.0476041, 3.0476041
5: -8.9391508, -5.7702627, -8.9391508, -5.7702627, -2.5555353, 2.5555351
6: -12.4553823, -8.9797430, -12.4553823, -8.9797430, -2.2492852, 2.2492847
7: -5.5616713, -2.8372831, -5.5616713, -2.8372831, -2.4468436, 2.4468439
8: -1.1583488, 1.9369686, -1.1583488, 1.9369686, -3.0953174, 3.0953174
9: -6.5155468, -3.9069538, -6.5155468, -3.9069538, -2.4177794, 2.4177794

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 317
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 1753
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 2369
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 2642
type: B, layer: 3, pos: 2642
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2860
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1432
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 1396
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 611
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2594
type: B, layer: 3, pos: 2594
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1165
type: B, layer: 3, pos: 1165
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850

Time for candidate selection: 12.83 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5604222, upper bound: 1.5646272
time: 5.83 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5605463, upper bound: 1.5605460
time: 4.85 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.2260590, -8.9944620, -13.1549864, -9.1414566, -3.3054113, 3.1320915
1: -7.2746186, -3.5164707, -7.1668282, -3.5483451, -3.2731991, 3.1880322
2: -9.9830484, -7.2956896, -9.9397202, -7.3311028, -2.6519456, 2.6440306
3: -12.5499239, -9.4796906, -12.4820814, -9.5253382, -2.5936115, 2.6144176
4: 5.3445454, 8.7028666, 5.4385414, 8.5623474, -3.1071544, 3.0690234
5: -8.9599285, -5.7464428, -8.9218502, -5.8169274, -2.5515618, 2.5496876
6: -12.4197655, -8.9672508, -12.3745804, -8.9950542, -2.2653880, 2.2431455
7: -5.6799746, -2.8010478, -5.5388203, -2.8817828, -2.4616559, 2.5242410
8: -1.1553364, 1.9808817, -1.1019375, 1.9122753, -3.0676117, 3.0828192
9: -6.5711851, -3.8441548, -6.5032806, -3.9169111, -2.4657087, 2.4175062

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5598135, upper bound: 1.6014885
time: 5.29 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 317
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 1199
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1451
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 2123
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 2564
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 1384
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1845
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 2642
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 2570
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1432
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 234
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 611
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 1396
type: B, layer: 3, pos: 611
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2594
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1782
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850

Time for candidate selection: 18.43 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B1_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5603774, upper bound: 1.5837819
time: 5.20 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5605023, upper bound: 1.5798958
time: 4.88 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3150902, -8.9753246, -13.1549864, -9.1414566, -3.3494015, 3.1455531
1: -7.2942314, -3.5093179, -7.1668282, -3.5483451, -3.2928247, 3.1973209
2: -10.0563908, -7.2667065, -9.9397202, -7.3311028, -2.6972308, 2.6730137
3: -12.5696096, -9.4232502, -12.4820814, -9.5253382, -2.6137187, 2.6532474
4: 5.3144255, 8.7115898, 5.4385414, 8.5623474, -3.1355557, 3.0781076
5: -8.9769230, -5.7021275, -8.9218502, -5.8169274, -2.5682192, 2.5699296
6: -12.4993305, -8.9522085, -12.3745804, -8.9950542, -2.3151116, 2.2577634
7: -5.7017422, -2.7566206, -5.5388203, -2.8817828, -2.4793217, 2.5425835
8: -1.2117124, 2.0039647, -1.1019375, 1.9122753, -3.1239877, 3.1059022
9: -6.5831780, -3.8341756, -6.5032806, -3.9169111, -2.4817452, 2.4277310

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5598136, upper bound: 1.6014886
time: 4.69 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 317
type: B, layer: 3, pos: 1199
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1451
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 2123
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 1384
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 2642
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2334
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 2570
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 654
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 234
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1396
type: B, layer: 3, pos: 611
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1782
type: A, layer: 3, pos: 1165
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850

Time for candidate selection: 17.47 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B1_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5603774, upper bound: 1.5837819
time: 5.21 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5605021, upper bound: 1.5798960
time: 4.44 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.2260590, -8.9944620, -13.2442513, -9.1223488, -3.3278017, 3.1583633
1: -7.2746186, -3.5164707, -7.1861649, -3.5412786, -3.2804637, 3.2086635
2: -9.9830484, -7.2956896, -10.0130043, -7.3020515, -2.6809969, 2.7173147
3: -12.5499239, -9.4796906, -12.5013123, -9.4687214, -2.6148353, 2.6403012
4: 5.3445454, 8.7028666, 5.4085526, 8.5710869, -3.1163540, 3.0974081
5: -8.9599285, -5.7464428, -8.9390907, -5.7728987, -2.5799222, 2.5632415
6: -12.4197655, -8.9672508, -12.4539185, -8.9801397, -2.2800407, 2.3019547
7: -5.6799746, -2.8010478, -5.5602961, -2.8373384, -2.4799664, 2.5419140
8: -1.1553364, 1.9808817, -1.1582379, 1.9351881, -3.0905244, 3.1391196
9: -6.5711851, -3.8441548, -6.5150528, -3.9069672, -2.4764891, 2.4317312

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5598131, upper bound: 1.6014887
time: 5.14 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 317
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2333
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 765
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 2123
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 2564
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 1384
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1845
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 2642
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 2570
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 234
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 611
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 611
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2594
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 2314
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 1782
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 1165
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850

Time for candidate selection: 17.91 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B1_A2_B2_A1_A1

### Relational analysis result of IS_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5603769, upper bound: 1.5837794
time: 4.80 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2

### Relational analysis result of IS_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5605018, upper bound: 1.5798958
time: 4.69 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3151875, -8.9740076, -13.2443399, -9.1210403, -3.3674159, 3.1910312
1: -7.2944484, -3.5088401, -7.1863809, -3.5407999, -3.3045907, 3.2217073
2: -10.0566378, -7.2625899, -10.0132484, -7.2979493, -2.7354341, 2.7396173
3: -12.5696898, -9.4198027, -12.5013933, -9.4653034, -2.6549048, 2.6779304
4: 5.3136625, 8.7125702, 5.4078064, 8.5721302, -3.1467772, 3.1085103
5: -8.9769812, -5.6994276, -8.9391508, -5.7702627, -2.6086259, 2.6046214
6: -12.5007906, -8.9518204, -12.4553823, -8.9797430, -2.2982459, 2.2760217
7: -5.7031479, -2.7565625, -5.5616713, -2.8372831, -2.5009761, 2.5635252
8: -1.2118309, 2.0057642, -1.1583488, 1.9369686, -3.1487994, 3.1641130
9: -6.5837078, -3.8341632, -6.5155468, -3.9069538, -2.5004601, 2.4501572

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5598130, upper bound: 1.6014890
time: 4.88 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 317
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 1199
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1451
type: B, layer: 3, pos: 310
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 709
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 1753
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 2123
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 2564
type: B, layer: 3, pos: 2564
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 1384
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 1384
type: B, layer: 3, pos: 2369
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1845
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1676
type: B, layer: 3, pos: 2642
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 2570
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 654
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1432
type: B, layer: 3, pos: 1432
type: B, layer: 3, pos: 234
type: A, layer: 3, pos: 234
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 611
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 1396
type: B, layer: 3, pos: 611
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2594
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 2118
type: A, layer: 3, pos: 2118
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 1782
type: A, layer: 3, pos: 1165
type: B, layer: 3, pos: 1165
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 417
type: B, layer: 3, pos: 417
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850

Time for candidate selection: 17.56 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B1_A2_B2_A2_A1

### Relational analysis result of IS_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5603749, upper bound: 1.5837819
time: 6.67 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2

### Relational analysis result of IS_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5605019, upper bound: 1.5798959
time: 4.78 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -13.2312775, -8.9941607, -13.2305546, -8.9946795, -3.4711914, 3.4812922
1: -7.2848234, -3.5164182, -7.2861433, -3.5166163, -3.3582387, 3.3165817
2: -9.9833584, -7.2928162, -9.9831047, -7.2933102, -2.6900482, 2.6902885
3: -12.5504599, -9.4762192, -12.5547447, -9.4765911, -2.7326889, 2.7046831
4: 5.3418427, 8.7027092, 5.3429298, 8.7024441, -3.2514544, 3.2542586
5: -8.9612389, -5.7471762, -8.9598188, -5.7481189, -2.6160946, 2.6270912
6: -12.4201756, -8.9666452, -12.4188137, -8.9664268, -2.3115010, 2.2940810
7: -5.6805115, -2.7951798, -5.6815777, -2.7953858, -2.6727672, 2.6450777
8: -1.1592495, 1.9809299, -1.1587684, 1.9807363, -3.1399858, 3.1396983
9: -6.5757303, -3.8429141, -6.5751863, -3.8427639, -2.5463572, 2.5689702

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 1690
type: B, layer: 3, pos: 1145
type: A, layer: 3, pos: 1145
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1704
type: B, layer: 3, pos: 1704
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1199
type: B, layer: 3, pos: 1199
type: A, layer: 3, pos: 317
type: B, layer: 3, pos: 317
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 1746
type: B, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: B, layer: 3, pos: 1451
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 310
type: B, layer: 3, pos: 310
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2333
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2488
type: B, layer: 3, pos: 2488
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 709
type: B, layer: 3, pos: 709
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 1753
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2123
type: A, layer: 3, pos: 2123
type: B, layer: 3, pos: 2572
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2564
type: B, layer: 3, pos: 669
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1516
type: B, layer: 3, pos: 1516
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 669
type: B, layer: 3, pos: 1384
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 1845
type: B, layer: 3, pos: 2369
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1845
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2371
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 2642
type: B, layer: 3, pos: 2642
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 2371
type: B, layer: 3, pos: 2860
type: A, layer: 3, pos: 2860
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2334
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 2570
type: B, layer: 3, pos: 2570
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1396
type: B, layer: 3, pos: 1432
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1396
type: A, layer: 3, pos: 234
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 611
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 2594
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 2314
type: A, layer: 3, pos: 2314
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 1782
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 1782
type: B, layer: 3, pos: 431
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 431
type: B, layer: 3, pos: 417
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850

Time for candidate selection: 0.40 seconds

### Candidate
type: A, layer: 3, pos: 1690

## Relational analysis of IS_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5637691, upper bound: 1.5862748
time: 4.85 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5639024, upper bound: 1.5822711
time: 5.10 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -13.2316961, -8.9936991, -13.2316856, -8.9937220, -3.4722953, 3.4844558
1: -7.2850485, -3.5166883, -7.2850313, -3.5170465, -3.3604035, 3.3186917
2: -9.9834356, -7.2925549, -9.9834280, -7.2925596, -2.6908760, 2.6908731
3: -12.5505381, -9.4770069, -12.5505285, -9.4780045, -2.7390738, 2.6995821
4: 5.3413124, 8.7029972, 5.3413219, 8.7029848, -3.2522831, 3.2572279
5: -8.9616623, -5.7460117, -8.9616604, -5.7460260, -2.6188293, 2.6299984
6: -12.4220276, -8.9665165, -12.4220219, -8.9667835, -2.3149638, 2.2943146
7: -5.6807051, -2.7953498, -5.6806822, -2.7956779, -2.6754360, 2.6439228
8: -1.1593835, 1.9811039, -1.1593785, 1.9810958, -3.1404793, 3.1404824
9: -6.5759749, -3.8428721, -6.5759697, -3.8428817, -2.5462689, 2.5698967

Time for backsubstitution: 14.51 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=3.2979745864868164
rel_dist={4: [-1.6115673031162494, 1.6115673021877095]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2423.62 seconds
