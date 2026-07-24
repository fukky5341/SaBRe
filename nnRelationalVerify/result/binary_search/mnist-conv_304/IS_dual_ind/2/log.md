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
execution time: IAR + LP analysis = 15.37 + 34.46 = 49.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -2.5504220, upper bound: 2.5504211


# Binary Search by BASE starts (time budget: 3550.17 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.402289390563965
rel_dist={4: [-1.981378173882315, 1.9813780589154808]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=3.2979745864868164
rel_dist={4: [-1.6115652775740719, 1.611567303353243]}

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
Binary search time: 212.62 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3337.55 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0856247, upper bound: 2.0438436
time: 4.24 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0911799, upper bound: 2.0911775
time: 4.50 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.94 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.94
Output dim: 4, lower bound: -2.0856247, upper bound: 2.0438436
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.94
Output dim: 4, lower bound: -2.0911799, upper bound: 2.0911775

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.2443447, -9.1210403, -13.3082533, -9.0077085, -3.7532387, 3.8894961
1: -7.1863823, -3.5408008, -7.2750816, -3.5116777, -3.5092392, 3.6059470
2: -10.0132523, -7.2979498, -10.0466843, -7.2668419, -2.7464104, 2.7487345
3: -12.5013924, -9.4653015, -12.5522881, -9.4253349, -3.0022178, 2.9792039
4: 5.4078045, 8.5721302, 5.3250570, 8.6787567, -3.2709522, 3.2470732
5: -8.9391518, -5.7702613, -8.9715576, -5.7159948, -2.9185243, 2.9114060
6: -12.4553852, -8.9797430, -12.4917459, -8.9564810, -2.7732387, 2.7835300
7: -5.5616732, -2.8372822, -5.6693201, -2.7628076, -2.7988656, 2.8320379
8: -1.1583521, 1.9369693, -1.2075975, 1.9895597, -3.1479118, 3.1445668
9: -6.5155468, -3.9069538, -6.5773993, -3.8508630, -2.6646838, 2.6704454

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438416, upper bound: 2.0438418
time: 4.55 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438436, upper bound: 2.0438415
time: 5.09 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.3209581, -8.9732561, -13.3209705, -8.9732313, -4.0759096, 4.0634732
1: -7.3048439, -3.5086877, -7.3048649, -3.5086842, -3.6606903, 3.6828380
2: -10.0570183, -7.2594490, -10.0570240, -7.2594433, -2.7975750, 2.7975750
3: -12.5703068, -9.4160843, -12.5703182, -9.4160767, -3.0888529, 3.1258359
4: 5.3104267, 8.7126923, 5.3104191, 8.7127085, -3.4022818, 3.4022732
5: -8.9787130, -5.6990013, -8.9787197, -5.6989894, -2.9914393, 2.9880135
6: -12.5030413, -8.9513454, -12.5030499, -8.9509478, -2.8292618, 2.8489447
7: -5.7038751, -2.7505379, -5.7039032, -2.7505317, -2.9533434, 2.9533653
8: -1.2158689, 2.0059829, -1.2158751, 2.0059915, -3.2218604, 3.2218580
9: -6.5884991, -3.8328958, -6.5885048, -3.8328829, -2.7556162, 2.7556090

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438416, upper bound: 2.0856251
time: 4.64 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438416, upper bound: 2.0856246
time: 5.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.28 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.28
Output dim: 4, lower bound: -2.0438416, upper bound: 2.0438418
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.28
Output dim: 4, lower bound: -2.0438436, upper bound: 2.0438415
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.28
Output dim: 4, lower bound: -2.0438416, upper bound: 2.0856251
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.28
Output dim: 4, lower bound: -2.0438416, upper bound: 2.0856246

## BFS IS instance: IS_A1_B1

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

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438419
time: 4.72 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438415
time: 5.00 seconds

## BFS IS instance: IS_A1_B2

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

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438415
time: 5.60 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438455, upper bound: 2.0438411
time: 5.66 seconds

## BFS IS instance: IS_A2_B1

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

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438432, upper bound: 2.0856225
time: 4.72 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438411, upper bound: 2.0856221
time: 4.78 seconds

## BFS IS instance: IS_A2_B2

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

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438432, upper bound: 2.0911797
time: 6.70 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438412, upper bound: 2.0911792
time: 5.04 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.16 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.16
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438419
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.16
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438415
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.16
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438415
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.16
Output dim: 4, lower bound: -2.0438455, upper bound: 2.0438411
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.16
Output dim: 4, lower bound: -2.0438432, upper bound: 2.0856225
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.16
Output dim: 4, lower bound: -2.0438411, upper bound: 2.0856221
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.16
Output dim: 4, lower bound: -2.0438432, upper bound: 2.0911797
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.16
Output dim: 4, lower bound: -2.0438412, upper bound: 2.0911792

## BFS IS instance: IS_A1_B1_A1

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

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438458
time: 4.49 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438476, upper bound: 2.0438458
time: 4.79 seconds

## BFS IS instance: IS_A1_B1_A2

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

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438476, upper bound: 2.0438460
time: 4.78 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438457
time: 4.80 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.3063345, -8.9747963, -3.6708636, 3.8454046
1: -7.1668282, -3.5483451, -7.3018589, -3.5093508, -3.4924021, 3.6100931
2: -9.9397202, -7.3311028, -10.0451736, -7.2622795, -2.6774406, 2.7140708
3: -12.4820814, -9.5253382, -12.5678434, -9.4256077, -2.9626188, 2.9364853
4: 5.4385414, 8.5623474, 5.3144875, 8.7117653, -3.2732239, 3.2478600
5: -8.9218502, -5.8169274, -8.9765530, -5.7065763, -2.8888688, 2.8669572
6: -12.3745804, -8.9950542, -12.4898367, -8.9523640, -2.6943111, 2.7702141
7: -5.5388203, -2.8817828, -5.7020702, -2.7577899, -2.7810304, 2.7927291
8: -1.1019375, 1.9122753, -1.2068169, 2.0037317, -3.1056693, 3.1190922
9: -6.5032806, -3.9169111, -6.5870600, -3.8344464, -2.6682577, 2.6701488

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0856237, upper bound: 2.0438415
time: 4.86 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0856217, upper bound: 2.0438414
time: 4.68 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.2443399, -9.1210403, -13.3209581, -8.9732561, -3.7243257, 3.8972080
1: -7.1863809, -3.5407999, -7.3048439, -3.5086877, -3.5147858, 3.6243293
2: -10.0132484, -7.2979493, -10.0570183, -7.2594490, -2.7537994, 2.7590690
3: -12.5013933, -9.4653034, -12.5703068, -9.4160843, -3.0024223, 2.9849074
4: 5.4078064, 8.5721302, 5.3104267, 8.7126923, -3.3048859, 3.2617035
5: -8.9391508, -5.7702627, -8.9787130, -5.6990013, -2.9243526, 2.9096310
6: -12.4553823, -8.9797430, -12.5030413, -8.9513454, -2.7261047, 2.7989187
7: -5.5616713, -2.8372831, -5.7038751, -2.7505379, -2.8111334, 2.8164794
8: -1.1583488, 1.9369686, -1.2158689, 2.0059829, -3.1643317, 3.1528375
9: -6.5155468, -3.9069538, -6.5884991, -3.8328958, -2.6826510, 2.6815453

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0856217, upper bound: 2.0438413
time: 5.07 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0856218, upper bound: 2.0438412
time: 4.98 seconds

## BFS IS instance: IS_A2_B1_A1

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

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0269124, upper bound: 2.0855010
time: 5.99 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438412, upper bound: 2.0856221
time: 4.88 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438412, upper bound: 2.0856221
time: 4.78 seconds

## BFS IS instance: IS_A2_B1_A2

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

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0269123, upper bound: 2.0855007
time: 4.99 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438432, upper bound: 2.0856221
time: 5.21 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0438411, upper bound: 2.0856220
time: 5.22 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.2316914, -8.9937172, -13.3063345, -8.9747963, -3.9851618, 4.0214453
1: -7.2850342, -3.5163171, -7.3018589, -3.5093508, -3.6386108, 3.6467400
2: -9.9834318, -7.2925563, -10.0451736, -7.2622795, -2.7211523, 2.7526174
3: -12.5505304, -9.4759731, -12.5678434, -9.4256077, -3.0848236, 3.0627868
4: 5.3413177, 8.7029877, 5.3144875, 8.7117653, -3.3704476, 3.3885002
5: -8.9616632, -5.7460165, -8.9765530, -5.7065763, -2.9627771, 2.9394865
6: -12.4220238, -8.9667816, -12.4898367, -8.9523640, -2.7691717, 2.8202672
7: -5.6806841, -2.7950184, -5.7020702, -2.7577899, -2.9228942, 2.9070518
8: -1.1593797, 1.9811001, -1.2068169, 2.0037317, -3.1631114, 3.1879170
9: -6.5759706, -3.8428798, -6.5870600, -3.8344464, -2.7415242, 2.7441802

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0534511, upper bound: 2.0911794
time: 4.92 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0534511, upper bound: 2.0911794
time: 4.95 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3209572, -8.9732580, -13.3209581, -8.9732561, -4.0414004, 4.0744605
1: -7.3048396, -3.5086896, -7.3048439, -3.5086877, -3.6606784, 3.6627378
2: -10.0570154, -7.2594500, -10.0570183, -7.2594490, -2.7975664, 2.7975683
3: -12.5703049, -9.4160862, -12.5703068, -9.4160843, -3.1245222, 3.1133647
4: 5.3104296, 8.7126913, 5.3104267, 8.7126923, -3.4022627, 3.4022646
5: -8.9787140, -5.6990032, -8.9787130, -5.6990013, -2.9880028, 2.9828882
6: -12.5030384, -8.9513435, -12.5030413, -8.9513454, -2.8008947, 2.8489347
7: -5.7038755, -2.7505393, -5.7038751, -2.7505379, -2.9533377, 2.9533358
8: -1.2158673, 2.0059822, -1.2158689, 2.0059829, -3.2218502, 3.2218511
9: -6.5884991, -3.8328962, -6.5884991, -3.8328958, -2.7556033, 2.7556028

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0534510, upper bound: 2.0911772
time: 5.01 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0534510, upper bound: 2.0911795
time: 5.24 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.69 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438458
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0438476, upper bound: 2.0438458
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0438476, upper bound: 2.0438460
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0438456, upper bound: 2.0438457
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0856237, upper bound: 2.0438415
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0856217, upper bound: 2.0438414
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0856217, upper bound: 2.0438413
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0856218, upper bound: 2.0438412
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0438412, upper bound: 2.0856221
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0438412, upper bound: 2.0856221
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0438432, upper bound: 2.0856221
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0438411, upper bound: 2.0856220
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0534511, upper bound: 2.0911794
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0534511, upper bound: 2.0911794
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0534510, upper bound: 2.0911772
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 4, lower bound: -2.0534510, upper bound: 2.0911795

## BFS IS instance: IS_A1_B1_A1_B1

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

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.57 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0095474, upper bound: 2.0155213
time: 4.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0155209, upper bound: 2.0155217
time: 4.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2

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

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.39 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0095474, upper bound: 2.0155213
time: 4.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0155209, upper bound: 2.0155216
time: 5.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1

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

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.40 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0095474, upper bound: 2.0155209
time: 4.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0155209, upper bound: 2.0155212
time: 4.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2

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

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.57 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0095494, upper bound: 2.0155209
time: 4.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0155209, upper bound: 2.0155212
time: 5.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.2316914, -8.9937172, -3.6475077, 3.7843957
1: -7.1668282, -3.5483451, -7.2850342, -3.5163171, -3.4828844, 3.5926144
2: -9.9397202, -7.3311028, -9.9834318, -7.2925563, -2.6471639, 2.6523290
3: -12.4820814, -9.5253382, -12.5505304, -9.4759731, -2.9158688, 2.9116979
4: 5.4385414, 8.5623474, 5.3413177, 8.7029877, -3.2644463, 3.2210298
5: -8.9218502, -5.8169274, -8.9616632, -5.7460165, -2.8636465, 2.8513660
6: -12.3745804, -8.9950542, -12.4220238, -8.9667816, -2.6804132, 2.7052231
7: -5.5388203, -2.8817828, -5.6806841, -2.7950184, -2.7438018, 2.7692285
8: -1.1019375, 1.9122753, -1.1593797, 1.9811001, -3.0830376, 3.0716550
9: -6.5032806, -3.9169111, -6.5759706, -3.8428798, -2.6562624, 2.6590595

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0855004, upper bound: 2.0269129
time: 5.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 11.95 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0496056, upper bound: 2.0155157
time: 4.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0579721, upper bound: 2.0155159
time: 5.16 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.3209562, -8.9732943, -3.6626420, 3.8478241
1: -7.1668282, -3.5483451, -7.3048363, -3.5087011, -3.4932127, 3.6129827
2: -9.9397202, -7.3311028, -10.0570078, -7.2595644, -2.6801558, 2.7259049
3: -12.4820814, -9.5253382, -12.5703030, -9.4161816, -2.9713597, 2.9331470
4: 5.4385414, 8.5623474, 5.3104515, 8.7126656, -3.2741241, 3.2518959
5: -8.9218502, -5.8169274, -8.9787130, -5.6990790, -2.8914673, 2.8688674
6: -12.3745804, -8.9950542, -12.5030003, -8.9513521, -2.6952834, 2.7723827
7: -5.5388203, -2.8817828, -5.7038364, -2.7505407, -2.7882795, 2.7891338
8: -1.1019375, 1.9122753, -1.2158635, 2.0059309, -3.1078684, 3.1281388
9: -6.5032806, -3.9169111, -6.5884829, -3.8328977, -2.6673679, 2.6715717

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0855003, upper bound: 2.0269128
time: 5.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 12.18 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0496080, upper bound: 2.0155156
time: 4.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0579720, upper bound: 2.0155159
time: 5.96 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.2443371, -9.1210804, -13.2316914, -8.9937172, -3.6873355, 3.8084111
1: -7.1863713, -3.5408142, -7.2850342, -3.5163171, -3.5042458, 3.6009924
2: -10.0132427, -7.2980623, -9.9834318, -7.2925563, -2.7206864, 2.6853695
3: -12.5013895, -9.4654007, -12.5505304, -9.4759731, -2.9424782, 2.9444890
4: 5.4078264, 8.5721016, 5.3413177, 8.7029877, -3.2951612, 3.2307839
5: -8.9391499, -5.7703357, -8.9616632, -5.7460165, -2.8781004, 2.8928230
6: -12.4553404, -8.9797535, -12.4220238, -8.9667816, -2.7551584, 2.7201254
7: -5.5616341, -2.8372846, -5.6806841, -2.7950184, -2.7666156, 2.7936130
8: -1.1583457, 1.9369197, -1.1593797, 1.9811001, -3.1394458, 3.0962994
9: -6.5155330, -3.9069543, -6.5759706, -3.8428798, -2.6724010, 2.6690164

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0855003, upper bound: 2.0269125
time: 5.48 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 12.12 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0496080, upper bound: 2.0155153
time: 4.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0579720, upper bound: 2.0155155
time: 5.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.2443399, -9.1210403, -13.3209572, -8.9732580, -3.7243261, 3.8642159
1: -7.1863809, -3.5407999, -7.3048396, -3.5086896, -3.5168452, 3.6243281
2: -10.0132484, -7.2979493, -10.0570154, -7.2594500, -2.7537985, 2.7590661
3: -12.5013933, -9.4653034, -12.5703049, -9.4160862, -2.9907436, 2.9849057
4: 5.4078064, 8.5721302, 5.3104296, 8.7126913, -3.3048849, 3.2617006
5: -8.9391508, -5.7702627, -8.9787140, -5.6990032, -2.9199157, 2.9096308
6: -12.4553823, -8.9797430, -12.5030384, -8.9513435, -2.7261052, 2.7508790
7: -5.5616713, -2.8372831, -5.7038755, -2.7505393, -2.8111320, 2.8140161
8: -1.1583488, 1.9369686, -1.2158673, 2.0059822, -3.1643310, 3.1528358
9: -6.5155468, -3.9069538, -6.5884991, -3.8328962, -2.6826506, 2.6815453

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0855003, upper bound: 2.0269125
time: 5.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 12.17 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0496056, upper bound: 2.0155152
time: 5.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0579720, upper bound: 2.0155155
time: 5.20 seconds

## BFS IS instance: IS_A2_B1_A1_B1

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

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.74 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0095450, upper bound: 2.0579728
time: 4.84 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0095470, upper bound: 2.0579747
time: 6.46 seconds

## BFS IS instance: IS_A2_B1_A1_B2

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

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.76 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0095451, upper bound: 2.0579727
time: 5.13 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0155153, upper bound: 2.0579728
time: 5.02 seconds

## BFS IS instance: IS_A2_B1_A2_B1

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

Time for backsubstitution: 14.46 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=3.402289390563965
rel_dist={4: [-2.091210306910777, 2.0912106090343423]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7298020, upper bound: 1.7040483
time: 5.01 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7405272, upper bound: 1.7405245
time: 4.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.04 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.04
Output dim: 4, lower bound: -1.7298020, upper bound: 1.7040483
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.04
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

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040491, upper bound: 1.7040487
time: 6.33 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040491, upper bound: 1.7040490
time: 6.10 seconds

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

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040489, upper bound: 1.7298040
time: 4.77 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040490, upper bound: 1.7405274
time: 7.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.98 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.98
Output dim: 4, lower bound: -1.7040491, upper bound: 1.7040487
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.98
Output dim: 4, lower bound: -1.7040491, upper bound: 1.7040490
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.98
Output dim: 4, lower bound: -1.7040489, upper bound: 1.7298040
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.98
Output dim: 4, lower bound: -1.7040490, upper bound: 1.7405274

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

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040511
time: 4.74 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040862, upper bound: 1.7040506
time: 4.72 seconds

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

Time for backsubstitution: 14.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040882, upper bound: 1.7040487
time: 6.29 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040484
time: 5.68 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.3175774, -8.9732952, -13.2443447, -9.1210403, -3.5402880, 3.3749557
1: -7.3003254, -3.5087399, -7.1863823, -3.5408008, -3.3861179, 3.2939258
2: -10.0569267, -7.2605362, -10.0132523, -7.2979498, -2.7589769, 2.7527161
3: -12.5701885, -9.4177761, -12.5013924, -9.4653015, -2.7610190, 2.7783515
4: 5.3111296, 8.7126455, 5.4078045, 8.5721302, -3.2121062, 3.1788499
5: -8.9782515, -5.6990557, -8.9391518, -5.7702613, -2.6910620, 2.6923499
6: -12.5023689, -8.9515209, -12.4553852, -8.9797430, -2.4702973, 2.4460077
7: -5.7035522, -2.7524328, -5.5616732, -2.8372822, -2.6079898, 2.6615784
8: -1.2141044, 2.0058858, -1.1583521, 1.9369693, -3.1510737, 3.1642380
9: -6.5864849, -3.8332114, -6.5155468, -3.9069538, -2.5557513, 2.5090504

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040484, upper bound: 1.7298012
time: 5.22 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298008
time: 4.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.3209581, -8.9732561, -13.3209581, -8.9732561, -3.7172632, 3.7172635
1: -7.3048439, -3.5086877, -7.3048439, -3.5086877, -3.4278550, 3.4278541
2: -10.0570183, -7.2594490, -10.0570183, -7.2594490, -2.7975693, 2.7975693
3: -12.5703068, -9.4160843, -12.5703068, -9.4160843, -2.8975124, 2.8975127
4: 5.3104267, 8.7126923, 5.3104267, 8.7126923, -3.3551750, 3.3551760
5: -8.9787130, -5.6990013, -8.9787130, -5.6990013, -2.7591643, 2.7591646
6: -12.5030413, -8.9513454, -12.5030413, -8.9513454, -2.5186992, 2.5186992
7: -5.7038751, -2.7505379, -5.7038751, -2.7505379, -2.7803173, 2.7803178
8: -1.2158689, 2.0059829, -1.2158689, 2.0059829, -3.2218518, 3.2218518
9: -6.5884991, -3.8328958, -6.5884991, -3.8328958, -2.6600609, 2.6600611

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298014
time: 9.71 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040485, upper bound: 1.7405269
time: 5.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 29.82 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.82
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040511
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.82
Output dim: 4, lower bound: -1.7040862, upper bound: 1.7040506
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.82
Output dim: 4, lower bound: -1.7040882, upper bound: 1.7040487
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.82
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040484
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.82
Output dim: 4, lower bound: -1.7040484, upper bound: 1.7298012
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.82
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298008
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.82
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298014
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.82
Output dim: 4, lower bound: -1.7040485, upper bound: 1.7405269

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.2182474, -9.1241589, -3.1738133, 3.2073405
1: -7.1668282, -3.5483451, -7.1810794, -3.5420969, -3.2298760, 3.2366352
2: -9.9397202, -7.3311028, -9.9921074, -7.3042269, -2.6354933, 2.6610045
3: -12.4820814, -9.5253382, -12.4973030, -9.4830341, -2.6418295, 2.6290023
4: 5.4385414, 8.5623474, 5.4152508, 8.5701513, -3.0772381, 3.0911365
5: -8.9218502, -5.8169274, -8.9352074, -5.7842555, -2.6049943, 2.5871468
6: -12.3745804, -8.9950542, -12.4316349, -8.9816055, -2.3386512, 2.3789916
7: -5.5388203, -2.8817828, -5.5581160, -2.8502276, -2.5045276, 2.4954948
8: -1.1019375, 1.9122753, -1.1422050, 1.9324846, -3.0344222, 3.0544803
9: -6.5032806, -3.9169111, -6.5129008, -3.9097018, -2.4492230, 2.4525108

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040881
time: 4.72 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040882, upper bound: 1.7040881
time: 4.94 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.2443399, -9.1210403, -13.2443447, -9.1210403, -3.2186108, 3.2646470
1: -7.1863809, -3.5407999, -7.1863823, -3.5408008, -3.2533355, 3.2551789
2: -10.0132484, -7.2979493, -10.0132523, -7.2979498, -2.7152987, 2.7153029
3: -12.5013933, -9.4653034, -12.5013924, -9.4653015, -2.6937180, 2.6735430
4: 5.4078064, 8.5721302, 5.4078045, 8.5721302, -3.1093459, 3.1093569
5: -8.9391508, -5.7702627, -8.9391518, -5.7702613, -2.6367602, 2.6307430
6: -12.4553823, -8.9797430, -12.4553852, -8.9797430, -2.3616948, 2.4189897
7: -5.5616713, -2.8372831, -5.5616732, -2.8372822, -2.5433824, 2.5175996
8: -1.1583488, 1.9369686, -1.1583521, 1.9369693, -3.0953181, 3.0953207
9: -6.5155468, -3.9069538, -6.5155468, -3.9069538, -2.4695034, 2.4741230

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040882
time: 5.02 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040882
time: 5.14 seconds

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

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7297986, upper bound: 1.7040480
time: 4.66 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7297986, upper bound: 1.7040480
time: 4.73 seconds

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

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7298015, upper bound: 1.7040484
time: 4.79 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7298014, upper bound: 1.7040482
time: 4.63 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.2284451, -8.9937534, -13.2182474, -9.1241589, -3.4474840, 3.2867582
1: -7.2805090, -3.5163677, -7.1810794, -3.5420969, -3.3628492, 3.2771535
2: -9.9833384, -7.2936392, -9.9921074, -7.3042269, -2.6791115, 2.6984682
3: -12.5504169, -9.4776659, -12.4973030, -9.4830341, -2.6950557, 2.7126031
4: 5.3420167, 8.7029409, 5.4152508, 8.5701513, -3.1800332, 3.1605999
5: -8.9611988, -5.7460699, -8.9352074, -5.7842555, -2.6559596, 2.6424079
6: -12.4213486, -8.9669552, -12.4316349, -8.9816055, -2.3896713, 2.4060445
7: -5.6803694, -2.7969165, -5.5581160, -2.8502276, -2.5551262, 2.6134562
8: -1.1576118, 1.9810028, -1.1422050, 1.9324846, -3.0900965, 3.1232078
9: -6.5739584, -3.8432016, -6.5129008, -3.9097018, -2.5350342, 2.4918346

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6876568, upper bound: 1.7296449
time: 4.93 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298008
time: 5.08 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298008
time: 4.87 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3175745, -8.9732962, -13.2443447, -9.1210403, -3.4925098, 3.3648777
1: -7.3003244, -3.5087399, -7.1863823, -3.5408008, -3.3861170, 3.2957678
2: -10.0569267, -7.2605367, -10.0132523, -7.2979498, -2.7589769, 2.7527156
3: -12.5701885, -9.4177771, -12.5013924, -9.4653015, -2.7546036, 2.7569039
4: 5.3111305, 8.7126436, 5.4078045, 8.5721302, -3.2121034, 3.1782444
5: -8.9782515, -5.6990600, -8.9391518, -5.7702613, -2.6910615, 2.6863286
6: -12.5023670, -8.9515228, -12.4553852, -8.9797430, -2.4126511, 2.4460075
7: -5.7035513, -2.7524340, -5.5616732, -2.8372822, -2.6024561, 2.6352952
8: -1.2141023, 2.0058866, -1.1583521, 1.9369693, -3.1510715, 3.1642387
9: -6.5864844, -3.8332124, -6.5155468, -3.9069538, -2.5554709, 2.5112135

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6876569, upper bound: 1.7296445
time: 5.07 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298008
time: 4.97 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7040507, upper bound: 1.7298008
time: 5.27 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.2316914, -8.9937172, -13.2948923, -8.9763851, -3.6244855, 3.6478264
1: -7.2850342, -3.5163171, -7.2994795, -3.5100107, -3.4046993, 3.4111743
2: -9.9834318, -7.2925563, -10.0358582, -7.2656765, -2.7177553, 2.7433019
3: -12.5505304, -9.4759731, -12.5658693, -9.4337883, -2.8445415, 2.8317866
4: 5.3413177, 8.7029877, 5.3178964, 8.7107563, -3.3230352, 3.3368549
5: -8.9616632, -5.7460165, -8.9748392, -5.7131019, -2.7271667, 2.7088523
6: -12.4220238, -8.9667816, -12.4792290, -8.9532738, -2.4380970, 2.4787538
7: -5.6806841, -2.7950184, -5.7002420, -2.7634637, -2.7377503, 2.7322264
8: -1.1593797, 1.9811001, -1.1997190, 2.0014510, -3.1608307, 3.1808190
9: -6.5759706, -3.8428798, -6.5857782, -3.8356547, -2.6393290, 2.6427557

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7073671, upper bound: 1.7332400
time: 5.31 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7188936, upper bound: 1.7405257
time: 7.85 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3209572, -8.9732580, -13.3209581, -8.9732561, -3.6694217, 3.7158489
1: -7.3048396, -3.5086896, -7.3048439, -3.5086877, -3.4278531, 3.4296966
2: -10.0570154, -7.2594500, -10.0570183, -7.2594490, -2.7975664, 2.7975683
3: -12.5703049, -9.4160862, -12.5703068, -9.4160843, -2.8962154, 2.8765383
4: 5.3104296, 8.7126913, 5.3104267, 8.7126923, -3.3551750, 3.3551850
5: -8.9787140, -5.6990032, -8.9787130, -5.6990013, -2.7591643, 2.7531476
6: -12.5030384, -8.9513435, -12.5030413, -8.9513454, -2.4610519, 2.5186996
7: -5.7038755, -2.7505393, -5.7038751, -2.7505379, -2.7803173, 2.7540350
8: -1.2158673, 2.0059822, -1.2158689, 2.0059829, -3.2218502, 3.2218511
9: -6.5884991, -3.8328962, -6.5884991, -3.8328958, -2.6582651, 2.6619678

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7188979, upper bound: 1.7405263
time: 10.35 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7188961, upper bound: 1.7405241
time: 9.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 34.37 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040881
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7040882, upper bound: 1.7040881
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040882
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7040860, upper bound: 1.7040882
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7297986, upper bound: 1.7040480
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7297986, upper bound: 1.7040480
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7298015, upper bound: 1.7040484
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7298014, upper bound: 1.7040482
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298008
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298008
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7040486, upper bound: 1.7298008
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7040507, upper bound: 1.7298008
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7073671, upper bound: 1.7332400
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7188936, upper bound: 1.7405257
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7188979, upper bound: 1.7405263
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.37
Output dim: 4, lower bound: -1.7188961, upper bound: 1.7405241

## BFS IS instance: IS_A1_B1_A1_B1

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

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.45 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6828735, upper bound: 1.6864305
time: 4.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6864784, upper bound: 1.6864784
time: 5.25 seconds

## BFS IS instance: IS_A1_B1_A1_B2

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

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.55 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6828714, upper bound: 1.6864305
time: 4.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6864784, upper bound: 1.6864784
time: 4.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1

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

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.56 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6828714, upper bound: 1.6864300
time: 6.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6864784, upper bound: 1.6864780
time: 5.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2

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

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.71 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6828735, upper bound: 1.6864300
time: 4.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6864784, upper bound: 1.6864779
time: 5.27 seconds

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

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7296477, upper bound: 1.6876568
time: 4.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 11.51 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7075508, upper bound: 1.6863911
time: 4.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7128579, upper bound: 1.6864397
time: 6.21 seconds

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

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7296477, upper bound: 1.6876569
time: 5.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 12.63 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7075481, upper bound: 1.6863911
time: 5.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7128550, upper bound: 1.6864397
time: 6.09 seconds

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

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7296477, upper bound: 1.6876564
time: 5.04 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 11.65 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7075507, upper bound: 1.6863905
time: 6.05 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7128579, upper bound: 1.6864392
time: 6.94 seconds

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

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7296477, upper bound: 1.6876566
time: 5.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 12.60 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7075506, upper bound: 1.6863906
time: 6.13 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7128549, upper bound: 1.6864393
time: 7.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.2284451, -8.9937534, -13.1549864, -9.1414566, -3.4260550, 3.2613232
1: -7.2805090, -3.5163677, -7.1668282, -3.5483451, -3.3544559, 3.2620196
2: -9.9833384, -7.2936392, -9.9397202, -7.3311028, -2.6522355, 2.6460810
3: -12.5504169, -9.4776659, -12.4820814, -9.5253382, -2.6736801, 2.6905494
4: 5.3420167, 8.7029409, 5.4385414, 8.5623474, -3.1713443, 3.1379850
5: -8.9611988, -5.7460699, -8.9218502, -5.8169274, -2.6276827, 2.6284115
6: -12.4213486, -8.9669552, -12.3745804, -8.9950542, -2.3765984, 2.3526311
7: -5.6803694, -2.7969165, -5.5388203, -2.8817828, -2.5389001, 2.5923250
8: -1.1576118, 1.9810028, -1.1019375, 1.9122753, -3.0698872, 3.0829403
9: -6.5739584, -3.8432016, -6.5032806, -3.9169111, -2.5242672, 2.4777615

Time for backsubstitution: 14.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.90 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6828372, upper bound: 1.7128313
time: 4.99 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6864397, upper bound: 1.7128553
time: 5.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -13.2284451, -8.9937534, -13.2442789, -9.1219788, -3.4489217, 3.2909920
1: -7.2805090, -3.5163677, -7.1862268, -3.5411444, -3.3620462, 3.2828650
2: -9.9833384, -7.2936392, -10.0130739, -7.3008842, -2.6824541, 2.7194347
3: -12.5504169, -9.4776659, -12.5013351, -9.4677496, -2.6977530, 2.7166448
4: 5.3420167, 8.7029409, 5.4083400, 8.5713844, -3.1814518, 3.1665657
5: -8.9611988, -5.7460699, -8.9391069, -5.7721491, -2.6594212, 2.6422291
6: -12.4213486, -8.9669552, -12.4543362, -8.9800262, -2.3913236, 2.4154677
7: -5.6803694, -2.7969165, -5.5606880, -2.8373220, -2.5587425, 2.6124587
8: -1.1576118, 1.9810028, -1.1582696, 1.9356942, -3.0933061, 3.1392725
9: -6.5739584, -3.8432016, -6.5151930, -3.9069624, -2.5352726, 2.4925466

Time for backsubstitution: 14.54 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=3.359607696533203
rel_dist={4: [-1.74054476319957, 1.7405458869370563]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6016579, upper bound: 1.5823060
time: 4.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115537, upper bound: 1.6115487
time: 8.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.12 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.12
Output dim: 4, lower bound: -1.6016579, upper bound: 1.5823060
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.12
Output dim: 4, lower bound: -1.6115537, upper bound: 1.6115487

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.2443447, -9.1210403, -13.2960911, -9.0434008, -3.2316456, 3.4012187
1: -7.1863823, -3.5408008, -7.2447996, -3.5148530, -3.2092571, 3.2778378
2: -10.0132523, -7.2979498, -10.0359745, -7.2747850, -2.7384672, 2.7380247
3: -12.5013924, -9.4653015, -12.5336781, -9.4353437, -2.6954236, 2.6575131
4: 5.4078045, 8.5721302, 5.3407826, 8.6435699, -3.1013341, 3.1224799
5: -8.9391518, -5.7702613, -8.9638901, -5.7333999, -2.5981917, 2.6041977
6: -12.4553852, -8.9797430, -12.4801884, -8.9622803, -2.3313751, 2.3328354
7: -5.5616732, -2.8372822, -5.6334696, -2.7762892, -2.5753517, 2.5207431
8: -1.1583521, 1.9369693, -1.1986203, 1.9725804, -3.1309326, 3.1355896
9: -6.5155468, -3.9069538, -6.5652227, -3.8694601, -2.4450283, 2.4883504

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823061, upper bound: 1.5823082
time: 5.14 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823061, upper bound: 1.5823059
time: 4.68 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.3209581, -8.9732561, -13.3209705, -8.9732380, -3.5977516, 3.5866735
1: -7.3048439, -3.5086877, -7.3048601, -3.5086854, -3.3502526, 3.3905706
2: -10.0570183, -7.2594490, -10.0570221, -7.2594457, -2.7975726, 2.7975731
3: -12.5703068, -9.4160843, -12.5703163, -9.4160786, -2.7884669, 2.8214233
4: 5.3104267, 8.7126923, 5.3104210, 8.7127037, -3.2979622, 3.2933297
5: -8.9787130, -5.6990013, -8.9787188, -5.6989913, -2.6940875, 2.6828935
6: -12.5030413, -8.9513454, -12.5030479, -8.9510479, -2.3910866, 2.4086285
7: -5.7038751, -2.7505379, -5.7038965, -2.7505322, -2.7138257, 2.7433290
8: -1.2158689, 2.0059829, -1.2158730, 2.0059884, -3.2218573, 3.2218559
9: -6.5884991, -3.8328958, -6.5885034, -3.8328872, -2.6015458, 2.5779166

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823061, upper bound: 1.6016601
time: 4.95 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016576
time: 4.51 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.07 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.07
Output dim: 4, lower bound: -1.5823061, upper bound: 1.5823082
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.07
Output dim: 4, lower bound: -1.5823061, upper bound: 1.5823059
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.07
Output dim: 4, lower bound: -1.5823061, upper bound: 1.6016601
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.07
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016576

## BFS IS instance: IS_A1_B1

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

Time for backsubstitution: 13.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823505, upper bound: 1.5823083
time: 6.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823505, upper bound: 1.5823078
time: 5.88 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.2443447, -9.1210403, -13.3151913, -8.9740067, -3.2457232, 3.4196460
1: -7.1863823, -3.5408008, -7.2944493, -3.5088406, -3.2199392, 3.3048549
2: -10.0132523, -7.2979498, -10.0566397, -7.2625914, -2.7506609, 2.7586899
3: -12.5013924, -9.4653015, -12.5696917, -9.4197979, -2.7022214, 2.6809354
4: 5.4078045, 8.5721302, 5.3136606, 8.7125721, -3.1098795, 3.1482015
5: -8.9391518, -5.7702613, -8.9769831, -5.6994257, -2.6135576, 2.6149433
6: -12.4553852, -8.9797430, -12.5007944, -8.9518213, -2.3365202, 2.3590915
7: -5.5616732, -2.8372822, -5.7031474, -2.7565615, -2.5936899, 2.5307379
8: -1.1583521, 1.9369693, -1.2118325, 2.0057650, -3.1641171, 3.1488018
9: -6.5155468, -3.9069538, -6.5837078, -3.8341618, -2.4487901, 2.4971910

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823505, upper bound: 1.5823059
time: 4.55 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823504, upper bound: 1.5823054
time: 4.59 seconds

## BFS IS instance: IS_A2_B1

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

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823054, upper bound: 1.6016573
time: 5.34 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823057, upper bound: 1.6016568
time: 5.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.3209581, -8.9732561, -13.3209581, -8.9732561, -3.5977259, 3.5977263
1: -7.3048439, -3.5086877, -7.3048439, -3.5086877, -3.3502450, 3.3502455
2: -10.0570183, -7.2594490, -10.0570183, -7.2594490, -2.7975693, 2.7975693
3: -12.5703068, -9.4160843, -12.5703068, -9.4160843, -2.8214107, 2.8214104
4: 5.3104267, 8.7126923, 5.3104267, 8.7126923, -3.2933216, 3.2933221
5: -8.9787130, -5.6990013, -8.9787130, -5.6990013, -2.6828856, 2.6828852
6: -12.5030413, -8.9513454, -12.5030413, -8.9513454, -2.4086213, 2.4086208
7: -5.7038751, -2.7505379, -5.7038751, -2.7505379, -2.7138195, 2.7138205
8: -1.2158689, 2.0059829, -1.2158689, 2.0059829, -3.2218518, 3.2218518
9: -6.5884991, -3.8328958, -6.5884991, -3.8328958, -2.6015444, 2.6015441

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823057, upper bound: 1.6115516
time: 4.67 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823056, upper bound: 1.6115511
time: 4.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.83 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.83
Output dim: 4, lower bound: -1.5823505, upper bound: 1.5823083
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.83
Output dim: 4, lower bound: -1.5823505, upper bound: 1.5823078
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.83
Output dim: 4, lower bound: -1.5823505, upper bound: 1.5823059
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.83
Output dim: 4, lower bound: -1.5823504, upper bound: 1.5823054
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.83
Output dim: 4, lower bound: -1.5823054, upper bound: 1.6016573
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.83
Output dim: 4, lower bound: -1.5823057, upper bound: 1.6016568
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.83
Output dim: 4, lower bound: -1.5823057, upper bound: 1.6115516
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.83
Output dim: 4, lower bound: -1.5823056, upper bound: 1.6115511

## BFS IS instance: IS_A1_B1_A1

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

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823505, upper bound: 1.5823526
time: 4.97 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823505, upper bound: 1.5823526
time: 4.68 seconds

## BFS IS instance: IS_A1_B1_A2

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

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823505, upper bound: 1.5823526
time: 5.09 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823505, upper bound: 1.5823526
time: 5.27 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.2824707, -8.9780769, -3.1522751, 3.3441322
1: -7.1668282, -3.5483451, -7.2876921, -3.5105543, -3.1958027, 3.2867517
2: -9.9397202, -7.3311028, -10.0300150, -7.2708406, -2.6688795, 2.6926894
3: -12.4820814, -9.5253382, -12.5642195, -9.4422235, -2.6421227, 2.6137574
4: 5.4385414, 8.5623474, 5.3231411, 8.7100344, -3.0770078, 3.1277707
5: -8.9218502, -5.8169274, -8.9720974, -5.7173042, -2.5655417, 2.5643377
6: -12.3745804, -8.9950542, -12.4707890, -8.9542923, -2.2557006, 2.3111703
7: -5.5388203, -2.8817828, -5.6984339, -2.7728162, -2.5380812, 2.4816372
8: -1.1019375, 1.9122753, -1.1915166, 1.9998784, -3.1018159, 3.1037920
9: -6.5032806, -3.9169111, -6.5802326, -3.8376260, -2.4274840, 2.4786286

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6016576, upper bound: 1.5823055
time: 4.91 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6016576, upper bound: 1.5823054
time: 4.75 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.2443399, -9.1210403, -13.3151913, -8.9740067, -3.1910315, 3.4182255
1: -7.1863809, -3.5407999, -7.2944493, -3.5088406, -3.2199373, 3.3045921
2: -10.0132484, -7.2979493, -10.0566397, -7.2625914, -2.7396178, 2.7586904
3: -12.5013933, -9.4653034, -12.5696917, -9.4197979, -2.7009721, 2.6549067
4: 5.4078064, 8.5721302, 5.3136606, 8.7125721, -3.1091051, 3.1475699
5: -8.9391508, -5.7702627, -8.9769831, -5.6994257, -2.6102641, 2.6086261
6: -12.4553823, -8.9797430, -12.5007944, -8.9518213, -2.2760224, 2.3590920
7: -5.5616713, -2.8372831, -5.7031474, -2.7565615, -2.5882082, 2.5034351
8: -1.1583488, 1.9369686, -1.2118325, 2.0057650, -3.1641138, 3.1488011
9: -6.5155468, -3.9069538, -6.5837078, -3.8341618, -2.4469929, 2.5004611

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6016548, upper bound: 1.5823052
time: 5.41 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6016545, upper bound: 1.5823050
time: 5.20 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.2260590, -8.9944620, -13.2115250, -9.1251011, -3.3256063, 3.1530426
1: -7.2746186, -3.5164707, -7.1797142, -3.5424833, -3.2809477, 3.2017612
2: -9.9830484, -7.2956896, -9.9866447, -7.3062539, -2.6767945, 2.6909552
3: -12.5499239, -9.4796906, -12.4963064, -9.4877596, -2.6114438, 2.6348655
4: 5.3445454, 8.7028666, 5.4172583, 8.5695353, -3.1152263, 3.0896912
5: -8.9599285, -5.7464428, -8.9341793, -5.7880020, -2.5755715, 2.5626242
6: -12.4197655, -8.9672508, -12.4254637, -8.9821262, -2.2779593, 2.2899165
7: -5.6799746, -2.8010478, -5.5570507, -2.8535593, -2.4754305, 2.5441074
8: -1.1553364, 1.9808817, -1.1380415, 1.9311433, -3.0864797, 3.1189232
9: -6.5711851, -3.8441548, -6.5121675, -3.9104023, -2.4757204, 2.4303694

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5598131, upper bound: 1.6014896
time: 5.18 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823078, upper bound: 1.6016568
time: 6.16 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823056, upper bound: 1.6016568
time: 5.24 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3151875, -8.9740076, -13.2443447, -9.1210403, -3.3674164, 3.2356439
1: -7.2944484, -3.5088401, -7.1863823, -3.5408008, -3.3048534, 3.2217088
2: -10.0566378, -7.2625899, -10.0132523, -7.2979498, -2.7374105, 2.7506623
3: -12.5696898, -9.4198027, -12.5013924, -9.4653015, -2.6745198, 2.6779318
4: 5.3136625, 8.7125702, 5.4078045, 8.5721302, -3.1473875, 3.1092649
5: -8.9769812, -5.6994276, -8.9391518, -5.7702613, -2.6149435, 2.6072352
6: -12.5007906, -8.9518204, -12.4553852, -8.9797430, -2.2982459, 2.3365195
7: -5.7031479, -2.7565625, -5.5616732, -2.8372822, -2.5252061, 2.5659344
8: -1.2118309, 2.0057642, -1.1583521, 1.9369693, -3.1488001, 3.1641164
9: -6.5837078, -3.8341632, -6.5155468, -3.9069538, -2.4969106, 2.4501579

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5598131, upper bound: 1.6014880
time: 4.70 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823056, upper bound: 1.6016568
time: 5.29 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823056, upper bound: 1.6016568
time: 4.77 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.2316914, -8.9937172, -13.2881813, -8.9773293, -3.5037212, 3.5227876
1: -7.2850342, -3.5163171, -7.2980933, -3.5104015, -3.3264427, 3.3321953
2: -9.9834318, -7.2925563, -10.0303936, -7.2677031, -2.7157288, 2.7378373
3: -12.5505304, -9.4759731, -12.5648270, -9.4385090, -2.7613688, 2.7540846
4: 5.3413177, 8.7029877, 5.3199091, 8.7101555, -3.2604680, 3.2730436
5: -8.9616632, -5.7460165, -8.9738293, -5.7168784, -2.6470017, 2.6315098
6: -12.4220238, -8.9667816, -12.4730406, -8.9538136, -2.3275218, 2.3620372
7: -5.6806841, -2.7950184, -5.6991549, -2.7667899, -2.6671619, 2.6644120
8: -1.1593797, 1.9811001, -1.1955547, 2.0000935, -3.1594732, 3.1766548
9: -6.5759706, -3.8428798, -6.5850220, -3.8363581, -2.5800605, 2.5830142

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5857452, upper bound: 1.6046488
time: 5.62 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954324, upper bound: 1.6115525
time: 5.57 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3209572, -8.9732580, -13.3209581, -8.9732561, -3.5454278, 3.5963118
1: -7.3048396, -3.5086896, -7.3048439, -3.5086877, -3.3502440, 3.3520164
2: -10.0570154, -7.2594500, -10.0570183, -7.2594490, -2.7975664, 2.7975683
3: -12.5703049, -9.4160862, -12.5703068, -9.4160843, -2.8201137, 2.7975962
4: 5.3104296, 8.7126913, 5.3104267, 8.7126923, -3.2933216, 3.2933240
5: -8.9787140, -5.6990032, -8.9787130, -5.6990013, -2.6828856, 2.6765676
6: -12.5030384, -8.9513435, -12.5030413, -8.9513454, -2.3477716, 2.4086213
7: -5.7038755, -2.7505393, -5.7038751, -2.7505379, -2.7138205, 2.6861777
8: -1.2158673, 2.0059822, -1.2158689, 2.0059829, -3.2218502, 3.2218511
9: -6.5884991, -3.8328962, -6.5884991, -3.8328958, -2.5997481, 2.6026545

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954366, upper bound: 1.6115509
time: 5.67 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954345, upper bound: 1.6115532
time: 5.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.53 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.5823505, upper bound: 1.5823526
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.5823505, upper bound: 1.5823526
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.5823505, upper bound: 1.5823526
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.5823505, upper bound: 1.5823526
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.6016576, upper bound: 1.5823055
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.6016576, upper bound: 1.5823054
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.6016548, upper bound: 1.5823052
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.6016545, upper bound: 1.5823050
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.5823078, upper bound: 1.6016568
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.5823056, upper bound: 1.6016568
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.5823056, upper bound: 1.6016568
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.5823056, upper bound: 1.6016568
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.5857452, upper bound: 1.6046488
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.5954324, upper bound: 1.6115525
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.5954366, upper bound: 1.6115509
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.5954345, upper bound: 1.6115532

## BFS IS instance: IS_A1_B1_A1_B1

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

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.42 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5654625, upper bound: 1.5681026
time: 5.04 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5681476, upper bound: 1.5681476
time: 4.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2

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

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.48 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5654625, upper bound: 1.5681026
time: 4.99 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5681476, upper bound: 1.5681476
time: 5.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1

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

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.39 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5654625, upper bound: 1.5681022
time: 5.25 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5681476, upper bound: 1.5681470
time: 5.46 seconds

## BFS IS instance: IS_A1_B1_A2_B2

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

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.47 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5654625, upper bound: 1.5681022
time: 5.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5681476, upper bound: 1.5681470
time: 5.50 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.2260590, -8.9944620, -3.1320915, 3.3054118
1: -7.1668282, -3.5483451, -7.2746186, -3.5164707, -3.1880326, 3.2731986
2: -9.9397202, -7.3311028, -9.9830484, -7.2956896, -2.6440306, 2.6519456
3: -12.4820814, -9.5253382, -12.5499239, -9.4796906, -2.6144176, 2.5936117
4: 5.4385414, 8.5623474, 5.3445454, 8.7028666, -3.0690236, 3.1071541
5: -8.9218502, -5.8169274, -8.9599285, -5.7464428, -2.5496871, 2.5515621
6: -12.3745804, -8.9950542, -12.4197655, -8.9672508, -2.2431450, 2.2653878
7: -5.5388203, -2.8817828, -5.6799746, -2.8010478, -2.5242410, 2.4616556
8: -1.1019375, 1.9122753, -1.1553364, 1.9808817, -3.0828192, 3.0676117
9: -6.5032806, -3.9169111, -6.5711851, -3.8441548, -2.4175067, 2.4657092

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6014916, upper bound: 1.5598129
time: 5.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 11.54 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5839979, upper bound: 1.5680542
time: 4.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5878731, upper bound: 1.5681021
time: 5.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.1549864, -9.1414566, -13.3150902, -8.9753246, -3.1455531, 3.3494012
1: -7.1668282, -3.5483451, -7.2942314, -3.5093179, -3.1973209, 3.2928245
2: -9.9397202, -7.3311028, -10.0563908, -7.2667065, -2.6730137, 2.6972308
3: -12.4820814, -9.5253382, -12.5696096, -9.4232502, -2.6532476, 2.6137187
4: 5.4385414, 8.5623474, 5.3144255, 8.7115898, -3.0781078, 3.1355553
5: -8.9218502, -5.8169274, -8.9769230, -5.7021275, -2.5699296, 2.5682185
6: -12.3745804, -8.9950542, -12.4993305, -8.9522085, -2.2577629, 2.3151119
7: -5.5388203, -2.8817828, -5.7017422, -2.7566206, -2.5425830, 2.4793220
8: -1.1019375, 1.9122753, -1.2117124, 2.0039647, -3.1059022, 3.1239877
9: -6.5032806, -3.9169111, -6.5831780, -3.8341756, -2.4277310, 2.4817457

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6014916, upper bound: 1.5598129
time: 4.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 11.53 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5840005, upper bound: 1.5680541
time: 5.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5878703, upper bound: 1.5681022
time: 7.14 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.2442513, -9.1223488, -13.2260590, -8.9944620, -3.1583633, 3.3278019
1: -7.1861649, -3.5412786, -7.2746186, -3.5164707, -3.2086635, 3.2804642
2: -10.0130043, -7.3020515, -9.9830484, -7.2956896, -2.7173147, 2.6809969
3: -12.5013123, -9.4687214, -12.5499239, -9.4796906, -2.6403012, 2.6148350
4: 5.4085526, 8.5710869, 5.3445454, 8.7028666, -3.0974088, 3.1163540
5: -8.9390907, -5.7728987, -8.9599285, -5.7464428, -2.5632412, 2.5799220
6: -12.4539185, -8.9801397, -12.4197655, -8.9672508, -2.3019552, 2.2800405
7: -5.5602961, -2.8373384, -5.6799746, -2.8010478, -2.5419145, 2.4799666
8: -1.1582379, 1.9351881, -1.1553364, 1.9808817, -3.1391196, 3.0905244
9: -6.5150528, -3.9069672, -6.5711851, -3.8441548, -2.4317312, 2.4764891

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6014916, upper bound: 1.5598124
time: 4.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 11.19 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5839979, upper bound: 1.5680537
time: 5.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5878703, upper bound: 1.5681017
time: 5.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.2443399, -9.1210403, -13.3151875, -8.9740076, -3.1910315, 3.3674147
1: -7.1863809, -3.5407999, -7.2944484, -3.5088401, -3.2217083, 3.3045907
2: -10.0132484, -7.2979493, -10.0566378, -7.2625899, -2.7396173, 2.7354336
3: -12.5013933, -9.4653034, -12.5696898, -9.4198027, -2.6779304, 2.6549048
4: 5.4078064, 8.5721302, 5.3136625, 8.7125702, -3.1085110, 3.1467772
5: -8.9391508, -5.7702627, -8.9769812, -5.6994276, -2.6046216, 2.6086257
6: -12.4553823, -8.9797430, -12.5007906, -8.9518204, -2.2760215, 2.2982459
7: -5.5616713, -2.8372831, -5.7031479, -2.7565625, -2.5635257, 2.5009758
8: -1.1583488, 1.9369686, -1.2118309, 2.0057642, -3.1641130, 3.1487994
9: -6.5155468, -3.9069538, -6.5837078, -3.8341632, -2.4501567, 2.5004601

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6014917, upper bound: 1.5598125
time: 5.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1165
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 12.37 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5840005, upper bound: 1.5680537
time: 4.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5878705, upper bound: 1.5681018
time: 4.97 seconds

## BFS IS instance: IS_A2_B1_A1_B1

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

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.64 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5654193, upper bound: 1.5877617
time: 5.14 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5681019, upper bound: 1.5878700
time: 4.86 seconds

## BFS IS instance: IS_A2_B1_A1_B2

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

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1145
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1199
type: A, layer: 3, pos: 1704
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2564
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 310
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2642
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2860
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 709
type: A, layer: 3, pos: 2123
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1845
type: A, layer: 3, pos: 1451
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2570
type: A, layer: 3, pos: 1746
type: A, layer: 3, pos: 2369
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 1384
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1516
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 1432
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 611
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2488
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2594
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 431
type: A, layer: 3, pos: 1396
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 2314
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1782

Time for candidate selection: 6.67 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5654192, upper bound: 1.5877617
time: 4.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5681018, upper bound: 1.5878701
time: 4.82 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 30.63 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5654625, upper bound: 1.5681026
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5681476, upper bound: 1.5681476
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5654625, upper bound: 1.5681026
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5681476, upper bound: 1.5681476
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5654625, upper bound: 1.5681022
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5681476, upper bound: 1.5681470
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5654625, upper bound: 1.5681022
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5681476, upper bound: 1.5681470
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5839979, upper bound: 1.5680542
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5878731, upper bound: 1.5681021
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5840005, upper bound: 1.5680541
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5878703, upper bound: 1.5681022
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5839979, upper bound: 1.5680537
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5878703, upper bound: 1.5681017
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5840005, upper bound: 1.5680537
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5878705, upper bound: 1.5681018
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5654193, upper bound: 1.5877617
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5681019, upper bound: 1.5878700
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5654192, upper bound: 1.5877617
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.63
Output dim: 4, lower bound: -1.5681018, upper bound: 1.5878701
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 4, lower bound: -1.5823056, upper bound: 1.6016568
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 4, lower bound: -1.5823056, upper bound: 1.6016568
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 4, lower bound: -1.5857452, upper bound: 1.6046488
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 4, lower bound: -1.5954324, upper bound: 1.6115525
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 4, lower bound: -1.5954366, upper bound: 1.6115509
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 4, lower bound: -1.5954345, upper bound: 1.6115532
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=3.2979745864868164
rel_dist={4: [-1.6115652775740719, 1.611567303353243]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2414.16 seconds
