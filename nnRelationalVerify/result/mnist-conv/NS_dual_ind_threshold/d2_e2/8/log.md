## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.30999355


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.8070443, 0.8070445)
1: (-16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6663737, 0.6663735)
2: (-11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.6031795, 0.6031797)
3: (-10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4897735, 0.4897735)
4: (-2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4953128, 0.4953129)
5: (-10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4873219, 0.4873219)
6: (-19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5179894, 0.5179895)
7: (-2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4690161, 0.4690161)
8: (-1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5286696, 0.5286698)
9: (5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4882252, 0.4882252)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.44 + 34.83 = 58.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.3263090, upper bound: 0.3263092

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4600

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 4600

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3263059, upper bound: 0.3257033
time: 3.65 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3263059, upper bound: 0.3263060
time: 3.41 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.32 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.32
Output dim: 9, lower bound: -0.3263059, upper bound: 0.3257033
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.32
Output dim: 9, lower bound: -0.3263059, upper bound: 0.3263060

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -19.7804489, -18.2295799, -19.7822361, -18.2242451, -0.8031387, 0.7996573
1: -16.7279129, -15.4544535, -16.7283325, -15.4543200, -0.6657553, 0.6660368
2: -11.4483500, -10.3897362, -11.4504757, -10.3892050, -0.6003127, 0.6018338
3: -10.8654652, -9.8215370, -10.8671989, -9.8208866, -0.4873800, 0.4884521
4: -2.1144702, -1.3062899, -2.1178424, -1.3056178, -0.4906970, 0.4934237
5: -10.2583523, -9.3242302, -10.2593727, -9.3200884, -0.4847546, 0.4817238
6: -19.7635918, -18.5511799, -19.7649879, -18.5461559, -0.5147337, 0.5111178
7: -2.9937034, -2.1956198, -2.9937968, -2.1951816, -0.4687788, 0.4684381
8: -1.9131064, -1.1444364, -1.9133224, -1.1442099, -0.5283411, 0.5283070
9: 5.6436167, 6.5506744, 5.6431189, 6.5523229, -0.4869839, 0.4859133

Time for backsubstitution: 21.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4600

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4600

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3257023, upper bound: 0.3257025
time: 3.82 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3257023, upper bound: 0.3257026
time: 3.87 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.8070266, 0.8007255
1: -16.7284927, -15.4543095, -16.7284927, -15.4543056, -0.6658545, 0.6663723
2: -11.4512243, -10.3891983, -11.4512272, -10.3891983, -0.6009426, 0.6031761
3: -10.8678341, -9.8208590, -10.8678350, -9.8208580, -0.4876845, 0.4897736
4: -2.1190419, -1.3056176, -2.1190450, -1.3056176, -0.4919183, 0.4953115
5: -10.2593746, -9.3186073, -10.2593746, -9.3186054, -0.4873126, 0.4828215
6: -19.7649994, -18.5443192, -19.7650013, -18.5443192, -0.5179894, 0.5119451
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4690161, 0.4686100
8: -1.9133396, -1.1441288, -1.9133406, -1.1441293, -0.5286696, 0.5285559
9: 5.6430740, 6.5529265, 5.6430740, 6.5529280, -0.4881804, 0.4864888

Time for backsubstitution: 21.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4600

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4600

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3257023, upper bound: 0.3263062
time: 3.30 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3257023, upper bound: 0.3263061
time: 3.42 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.68 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.68
Output dim: 9, lower bound: -0.3257023, upper bound: 0.3257025
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.68
Output dim: 9, lower bound: -0.3257023, upper bound: 0.3257026
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.68
Output dim: 9, lower bound: -0.3257023, upper bound: 0.3263062
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.68
Output dim: 9, lower bound: -0.3257023, upper bound: 0.3263061

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -19.7804489, -18.2295799, -19.7804489, -18.2295799, -0.7978327, 0.7978327
1: -16.7279129, -15.4544535, -16.7279129, -15.4544535, -0.6655948, 0.6655953
2: -11.4483500, -10.3897362, -11.4483500, -10.3897362, -0.5997705, 0.5997705
3: -10.8654652, -9.8215370, -10.8654652, -9.8215370, -0.4867229, 0.4867229
4: -2.1144702, -1.3062899, -2.1144702, -1.3062899, -0.4900119, 0.4900119
5: -10.2583523, -9.3242302, -10.2583523, -9.3242302, -0.4807076, 0.4807076
6: -19.7635918, -18.5511799, -19.7635918, -18.5511799, -0.5097311, 0.5097311
7: -2.9937034, -2.1956198, -2.9937034, -2.1956198, -0.4683464, 0.4683464
8: -1.9131064, -1.1444364, -1.9131064, -1.1444364, -0.5280933, 0.5280931
9: 5.6436167, 6.5506744, 5.6436167, 6.5506744, -0.4853463, 0.4853463

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 3, pos: 1101

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3202554, upper bound: 0.3093427
time: 3.37 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3210103, upper bound: 0.3210110
time: 3.35 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -19.7804489, -18.2295799, -19.7823353, -18.2222900, -0.8051069, 0.7997479
1: -16.7279129, -15.4544535, -16.7284927, -15.4543095, -0.6657691, 0.6661980
2: -11.4483500, -10.3897362, -11.4512243, -10.3891983, -0.6003182, 0.6026282
3: -10.8654652, -9.8215370, -10.8678341, -9.8208590, -0.4874001, 0.4890954
4: -2.1144702, -1.3062899, -2.1190419, -1.3056176, -0.4906958, 0.4946251
5: -10.2583523, -9.3242302, -10.2593746, -9.3186073, -0.4863014, 0.4817162
6: -19.7635918, -18.5511799, -19.7649994, -18.5443192, -0.5165875, 0.5111321
7: -2.9937034, -2.1956198, -2.9937968, -2.1950381, -0.4689240, 0.4684381
8: -1.9131064, -1.1444364, -1.9133396, -1.1441288, -0.5284300, 0.5283327
9: 5.6436167, 6.5506744, 5.6430740, 6.5529265, -0.4876050, 0.4859214

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 1101

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3202554, upper bound: 0.3093427
time: 3.51 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3210103, upper bound: 0.3210110
time: 3.71 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -19.7823353, -18.2222900, -19.7804489, -18.2295799, -0.7997482, 0.8051069
1: -16.7284927, -15.4543095, -16.7279129, -15.4544535, -0.6661975, 0.6657691
2: -11.4512243, -10.3891983, -11.4483500, -10.3897362, -0.6026282, 0.6003184
3: -10.8678341, -9.8208590, -10.8654652, -9.8215370, -0.4890954, 0.4874002
4: -2.1190419, -1.3056176, -2.1144702, -1.3062899, -0.4946251, 0.4906957
5: -10.2593746, -9.3186073, -10.2583523, -9.3242302, -0.4817162, 0.4863014
6: -19.7649994, -18.5443192, -19.7635918, -18.5511799, -0.5111322, 0.5165875
7: -2.9937968, -2.1950381, -2.9937034, -2.1956198, -0.4684381, 0.4689240
8: -1.9133396, -1.1441288, -1.9131064, -1.1444364, -0.5283327, 0.5284302
9: 5.6430740, 6.5529265, 5.6436167, 6.5506744, -0.4859214, 0.4876049

Time for backsubstitution: 21.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 1101

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3202554, upper bound: 0.3099416
time: 3.49 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3210103, upper bound: 0.3216137
time: 3.29 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.8007255, 0.8007255
1: -16.7284927, -15.4543095, -16.7284927, -15.4543095, -0.6658545, 0.6658545
2: -11.4512243, -10.3891983, -11.4512243, -10.3891983, -0.6009433, 0.6009433
3: -10.8678341, -9.8208590, -10.8678341, -9.8208590, -0.4876845, 0.4876846
4: -2.1190419, -1.3056176, -2.1190419, -1.3056176, -0.4919182, 0.4919183
5: -10.2593746, -9.3186073, -10.2593746, -9.3186073, -0.4828215, 0.4828215
6: -19.7649994, -18.5443192, -19.7649994, -18.5443192, -0.5119449, 0.5119449
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4686100, 0.4686100
8: -1.9133396, -1.1441288, -1.9133396, -1.1441288, -0.5285561, 0.5285559
9: 5.6430740, 6.5529265, 5.6430740, 6.5529265, -0.4864885, 0.4864886

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.50 seconds

### Candidate
type: A, layer: 3, pos: 1101

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3202554, upper bound: 0.3099417
time: 3.40 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3210103, upper bound: 0.3216138
time: 3.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.72 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.72
Output dim: 9, lower bound: -0.3202554, upper bound: 0.3093427
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.72
Output dim: 9, lower bound: -0.3210103, upper bound: 0.3210110
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.72
Output dim: 9, lower bound: -0.3202554, upper bound: 0.3093427
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.72
Output dim: 9, lower bound: -0.3210103, upper bound: 0.3210110
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.72
Output dim: 9, lower bound: -0.3202554, upper bound: 0.3099416
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.72
Output dim: 9, lower bound: -0.3210103, upper bound: 0.3216137
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.72
Output dim: 9, lower bound: -0.3202554, upper bound: 0.3099417
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.72
Output dim: 9, lower bound: -0.3210103, upper bound: 0.3216138

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -19.7794228, -18.2286510, -19.7795830, -18.2295799, -0.7970855, 0.7977486
1: -16.7523956, -15.4865217, -16.7279129, -15.4637413, -0.6427803, 0.6169920
2: -11.4415169, -10.3853369, -11.4464054, -10.3898468, -0.5888402, 0.5924447
3: -10.8715487, -9.8279982, -10.8653755, -9.8234806, -0.4854064, 0.4777460
4: -2.1206243, -1.3105998, -2.1135755, -1.3075087, -0.4831038, 0.4807448
5: -10.2651567, -9.3359737, -10.2583523, -9.3277464, -0.4708843, 0.4611335
6: -19.7608681, -18.5494804, -19.7628136, -18.5512161, -0.5068874, 0.5103133
7: -3.0061274, -2.2037203, -2.9937034, -2.1980014, -0.4786630, 0.4614239
8: -1.9036922, -1.1305389, -1.9101110, -1.1451831, -0.5122058, 0.5225165
9: 5.6415844, 6.5266685, 5.6442223, 6.5433149, -0.4671388, 0.4564157

Time for backsubstitution: 21.90 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1678
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2129
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.35 seconds

### Candidate
type: B, layer: 3, pos: 1248

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3104425, upper bound: 0.2606582
time: 4.20 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3166108, upper bound: 0.3043502
time: 3.74 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -19.7802391, -18.2295799, -19.7804489, -18.2295799, -0.7977064, 0.7978327
1: -16.7279129, -15.4639778, -16.7279129, -15.4544535, -0.6655948, 0.6087806
2: -11.4465771, -10.3897629, -11.4483500, -10.3897362, -0.5845380, 0.5997612
3: -10.8654442, -9.8238487, -10.8654652, -9.8215370, -0.4867029, 0.4784310
4: -2.1142533, -1.3085387, -2.1144702, -1.3062899, -0.4899087, 0.4775398
5: -10.2583523, -9.3272800, -10.2583523, -9.3242302, -0.4807076, 0.4540632
6: -19.7627831, -18.5511932, -19.7635918, -18.5511799, -0.5080373, 0.5097244
7: -2.9937034, -2.1994653, -2.9937034, -2.1956198, -0.4683464, 0.4728115
8: -1.9080586, -1.1446199, -1.9131064, -1.1444364, -0.5076621, 0.5279992
9: 5.6437664, 6.5480433, 5.6436167, 6.5506744, -0.4852722, 0.4537497

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1678
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2129
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.35 seconds

### Candidate
type: B, layer: 3, pos: 1101

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3093425, upper bound: 0.3202563
time: 3.90 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3093425, upper bound: 0.3210112
time: 3.68 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -19.7794228, -18.2286510, -19.7814636, -18.2222900, -0.8043597, 0.7996554
1: -16.7523956, -15.4865217, -16.7284927, -15.4635954, -0.6429534, 0.6175945
2: -11.4415169, -10.3853369, -11.4492760, -10.3893099, -0.5893874, 0.5953012
3: -10.8715487, -9.8279982, -10.8677444, -9.8228045, -0.4860835, 0.4801185
4: -2.1206243, -1.3105998, -2.1181488, -1.3068371, -0.4837872, 0.4853578
5: -10.2651567, -9.3359737, -10.2593746, -9.3221235, -0.4764776, 0.4621421
6: -19.7608681, -18.5494804, -19.7642155, -18.5443497, -0.5137434, 0.5117146
7: -3.0061274, -2.2037203, -2.9937968, -2.1974173, -0.4792407, 0.4615157
8: -1.9036922, -1.1305389, -1.9103451, -1.1448746, -0.5125437, 0.5227566
9: 5.6415844, 6.5266685, 5.6436825, 6.5455670, -0.4693975, 0.4569867

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1678
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2129
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 1248

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3110446, upper bound: 0.2606582
time: 4.13 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3172135, upper bound: 0.3043502
time: 3.90 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -19.7802391, -18.2295799, -19.7823353, -18.2222900, -0.8049798, 0.7997479
1: -16.7279129, -15.4639778, -16.7284927, -15.4543095, -0.6657691, 0.6093836
2: -11.4465771, -10.3897629, -11.4512243, -10.3891983, -0.5850859, 0.6026187
3: -10.8654442, -9.8238487, -10.8678341, -9.8208590, -0.4873798, 0.4808030
4: -2.1142533, -1.3085387, -2.1190419, -1.3056176, -0.4905921, 0.4821565
5: -10.2583523, -9.3272800, -10.2593746, -9.3186073, -0.4863014, 0.4550723
6: -19.7627831, -18.5511932, -19.7649994, -18.5443192, -0.5148937, 0.5111253
7: -2.9937034, -2.1994653, -2.9937968, -2.1950381, -0.4689240, 0.4729028
8: -1.9080586, -1.1446199, -1.9133396, -1.1441288, -0.5079958, 0.5282385
9: 5.6437664, 6.5480433, 5.6430740, 6.5529265, -0.4875306, 0.4543257

Time for backsubstitution: 21.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1678
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2129
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.35 seconds

### Candidate
type: B, layer: 3, pos: 1101

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3099414, upper bound: 0.3202563
time: 3.52 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3099414, upper bound: 0.3210112
time: 3.73 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -19.7812862, -18.2213650, -19.7795830, -18.2295799, -0.7989728, 0.8050220
1: -16.7529755, -15.4863758, -16.7279129, -15.4637413, -0.6433830, 0.6171675
2: -11.4443865, -10.3847990, -11.4464054, -10.3898468, -0.5916898, 0.5929887
3: -10.8739176, -9.8273172, -10.8653755, -9.8234806, -0.4877784, 0.4784237
4: -2.1251965, -1.3099283, -2.1135755, -1.3075087, -0.4877119, 0.4814281
5: -10.2661734, -9.3303528, -10.2583523, -9.3277464, -0.4718938, 0.4667236
6: -19.7622757, -18.5426140, -19.7628136, -18.5512161, -0.5082896, 0.5171696
7: -3.0062175, -2.2031384, -2.9937034, -2.1980014, -0.4787543, 0.4620018
8: -1.9039264, -1.1302276, -1.9101110, -1.1451831, -0.5124471, 0.5228522
9: 5.6410561, 6.5289207, 5.6442223, 6.5433149, -0.4676926, 0.4586744

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1678
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2129
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.35 seconds

### Candidate
type: B, layer: 3, pos: 1248

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3104425, upper bound: 0.2612522
time: 3.25 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3166108, upper bound: 0.3049520
time: 3.91 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -19.7821236, -18.2222900, -19.7804489, -18.2295799, -0.7996199, 0.8051069
1: -16.7284927, -15.4638252, -16.7279129, -15.4544535, -0.6661975, 0.6089582
2: -11.4494514, -10.3892241, -11.4483500, -10.3897362, -0.5873952, 0.6003091
3: -10.8678131, -9.8231716, -10.8654652, -9.8215370, -0.4890754, 0.4791074
4: -2.1188254, -1.3078673, -2.1144702, -1.3062899, -0.4945221, 0.4782231
5: -10.2593746, -9.3216591, -10.2583523, -9.3242302, -0.4817162, 0.4596573
6: -19.7641888, -18.5443287, -19.7635918, -18.5511799, -0.5094380, 0.5165807
7: -2.9937968, -2.1988833, -2.9937034, -2.1956198, -0.4684381, 0.4733899
8: -1.9082932, -1.1443114, -1.9131064, -1.1444364, -0.5079007, 0.5283363
9: 5.6432228, 6.5502934, 5.6436167, 6.5506744, -0.4858458, 0.4560084

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1678
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2129
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 1101

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3093425, upper bound: 0.3208585
time: 3.70 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3093425, upper bound: 0.3208585
time: 4.22 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -19.7812862, -18.2213650, -19.7814636, -18.2222900, -0.7999744, 0.8006384
1: -16.7529755, -15.4863758, -16.7284927, -15.4635954, -0.6430402, 0.6172581
2: -11.4443865, -10.3847990, -11.4492760, -10.3893099, -0.5900128, 0.5936167
3: -10.8739176, -9.8273172, -10.8677444, -9.8228045, -0.4863670, 0.4787086
4: -2.1251965, -1.3099283, -2.1181488, -1.3068371, -0.4850056, 0.4826508
5: -10.2661734, -9.3303528, -10.2593746, -9.3221235, -0.4729998, 0.4632494
6: -19.7622757, -18.5426140, -19.7642155, -18.5443497, -0.5091019, 0.5125273
7: -3.0062175, -2.2031384, -2.9937968, -2.1974173, -0.4789264, 0.4616880
8: -1.9039264, -1.1302276, -1.9103451, -1.1448746, -0.5126715, 0.5229781
9: 5.6410561, 6.5289207, 5.6436825, 6.5455670, -0.4682792, 0.4575582

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1678
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2129
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 1248

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3104418, upper bound: 0.2612523
time: 3.18 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3166101, upper bound: 0.3049522
time: 3.74 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -19.7821236, -18.2222900, -19.7823353, -18.2222900, -0.8005984, 0.8007255
1: -16.7284927, -15.4638252, -16.7284927, -15.4543095, -0.6658545, 0.6090438
2: -11.4494514, -10.3892241, -11.4512243, -10.3891983, -0.5857110, 0.6009336
3: -10.8678131, -9.8231716, -10.8678341, -9.8208590, -0.4876645, 0.4793916
4: -2.1188254, -1.3078673, -2.1190419, -1.3056176, -0.4918157, 0.4794495
5: -10.2593746, -9.3216591, -10.2593746, -9.3186073, -0.4828215, 0.4561781
6: -19.7641888, -18.5443287, -19.7649994, -18.5443192, -0.5102508, 0.5119385
7: -2.9937968, -2.1988833, -2.9937968, -2.1950381, -0.4686100, 0.4730752
8: -1.9082932, -1.1443114, -1.9133396, -1.1441288, -0.5081215, 0.5284624
9: 5.6432228, 6.5502934, 5.6430740, 6.5529265, -0.4864148, 0.4548924

Time for backsubstitution: 22.37 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1678
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2129
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.52 seconds

### Candidate
type: B, layer: 3, pos: 1101

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3093418, upper bound: 0.3208586
time: 3.84 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3093418, upper bound: 0.3216141
time: 3.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.42 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3104425, upper bound: 0.2606582
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3166108, upper bound: 0.3043502
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3093425, upper bound: 0.3202563
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3093425, upper bound: 0.3210112
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3110446, upper bound: 0.2606582
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3172135, upper bound: 0.3043502
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3099414, upper bound: 0.3202563
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3099414, upper bound: 0.3210112
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3104425, upper bound: 0.2612522
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3166108, upper bound: 0.3049520
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3093425, upper bound: 0.3208585
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3093425, upper bound: 0.3208585
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3104418, upper bound: 0.2612523
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3166101, upper bound: 0.3049522
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3093418, upper bound: 0.3208586
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.42
Output dim: 9, lower bound: -0.3093418, upper bound: 0.3216141

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -19.7707195, -18.2286510, -19.7519608, -18.2293510, -0.7901514, 0.7707655
1: -16.7437840, -15.4869471, -16.6979713, -15.4454727, -0.6350834, 0.5753555
2: -11.4404650, -10.3909454, -11.4526062, -10.4086399, -0.5574136, 0.5704653
3: -10.8684177, -9.8296499, -10.8548403, -9.8236084, -0.4780973, 0.4609170
4: -2.1119483, -1.3105998, -2.0841238, -1.2932267, -0.4665923, 0.4331923
5: -10.2612915, -9.3373852, -10.2445478, -9.3235111, -0.4540145, 0.4375006
6: -19.7576008, -18.5494919, -19.7540359, -18.5479431, -0.5094042, 0.5041009
7: -3.0061274, -2.2165394, -3.0092278, -2.2361903, -0.4214835, 0.4440620
8: -1.9021091, -1.1324296, -1.9140677, -1.1527338, -0.4972885, 0.5269918
9: 5.6466780, 6.5266685, 5.6644077, 6.5568628, -0.4696822, 0.4280547

Time for backsubstitution: 21.81 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 3, pos: 1101

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3104425, upper bound: 0.2606582
time: 4.12 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3104425, upper bound: 0.2606583
time: 3.96 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -19.7794228, -18.2286510, -19.7756958, -18.2295799, -0.7970855, 0.7933230
1: -16.7523956, -15.4865217, -16.7208366, -15.4547234, -0.6572537, 0.5875936
2: -11.4415169, -10.3853369, -11.4480705, -10.3946228, -0.5547552, 0.5939894
3: -10.8715487, -9.8279982, -10.8630171, -9.8219709, -0.4870317, 0.4691156
4: -2.1206243, -1.3105998, -2.1039424, -1.3062899, -0.4856224, 0.4531782
5: -10.2651567, -9.3359737, -10.2552910, -9.3246346, -0.4743152, 0.4350365
6: -19.7608681, -18.5494804, -19.7615738, -18.5511932, -0.5068723, 0.5140402
7: -3.0061274, -2.2037203, -2.9937034, -2.2033799, -0.4236093, 0.4614239
8: -1.9036922, -1.1305389, -1.9126911, -1.1476374, -0.5085340, 0.5268440
9: 5.6415844, 6.5266685, 5.6478348, 6.5506744, -0.4756470, 0.4460704

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 1101

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3166108, upper bound: 0.3043502
time: 3.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3166109, upper bound: 0.3043502
time: 3.97 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -19.7802391, -18.2295799, -19.7794228, -18.2286510, -0.7977846, 0.7970853
1: -16.7279129, -15.4639778, -16.7523956, -15.4865217, -0.6169920, 0.6555181
2: -11.4465771, -10.3897629, -11.4415169, -10.3853369, -0.5954404, 0.5888691
3: -10.8654442, -9.8238487, -10.8715487, -9.8279982, -0.4778092, 0.4871805
4: -2.1142533, -1.3085387, -2.1206243, -1.3105998, -0.4810679, 0.4848094
5: -10.2583523, -9.3272800, -10.2651567, -9.3359737, -0.4611335, 0.4766487
6: -19.7627831, -18.5511932, -19.7608681, -18.5494804, -0.5105165, 0.5069075
7: -2.9937034, -2.1994653, -3.0061274, -2.2037203, -0.4614239, 0.4762394
8: -1.9080586, -1.1446199, -1.9036922, -1.1305389, -0.5250020, 0.5124962
9: 5.6437664, 6.5480433, 5.6415844, 6.5266685, -0.4566448, 0.4741968

Time for backsubstitution: 22.22 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.46 seconds

### Candidate
type: A, layer: 3, pos: 1248

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2606579, upper bound: 0.3104422
time: 3.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3043500, upper bound: 0.3166105
time: 3.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -19.7802391, -18.2295799, -19.7802391, -18.2295799, -0.7977064, 0.7977061
1: -16.7279129, -15.4639778, -16.7279129, -15.4639778, -0.6087806, 0.6087806
2: -11.4465771, -10.3897629, -11.4465771, -10.3897629, -0.5845287, 0.5845284
3: -10.8654442, -9.8238487, -10.8654442, -9.8238487, -0.4784093, 0.4784093
4: -2.1142533, -1.3085387, -2.1142533, -1.3085387, -0.4774454, 0.4774456
5: -10.2583523, -9.3272800, -10.2583523, -9.3272800, -0.4540632, 0.4540632
6: -19.7627831, -18.5511932, -19.7627831, -18.5511932, -0.5080308, 0.5080309
7: -2.9937034, -2.1994653, -2.9937034, -2.1994653, -0.4728115, 0.4728115
8: -1.9080586, -1.1446199, -1.9080586, -1.1446199, -0.5075569, 0.5075567
9: 5.6437664, 6.5480433, 5.6437664, 6.5480433, -0.4536772, 0.4536772

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 1101

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3202561, upper bound: 0.3093428
time: 3.33 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3210110, upper bound: 0.3210110
time: 3.25 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -19.7707195, -18.2286510, -19.7538338, -18.2220650, -0.7974260, 0.7726796
1: -16.7437840, -15.4869471, -16.6985512, -15.4453259, -0.6352456, 0.5759587
2: -11.4404650, -10.3909454, -11.4555130, -10.4081011, -0.5579634, 0.5733919
3: -10.8684177, -9.8296499, -10.8572063, -9.8229628, -0.4787464, 0.4632889
4: -2.1119483, -1.3105998, -2.0887146, -1.2925551, -0.4672759, 0.4378581
5: -10.2612915, -9.3373852, -10.2455702, -9.3178463, -0.4596967, 0.4385095
6: -19.7576008, -18.5494919, -19.7554283, -18.5410805, -0.5162601, 0.5054839
7: -3.0061274, -2.2165394, -3.0093193, -2.2355952, -0.4220741, 0.4441535
8: -1.9021091, -1.1324296, -1.9142756, -1.1524363, -0.4976163, 0.5271962
9: 5.6466780, 6.5266685, 5.6638527, 6.5591135, -0.4719406, 0.4286413

Time for backsubstitution: 21.83 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.27 + 563.45 = 621.72 seconds
