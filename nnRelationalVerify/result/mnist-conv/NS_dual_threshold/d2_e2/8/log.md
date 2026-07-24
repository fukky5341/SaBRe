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
execution time: IAR + RelationalAnalysis = 23.14 + 33.30 = 56.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.3263090, upper bound: 0.3263092

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4600
type: B, layer: 1, pos: 4600

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4600

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3263059, upper bound: 0.3257033
time: 3.31 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3263059, upper bound: 0.3263060
time: 3.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.60 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.60
Output dim: 9, lower bound: -0.3263059, upper bound: 0.3257033
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.60
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

Time for backsubstitution: 20.73 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1678
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1445
type: A, layer: 3, pos: 1445
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 1404
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 1990
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 1101

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3208584, upper bound: 0.3093427
time: 3.39 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3216138, upper bound: 0.3210110
time: 3.29 seconds

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

Time for backsubstitution: 21.61 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1678
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1445
type: A, layer: 3, pos: 1445
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 1404
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 1990
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 1101

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3208584, upper bound: 0.3099417
time: 3.36 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3216138, upper bound: 0.3216136
time: 4.10 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.47 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 29.47
Output dim: 9, lower bound: -0.3208584, upper bound: 0.3093427
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 29.47
Output dim: 9, lower bound: -0.3216138, upper bound: 0.3210110
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 29.47
Output dim: 9, lower bound: -0.3208584, upper bound: 0.3099417
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 29.47
Output dim: 9, lower bound: -0.3216138, upper bound: 0.3216136

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -19.7794228, -18.2286510, -19.7813683, -18.2242451, -0.8023911, 0.7995677
1: -16.7523956, -15.4865217, -16.7283325, -15.4636059, -0.6429400, 0.6174333
2: -11.4415169, -10.3853369, -11.4485264, -10.3893185, -0.5893826, 0.5945063
3: -10.8715487, -9.8279982, -10.8671103, -9.8228312, -0.4860644, 0.4794754
4: -2.1206243, -1.3105998, -2.1169474, -1.3068374, -0.4837888, 0.4841564
5: -10.2651567, -9.3359737, -10.2593727, -9.3236046, -0.4749305, 0.4621499
6: -19.7608681, -18.5494804, -19.7642059, -18.5461884, -0.5118897, 0.5117003
7: -3.0061274, -2.2037203, -2.9937968, -2.1975629, -0.4790955, 0.4615157
8: -1.9036922, -1.1305389, -1.9103274, -1.1449542, -0.5124555, 0.5227313
9: 5.6415844, 6.5266685, 5.6437263, 6.5449624, -0.4687765, 0.4569798

Time for backsubstitution: 21.57 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1678
type: B, layer: 3, pos: 1445
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 2129
type: B, layer: 3, pos: 2129
type: B, layer: 3, pos: 1404
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 1241

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3153784, upper bound: 0.3023805
time: 3.56 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3153899, upper bound: 0.3053366
time: 3.90 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -19.7802391, -18.2295799, -19.7822361, -18.2242451, -0.8030121, 0.7996573
1: -16.7279129, -15.4639778, -16.7283325, -15.4543200, -0.6657553, 0.6092224
2: -11.4465771, -10.3897629, -11.4504757, -10.3892050, -0.5850806, 0.6018243
3: -10.8654442, -9.8238487, -10.8671989, -9.8208866, -0.4873598, 0.4801600
4: -2.1142533, -1.3085387, -2.1178424, -1.3056178, -0.4905938, 0.4809542
5: -10.2583523, -9.3272800, -10.2593727, -9.3200884, -0.4847546, 0.4550798
6: -19.7627831, -18.5511932, -19.7649879, -18.5461559, -0.5130398, 0.5111110
7: -2.9937034, -2.1994653, -2.9937968, -2.1951816, -0.4687788, 0.4729028
8: -1.9080586, -1.1446199, -1.9133224, -1.1442099, -0.5079083, 0.5282128
9: 5.6437664, 6.5480433, 5.6431189, 6.5523229, -0.4869095, 0.4543171

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1678
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: B, layer: 3, pos: 1404
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: B, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 1101

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3099415, upper bound: 0.3202563
time: 3.52 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3099415, upper bound: 0.3210112
time: 3.61 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -19.7812862, -18.2213650, -19.7814598, -18.2222900, -0.8062518, 0.8006392
1: -16.7529755, -15.4863758, -16.7284927, -15.4635954, -0.6430399, 0.6177704
2: -11.4443865, -10.3847990, -11.4492779, -10.3893118, -0.5900133, 0.5958459
3: -10.8739176, -9.8273172, -10.8677464, -9.8228035, -0.4863670, 0.4807969
4: -2.1251965, -1.3099283, -2.1181505, -1.3068371, -0.4850055, 0.4860435
5: -10.2661734, -9.3303528, -10.2593746, -9.3221216, -0.4774897, 0.4632493
6: -19.7622757, -18.5426140, -19.7642155, -18.5443497, -0.5151466, 0.5125276
7: -3.0062175, -2.2031384, -2.9937968, -2.1974163, -0.4793324, 0.4616880
8: -1.9039264, -1.1302276, -1.9103432, -1.1448746, -0.5127852, 0.5229783
9: 5.6410561, 6.5289207, 5.6436825, 6.5455661, -0.4699519, 0.4575584

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1678
type: B, layer: 3, pos: 1445
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: A, layer: 3, pos: 2129
type: B, layer: 3, pos: 2129
type: B, layer: 3, pos: 1404
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 1241

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3153784, upper bound: 0.3029789
time: 3.44 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3153899, upper bound: 0.3059358
time: 3.43 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -19.7821236, -18.2222900, -19.7823353, -18.2222900, -0.8068986, 0.8007255
1: -16.7284927, -15.4638252, -16.7284927, -15.4543056, -0.6658545, 0.6095614
2: -11.4494514, -10.3892241, -11.4512272, -10.3891983, -0.5857115, 0.6031668
3: -10.8678131, -9.8231716, -10.8678350, -9.8208580, -0.4876645, 0.4814805
4: -2.1188254, -1.3078673, -2.1190450, -1.3056176, -0.4918158, 0.4828422
5: -10.2593746, -9.3216591, -10.2593746, -9.3186054, -0.4873126, 0.4561779
6: -19.7641888, -18.5443287, -19.7650013, -18.5443192, -0.5162953, 0.5119386
7: -2.9937968, -2.1988833, -2.9937968, -2.1950381, -0.4690161, 0.4730752
8: -1.9082932, -1.1443114, -1.9133406, -1.1441293, -0.5082347, 0.5284624
9: 5.6432228, 6.5502934, 5.6430740, 6.5529280, -0.4881051, 0.4548926

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1678
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: B, layer: 3, pos: 1404
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: B, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 1101

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3099415, upper bound: 0.3208586
time: 3.55 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3099415, upper bound: 0.3216140
time: 4.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.67 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 29.67
Output dim: 9, lower bound: -0.3153784, upper bound: 0.3023805
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 29.67
Output dim: 9, lower bound: -0.3153899, upper bound: 0.3053366
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.67
Output dim: 9, lower bound: -0.3099415, upper bound: 0.3202563
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.67
Output dim: 9, lower bound: -0.3099415, upper bound: 0.3210112
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 29.67
Output dim: 9, lower bound: -0.3153784, upper bound: 0.3029789
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 29.67
Output dim: 9, lower bound: -0.3153899, upper bound: 0.3059358
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.67
Output dim: 9, lower bound: -0.3099415, upper bound: 0.3208586
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.67
Output dim: 9, lower bound: -0.3099415, upper bound: 0.3216140

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -19.7792587, -18.2286510, -19.7816925, -18.2240829, -0.8021348, 0.7993081
1: -16.7479687, -15.4865713, -16.7162170, -15.4366856, -0.6604986, 0.6005116
2: -11.4362974, -10.3853569, -11.4373512, -10.3746881, -0.5797341, 0.5731168
3: -10.8685980, -9.8280010, -10.8575420, -9.8176136, -0.4896498, 0.4703107
4: -2.1199722, -1.3105998, -2.1160982, -1.3002616, -0.4880568, 0.4812796
5: -10.2651567, -9.3378382, -10.2697039, -9.3272896, -0.4647572, 0.4538643
6: -19.7498055, -18.5494881, -19.7368050, -18.5263844, -0.4821262, 0.4586666
7: -3.0061274, -2.2141078, -2.9953432, -2.2282357, -0.4511237, 0.4509399
8: -1.9036875, -1.1306958, -1.9139843, -1.1446676, -0.5124071, 0.5277758
9: 5.6469383, 6.5266685, 5.6591997, 6.5631166, -0.4788659, 0.4413846

Time for backsubstitution: 21.57 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1678
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: B, layer: 3, pos: 1404
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1101

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3067723, upper bound: 0.3023806
time: 3.89 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3067723, upper bound: 0.3023804
time: 4.16 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -19.7794228, -18.2286510, -19.7821770, -18.2242451, -0.8023911, 0.7999635
1: -16.7523956, -15.4865217, -16.7212753, -15.4543371, -0.6587493, 0.6021323
2: -11.4415169, -10.3853369, -11.4461536, -10.3892088, -0.5894184, 0.5751879
3: -10.8715487, -9.8279982, -10.8650703, -9.8208904, -0.4890575, 0.4757794
4: -2.1206243, -1.3105998, -2.1161978, -1.3056178, -0.4863073, 0.4826112
5: -10.2651567, -9.3359737, -10.2593727, -9.3239975, -0.4598482, 0.4621499
6: -19.7608681, -18.5494804, -19.7581654, -18.5461597, -0.5119150, 0.4498335
7: -3.0061274, -2.2037203, -2.9937968, -2.1990900, -0.4658921, 0.4615157
8: -1.9036922, -1.1305389, -1.9133186, -1.1442637, -0.5124016, 0.5278327
9: 5.6415844, 6.5266685, 5.6482487, 6.5523229, -0.4772843, 0.4455209

Time for backsubstitution: 21.77 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1678
type: B, layer: 3, pos: 1445
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: B, layer: 3, pos: 1404
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 1101

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3068337, upper bound: 0.3053366
time: 3.54 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3068337, upper bound: 0.3053366
time: 3.50 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -19.7802391, -18.2295799, -19.7811890, -18.2233162, -0.8030903, 0.7988927
1: -16.7279129, -15.4639778, -16.7528152, -15.4863901, -0.6171536, 0.6559598
2: -11.4465771, -10.3897629, -11.4436378, -10.3848057, -0.5959804, 0.5909264
3: -10.8654442, -9.8238487, -10.8732834, -9.8273439, -0.4784679, 0.4889098
4: -2.1142533, -1.3085387, -2.1239953, -1.3099284, -0.4817520, 0.4882171
5: -10.2583523, -9.3272800, -10.2661743, -9.3318319, -0.4651778, 0.4776654
6: -19.7627831, -18.5511932, -19.7622662, -18.5444527, -0.5155196, 0.5082949
7: -2.9937034, -2.1994653, -3.0062175, -2.2032838, -0.4618566, 0.4763308
8: -1.9080586, -1.1446199, -1.9039092, -1.1303072, -0.5252497, 0.5127127
9: 5.6437664, 6.5480433, 5.6410942, 6.5283155, -0.4582827, 0.4747506

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1678
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 1404
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 1241

## Relational analysis of NS_A1_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3029786, upper bound: 0.3147758
time: 5.74 seconds

## Relational analysis of NS_A1_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3059356, upper bound: 0.3147874
time: 3.80 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -19.7802391, -18.2295799, -19.7820225, -18.2242451, -0.8030121, 0.7995300
1: -16.7279129, -15.4639778, -16.7283325, -15.4638376, -0.6089432, 0.6092224
2: -11.4465771, -10.3897629, -11.4487000, -10.3892317, -0.5850711, 0.5865910
3: -10.8654442, -9.8238487, -10.8671780, -9.8231964, -0.4790668, 0.4801378
4: -2.1142533, -1.3085387, -2.1176257, -1.3078663, -0.4781300, 0.4808593
5: -10.2583523, -9.3272800, -10.2593727, -9.3231392, -0.4581106, 0.4550798
6: -19.7627831, -18.5511932, -19.7641792, -18.5461655, -0.5130332, 0.5094172
7: -2.9937034, -2.1994653, -2.9937968, -2.1990292, -0.4732444, 0.4729028
8: -1.9080586, -1.1446199, -1.9082756, -1.1443906, -0.5078025, 0.5077708
9: 5.6437664, 6.5480433, 5.6432667, 6.5496898, -0.4553149, 0.4542437

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1678
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1445
type: A, layer: 3, pos: 1445
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 1404
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 1990
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 1101

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3208584, upper bound: 0.3093428
time: 3.33 seconds

## Relational analysis of NS_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3216138, upper bound: 0.3210110
time: 3.34 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -19.7811222, -18.2213650, -19.7817917, -18.2221298, -0.8059931, 0.8003786
1: -16.7485542, -15.4864292, -16.7163754, -15.4366741, -0.6605966, 0.6008482
2: -11.4391699, -10.3848219, -11.4381046, -10.3746796, -0.5803638, 0.5744567
3: -10.8709698, -9.8273201, -10.8581791, -9.8175869, -0.4899526, 0.4716326
4: -2.1245456, -1.3099283, -2.1173000, -1.3002623, -0.4892793, 0.4831681
5: -10.2661734, -9.3322201, -10.2697067, -9.3258104, -0.4673157, 0.4549630
6: -19.7512169, -18.5426216, -19.7368126, -18.5245476, -0.4853829, 0.4594936
7: -3.0062175, -2.2135229, -2.9953432, -2.2280884, -0.4513614, 0.4511123
8: -1.9039240, -1.1303849, -1.9140019, -1.1445866, -0.5127368, 0.5280230
9: 5.6464081, 6.5289207, 5.6591530, 6.5637217, -0.4800403, 0.4419627

Time for backsubstitution: 21.90 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1678
type: B, layer: 3, pos: 1445
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: B, layer: 3, pos: 1404
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1101

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3067723, upper bound: 0.3029789
time: 4.09 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3067723, upper bound: 0.3029789
time: 3.66 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -19.7812862, -18.2213650, -19.7822762, -18.2222900, -0.8062518, 0.8010316
1: -16.7529755, -15.4863758, -16.7214355, -15.4543228, -0.6588478, 0.6024683
2: -11.4443865, -10.3847990, -11.4469032, -10.3892031, -0.5900488, 0.5765259
3: -10.8739176, -9.8273172, -10.8657084, -9.8208628, -0.4893606, 0.4771018
4: -2.1251965, -1.3099283, -2.1174006, -1.3056176, -0.4875245, 0.4844999
5: -10.2661734, -9.3303528, -10.2593746, -9.3225174, -0.4624062, 0.4632493
6: -19.7622757, -18.5426140, -19.7581730, -18.5443211, -0.5151718, 0.4506598
7: -3.0062175, -2.2031384, -2.9937968, -2.1989450, -0.4661303, 0.4616880
8: -1.9039264, -1.1302276, -1.9133368, -1.1441860, -0.5127323, 0.5280807
9: 5.6410561, 6.5289207, 5.6482010, 6.5529280, -0.4784598, 0.4460976

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1678
type: B, layer: 3, pos: 1445
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: B, layer: 3, pos: 1404
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 1101

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3068337, upper bound: 0.3059358
time: 3.44 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3068337, upper bound: 0.3059358
time: 3.34 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -19.7821236, -18.2222900, -19.7812881, -18.2213573, -0.8069763, 0.7999752
1: -16.7284927, -15.4638252, -16.7529755, -15.4863758, -0.6172578, 0.6562943
2: -11.4494514, -10.3892241, -11.4443884, -10.3847990, -0.5966125, 0.5922673
3: -10.8678131, -9.8231716, -10.8739185, -9.8273172, -0.4787717, 0.4902307
4: -2.1188254, -1.3078673, -2.1251988, -1.3099283, -0.4829745, 0.4901037
5: -10.2593746, -9.3216591, -10.2661734, -9.3303509, -0.4677346, 0.4787626
6: -19.7641888, -18.5443287, -19.7622776, -18.5426159, -0.5187758, 0.5091226
7: -2.9937968, -2.1988833, -3.0062175, -2.2031384, -0.4620943, 0.4765024
8: -1.9082932, -1.1443114, -1.9039278, -1.1302285, -0.5255780, 0.5129614
9: 5.6432228, 6.5502934, 5.6410542, 6.5289197, -0.4594777, 0.4753377

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1678
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 1404
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1241

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3029786, upper bound: 0.3153779
time: 5.36 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3059356, upper bound: 0.3153897
time: 4.00 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -19.7821236, -18.2222900, -19.7821236, -18.2222900, -0.8068986, 0.8005984
1: -16.7284927, -15.4638252, -16.7284927, -15.4638262, -0.6090443, 0.6095614
2: -11.4494514, -10.3892241, -11.4494524, -10.3892241, -0.5857010, 0.5879340
3: -10.8678131, -9.8231716, -10.8678131, -9.8231707, -0.4793699, 0.4814584
4: -2.1188254, -1.3078673, -2.1188288, -1.3078673, -0.4793553, 0.4827478
5: -10.2593746, -9.3216591, -10.2593746, -9.3216553, -0.4606693, 0.4561779
6: -19.7641888, -18.5443287, -19.7641888, -18.5443249, -0.5162886, 0.5102443
7: -2.9937968, -2.1988833, -2.9937968, -2.1988819, -0.4734817, 0.4730752
8: -1.9082932, -1.1443114, -1.9082923, -1.1443119, -0.5081296, 0.5080154
9: 5.6432228, 6.5502934, 5.6432238, 6.5502968, -0.4565108, 0.4548199

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1678
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1445
type: A, layer: 3, pos: 1445
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 1404
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 1990
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1101

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3208584, upper bound: 0.3099417
time: 3.62 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3216138, upper bound: 0.3216136
time: 3.98 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.60 seconds
NS_A1_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3067723, upper bound: 0.3023806
NS_A1_A1_B1_B2, status: Status.VERIFIED, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3067723, upper bound: 0.3023804
NS_A1_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3068337, upper bound: 0.3053366
NS_A1_A1_B2_B2, status: Status.VERIFIED, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3068337, upper bound: 0.3053366
NS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3029786, upper bound: 0.3147758
NS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3059356, upper bound: 0.3147874
NS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3208584, upper bound: 0.3093428
NS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3216138, upper bound: 0.3210110
NS_A2_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3067723, upper bound: 0.3029789
NS_A2_A1_B1_B2, status: Status.VERIFIED, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3067723, upper bound: 0.3029789
NS_A2_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3068337, upper bound: 0.3059358
NS_A2_A1_B2_B2, status: Status.VERIFIED, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3068337, upper bound: 0.3059358
NS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3029786, upper bound: 0.3153779
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3059356, upper bound: 0.3153897
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3208584, upper bound: 0.3099417
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.60
Output dim: 9, lower bound: -0.3216138, upper bound: 0.3216136

## BFS NS instance: NS_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -19.7796993, -18.2294235, -19.7810230, -18.2233162, -0.8020389, 0.7986350
1: -16.7157936, -15.4451513, -16.7483883, -15.4864378, -0.6002312, 0.6591303
2: -11.4334564, -10.3752480, -11.4384184, -10.3848295, -0.5711126, 0.5812371
3: -10.8557854, -9.8204451, -10.8703337, -9.8273487, -0.4692202, 0.4896138
4: -2.1125054, -1.3028862, -2.1233442, -1.3099284, -0.4784595, 0.4903126
5: -10.2686853, -9.3345718, -10.2661743, -9.3336945, -0.4568918, 0.4611754
6: -19.7344685, -18.5314217, -19.7512074, -18.5444641, -0.4611329, 0.4785033
7: -2.9952521, -2.2327614, -3.0062175, -2.2136695, -0.4512811, 0.4456353
8: -1.9087205, -1.1450815, -1.9039044, -1.1304660, -0.5252237, 0.5122788
9: 5.6598415, 6.5588355, 5.6464491, 6.5283155, -0.4423866, 0.4774704

Time for backsubstitution: 20.91 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1678
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 1404
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 1101

## Relational analysis of NS_A1_A2_B1_A1_A1

### Relational analysis result of NS_A1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3029787, upper bound: 0.3061734
time: 4.16 seconds

## Relational analysis of NS_A1_A2_B1_A1_A2

### Relational analysis result of NS_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3029787, upper bound: 0.3147761
time: 5.30 seconds

## BFS NS instance: NS_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -19.7801838, -18.2295799, -19.7811890, -18.2233162, -0.8026929, 0.7988927
1: -16.7208519, -15.4639912, -16.7528152, -15.4863901, -0.6018512, 0.6559467
2: -11.4422541, -10.3897696, -11.4436378, -10.3848057, -0.5731838, 0.5909235
3: -10.8633156, -9.8238544, -10.8732834, -9.8273439, -0.4746895, 0.4889020
4: -2.1126103, -1.3085387, -2.1239953, -1.3099284, -0.4797912, 0.4882171
5: -10.2583523, -9.3310223, -10.2661743, -9.3318319, -0.4651778, 0.4562664
6: -19.7560291, -18.5511951, -19.7622662, -18.5444527, -0.4522988, 0.5082934
7: -2.9937034, -2.2033720, -3.0062175, -2.2032838, -0.4618566, 0.4604051
8: -1.9080534, -1.1446733, -1.9039092, -1.1303072, -0.5252450, 0.5122747
9: 5.6488924, 6.5480433, 5.6410942, 6.5283155, -0.4465257, 0.4747506

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1678
type: A, layer: 3, pos: 1678
type: A, layer: 3, pos: 1445
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 1404
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1101

## Relational analysis of NS_A1_A2_B1_A2_A1

### Relational analysis result of NS_A1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3059356, upper bound: 0.3062335
time: 4.60 seconds

## Relational analysis of NS_A1_A2_B1_A2_A2

### Relational analysis result of NS_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3059356, upper bound: 0.3147877
time: 4.36 seconds

## BFS NS instance: NS_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -19.7794228, -18.2286510, -19.7820225, -18.2242451, -0.8023911, 0.7996080
1: -16.7523956, -15.4865217, -16.7283325, -15.4638376, -0.6556773, 0.6174333
2: -11.4415169, -10.3853369, -11.4487000, -10.3892317, -0.5894117, 0.5975029
3: -10.8715487, -9.8279982, -10.8671780, -9.8231964, -0.4878378, 0.4795382
4: -2.1206243, -1.3105998, -2.1176257, -1.3078663, -0.4854943, 0.4844799
5: -10.2651567, -9.3359737, -10.2593727, -9.3231392, -0.4806955, 0.4621499
6: -19.7608681, -18.5494804, -19.7641792, -18.5461655, -0.5119100, 0.5119034
7: -3.0061274, -2.2037203, -2.9937968, -2.1990292, -0.4766717, 0.4615157
8: -1.9036922, -1.1305389, -1.9082756, -1.1443906, -0.5127451, 0.5252163
9: 5.6415844, 6.5266685, 5.6432667, 6.5496898, -0.4758345, 0.4572115

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 1241
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 2383
type: B, layer: 3, pos: 2383
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 1445
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1678
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1445
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2384
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: B, layer: 3, pos: 1404
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 2819
type: B, layer: 3, pos: 2819
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1990
type: A, layer: 3, pos: 3109
type: B, layer: 3, pos: 3109

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 1241

## Relational analysis of NS_A1_A2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3058822, upper bound: 0.3023805
time: 3.61 seconds

## Relational analysis of NS_A1_A2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3059358, upper bound: 0.3053363
time: 3.53 seconds

## BFS NS instance: NS_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -19.7802391, -18.2295799, -19.7820225, -18.2242451, -0.8030121, 0.7995300
1: -16.7279129, -15.4639778, -16.7283325, -15.4638376, -0.6089432, 0.6092224
2: -11.4465771, -10.3897629, -11.4487000, -10.3892317, -0.5850711, 0.5865910
3: -10.8654442, -9.8238487, -10.8671780, -9.8231964, -0.4790668, 0.4801378
4: -2.1142533, -1.3085387, -2.1176257, -1.3078663, -0.4781300, 0.4808593
5: -10.2583523, -9.3272800, -10.2593727, -9.3231392, -0.4581106, 0.4550798
6: -19.7627831, -18.5511932, -19.7641792, -18.5461655, -0.5130332, 0.5094172
7: -2.9937034, -2.1994653, -2.9937968, -2.1990292, -0.4732444, 0.4729028
8: -1.9080586, -1.1446199, -1.9082756, -1.1443906, -0.5078025, 0.5077708
9: 5.6437664, 6.5480433, 5.6432667, 6.5496898, -0.4553149, 0.4542437

Time for backsubstitution: 21.77 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 2383
type: A, layer: 3, pos: 2383
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1678
type: A, layer: 3, pos: 1678
type: B, layer: 3, pos: 1445
type: A, layer: 3, pos: 1445
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2384
type: B, layer: 3, pos: 2384
type: B, layer: 3, pos: 2129
type: A, layer: 3, pos: 2129
type: A, layer: 3, pos: 1404
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1450
type: A, layer: 3, pos: 1450
type: B, layer: 3, pos: 941
type: A, layer: 3, pos: 941
type: B, layer: 3, pos: 2819
type: A, layer: 3, pos: 2819
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 1990
type: B, layer: 3, pos: 1990
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 3109
type: A, layer: 3, pos: 3109

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 1101

## Relational analysis of NS_A1_A2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3099415, upper bound: 0.3202563
time: 3.57 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3216136, upper bound: 0.3210112
time: 3.86 seconds

## BFS NS instance: NS_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -19.7815742, -18.2221336, -19.7811203, -18.2213573, -0.8059187, 0.7997189
1: -16.7163754, -15.4450083, -16.7485504, -15.4864264, -0.6003361, 0.6594620
2: -11.4363289, -10.3747101, -11.4391718, -10.3848209, -0.5717444, 0.5825753
3: -10.8581562, -9.8197632, -10.8709698, -9.8273211, -0.4695239, 0.4909339
4: -2.1170812, -1.3022156, -2.1245468, -1.3099283, -0.4796860, 0.4922004
5: -10.2697067, -9.3289461, -10.2661734, -9.3322182, -0.4594486, 0.4622734
6: -19.7358723, -18.5245552, -19.7512169, -18.5426235, -0.4643879, 0.4793307
7: -2.9953432, -2.2321763, -3.0062175, -2.2135241, -0.4515185, 0.4458084
8: -1.9089537, -1.1447730, -1.9039235, -1.1303849, -0.5255516, 0.5125282
9: 5.6593027, 6.5610895, 5.6464081, 6.5289197, -0.4435811, 0.4780577

Time for backsubstitution: 21.79 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.44 + 561.46 = 617.90 seconds
