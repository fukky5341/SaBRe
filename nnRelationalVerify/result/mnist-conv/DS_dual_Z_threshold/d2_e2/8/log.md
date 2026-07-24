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
execution time: IAR + RelationalAnalysis = 22.25 + 33.12 = 55.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.3263090, upper bound: 0.3263092

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4600

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 4600

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3263059, upper bound: 0.3257025
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3257023, upper bound: 0.3263062
time: 3.21 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.82 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.82
Output dim: 9, lower bound: -0.3263059, upper bound: 0.3257025
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.82
Output dim: 9, lower bound: -0.3257023, upper bound: 0.3263062

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.8007255, 0.7985499
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6656971, 0.6658542
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.6000938, 0.6009429
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4869807, 0.4876847
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4907949, 0.4919190
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4828219, 0.4811494
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5119451, 0.5098975
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4686098, 0.4684741
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5285563, 0.5285192
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4864887, 0.4858084

Time for backsubstitution: 20.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3202123, upper bound: 0.3209272
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3215308, upper bound: 0.3196088
time: 5.91 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7985501, 0.8007255
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6658545, 0.6656971
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.6009426, 0.6000938
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4876847, 0.4869807
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4919188, 0.4907949
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4811494, 0.4828219
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5098976, 0.5119451
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4684741, 0.4686100
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5285187, 0.5285563
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4858083, 0.4864888

Time for backsubstitution: 20.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3196087, upper bound: 0.3215307
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3209273, upper bound: 0.3202122
time: 4.93 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.34 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.34
Output dim: 9, lower bound: -0.3202123, upper bound: 0.3209272
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.34
Output dim: 9, lower bound: -0.3215308, upper bound: 0.3196088
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.34
Output dim: 9, lower bound: -0.3196087, upper bound: 0.3215307
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.34
Output dim: 9, lower bound: -0.3209273, upper bound: 0.3202122

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.8008921, 0.7988687
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6157165, 0.6185639
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5650759, 0.5655062
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4744020, 0.4760672
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4688745, 0.4662457
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4558225, 0.4529366
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5140729, 0.5134021
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4211216, 0.4148743
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5241847, 0.5263593
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4709312, 0.4705660

Time for backsubstitution: 20.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1101

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3148089, upper bound: 0.2996035
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3056377, upper bound: 0.3169676
time: 3.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.8010442, 0.7987165
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6184068, 0.6158736
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5646572, 0.5659251
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4753633, 0.4751061
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4651215, 0.4699986
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4546089, 0.4541500
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5154495, 0.5120252
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4150102, 0.4209859
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5263968, 0.5241475
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4712464, 0.4702506

Time for backsubstitution: 21.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1101

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3175756, upper bound: 0.3050346
time: 4.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3002067, upper bound: 0.3142057
time: 3.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7987168, 0.8010442
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6158736, 0.6184068
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5659251, 0.5646572
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4751062, 0.4753633
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4699986, 0.4651217
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4541502, 0.4546089
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5120254, 0.5154496
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4209859, 0.4150102
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5241475, 0.5263968
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4702507, 0.4712465

Time for backsubstitution: 21.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1101

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3142054, upper bound: 0.3002069
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3050344, upper bound: 0.3175759
time: 3.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7988689, 0.8008919
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6185641, 0.6157165
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5655062, 0.5650759
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4760671, 0.4744021
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4662457, 0.4688745
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4529366, 0.4558223
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5134020, 0.5140728
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4148743, 0.4211214
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5263591, 0.5241849
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4705659, 0.4709311

Time for backsubstitution: 21.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1101

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3169674, upper bound: 0.3056379
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2996033, upper bound: 0.3148092
time: 3.31 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.41 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.41
Output dim: 9, lower bound: -0.3148089, upper bound: 0.2996035
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.41
Output dim: 9, lower bound: -0.3056377, upper bound: 0.3169676
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.41
Output dim: 9, lower bound: -0.3175756, upper bound: 0.3050346
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.41
Output dim: 9, lower bound: -0.3002067, upper bound: 0.3142057
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.41
Output dim: 9, lower bound: -0.3142054, upper bound: 0.3002069
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.41
Output dim: 9, lower bound: -0.3050344, upper bound: 0.3175759
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.41
Output dim: 9, lower bound: -0.3169674, upper bound: 0.3056379
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.41
Output dim: 9, lower bound: -0.2996033, upper bound: 0.3148092

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.8007042, 0.7985642
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6089265, 0.6092339
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5851169, 0.5857236
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4786978, 0.4804564
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4783258, 0.4815276
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4562223, 0.4565358
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5106342, 0.5082104
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4731388, 0.4750829
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5128233, 0.5080981
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4548926, 0.4499440

Time for backsubstitution: 20.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Candidate
type: DSZ, layer: 3, pos: 2383

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3054875, upper bound: 0.2977512
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3131706, upper bound: 0.2914453
time: 3.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.8007395, 0.7985287
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6090765, 0.6090837
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5848746, 0.5859656
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4797525, 0.4794017
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4804034, 0.4794497
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4582081, 0.4545500
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5102580, 0.5085866
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4752185, 0.4730031
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5081351, 0.5127859
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4506245, 0.4542122

Time for backsubstitution: 21.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Candidate
type: DSZ, layer: 3, pos: 2383

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2973370, upper bound: 0.3153674
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3038194, upper bound: 0.3070327
time: 3.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.8007042, 0.7985642
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6089265, 0.6092339
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5851169, 0.5857236
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4786978, 0.4804564
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4783258, 0.4815276
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4562223, 0.4565358
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5106342, 0.5082104
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4731388, 0.4750829
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5128233, 0.5080981
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4548926, 0.4499440

Time for backsubstitution: 21.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Candidate
type: DSZ, layer: 3, pos: 2383

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3076363, upper bound: 0.3032162
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3159754, upper bound: 0.2967338
time: 5.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.8007395, 0.7985287
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6090765, 0.6090837
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5848746, 0.5859656
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4797525, 0.4794017
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4804034, 0.4794497
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4582081, 0.4545500
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5102580, 0.5085866
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4752185, 0.4730031
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5081351, 0.5127859
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4506245, 0.4542122

Time for backsubstitution: 21.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Candidate
type: DSZ, layer: 3, pos: 2383

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2920483, upper bound: 0.3125671
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2983545, upper bound: 0.3048845
time: 3.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7985289, 0.8007395
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6090837, 0.6090767
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5859656, 0.5848746
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4794016, 0.4797524
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4794497, 0.4804034
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4545500, 0.4582081
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5085866, 0.5102581
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4730031, 0.4752185
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5127861, 0.5081353
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4542122, 0.4506245

Time for backsubstitution: 20.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Candidate
type: DSZ, layer: 3, pos: 2383

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3048843, upper bound: 0.2983545
time: 7.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3125671, upper bound: 0.2920484
time: 3.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7985642, 0.8007042
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6092339, 0.6089265
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5857234, 0.5851166
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4804564, 0.4786978
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4815273, 0.4783258
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4565358, 0.4562223
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5082104, 0.5106343
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4750829, 0.4731388
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5080979, 0.5128233
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4499440, 0.4548926

Time for backsubstitution: 21.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Candidate
type: DSZ, layer: 3, pos: 2383

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2967339, upper bound: 0.3159757
time: 3.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3032160, upper bound: 0.3076366
time: 3.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7985289, 0.8007395
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6090837, 0.6090767
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5859656, 0.5848746
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4794016, 0.4797524
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4794497, 0.4804034
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4545500, 0.4582081
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5085866, 0.5102581
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4730031, 0.4752185
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5127861, 0.5081353
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4542122, 0.4506245

Time for backsubstitution: 21.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Candidate
type: DSZ, layer: 3, pos: 2383

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3070325, upper bound: 0.3038196
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3153671, upper bound: 0.2973372
time: 3.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7985642, 0.8007042
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6092339, 0.6089265
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5857234, 0.5851166
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4804564, 0.4786978
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4815273, 0.4783258
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4565358, 0.4562223
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5082104, 0.5106343
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4750829, 0.4731388
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5080979, 0.5128233
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4499440, 0.4548926

Time for backsubstitution: 20.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Candidate
type: DSZ, layer: 3, pos: 2383

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2914451, upper bound: 0.3131706
time: 6.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2977510, upper bound: 0.3054878
time: 3.22 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.87 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.3054875, upper bound: 0.2977512
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.3131706, upper bound: 0.2914453
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.2973370, upper bound: 0.3153674
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.3038194, upper bound: 0.3070327
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.3076363, upper bound: 0.3032162
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.3159754, upper bound: 0.2967338
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.2920483, upper bound: 0.3125671
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.2983545, upper bound: 0.3048845
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.3048843, upper bound: 0.2983545
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.3125671, upper bound: 0.2920484
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.2967339, upper bound: 0.3159757
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.3032160, upper bound: 0.3076366
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.3070325, upper bound: 0.3038196
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.3153671, upper bound: 0.2973372
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.2914451, upper bound: 0.3131706
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.87
Output dim: 9, lower bound: -0.2977510, upper bound: 0.3054878

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7592688, 0.7365670
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6335282, 0.6262672
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5113864, 0.5194440
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4758573, 0.4824461
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4894052, 0.4896178
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4683659, 0.4682091
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5083736, 0.5041275
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4661317, 0.4641602
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.4893191, 0.4786792
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4682121, 0.4643228

Time for backsubstitution: 20.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Candidate
type: DSZ, layer: 3, pos: 1101

### Candidate
type: DSZ, layer: 3, pos: 2384

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3048551, upper bound: 0.2836135
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3048551, upper bound: 0.2836135
time: 3.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7387424, 0.7570932
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6261096, 0.6336856
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5185940, 0.5122356
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4817424, 0.4765608
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4884939, 0.4905293
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4698815, 0.4666936
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5061752, 0.5063262
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4642959, 0.4659960
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.4787164, 0.4892812
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4650033, 0.4675314

Time for backsubstitution: 20.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Candidate
type: DSZ, layer: 3, pos: 1101

### Candidate
type: DSZ, layer: 3, pos: 2384

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2895140, upper bound: 0.3070927
time: 5.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2895140, upper bound: 0.3070927
time: 5.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7592688, 0.7365670
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6335282, 0.6262672
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5113864, 0.5194440
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4758573, 0.4824461
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4894052, 0.4896178
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4683659, 0.4682091
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5083736, 0.5041275
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4661317, 0.4641602
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.4893191, 0.4786792
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4682121, 0.4643228

Time for backsubstitution: 20.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Candidate
type: DSZ, layer: 3, pos: 1101

### Candidate
type: DSZ, layer: 3, pos: 2384

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3076958, upper bound: 0.2889112
time: 4.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3076958, upper bound: 0.2889112
time: 4.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7387424, 0.7570932
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6261096, 0.6336856
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5185940, 0.5122356
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4817424, 0.4765608
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4884939, 0.4905293
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4698815, 0.4666936
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5061752, 0.5063262
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4642959, 0.4659960
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.4787164, 0.4892812
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4650033, 0.4675314

Time for backsubstitution: 20.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Candidate
type: DSZ, layer: 3, pos: 1101

### Candidate
type: DSZ, layer: 3, pos: 2384

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2842162, upper bound: 0.3042521
time: 4.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2842162, upper bound: 0.3042521
time: 4.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7570930, 0.7387426
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6336856, 0.6261096
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5122356, 0.5185940
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4765606, 0.4817421
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4905291, 0.4884939
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4666936, 0.4698814
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5063261, 0.5061752
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4659960, 0.4642959
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.4892812, 0.4787164
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4675314, 0.4650033

Time for backsubstitution: 20.87 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.38 + 552.16 = 607.54 seconds
