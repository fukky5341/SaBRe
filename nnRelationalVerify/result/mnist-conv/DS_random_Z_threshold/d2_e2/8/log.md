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
execution time: IAR + RelationalAnalysis = 24.98 + 33.78 = 58.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.3263090, upper bound: 0.3263092

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4600

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4600

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3263059, upper bound: 0.3257025
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3257023, upper bound: 0.3263062
time: 3.23 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.68 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.68
Output dim: 9, lower bound: -0.3263059, upper bound: 0.3257025
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.68
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

Time for backsubstitution: 23.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1450

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3245568, upper bound: 0.3254089
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3260125, upper bound: 0.3239534
time: 3.44 seconds

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

Time for backsubstitution: 23.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 961

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2819

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3254186, upper bound: 0.3167286
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3161243, upper bound: 0.3260224
time: 3.31 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.14 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.14
Output dim: 9, lower bound: -0.3245568, upper bound: 0.3254089
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.14
Output dim: 9, lower bound: -0.3260125, upper bound: 0.3239534
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.14
Output dim: 9, lower bound: -0.3254186, upper bound: 0.3167286
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.14
Output dim: 9, lower bound: -0.3161243, upper bound: 0.3260224

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.8000722, 0.7977681
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6617143, 0.6616786
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5981216, 0.5997581
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4869268, 0.4875612
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4901781, 0.4919076
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4826515, 0.4822314
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5077759, 0.5071939
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4679592, 0.4677942
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5241134, 0.5224056
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4860094, 0.4854136

Time for backsubstitution: 23.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 3109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2327

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3181483, upper bound: 0.3231893
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3223377, upper bound: 0.3189524
time: 3.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7999434, 0.7978969
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6615217, 0.6618717
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5989094, 0.5989707
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4868572, 0.4876306
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4907837, 0.4913023
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4839036, 0.4809792
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5092417, 0.5057284
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4679298, 0.4678235
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5224431, 0.5240760
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4860940, 0.4853289

Time for backsubstitution: 23.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2819

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2384

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3178769, upper bound: 0.3158346
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3178769, upper bound: 0.3158346
time: 3.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7969356, 0.7994640
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6571028, 0.6586466
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5934522, 0.5947297
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4820566, 0.4814764
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4869130, 0.4863703
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4808109, 0.4825387
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5063993, 0.5089444
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4673648, 0.4676371
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5245628, 0.5246878
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4848330, 0.4853005

Time for backsubstitution: 23.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1445

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2384

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3172974, upper bound: 0.3091888
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3172974, upper bound: 0.3091888
time: 3.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7972884, 0.7991111
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6588037, 0.6569457
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5955789, 0.5926032
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4821804, 0.4813524
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4874942, 0.4857886
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4808662, 0.4824831
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5068969, 0.5084469
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4675016, 0.4675004
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5246506, 0.5246000
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4846201, 0.4855134

Time for backsubstitution: 23.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2384

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1450

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3143752, upper bound: 0.3257290
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3158313, upper bound: 0.3242731
time: 3.91 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.64 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 9, lower bound: -0.3181483, upper bound: 0.3231893
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 9, lower bound: -0.3223377, upper bound: 0.3189524
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 9, lower bound: -0.3178769, upper bound: 0.3158346
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 9, lower bound: -0.3178769, upper bound: 0.3158346
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 9, lower bound: -0.3172974, upper bound: 0.3091888
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 9, lower bound: -0.3172974, upper bound: 0.3091888
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 9, lower bound: -0.3143752, upper bound: 0.3257290
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 9, lower bound: -0.3158313, upper bound: 0.3242731

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7943153, 0.7896883
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6582034, 0.6570852
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.6004932, 0.6015017
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4885266, 0.4887464
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4912984, 0.4921489
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4592054, 0.4560356
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5055602, 0.5025861
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4618201, 0.4570041
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5244384, 0.5245054
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4708689, 0.4721816

Time for backsubstitution: 23.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1990

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1248

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3123428, upper bound: 0.3184201
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3132463, upper bound: 0.3170990
time: 3.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7918420, 0.7921398
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6569188, 0.6583605
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.6006515, 0.6013420
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4880424, 0.4892187
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4910247, 0.4924212
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4576905, 0.4575331
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5046134, 0.5035126
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4571397, 0.4616675
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5245428, 0.5243950
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4728421, 0.4701886

Time for backsubstitution: 23.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3222962, upper bound: 0.3124162
time: 3.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3156323, upper bound: 0.3189231
time: 3.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.8006837, 0.7975724
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6670201, 0.6650608
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5983953, 0.5931914
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4846902, 0.4952865
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4877930, 0.4904301
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4827955, 0.4810383
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5132688, 0.5087494
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4711185, 0.4679966
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5274277, 0.5346341
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4885063, 0.4849091

Time for backsubstitution: 23.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1404

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1450

### Candidate
type: DSZ, layer: 3, pos: 1678

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3160725, upper bound: 0.3150396
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3170808, upper bound: 0.3140264
time: 3.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7997477, 0.7985499
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6649034, 0.6658542
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.6000938, 0.5992444
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4869807, 0.4853941
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4893060, 0.4919190
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4827106, 0.4811494
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5107969, 0.5098975
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4681323, 0.4684741
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5285563, 0.5273905
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4855895, 0.4858084

Time for backsubstitution: 22.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2819

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3175930, upper bound: 0.3068375
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3088789, upper bound: 0.3155501
time: 3.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7985084, 0.7997479
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6671774, 0.6649034
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5992441, 0.5923426
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4853940, 0.4945830
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4889174, 0.4893060
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4811232, 0.4827106
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5112213, 0.5107969
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4709828, 0.4681323
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5273905, 0.5346720
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4878261, 0.4855895

Time for backsubstitution: 23.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 1248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3139549, upper bound: 0.3065025
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3146333, upper bound: 0.3057880
time: 3.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7975724, 0.8007255
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6650608, 0.6656971
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.6009426, 0.5983953
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4876847, 0.4846902
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4904299, 0.4907949
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4810383, 0.4828219
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5087494, 0.5119451
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4679966, 0.4686100
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5285187, 0.5274277
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4849091, 0.4864888

Time for backsubstitution: 22.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 2341

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1101

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3126124, upper bound: 0.2959445
time: 3.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3040519, upper bound: 0.3045044
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7978969, 0.7999434
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6618717, 0.6615214
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5989704, 0.5989091
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4876306, 0.4868573
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4913020, 0.4907835
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4809792, 0.4839036
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5057284, 0.5092416
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4678233, 0.4679298
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5240762, 0.5224431
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4853289, 0.4860940

Time for backsubstitution: 23.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 1404

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2129

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3062206, upper bound: 0.3246519
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3134494, upper bound: 0.3160147
time: 3.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.7977681, 0.8000722
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6616786, 0.6617146
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5997581, 0.5981216
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4875612, 0.4869267
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4919076, 0.4901781
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4822314, 0.4826515
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5071939, 0.5077759
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4677942, 0.4679592
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5224054, 0.5241134
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4854138, 0.4860094

Time for backsubstitution: 23.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 1248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2129

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3061163, upper bound: 0.3233475
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3147539, upper bound: 0.3161187
time: 6.02 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 34.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3123428, upper bound: 0.3184201
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3132463, upper bound: 0.3170990
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3222962, upper bound: 0.3124162
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3156323, upper bound: 0.3189231
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3160725, upper bound: 0.3150396
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3170808, upper bound: 0.3140264
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3175930, upper bound: 0.3068375
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3088789, upper bound: 0.3155501
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3139549, upper bound: 0.3065025
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3146333, upper bound: 0.3057880
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3126124, upper bound: 0.2959445
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3040519, upper bound: 0.3045044
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3062206, upper bound: 0.3246519
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3134494, upper bound: 0.3160147
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3061163, upper bound: 0.3233475
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.22
Output dim: 9, lower bound: -0.3147539, upper bound: 0.3161187

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 23.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1445

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3085680, upper bound: 0.3155959
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3095186, upper bound: 0.3146452
time: 3.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 22.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2384

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3055328, upper bound: 0.3088881
time: 4.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.3055328, upper bound: 0.3088881
time: 4.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.8027668, 0.8003039
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6670814, 0.6674891
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5989544, 0.6000309
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4867487, 0.4876146
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4893773, 0.4905717
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4813499, 0.4798830
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5134445, 0.5110340
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4669847, 0.4677939
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5354662, 0.5334690
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4855452, 0.4847379

Time for backsubstitution: 23.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2819
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1678
type: DSZ, layer: 3, pos: 2384

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 961

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3221213, upper bound: 0.3092171
time: 4.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.3207610, upper bound: 0.3121491
time: 3.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -19.7823353, -18.2222900, -19.7823353, -18.2222900, -0.8024793, 0.8005912
1: -16.7284927, -15.4543056, -16.7284927, -15.4543056, -0.6673317, 0.6672385
2: -11.4512272, -10.3891983, -11.4512272, -10.3891983, -0.5991819, 0.5998037
3: -10.8678350, -9.8208580, -10.8678350, -9.8208580, -0.4869108, 0.4874526
4: -2.1190450, -1.3056176, -2.1190450, -1.3056176, -0.4894474, 0.4905012
5: -10.2593746, -9.3186054, -10.2593746, -9.3186054, -0.4815552, 0.4796777
6: -19.7650013, -18.5443192, -19.7650013, -18.5443192, -0.5130817, 0.5113968
7: -2.9937968, -2.1950381, -2.9937968, -2.1950381, -0.4679296, 0.4668491
8: -1.9133406, -1.1441293, -1.9133406, -1.1441293, -0.5335064, 0.5354290
9: 5.6430740, 6.5529280, 5.6430740, 6.5529280, -0.4854183, 0.4848647

Time for backsubstitution: 23.49 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.76 + 554.27 = 613.03 seconds
