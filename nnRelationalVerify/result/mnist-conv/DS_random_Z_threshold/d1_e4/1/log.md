## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.3552878565


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7229214, 0.7229214)
1: (-6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.1029615, 1.1029606)
2: (-4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.8028684, 0.8028684)
3: (-5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7839131, 0.7839131)
4: (-10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6515818, 0.6515818)
5: (1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6103714, 0.6103711)
6: (0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6167908, 0.6167908)
7: (-12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8833489, 0.8833489)
8: (6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8516474, 0.8516474)
9: (-8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0465469, 1.0465469)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.98 + 34.76 = 57.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.3556433, upper bound: 0.3556435

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 4614
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 6212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 522

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3543486, upper bound: 0.3556386
time: 8.60 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556393, upper bound: 0.3543492
time: 4.78 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 13.40 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 13.40
Output dim: 8, lower bound: -0.3543486, upper bound: 0.3556386
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 13.40
Output dim: 8, lower bound: -0.3556393, upper bound: 0.3543492

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7230740, 0.7224188
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.1032887, 1.1018810
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.8035774, 0.8005228
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7792282, 0.7853281
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6512651, 0.6516771
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6105952, 0.6096225
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6178241, 0.6133671
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8833756, 0.8832650
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8502254, 0.8520775
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0467434, 1.0458961

Time for backsubstitution: 21.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4614
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4614

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3543481, upper bound: 0.3556234
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3543325, upper bound: 0.3556382
time: 4.49 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7224188, 0.7229214
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.1018810, 1.1029606
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.8005233, 0.8028684
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7839131, 0.7792282
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6515818, 0.6512651
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6096225, 0.6103711
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6133671, 0.6167908
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8832655, 0.8833489
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8516474, 0.8502254
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0458961, 1.0465469

Time for backsubstitution: 21.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4614
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4614

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556388, upper bound: 0.3543331
time: 4.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556228, upper bound: 0.3543479
time: 5.63 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.71 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.71
Output dim: 8, lower bound: -0.3543481, upper bound: 0.3556234
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.71
Output dim: 8, lower bound: -0.3543325, upper bound: 0.3556382
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.71
Output dim: 8, lower bound: -0.3556388, upper bound: 0.3543331
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.71
Output dim: 8, lower bound: -0.3556228, upper bound: 0.3543479

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7218676, 0.7210431
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0932827, 1.0936680
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7868443, 0.7858806
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7725639, 0.7767403
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6506855, 0.6510167
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6119442, 0.6105015
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6137755, 0.6084578
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8816137, 0.8824434
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8525052, 0.8539934
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0426497, 1.0412169

Time for backsubstitution: 21.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6182

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3537414, upper bound: 0.3556220
time: 5.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3543474, upper bound: 0.3550145
time: 5.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7216978, 0.7212129
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0950756, 1.0918751
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7889352, 0.7837896
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7706404, 0.7786639
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6506045, 0.6510978
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6114740, 0.6109712
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6129148, 0.6093185
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8825531, 0.8815041
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8521414, 0.8543577
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0420637, 1.0418029

Time for backsubstitution: 23.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 542

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3543302, upper bound: 0.3542054
time: 4.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3529011, upper bound: 0.3556368
time: 4.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7212129, 0.7215457
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0918751, 1.0947475
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7837896, 0.7882271
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7772489, 0.7706404
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6510026, 0.6506047
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6109715, 0.6112494
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6093185, 0.6118810
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8815036, 0.8825278
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8539276, 0.8521414
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0418029, 1.0418677

Time for backsubstitution: 23.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 536

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3552789, upper bound: 0.3543319
time: 5.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556361, upper bound: 0.3539694
time: 4.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7210431, 0.7217155
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0936680, 1.0929546
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7858806, 0.7861362
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7753253, 0.7725639
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6509216, 0.6506853
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6105013, 0.6117191
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6084578, 0.6127417
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8824430, 0.8815885
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8535638, 0.8525052
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0412169, 1.0424542

Time for backsubstitution: 23.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 6212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 536

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3552632, upper bound: 0.3543475
time: 5.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556201, upper bound: 0.3539851
time: 5.07 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 33.38 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.38
Output dim: 8, lower bound: -0.3537414, upper bound: 0.3556220
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 33.38
Output dim: 8, lower bound: -0.3543474, upper bound: 0.3550145
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 33.38
Output dim: 8, lower bound: -0.3543302, upper bound: 0.3542054
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.38
Output dim: 8, lower bound: -0.3529011, upper bound: 0.3556368
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 33.38
Output dim: 8, lower bound: -0.3552789, upper bound: 0.3543319
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.38
Output dim: 8, lower bound: -0.3556361, upper bound: 0.3539694
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 33.38
Output dim: 8, lower bound: -0.3552632, upper bound: 0.3543475
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.38
Output dim: 8, lower bound: -0.3556201, upper bound: 0.3539851

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7218418, 0.7210245
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0893135, 1.0907421
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7865963, 0.7855468
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7692113, 0.7742677
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6483088, 0.6477914
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6118774, 0.6104128
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6119905, 0.6060357
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8806596, 0.8817387
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8508415, 0.8527679
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0403872, 1.0381484

Time for backsubstitution: 23.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 542

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3532046, upper bound: 0.3556206
time: 5.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3537402, upper bound: 0.3550873
time: 4.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7073016, 0.7047648
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0890770, 1.0850220
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7805810, 0.7740126
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7473135, 0.7582510
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6209512, 0.6171813
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.5935140, 0.5952535
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.5957050, 0.5943477
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8680663, 0.8649511
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8419428, 0.8454328
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0369220, 1.0359268

Time for backsubstitution: 22.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6212

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3523655, upper bound: 0.3556349
time: 5.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3528999, upper bound: 0.3551018
time: 4.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7212129, 0.7207327
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0913219, 1.0947475
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7827682, 0.7882271
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7772489, 0.7705762
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6510026, 0.6504974
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6106861, 0.6112494
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6070232, 0.6118810
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8815036, 0.8819599
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8539276, 0.8520465
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0418029, 1.0410333

Time for backsubstitution: 22.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 6212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 542

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556339, upper bound: 0.3525395
time: 4.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3542020, upper bound: 0.3539672
time: 4.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7210431, 0.7209020
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0931149, 1.0929546
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7848592, 0.7861362
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7753253, 0.7724998
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6509216, 0.6505780
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6102169, 0.6117191
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6061625, 0.6127417
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8824430, 0.8810205
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8535638, 0.8524103
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0412169, 1.0416193

Time for backsubstitution: 22.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3550125, upper bound: 0.3539834
time: 8.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556195, upper bound: 0.3533797
time: 4.73 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 35.49 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 35.49
Output dim: 8, lower bound: -0.3532046, upper bound: 0.3556206
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 35.49
Output dim: 8, lower bound: -0.3537402, upper bound: 0.3550873
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 35.49
Output dim: 8, lower bound: -0.3523655, upper bound: 0.3556349
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 35.49
Output dim: 8, lower bound: -0.3528999, upper bound: 0.3551018
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 35.49
Output dim: 8, lower bound: -0.3556339, upper bound: 0.3525395
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 35.49
Output dim: 8, lower bound: -0.3542020, upper bound: 0.3539672
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 35.49
Output dim: 8, lower bound: -0.3550125, upper bound: 0.3539834
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 35.49
Output dim: 8, lower bound: -0.3556195, upper bound: 0.3533797

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7180033, 0.7182293
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0888481, 1.0901041
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7828398, 0.7803941
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7689872, 0.7741032
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6445818, 0.6450777
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6096172, 0.6087654
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6097448, 0.6029506
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8789735, 0.8805108
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8491755, 0.8515525
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0388012, 1.0369935

Time for backsubstitution: 23.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 542

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3532022, upper bound: 0.3541883
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3517744, upper bound: 0.3556191
time: 4.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7034626, 0.7019691
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0886116, 1.0843830
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7768230, 0.7688589
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7470899, 0.7580874
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6172221, 0.6144667
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.5912552, 0.5936079
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.5934579, 0.5912607
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8663802, 0.8637228
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8402772, 0.8442187
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0353346, 1.0347705

Time for backsubstitution: 22.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 536

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3520017, upper bound: 0.3556322
time: 5.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3523642, upper bound: 0.3552763
time: 4.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7047648, 0.7063355
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0844679, 1.0887499
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7729907, 0.7798719
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7568355, 0.7472491
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6170864, 0.6208436
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.5949681, 0.5932899
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.5920522, 0.5946712
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8649507, 0.8674726
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8450031, 0.8418489
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0359268, 1.0358906

Time for backsubstitution: 22.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 6212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3550203, upper bound: 0.3525384
time: 4.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556332, upper bound: 0.3519331
time: 4.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7210245, 0.7208762
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0901890, 1.0889874
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7845249, 0.7858877
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7728519, 0.7691467
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6476965, 0.6482015
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6101284, 0.6116538
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6037400, 0.6109562
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8817387, 0.8800669
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8523374, 0.8507476
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0381484, 1.0393567

Time for backsubstitution: 23.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 542

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556173, upper bound: 0.3519501
time: 4.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3541871, upper bound: 0.3533775
time: 5.01 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 33.26 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 33.26
Output dim: 8, lower bound: -0.3532022, upper bound: 0.3541883
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 33.26
Output dim: 8, lower bound: -0.3517744, upper bound: 0.3556191
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 33.26
Output dim: 8, lower bound: -0.3520017, upper bound: 0.3556322
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 33.26
Output dim: 8, lower bound: -0.3523642, upper bound: 0.3552763
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 33.26
Output dim: 8, lower bound: -0.3550203, upper bound: 0.3525384
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 33.26
Output dim: 8, lower bound: -0.3556332, upper bound: 0.3519331
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 33.26
Output dim: 8, lower bound: -0.3556173, upper bound: 0.3519501
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 33.26
Output dim: 8, lower bound: -0.3541871, upper bound: 0.3533775

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7036057, 0.7017798
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0828495, 1.0832491
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7744851, 0.7706165
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7456608, 0.7536912
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6149263, 0.6111603
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.5916586, 0.5930500
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.5925326, 0.5879779
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8644862, 0.8639579
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8389778, 0.8426285
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0336585, 1.0311170

Time for backsubstitution: 23.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 536

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3514120, upper bound: 0.3556164
time: 5.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3517731, upper bound: 0.3552600
time: 5.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7026496, 0.7022667
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0888205, 1.0838280
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7771940, 0.7678366
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7470264, 0.7581100
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6171148, 0.6145058
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.5913594, 0.5933223
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.5942969, 0.5889654
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8658128, 0.8639302
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8401837, 0.8442535
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0345006, 1.0350790

Time for backsubstitution: 23.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3513954, upper bound: 0.3556314
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3520012, upper bound: 0.3550188
time: 5.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7047462, 0.7063098
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0815449, 1.0847807
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7726574, 0.7796259
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7543640, 0.7438958
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6138604, 0.6184669
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.5948796, 0.5932245
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.5896301, 0.5928869
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8642473, 0.8665190
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8437772, 0.8401856
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0328588, 1.0336289

Time for backsubstitution: 23.39 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.74 + 560.56 = 618.30 seconds
