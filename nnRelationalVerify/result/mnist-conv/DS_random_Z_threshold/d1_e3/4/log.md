## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.184724793


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3232119, 0.3232119)
1: (-8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3898010, 0.3898010)
2: (-0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3374567, 0.3374565)
3: (-4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4445329, 0.4445329)
4: (-11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.3013635, 0.3013635)
5: (-9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3966613, 0.3966613)
6: (-11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4207950, 0.4207947)
7: (-11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3741808, 0.3741808)
8: (9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4533181, 0.4533181)
9: (-5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2832100, 0.2832100)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.38 + 35.83 = 59.21 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1865907, upper bound: 0.1865909

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 63

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1862220, upper bound: 0.1865852
time: 3.77 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865850, upper bound: 0.1862220
time: 4.45 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.23 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.23
Output dim: 8, lower bound: -0.1862220, upper bound: 0.1865852
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.23
Output dim: 8, lower bound: -0.1865850, upper bound: 0.1862220

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3233876, 0.3235052
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3887706, 0.3885679
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3380611, 0.3378642
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4443502, 0.4444146
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.3009043, 0.3007915
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3957634, 0.3956122
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4217424, 0.4219868
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3728347, 0.3724446
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4533749, 0.4533858
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2822535, 0.2820616

Time for backsubstitution: 21.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 499

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1855707, upper bound: 0.1865844
time: 4.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1862213, upper bound: 0.1859338
time: 4.32 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3235052, 0.3233876
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3885679, 0.3887706
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3378637, 0.3380616
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4444146, 0.4443502
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.3007915, 0.3009043
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3956122, 0.3957634
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4219866, 0.4217427
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3724451, 0.3728347
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4533863, 0.4533749
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2820616, 0.2822535

Time for backsubstitution: 21.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 499

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858684, upper bound: 0.1862209
time: 4.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865836, upper bound: 0.1855058
time: 5.00 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.49 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.49
Output dim: 8, lower bound: -0.1855707, upper bound: 0.1865844
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.49
Output dim: 8, lower bound: -0.1862213, upper bound: 0.1859338
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.49
Output dim: 8, lower bound: -0.1858684, upper bound: 0.1862209
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.49
Output dim: 8, lower bound: -0.1865836, upper bound: 0.1855058

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3233874, 0.3235049
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3887691, 0.3885670
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3380611, 0.3378642
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4443507, 0.4444146
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.3009045, 0.3007910
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3957639, 0.3956122
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4217429, 0.4219863
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3728352, 0.3724446
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4533749, 0.4533863
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2822535, 0.2820621

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 499

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 499

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1855055, upper bound: 0.1865832
time: 6.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1855166, upper bound: 0.1858683
time: 3.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3233874, 0.3235049
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3887701, 0.3885660
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3380611, 0.3378642
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4443502, 0.4444146
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.3009040, 0.3007915
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3957634, 0.3956127
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4217420, 0.4219868
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3728347, 0.3724451
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4533753, 0.4533858
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2822540, 0.2820616

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 499

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 499

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1855052, upper bound: 0.1858798
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1862203, upper bound: 0.1858686
time: 3.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3085983, 0.3055010
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3854833, 0.3861995
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3344057, 0.3339143
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4308190, 0.4330387
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2887444, 0.2863750
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3748965, 0.3784976
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.3995481, 0.4030449
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3681178, 0.3676004
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4488168, 0.4495664
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2648602, 0.2615242

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858684, upper bound: 0.1862205
time: 5.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858681, upper bound: 0.1855166
time: 7.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3056185, 0.3084807
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3859968, 0.3856859
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3337166, 0.3346033
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4331031, 0.4307547
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2862620, 0.2888575
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3783464, 0.3750482
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4032888, 0.3993039
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3672099, 0.3685083
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4495773, 0.4488053
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2613325, 0.2650521

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858796, upper bound: 0.1855054
time: 3.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865833, upper bound: 0.1855057
time: 3.74 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.71 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 8, lower bound: -0.1855055, upper bound: 0.1865832
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 8, lower bound: -0.1855166, upper bound: 0.1858683
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 8, lower bound: -0.1855052, upper bound: 0.1858798
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 8, lower bound: -0.1862203, upper bound: 0.1858686
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 8, lower bound: -0.1858684, upper bound: 0.1862205
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 8, lower bound: -0.1858681, upper bound: 0.1855166
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 8, lower bound: -0.1858796, upper bound: 0.1855054
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 8, lower bound: -0.1865833, upper bound: 0.1855057

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3084807, 0.3056180
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3856854, 0.3859973
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3346031, 0.3337169
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4307556, 0.4331036
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2888570, 0.2862613
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3750491, 0.3783469
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.3993049, 0.4032891
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3685074, 0.3672085
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4488058, 0.4495778
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2650509, 0.2613318

Time for backsubstitution: 21.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 1425
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 718

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1709

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1855025, upper bound: 0.1864432
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1853652, upper bound: 0.1865805
time: 4.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3055005, 0.3085978
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3861990, 0.3854842
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3339145, 0.3344057
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4330392, 0.4308195
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2863748, 0.2887435
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3784976, 0.3748970
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4030442, 0.3995481
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3675995, 0.3681164
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4495664, 0.4488168
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2615230, 0.2648585

Time for backsubstitution: 21.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 1425
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 2334

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 722

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1855106, upper bound: 0.1838886
time: 4.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1835370, upper bound: 0.1858619
time: 6.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3084800, 0.3056180
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3856869, 0.3859963
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3346031, 0.3337169
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4307556, 0.4331036
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2888563, 0.2862618
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3750486, 0.3783464
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.3993039, 0.4032884
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3685064, 0.3672094
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4488058, 0.4495773
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2650502, 0.2613313

Time for backsubstitution: 21.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1425
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 718

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1255

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1799707, upper bound: 0.1810289
time: 6.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1806551, upper bound: 0.1803441
time: 5.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3055003, 0.3085983
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3862000, 0.3854828
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3339145, 0.3344059
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4330392, 0.4308195
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2863743, 0.2887440
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3784981, 0.3748980
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4030452, 0.3995485
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3675985, 0.3681173
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4495668, 0.4488168
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2615235, 0.2648590

Time for backsubstitution: 21.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1425
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 949

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1830394, upper bound: 0.1826856
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1830394, upper bound: 0.1826856
time: 3.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3085983, 0.3055003
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3854828, 0.3862000
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3344057, 0.3339143
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4308195, 0.4330392
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2887440, 0.2863743
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3748980, 0.3784986
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.3995485, 0.4030449
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3681173, 0.3675985
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4488168, 0.4495668
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2648590, 0.2615235

Time for backsubstitution: 21.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 1425
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 404

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1831892, upper bound: 0.1858439
time: 4.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1854938, upper bound: 0.1835461
time: 3.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3085978, 0.3055005
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3854842, 0.3861990
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3344057, 0.3339143
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4308195, 0.4330392
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2887435, 0.2863748
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3748970, 0.3784976
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.3995481, 0.4030445
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3681164, 0.3675995
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4488173, 0.4495664
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2648585, 0.2615230

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1425
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 718

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1840795, upper bound: 0.1837644
time: 6.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1841159, upper bound: 0.1837280
time: 5.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3056180, 0.3084800
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3859963, 0.3856869
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3337166, 0.3346033
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4331036, 0.4307556
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2862618, 0.2888563
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3783464, 0.3750486
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4032884, 0.3993039
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3672094, 0.3685064
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4495773, 0.4488058
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2613313, 0.2650502

Time for backsubstitution: 23.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1425
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 949

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1826970, upper bound: 0.1823249
time: 7.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1826970, upper bound: 0.1823249
time: 6.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3056180, 0.3084807
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3859978, 0.3856854
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3337171, 0.3346033
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4331036, 0.4307556
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2862613, 0.2888570
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3783469, 0.3750491
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4032893, 0.3993046
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3672085, 0.3685074
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4495778, 0.4488053
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2613318, 0.2650509

Time for backsubstitution: 22.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1425
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2334

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2124

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1859588, upper bound: 0.1843590
time: 3.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1854368, upper bound: 0.1848831
time: 5.02 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.35 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1855025, upper bound: 0.1864432
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1853652, upper bound: 0.1865805
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1855106, upper bound: 0.1838886
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1835370, upper bound: 0.1858619
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1799707, upper bound: 0.1810289
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1806551, upper bound: 0.1803441
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1830394, upper bound: 0.1826856
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1830394, upper bound: 0.1826856
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1831892, upper bound: 0.1858439
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1854938, upper bound: 0.1835461
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1840795, upper bound: 0.1837644
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1841159, upper bound: 0.1837280
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1826970, upper bound: 0.1823249
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1826970, upper bound: 0.1823249
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1859588, upper bound: 0.1843590
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.35
Output dim: 8, lower bound: -0.1854368, upper bound: 0.1848831

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3084786, 0.3056109
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3856678, 0.3859868
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3345759, 0.3336818
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4307432, 0.4330945
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2888472, 0.2862434
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3750339, 0.3783369
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.3993034, 0.4032874
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3685203, 0.3672047
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4488049, 0.4495769
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2650504, 0.2613308

Time for backsubstitution: 22.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1425
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 1255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2495

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1852157, upper bound: 0.1857914
time: 6.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1848507, upper bound: 0.1861564
time: 4.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3084736, 0.3056159
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3856750, 0.3859801
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3345683, 0.3336895
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4307466, 0.4330912
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2888391, 0.2862515
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3750391, 0.3783317
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.3993030, 0.4032879
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3685031, 0.3672209
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4488049, 0.4495769
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2650499, 0.2613311

Time for backsubstitution: 22.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1425
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 722

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 550

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1851760, upper bound: 0.1865738
time: 5.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1853586, upper bound: 0.1863913
time: 4.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3054979, 0.3085935
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3861761, 0.3854654
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3338680, 0.3343003
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4330254, 0.4308109
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2863619, 0.2887225
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3784256, 0.3748765
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4030447, 0.3995483
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3676677, 0.3681612
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4494715, 0.4487114
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2615168, 0.2648532

Time for backsubstitution: 22.58 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 59.21 + 542.05 = 601.26 seconds
