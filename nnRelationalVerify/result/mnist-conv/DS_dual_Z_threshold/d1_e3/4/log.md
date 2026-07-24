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
execution time: IAR + RelationalAnalysis = 21.75 + 35.92 = 57.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1865907, upper bound: 0.1865909

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 499

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858742, upper bound: 0.1865892
time: 6.02 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865893, upper bound: 0.1858744
time: 5.26 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 11.44 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 11.44
Output dim: 8, lower bound: -0.1858742, upper bound: 0.1865892
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 11.44
Output dim: 8, lower bound: -0.1865893, upper bound: 0.1858744

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3083053, 0.3053255
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3867168, 0.3872302
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3339984, 0.3333094
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4309363, 0.4332204
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2893164, 0.2868342
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3759460, 0.3793960
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.3983564, 0.4020977
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3698540, 0.3689461
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4487495, 0.4495101
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2660084, 0.2624805

Time for backsubstitution: 19.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 63

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1855055, upper bound: 0.1865836
time: 5.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858684, upper bound: 0.1862209
time: 5.17 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3053255, 0.3083053
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3872299, 0.3867166
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3333094, 0.3339984
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4332204, 0.4309363
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2868342, 0.2893164
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3793960, 0.3759460
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4020977, 0.3983564
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3689461, 0.3698540
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4495101, 0.4487495
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2624805, 0.2660084

Time for backsubstitution: 20.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 63

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1862206, upper bound: 0.1858684
time: 6.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865836, upper bound: 0.1855058
time: 5.19 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.67 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.67
Output dim: 8, lower bound: -0.1855055, upper bound: 0.1865836
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.67
Output dim: 8, lower bound: -0.1858684, upper bound: 0.1862209
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.67
Output dim: 8, lower bound: -0.1862206, upper bound: 0.1858684
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.67
Output dim: 8, lower bound: -0.1865836, upper bound: 0.1855058

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3084807, 0.3056185
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3856859, 0.3859968
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3346031, 0.3337169
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4307547, 0.4331031
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2888572, 0.2862620
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3750482, 0.3783464
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.3993039, 0.4032891
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3685079, 0.3672104
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4488053, 0.4495773
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2650521, 0.2613325

Time for backsubstitution: 20.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1855055, upper bound: 0.1865832
time: 6.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1855052, upper bound: 0.1858798
time: 4.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2

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

Time for backsubstitution: 20.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858684, upper bound: 0.1862205
time: 5.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858681, upper bound: 0.1855166
time: 7.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3055010, 0.3085983
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3861995, 0.3854833
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3339140, 0.3344057
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4330387, 0.4308190
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2863750, 0.2887444
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3784976, 0.3748965
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4030447, 0.3995481
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3676000, 0.3681183
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4495664, 0.4488168
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2615242, 0.2648602

Time for backsubstitution: 21.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1855166, upper bound: 0.1858683
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1862203, upper bound: 0.1858686
time: 4.13 seconds

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

Time for backsubstitution: 20.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858796, upper bound: 0.1855054
time: 4.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865833, upper bound: 0.1855057
time: 3.85 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.46 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 8, lower bound: -0.1855055, upper bound: 0.1865832
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 8, lower bound: -0.1855052, upper bound: 0.1858798
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 8, lower bound: -0.1858684, upper bound: 0.1862205
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 8, lower bound: -0.1858681, upper bound: 0.1855166
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 8, lower bound: -0.1855166, upper bound: 0.1858683
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 8, lower bound: -0.1862203, upper bound: 0.1858686
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 8, lower bound: -0.1858796, upper bound: 0.1855054
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.46
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

Time for backsubstitution: 20.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1425

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1847335, upper bound: 0.1859677
time: 4.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1848898, upper bound: 0.1858114
time: 3.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 20.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1425

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1847332, upper bound: 0.1852642
time: 4.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1848895, upper bound: 0.1851077
time: 4.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 20.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1425

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1850965, upper bound: 0.1856047
time: 5.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1852528, upper bound: 0.1854484
time: 4.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 21.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1425

Time for candidate selection: 0.41 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1850963, upper bound: 0.1849012
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1852525, upper bound: 0.1847446
time: 4.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 20.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1425

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1847444, upper bound: 0.1852527
time: 4.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1849010, upper bound: 0.1850965
time: 4.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 21.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1425

Time for candidate selection: 0.39 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1854481, upper bound: 0.1852531
time: 4.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1856044, upper bound: 0.1850968
time: 5.93 seconds

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

Time for backsubstitution: 20.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1425

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1851074, upper bound: 0.1848894
time: 6.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1852640, upper bound: 0.1847335
time: 5.62 seconds

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

Time for backsubstitution: 21.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1425

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858111, upper bound: 0.1848901
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1859675, upper bound: 0.1847334
time: 6.66 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 32.36 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1847335, upper bound: 0.1859677
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1848898, upper bound: 0.1858114
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1847332, upper bound: 0.1852642
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1848895, upper bound: 0.1851077
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1850965, upper bound: 0.1856047
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1852528, upper bound: 0.1854484
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1850963, upper bound: 0.1849012
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1852525, upper bound: 0.1847446
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1847444, upper bound: 0.1852527
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1849010, upper bound: 0.1850965
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1854481, upper bound: 0.1852531
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1856044, upper bound: 0.1850968
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1851074, upper bound: 0.1848894
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1852640, upper bound: 0.1847335
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1858111, upper bound: 0.1848901
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.1859675, upper bound: 0.1847334

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3087730, 0.3059149
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3895526, 0.3906550
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3354840, 0.3348927
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.3838701, 0.3922243
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2575327, 0.2613124
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.2953453, 0.2893934
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.3748646, 0.3743572
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3002822, 0.3109338
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4546199, 0.4551344
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2343938, 0.2331188

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1425

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 422

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1827370, upper bound: 0.1827452
time: 4.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1815110, upper bound: 0.1839700
time: 7.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3087776, 0.3059103
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3903432, 0.3898649
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3357792, 0.3345976
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.3898754, 0.3862181
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2639077, 0.2549372
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.2860956, 0.2986434
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.3703728, 0.3788488
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3122325, 0.2989838
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4543624, 0.4553919
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2368379, 0.2306745

Time for backsubstitution: 21.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1425

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 422

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1828934, upper bound: 0.1825889
time: 4.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1816673, upper bound: 0.1838140
time: 3.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3087726, 0.3059151
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3895540, 0.3906541
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3354840, 0.3348927
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.3838701, 0.3922243
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2575321, 0.2613128
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.2953446, 0.2893929
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.3748636, 0.3743565
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3002818, 0.3109345
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4546204, 0.4551339
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2343931, 0.2331183

Time for backsubstitution: 21.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1709
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1425

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 422

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1827368, upper bound: 0.1820417
time: 4.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1815108, upper bound: 0.1832668
time: 3.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3087771, 0.3059106
1: -8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3903446, 0.3898635
2: -0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3357792, 0.3345976
3: -4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.3898754, 0.3862176
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2639070, 0.2549375
5: -9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.2860951, 0.2986426
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.3703718, 0.3788483
7: -11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3122318, 0.2989845
8: 9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4543629, 0.4553914
9: -5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2368371, 0.2306740

Time for backsubstitution: 21.17 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.67 + 557.66 = 615.33 seconds
