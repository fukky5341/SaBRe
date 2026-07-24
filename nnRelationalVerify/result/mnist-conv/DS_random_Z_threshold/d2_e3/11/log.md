## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.36130252799999996


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7520418, 0.7520416)
1: (-15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8176699, 0.8176696)
2: (-3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6762986, 0.6762986)
3: (-9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0481901, 1.0481901)
4: (-5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8691096, 0.8691096)
5: (1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5810986, 0.5810986)
6: (6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7740793, 0.7740796)
7: (-19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8765984, 0.8765984)
8: (-1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6948266, 0.6948266)
9: (-6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6872334, 0.6872334)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.50 + 33.53 = 58.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3763568, upper bound: 0.3763567

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763568, upper bound: 0.3716006
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3716007, upper bound: 0.3763578
time: 3.08 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.21 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.21
Output dim: 6, lower bound: -0.3763568, upper bound: 0.3716006
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.21
Output dim: 6, lower bound: -0.3716007, upper bound: 0.3763578

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7520523, 0.7520509
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8176692, 0.8176706
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6763034, 0.6763043
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0482020, 1.0482039
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8691106, 0.8691101
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5811019, 0.5811024
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7740808, 0.7740796
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8766210, 0.8766248
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6948333, 0.6948321
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6872325, 0.6872323

Time for backsubstitution: 22.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3761860, upper bound: 0.3702025
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3749589, upper bound: 0.3714315
time: 3.24 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7520509, 0.7520523
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8176706, 0.8176692
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6763043, 0.6763034
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0482039, 1.0482018
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8691101, 0.8691106
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5811024, 0.5811019
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7740793, 0.7740810
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8766248, 0.8766210
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6948323, 0.6948330
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6872325, 0.6872327

Time for backsubstitution: 21.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 931

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3715691, upper bound: 0.3751888
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3704296, upper bound: 0.3763260
time: 3.13 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.66 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.66
Output dim: 6, lower bound: -0.3761860, upper bound: 0.3702025
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.66
Output dim: 6, lower bound: -0.3749589, upper bound: 0.3714315
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.66
Output dim: 6, lower bound: -0.3715691, upper bound: 0.3751888
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.66
Output dim: 6, lower bound: -0.3704296, upper bound: 0.3763260

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7520533, 0.7520516
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8176682, 0.8176694
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6763043, 0.6763051
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0482044, 1.0482054
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8691111, 0.8691106
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5811021, 0.5811028
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7740788, 0.7740774
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8766167, 0.8766212
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6948347, 0.6948335
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6872292, 0.6872282

Time for backsubstitution: 21.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 931

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3761542, upper bound: 0.3690322
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3750171, upper bound: 0.3701709
time: 3.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7520533, 0.7520516
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8176682, 0.8176696
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6763041, 0.6763051
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0482035, 1.0482061
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8691111, 0.8691101
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5811021, 0.5811028
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7740788, 0.7740777
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8766177, 0.8766208
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6948347, 0.6948338
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6872287, 0.6872289

Time for backsubstitution: 21.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 931

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3749272, upper bound: 0.3702605
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3737908, upper bound: 0.3714001
time: 3.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7512250, 0.7514138
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8161080, 0.8157947
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6761370, 0.6763008
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0485682, 1.0488048
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8717380, 0.8722763
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5806296, 0.5805349
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7755022, 0.7752612
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8739498, 0.8743932
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6941662, 0.6936731
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6893270, 0.6897571

Time for backsubstitution: 22.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3669961, upper bound: 0.3751885
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3715688, upper bound: 0.3706155
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7514124, 0.7512267
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8157961, 0.8161066
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6763015, 0.6761363
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0488067, 1.0485663
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8722758, 0.8717389
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5805357, 0.5806289
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7752600, 0.7755036
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8743970, 0.8739462
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6936722, 0.6941671
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6897571, 0.6893270

Time for backsubstitution: 21.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3702607, upper bound: 0.3749282
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3690323, upper bound: 0.3761553
time: 3.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 27.88 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.88
Output dim: 6, lower bound: -0.3761542, upper bound: 0.3690322
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.88
Output dim: 6, lower bound: -0.3750171, upper bound: 0.3701709
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.88
Output dim: 6, lower bound: -0.3749272, upper bound: 0.3702605
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.88
Output dim: 6, lower bound: -0.3737908, upper bound: 0.3714001
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.88
Output dim: 6, lower bound: -0.3669961, upper bound: 0.3751885
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.88
Output dim: 6, lower bound: -0.3715688, upper bound: 0.3706155
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.88
Output dim: 6, lower bound: -0.3702607, upper bound: 0.3749282
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.88
Output dim: 6, lower bound: -0.3690323, upper bound: 0.3761553

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7512279, 0.7514129
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8161058, 0.8157947
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6761370, 0.6763022
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0485692, 1.0488091
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8717399, 0.8722768
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5806298, 0.5805361
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7755022, 0.7752576
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8739426, 0.8743939
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6941681, 0.6936731
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6893239, 0.6897531

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3715820, upper bound: 0.3690330
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3761539, upper bound: 0.3644592
time: 3.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7514148, 0.7512259
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8157935, 0.8161068
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6763015, 0.6761377
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0488076, 1.0485706
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8722773, 0.8717389
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5805359, 0.5806301
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7752590, 0.7755001
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8743894, 0.8739469
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6936741, 0.6941671
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6897535, 0.6893229

Time for backsubstitution: 21.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3704449, upper bound: 0.3701717
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3750167, upper bound: 0.3655980
time: 3.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7512274, 0.7514131
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8161054, 0.8157949
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6761367, 0.6763022
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0485687, 1.0488095
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8717399, 0.8722763
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5806298, 0.5805361
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7755013, 0.7752581
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8739431, 0.8743935
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6941681, 0.6936731
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6893234, 0.6897535

Time for backsubstitution: 21.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3703549, upper bound: 0.3702614
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3749268, upper bound: 0.3656876
time: 3.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7514148, 0.7512259
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8157935, 0.8161068
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6763012, 0.6761377
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0488071, 1.0485711
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8722777, 0.8717389
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5805354, 0.5806301
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7752590, 0.7755003
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8743899, 0.8739464
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6936741, 0.6941671
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6897535, 0.6893234

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3692185, upper bound: 0.3714009
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3737904, upper bound: 0.3668270
time: 3.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7512236, 0.7513666
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8169513, 0.8168492
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6746764, 0.6750436
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0492229, 1.0496268
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8715706, 0.8722572
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5806262, 0.5805027
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7756677, 0.7754607
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8739419, 0.8744507
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6932197, 0.6925769
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6894741, 0.6898787

Time for backsubstitution: 21.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3668272, upper bound: 0.3737914
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3655980, upper bound: 0.3750178
time: 3.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7511778, 0.7514124
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8171625, 0.8166382
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6748798, 0.6748400
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0493908, 1.0494595
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8717184, 0.8721094
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5805972, 0.5805316
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7757015, 0.7754266
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8740077, 0.8743851
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6930699, 0.6927266
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6894484, 0.6899042

Time for backsubstitution: 22.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3713998, upper bound: 0.3692185
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3701706, upper bound: 0.3704459
time: 3.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7514129, 0.7512274
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8157949, 0.8161054
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6763022, 0.6761367
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0488100, 1.0485687
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8722768, 0.8717399
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5805364, 0.5806296
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7752581, 0.7755015
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8743932, 0.8739431
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6936731, 0.6941681
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6897535, 0.6893232

Time for backsubstitution: 22.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3656877, upper bound: 0.3749278
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3702603, upper bound: 0.3703547
time: 3.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7514129, 0.7512276
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8157949, 0.8161056
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6763022, 0.6761370
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0488091, 1.0485692
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8722768, 0.8717394
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5805364, 0.5806296
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7752581, 0.7755020
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8743942, 0.8739426
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6936731, 0.6941681
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6897531, 0.6893239

Time for backsubstitution: 22.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3644593, upper bound: 0.3761549
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3690319, upper bound: 0.3715816
time: 3.24 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.17 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3715820, upper bound: 0.3690330
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3761539, upper bound: 0.3644592
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3704449, upper bound: 0.3701717
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3750167, upper bound: 0.3655980
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3703549, upper bound: 0.3702614
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3749268, upper bound: 0.3656876
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3692185, upper bound: 0.3714009
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3737904, upper bound: 0.3668270
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3668272, upper bound: 0.3737914
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3655980, upper bound: 0.3750178
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3713998, upper bound: 0.3692185
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3701706, upper bound: 0.3704459
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3656877, upper bound: 0.3749278
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3702603, upper bound: 0.3703547
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3644593, upper bound: 0.3761549
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.17
Output dim: 6, lower bound: -0.3690319, upper bound: 0.3715816

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7512264, 0.7513657
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8169494, 0.8168492
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6746759, 0.6750445
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0492244, 1.0496314
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8715725, 0.8722572
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5806262, 0.5805039
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7756672, 0.7754569
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8739347, 0.8744519
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6932225, 0.6925771
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6894708, 0.6898746

Time for backsubstitution: 23.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2117
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1097
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2582
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2083
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 324
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1970

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3664821, upper bound: 0.3651226
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3676615, upper bound: 0.3639420
time: 3.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7511806, 0.7514114
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8171601, 0.8166382
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6748793, 0.6748412
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0493917, 1.0494640
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8717203, 0.8721094
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5805976, 0.5805326
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7757010, 0.7754230
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8740005, 0.8743863
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6930728, 0.6927271
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6894455, 0.6899002

Time for backsubstitution: 23.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 324
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2582
type: DSZ, layer: 3, pos: 1970
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 2117
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 1097
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2083
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 2487

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 417

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3753463, upper bound: 0.3609319
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3726166, upper bound: 0.3636571
time: 3.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7514133, 0.7511785
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8166370, 0.8171611
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6748405, 0.6748798
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0494628, 1.0493927
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8721104, 0.8717198
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5805323, 0.5805979
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7754245, 0.7756991
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8743820, 0.8740048
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6927285, 0.6930711
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6899009, 0.6894448

Time for backsubstitution: 23.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 2117
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 2083
type: DSZ, layer: 3, pos: 2582
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 1097
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 1970
type: DSZ, layer: 3, pos: 324
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2487

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3691929, upper bound: 0.3677259
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3679932, upper bound: 0.3689261
time: 3.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7513676, 0.7512245
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8168483, 0.8169503
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6750438, 0.6746767
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0496302, 1.0492256
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8722582, 0.8715720
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5805032, 0.5806267
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7754588, 0.7756653
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8744473, 0.8739393
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6925783, 0.6932211
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6898756, 0.6894703

Time for backsubstitution: 23.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 324
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 2582
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2083
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 1097
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 1970
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2117
type: DSZ, layer: 3, pos: 2494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 324

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3747761, upper bound: 0.3651578
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3746864, upper bound: 0.3653542
time: 3.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7512255, 0.7513657
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8169489, 0.8168492
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6746757, 0.6750445
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0492239, 1.0496318
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8715730, 0.8722572
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5806262, 0.5805039
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7756667, 0.7754571
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8739352, 0.8744514
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6932220, 0.6925774
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6894708, 0.6898751

Time for backsubstitution: 23.42 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.03 + 546.72 = 604.75 seconds
