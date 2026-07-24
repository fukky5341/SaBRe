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
execution time: IAR + RelationalAnalysis = 22.47 + 33.65 = 56.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3763568, upper bound: 0.3763567

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 931

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763250, upper bound: 0.3751877
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751878, upper bound: 0.3763248
time: 3.23 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.82 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.82
Output dim: 6, lower bound: -0.3763250, upper bound: 0.3751877
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.82
Output dim: 6, lower bound: -0.3751878, upper bound: 0.3763248

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7512159, 0.7514029
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8161068, 0.8157947
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6761322, 0.6762967
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0485549, 1.0487938
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8717389, 0.8722763
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5806255, 0.5805316
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7755027, 0.7752604
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8739233, 0.8743703
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6941605, 0.6936662
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6893291, 0.6897593

Time for backsubstitution: 21.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3761543, upper bound: 0.3737907
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3749272, upper bound: 0.3750170
time: 3.42 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7514029, 0.7512157
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8157949, 0.8161066
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6762967, 0.6761322
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0487938, 1.0485551
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8722763, 0.8717389
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5805316, 0.5806255
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7752604, 0.7755027
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8743706, 0.8739235
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6936665, 0.6941602
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6897593, 0.6893291

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3750171, upper bound: 0.3749270
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3737908, upper bound: 0.3761541
time: 3.35 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.72 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.72
Output dim: 6, lower bound: -0.3761543, upper bound: 0.3737907
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.72
Output dim: 6, lower bound: -0.3749272, upper bound: 0.3750170
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.72
Output dim: 6, lower bound: -0.3750171, upper bound: 0.3749270
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.72
Output dim: 6, lower bound: -0.3737908, upper bound: 0.3761541

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7512164, 0.7514033
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8161049, 0.8157928
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6761327, 0.6762969
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0485573, 1.0487952
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8717403, 0.8722782
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5806258, 0.5805318
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7755003, 0.7752576
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8739200, 0.8743675
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6941609, 0.6936669
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6893258, 0.6897554

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3761542, upper bound: 0.3690322
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3714001, upper bound: 0.3737918
time: 3.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7512159, 0.7514033
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8161049, 0.8157930
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6761324, 0.6762972
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0485568, 1.0487957
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8717403, 0.8722782
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5806258, 0.5805321
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7754998, 0.7752581
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8739204, 0.8743668
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6941609, 0.6936672
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6893258, 0.6897559

Time for backsubstitution: 21.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3749272, upper bound: 0.3702605
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3701710, upper bound: 0.3750181
time: 3.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7514033, 0.7512164
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8157930, 0.8161049
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6762972, 0.6761324
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0487957, 1.0485566
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8722782, 0.8717408
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5805318, 0.5806260
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7752581, 0.7755001
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8743668, 0.8739204
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6936669, 0.6941609
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6897559, 0.6893256

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3750171, upper bound: 0.3701709
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3702607, upper bound: 0.3749282
time: 3.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7514033, 0.7512164
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8157930, 0.8161051
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6762969, 0.6761327
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0487952, 1.0485570
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8722782, 0.8717403
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5805318, 0.5806260
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7752576, 0.7755003
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8743672, 0.8739200
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6936669, 0.6941612
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6897554, 0.6893260

Time for backsubstitution: 22.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3737908, upper bound: 0.3714001
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3690323, upper bound: 0.3761553
time: 3.37 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.86 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.86
Output dim: 6, lower bound: -0.3761542, upper bound: 0.3690322
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.86
Output dim: 6, lower bound: -0.3714001, upper bound: 0.3737918
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.86
Output dim: 6, lower bound: -0.3749272, upper bound: 0.3702605
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.86
Output dim: 6, lower bound: -0.3701710, upper bound: 0.3750181
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.86
Output dim: 6, lower bound: -0.3750171, upper bound: 0.3701709
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.86
Output dim: 6, lower bound: -0.3702607, upper bound: 0.3749282
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.86
Output dim: 6, lower bound: -0.3737908, upper bound: 0.3714001
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.86
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

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3715820, upper bound: 0.3690330
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3761539, upper bound: 0.3644592
time: 3.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7512259, 0.7514145
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8161068, 0.8157935
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6761377, 0.6763015
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0485711, 1.0488071
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8717389, 0.8722773
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5806303, 0.5805357
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7755003, 0.7752593
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8739464, 0.8743901
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6941671, 0.6936741
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6893234, 0.6897533

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3668272, upper bound: 0.3737914
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3713998, upper bound: 0.3692185
time: 3.41 seconds

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

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3703549, upper bound: 0.3702614
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3749268, upper bound: 0.3656876
time: 3.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7512259, 0.7514148
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8161068, 0.8157938
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6761377, 0.6763015
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0485706, 1.0488076
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8717389, 0.8722773
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5806303, 0.5805357
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7755003, 0.7752595
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8739469, 0.8743896
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6941671, 0.6936741
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6893229, 0.6897538

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3655980, upper bound: 0.3750178
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3701706, upper bound: 0.3704459
time: 3.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 21.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3704449, upper bound: 0.3701717
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3750167, upper bound: 0.3655980
time: 3.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3656877, upper bound: 0.3749278
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3702603, upper bound: 0.3703547
time: 3.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 22.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3692185, upper bound: 0.3714009
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3737904, upper bound: 0.3668270
time: 3.39 seconds

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

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3644593, upper bound: 0.3761549
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3690319, upper bound: 0.3715816
time: 3.35 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.89 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3715820, upper bound: 0.3690330
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3761539, upper bound: 0.3644592
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3668272, upper bound: 0.3737914
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3713998, upper bound: 0.3692185
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3703549, upper bound: 0.3702614
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3749268, upper bound: 0.3656876
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3655980, upper bound: 0.3750178
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3701706, upper bound: 0.3704459
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3704449, upper bound: 0.3701717
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3750167, upper bound: 0.3655980
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3656877, upper bound: 0.3749278
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3702603, upper bound: 0.3703547
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3692185, upper bound: 0.3714009
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3737904, upper bound: 0.3668270
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.89
Output dim: 6, lower bound: -0.3644593, upper bound: 0.3761549
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.89
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

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2117
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2083
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 324
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1097
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1970
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 2582

Time for candidate selection: 0.41 seconds

### Candidate
type: DSZ, layer: 3, pos: 417

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3707744, upper bound: 0.3655049
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3680453, upper bound: 0.3682309
time: 3.34 seconds

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

Time for backsubstitution: 21.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2117
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2083
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 324
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1097
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1970
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 2582

Time for candidate selection: 0.41 seconds

### Candidate
type: DSZ, layer: 3, pos: 417

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3753463, upper bound: 0.3609319
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3726166, upper bound: 0.3636571
time: 3.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7512245, 0.7513671
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8169503, 0.8168478
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6746767, 0.6750436
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0492263, 1.0496294
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8715720, 0.8722582
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5806267, 0.5805032
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7756658, 0.7754583
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8739386, 0.8744481
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6932216, 0.6925783
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6894708, 0.6898751

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2117
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2083
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 324
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1097
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1970
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 2582

Time for candidate selection: 0.47 seconds

### Candidate
type: DSZ, layer: 3, pos: 417

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3660244, upper bound: 0.3702547
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3632954, upper bound: 0.3729838
time: 3.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7511787, 0.7514129
1: -15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8171611, 0.8166370
2: -3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6748800, 0.6748402
3: -9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0493937, 1.0494621
4: -5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8717198, 0.8721104
5: 1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5805981, 0.5805321
6: 6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7756996, 0.7754245
7: -19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8740044, 0.8743825
8: -1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6930714, 0.6927280
9: -6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6894450, 0.6899004

Time for backsubstitution: 22.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2117
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2083
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 324
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1097
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 941
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 2129
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1970
type: DSZ, layer: 3, pos: 1450
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 2582

Time for candidate selection: 0.43 seconds

### Candidate
type: DSZ, layer: 3, pos: 417

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3705964, upper bound: 0.3656816
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3678673, upper bound: 0.3684107
time: 3.33 seconds

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

Time for backsubstitution: 22.16 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.13 + 547.91 = 604.04 seconds
