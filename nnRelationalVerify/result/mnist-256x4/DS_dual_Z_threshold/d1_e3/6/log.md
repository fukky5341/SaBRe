## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0009916


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0022326, 0.0022326)
1: (-0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0005563, 0.0005563)
2: (-0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0029481, 0.0029481)
3: (-0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0013418, 0.0013418)
4: (0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0005706, 0.0005706)
5: (-0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0037079, 0.0037079)
6: (0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0009411, 0.0009411)
7: (-0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0024349, 0.0024349)
8: (-0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0012805, 0.0012805)
9: (-0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0014848, 0.0014848)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 1.88 = 3.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0012395, upper bound: 0.0012395

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012395, upper bound: 0.0012385
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012385, upper bound: 0.0012395
time: 1.08 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.32 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.32
Output dim: 0, lower bound: -0.0012395, upper bound: 0.0012385
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.32
Output dim: 0, lower bound: -0.0012385, upper bound: 0.0012395

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0022288, 0.0022281
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0005553, 0.0005552
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0029422, 0.0029431
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0013396, 0.0013392
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0005695, 0.0005696
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0037005, 0.0037016
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0009395, 0.0009392
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0024308, 0.0024301
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0012783, 0.0012779
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0014818, 0.0014823

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011661, upper bound: 0.0011113
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011136, upper bound: 0.0011655
time: 1.07 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0022281, 0.0022288
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0005552, 0.0005553
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0029431, 0.0029422
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0013392, 0.0013396
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0005696, 0.0005695
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0037016, 0.0037005
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0009392, 0.0009395
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0024301, 0.0024308
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0012779, 0.0012783
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0014823, 0.0014818

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011655, upper bound: 0.0011136
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011113, upper bound: 0.0011661
time: 1.07 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.39 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -0.0011661, upper bound: 0.0011113
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -0.0011136, upper bound: 0.0011655
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -0.0011655, upper bound: 0.0011136
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -0.0011113, upper bound: 0.0011661

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0019208, 0.0018897
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004786, 0.0004709
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0024953, 0.0025365
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011545, 0.0011357
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004830, 0.0004909
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0031384, 0.0031902
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0008097, 0.0007966
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0020950, 0.0020609
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0011017, 0.0010838
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012568, 0.0012775

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011578, upper bound: 0.0011057
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011603, upper bound: 0.0010900
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0018903, 0.0019205
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004710, 0.0004785
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0025360, 0.0024962
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011361, 0.0011543
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004908, 0.0004831
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0031896, 0.0031395
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0007968, 0.0008096
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0020617, 0.0020946
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0010842, 0.0011015
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012773, 0.0012572

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010926, upper bound: 0.0011596
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011080, upper bound: 0.0011572
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0019205, 0.0018903
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004785, 0.0004710
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0024962, 0.0025360
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011543, 0.0011361
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004831, 0.0004908
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0031395, 0.0031896
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0008096, 0.0007968
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0020946, 0.0020617
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0011015, 0.0010842
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012572, 0.0012773

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011572, upper bound: 0.0011080
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011596, upper bound: 0.0010926
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0018897, 0.0019208
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004709, 0.0004786
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0025365, 0.0024953
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011357, 0.0011545
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004909, 0.0004830
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0031902, 0.0031384
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0007966, 0.0008097
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0020609, 0.0020950
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0010838, 0.0011017
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012775, 0.0012568

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010900, upper bound: 0.0011603
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011058, upper bound: 0.0011577
time: 1.15 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.70 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -0.0011578, upper bound: 0.0011057
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -0.0011603, upper bound: 0.0010900
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -0.0010926, upper bound: 0.0011596
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -0.0011080, upper bound: 0.0011572
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -0.0011572, upper bound: 0.0011080
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -0.0011596, upper bound: 0.0010926
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -0.0010900, upper bound: 0.0011603
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -0.0011058, upper bound: 0.0011577

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0019300, 0.0019238
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004809, 0.0004794
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0025404, 0.0025485
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011600, 0.0011563
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004917, 0.0004933
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0031952, 0.0032054
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0008136, 0.0008110
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0021049, 0.0020982
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0011070, 0.0011034
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012795, 0.0012836

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005594, upper bound: 0.0005432
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005594, upper bound: 0.0005432
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0019453, 0.0018988
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004847, 0.0004731
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0025074, 0.0025687
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011692, 0.0011413
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004853, 0.0004972
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0031536, 0.0032308
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0008200, 0.0008004
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0021216, 0.0020709
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0011157, 0.0010891
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012629, 0.0012937

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005583, upper bound: 0.0005423
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005583, upper bound: 0.0005423
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0018995, 0.0019449
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004733, 0.0004846
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0025683, 0.0025083
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011417, 0.0011690
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004971, 0.0004855
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0032302, 0.0031547
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0008007, 0.0008199
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0020717, 0.0021212
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0010895, 0.0011155
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012935, 0.0012633

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005430, upper bound: 0.0005579
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005430, upper bound: 0.0005579
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0019243, 0.0019297
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004795, 0.0004808
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0025481, 0.0025411
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011566, 0.0011598
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004932, 0.0004918
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0032049, 0.0031960
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0008112, 0.0008134
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0020988, 0.0021046
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0011037, 0.0011068
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012834, 0.0012798

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005437, upper bound: 0.0005589
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005437, upper bound: 0.0005589
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0019297, 0.0019243
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004808, 0.0004795
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0025411, 0.0025481
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011598, 0.0011566
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004918, 0.0004932
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0031960, 0.0032049
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0008134, 0.0008112
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0021046, 0.0020988
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0011068, 0.0011037
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012798, 0.0012834

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005589, upper bound: 0.0005437
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005589, upper bound: 0.0005437
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0019449, 0.0018995
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004846, 0.0004733
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0025083, 0.0025683
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011690, 0.0011417
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004855, 0.0004971
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0031547, 0.0032302
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0008199, 0.0008007
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0021212, 0.0020717
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0011155, 0.0010895
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012633, 0.0012935

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005579, upper bound: 0.0005430
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005579, upper bound: 0.0005430
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0018988, 0.0019453
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004731, 0.0004847
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0025687, 0.0025074
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011413, 0.0011692
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004972, 0.0004853
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0032308, 0.0031536
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0008004, 0.0008200
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0020709, 0.0021216
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0010891, 0.0011157
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012937, 0.0012629

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005423, upper bound: 0.0005583
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005423, upper bound: 0.0005583
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0019238, 0.0019300
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004794, 0.0004809
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0025485, 0.0025404
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011563, 0.0011600
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004933, 0.0004917
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0032054, 0.0031952
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0008110, 0.0008136
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0020982, 0.0021049
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0011034, 0.0011070
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012836, 0.0012795

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005431, upper bound: 0.0005594
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005431, upper bound: 0.0005594
time: 0.71 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.00 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005594, upper bound: 0.0005432
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005594, upper bound: 0.0005432
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005583, upper bound: 0.0005423
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005583, upper bound: 0.0005423
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005430, upper bound: 0.0005579
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005430, upper bound: 0.0005579
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005437, upper bound: 0.0005589
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005437, upper bound: 0.0005589
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005589, upper bound: 0.0005437
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005589, upper bound: 0.0005437
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005579, upper bound: 0.0005430
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005579, upper bound: 0.0005430
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005423, upper bound: 0.0005583
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005423, upper bound: 0.0005583
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005431, upper bound: 0.0005594
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0005431, upper bound: 0.0005594

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.30 + 46.36 = 49.66 seconds
