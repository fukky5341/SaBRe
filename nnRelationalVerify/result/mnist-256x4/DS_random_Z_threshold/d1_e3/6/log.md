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
execution time: IAR + RelationalAnalysis = 0.93 + 1.85 = 2.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0012395, upper bound: 0.0012395

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012041, upper bound: 0.0012026
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012025, upper bound: 0.0012041
time: 1.05 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.09 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.09
Output dim: 0, lower bound: -0.0012041, upper bound: 0.0012026
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.09
Output dim: 0, lower bound: -0.0012025, upper bound: 0.0012041

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0021984, 0.0021918
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0005478, 0.0005461
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0028942, 0.0029029
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0013213, 0.0013173
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0005602, 0.0005619
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0036402, 0.0036511
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0009267, 0.0009239
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0023976, 0.0023905
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0012609, 0.0012571
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0014577, 0.0014621

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011912, upper bound: 0.0011966
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011981, upper bound: 0.0011894
time: 1.03 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0021918, 0.0021984
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0005461, 0.0005478
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0029029, 0.0028942
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0013173, 0.0013213
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0005619, 0.0005602
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0036511, 0.0036402
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0009239, 0.0009267
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0023905, 0.0023976
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0012571, 0.0012609
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0014621, 0.0014577

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011894, upper bound: 0.0011982
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011967, upper bound: 0.0011912
time: 1.04 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.93 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 0, lower bound: -0.0011912, upper bound: 0.0011966
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 0, lower bound: -0.0011981, upper bound: 0.0011894
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 0, lower bound: -0.0011894, upper bound: 0.0011982
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 0, lower bound: -0.0011967, upper bound: 0.0011912

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0022746, 0.0022929
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0005668, 0.0005713
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0030277, 0.0030036
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0013671, 0.0013781
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0005860, 0.0005813
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0038081, 0.0037777
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0009588, 0.0009665
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0024808, 0.0025007
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0013046, 0.0013151
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0015249, 0.0015128

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011209, upper bound: 0.0010697
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010578, upper bound: 0.0011231
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0022996, 0.0022680
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0005730, 0.0005651
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0029949, 0.0030366
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0013821, 0.0013632
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0005797, 0.0005877
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0037668, 0.0038193
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0009694, 0.0009561
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0025081, 0.0024736
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0013190, 0.0013008
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0015084, 0.0015294

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007543, upper bound: 0.0007511
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007543, upper bound: 0.0007511
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0022680, 0.0022996
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0005651, 0.0005730
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0030366, 0.0029949
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0013632, 0.0013821
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0005877, 0.0005797
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0038193, 0.0037668
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0009561, 0.0009694
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0024736, 0.0025081
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0013008, 0.0013190
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0015294, 0.0015084

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007511, upper bound: 0.0007543
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007511, upper bound: 0.0007543
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0022929, 0.0022746
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0005713, 0.0005668
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0030036, 0.0030277
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0013781, 0.0013671
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0005813, 0.0005860
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0037777, 0.0038081
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0009665, 0.0009588
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0025007, 0.0024808
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0013151, 0.0013046
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0015128, 0.0015249

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007543, upper bound: 0.0007511
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007543, upper bound: 0.0007511
time: 0.82 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.35 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 0, lower bound: -0.0011209, upper bound: 0.0010697
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 0, lower bound: -0.0010578, upper bound: 0.0011231
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.35
Output dim: 0, lower bound: -0.0007543, upper bound: 0.0007511
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.35
Output dim: 0, lower bound: -0.0007543, upper bound: 0.0007511
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.35
Output dim: 0, lower bound: -0.0007511, upper bound: 0.0007543
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.35
Output dim: 0, lower bound: -0.0007511, upper bound: 0.0007543
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.35
Output dim: 0, lower bound: -0.0007543, upper bound: 0.0007511
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.35
Output dim: 0, lower bound: -0.0007543, upper bound: 0.0007511

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0018912, 0.0018810
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004712, 0.0004687
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0024839, 0.0024973
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011367, 0.0011306
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004808, 0.0004834
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0031241, 0.0031410
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0007972, 0.0007929
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0020627, 0.0020515
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0010847, 0.0010789
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012510, 0.0012578

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011209, upper bound: 0.0010680
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011207, upper bound: 0.0010697
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0018628, 0.0019020
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004642, 0.0004739
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0025115, 0.0024598
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011196, 0.0011431
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004861, 0.0004761
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0031588, 0.0030937
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0007852, 0.0008017
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0020316, 0.0020744
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0010684, 0.0010909
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012649, 0.0012389

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006330, upper bound: 0.0006692
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006330, upper bound: 0.0006692
time: 0.86 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.67 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.0011209, upper bound: 0.0010680
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.0011207, upper bound: 0.0010697
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.0006330, upper bound: 0.0006692
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.0006330, upper bound: 0.0006692

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0018888, 0.0018781
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004706, 0.0004680
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0024800, 0.0024941
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011352, 0.0011288
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004800, 0.0004827
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0031192, 0.0031370
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0007962, 0.0007917
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0020600, 0.0020483
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0010833, 0.0010772
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012491, 0.0012562

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006665, upper bound: 0.0006449
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006665, upper bound: 0.0006449
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0018885, 0.0018786
1: -0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0004706, 0.0004681
2: -0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0024807, 0.0024937
3: -0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0011350, 0.0011291
4: 0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0004801, 0.0004827
5: -0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0031200, 0.0031365
6: 0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0007961, 0.0007919
7: -0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0020597, 0.0020489
8: -0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0010832, 0.0010775
9: -0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0012494, 0.0012560

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006664, upper bound: 0.0006449
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006664, upper bound: 0.0006449
time: 0.85 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.64 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0006665, upper bound: 0.0006449
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0006665, upper bound: 0.0006449
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0006664, upper bound: 0.0006449
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0006664, upper bound: 0.0006449

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.78 + 33.58 = 36.36 seconds
