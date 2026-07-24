## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.001357455


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625)
1: (-0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128)
2: (0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554)
3: (-0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0126610, 0.0126610)
4: (-0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531)
5: (0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450)
6: (0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107)
7: (-0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458)
8: (0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171)
9: (0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0053181, 0.0053181)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.51 + 2.51 = 4.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0014289, upper bound: 0.0014289

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014208, upper bound: 0.0014129
time: 1.91 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014216, upper bound: 0.0014215
time: 1.41 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.45 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.45
Output dim: 2, lower bound: -0.0014208, upper bound: 0.0014129
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.45
Output dim: 2, lower bound: -0.0014216, upper bound: 0.0014215

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0042086, -0.0040458, -0.0042086, -0.0040462, -0.0001624, 0.0001628
1: -0.0102683, -0.0086585, -0.0102672, -0.0085730, -0.0016953, 0.0016086
2: 0.9641411, 0.9660729, 0.9641424, 0.9661755, -0.0020344, 0.0019304
3: -0.0181838, -0.0039348, -0.0181734, -0.0031781, -0.0124855, 0.0116178
4: -0.0003938, 0.0006900, -0.0004513, 0.0006892, -0.0010829, 0.0011413
5: 0.0168724, 0.0185485, 0.0168142, 0.0185462, -0.0016738, 0.0017343
6: 0.0016203, 0.0037001, 0.0016247, 0.0037284, -0.0021081, 0.0020753
7: -0.0067585, -0.0021471, -0.0069546, -0.0021522, -0.0046063, 0.0048075
8: 0.0113673, 0.0142969, 0.0112117, 0.0142948, -0.0029275, 0.0030852
9: 0.0181697, 0.0234390, 0.0178899, 0.0234351, -0.0049601, 0.0052575

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014129
time: 1.96 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014129
time: 1.73 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0042086, -0.0040464, -0.0042086, -0.0040461, -0.0001624, 0.0001622
1: -0.0102666, -0.0085844, -0.0102674, -0.0085546, -0.0017120, 0.0016830
2: 0.9641432, 0.9661617, 0.9641421, 0.9661976, -0.0020544, 0.0020195
3: -0.0181685, -0.0032789, -0.0181756, -0.0030149, -0.0126539, 0.0121791
4: -0.0004436, 0.0006888, -0.0004637, 0.0006893, -0.0011330, 0.0011525
5: 0.0168220, 0.0185451, 0.0168017, 0.0185467, -0.0017247, 0.0017434
6: 0.0016268, 0.0037246, 0.0016238, 0.0037345, -0.0021076, 0.0021008
7: -0.0069285, -0.0021546, -0.0069969, -0.0021511, -0.0047774, 0.0048423
8: 0.0112324, 0.0142938, 0.0111781, 0.0142952, -0.0030628, 0.0031156
9: 0.0179272, 0.0234333, 0.0178295, 0.0234360, -0.0051932, 0.0053155

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014208
time: 1.74 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014215
time: 2.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.29 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.29
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014129
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.29
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014129
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.29
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014208
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.29
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014215

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0042086, -0.0040458, -0.0042086, -0.0040458, -0.0001628, 0.0001628
1: -0.0102683, -0.0086585, -0.0102683, -0.0086585, -0.0016098, 0.0016098
2: 0.9641411, 0.9660729, 0.9641411, 0.9660729, -0.0019317, 0.0019317
3: -0.0181838, -0.0039348, -0.0181838, -0.0039348, -0.0116032, 0.0116032
4: -0.0003938, 0.0006900, -0.0003938, 0.0006900, -0.0010837, 0.0010837
5: 0.0168724, 0.0185485, 0.0168724, 0.0185485, -0.0016761, 0.0016761
6: 0.0016203, 0.0037001, 0.0016203, 0.0037001, -0.0020798, 0.0020798
7: -0.0067585, -0.0021471, -0.0067585, -0.0021471, -0.0046114, 0.0046114
8: 0.0113673, 0.0142969, 0.0113673, 0.0142969, -0.0029296, 0.0029296
9: 0.0181697, 0.0234390, 0.0181697, 0.0234390, -0.0049601, 0.0049601

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013980, upper bound: 0.0014087
time: 1.64 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0014088
time: 1.63 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042086, -0.0040458, -0.0042086, -0.0040464, -0.0001622, 0.0001628
1: -0.0102683, -0.0086585, -0.0102666, -0.0085844, -0.0016839, 0.0016081
2: 0.9641411, 0.9660729, 0.9641432, 0.9661617, -0.0020205, 0.0019297
3: -0.0181838, -0.0039348, -0.0181685, -0.0032789, -0.0123873, 0.0116129
4: -0.0003938, 0.0006900, -0.0004436, 0.0006888, -0.0010825, 0.0011336
5: 0.0168724, 0.0185485, 0.0168220, 0.0185451, -0.0016727, 0.0017266
6: 0.0016203, 0.0037001, 0.0016268, 0.0037246, -0.0021043, 0.0020732
7: -0.0067585, -0.0021471, -0.0069285, -0.0021546, -0.0046039, 0.0047814
8: 0.0113673, 0.0142969, 0.0112324, 0.0142938, -0.0029265, 0.0030645
9: 0.0181697, 0.0234390, 0.0179272, 0.0234333, -0.0049583, 0.0052203

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013980, upper bound: 0.0014088
time: 1.66 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0014088
time: 1.71 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042086, -0.0040464, -0.0042086, -0.0040458, -0.0001628, 0.0001622
1: -0.0102666, -0.0085844, -0.0102683, -0.0086585, -0.0016081, 0.0016839
2: 0.9641432, 0.9661617, 0.9641411, 0.9660729, -0.0019297, 0.0020205
3: -0.0181685, -0.0032789, -0.0181838, -0.0039348, -0.0116129, 0.0123873
4: -0.0004436, 0.0006888, -0.0003938, 0.0006900, -0.0011336, 0.0010825
5: 0.0168220, 0.0185451, 0.0168724, 0.0185485, -0.0017266, 0.0016727
6: 0.0016268, 0.0037246, 0.0016203, 0.0037001, -0.0020732, 0.0021043
7: -0.0069285, -0.0021546, -0.0067585, -0.0021471, -0.0047814, 0.0046039
8: 0.0112324, 0.0142938, 0.0113673, 0.0142969, -0.0030645, 0.0029265
9: 0.0179272, 0.0234333, 0.0181697, 0.0234390, -0.0052203, 0.0049583

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013969, upper bound: 0.0014164
time: 1.72 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014088, upper bound: 0.0014165
time: 1.89 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042086, -0.0040464, -0.0042086, -0.0040464, -0.0001622, 0.0001622
1: -0.0102666, -0.0085844, -0.0102666, -0.0085844, -0.0016822, 0.0016822
2: 0.9641432, 0.9661617, 0.9641432, 0.9661617, -0.0020185, 0.0020185
3: -0.0181685, -0.0032789, -0.0181685, -0.0032789, -0.0121720, 0.0121720
4: -0.0004436, 0.0006888, -0.0004436, 0.0006888, -0.0011324, 0.0011324
5: 0.0168220, 0.0185451, 0.0168220, 0.0185451, -0.0017231, 0.0017231
6: 0.0016268, 0.0037246, 0.0016268, 0.0037246, -0.0020978, 0.0020978
7: -0.0069285, -0.0021546, -0.0069285, -0.0021546, -0.0047739, 0.0047739
8: 0.0112324, 0.0142938, 0.0112324, 0.0142938, -0.0030613, 0.0030613
9: 0.0179272, 0.0234333, 0.0179272, 0.0234333, -0.0051906, 0.0051906

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013969, upper bound: 0.0014171
time: 1.62 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014088, upper bound: 0.0014172
time: 1.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.77 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 2, lower bound: -0.0013980, upper bound: 0.0014087
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0014088
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 2, lower bound: -0.0013980, upper bound: 0.0014088
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0014088
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 2, lower bound: -0.0013969, upper bound: 0.0014164
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 2, lower bound: -0.0014088, upper bound: 0.0014165
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 2, lower bound: -0.0013969, upper bound: 0.0014171
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 2, lower bound: -0.0014088, upper bound: 0.0014172

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042069, -0.0040674, -0.0042083, -0.0040496, -0.0001573, 0.0001410
1: -0.0102044, -0.0086723, -0.0102570, -0.0086599, -0.0015445, 0.0015848
2: 0.9642177, 0.9660563, 0.9641545, 0.9660711, -0.0018534, 0.0019018
3: -0.0176182, -0.0040566, -0.0180837, -0.0039474, -0.0110131, 0.0113404
4: -0.0003845, 0.0006469, -0.0003928, 0.0006823, -0.0010668, 0.0010397
5: 0.0168817, 0.0184221, 0.0168733, 0.0185261, -0.0016444, 0.0015487
6: 0.0018624, 0.0036955, 0.0016631, 0.0036996, -0.0018372, 0.0020324
7: -0.0067270, -0.0024249, -0.0067552, -0.0021963, -0.0045307, 0.0043303
8: 0.0113923, 0.0141806, 0.0113698, 0.0142763, -0.0028840, 0.0028108
9: 0.0182148, 0.0232298, 0.0181744, 0.0234020, -0.0048756, 0.0047448

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013739, upper bound: 0.0011900
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013951, upper bound: 0.0014066
time: 1.69 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042082, -0.0040511, -0.0042086, -0.0040458, -0.0001624, 0.0001575
1: -0.0102526, -0.0086629, -0.0102683, -0.0086585, -0.0015940, 0.0016054
2: 0.9641600, 0.9660675, 0.9641411, 0.9660729, -0.0019129, 0.0019264
3: -0.0180443, -0.0039738, -0.0181838, -0.0039348, -0.0112967, 0.0115636
4: -0.0003908, 0.0006793, -0.0003938, 0.0006900, -0.0010807, 0.0010731
5: 0.0168754, 0.0185173, 0.0168724, 0.0185485, -0.0016731, 0.0016449
6: 0.0016800, 0.0036986, 0.0016203, 0.0037001, -0.0020200, 0.0020783
7: -0.0067484, -0.0022157, -0.0067585, -0.0021471, -0.0046013, 0.0045428
8: 0.0113753, 0.0142682, 0.0113673, 0.0142969, -0.0029216, 0.0029009
9: 0.0181841, 0.0233874, 0.0181697, 0.0234390, -0.0049457, 0.0048922

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0013980
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0014095
time: 1.63 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042069, -0.0040674, -0.0042083, -0.0040502, -0.0001567, 0.0001409
1: -0.0102044, -0.0086723, -0.0102553, -0.0085860, -0.0016184, 0.0015830
2: 0.9642177, 0.9660563, 0.9641567, 0.9661599, -0.0019422, 0.0018997
3: -0.0176182, -0.0040566, -0.0180685, -0.0032928, -0.0117984, 0.0113516
4: -0.0003845, 0.0006469, -0.0004426, 0.0006812, -0.0010657, 0.0010895
5: 0.0168817, 0.0184221, 0.0168230, 0.0185227, -0.0016410, 0.0015990
6: 0.0018624, 0.0036955, 0.0016696, 0.0037241, -0.0018617, 0.0020259
7: -0.0067270, -0.0024249, -0.0069249, -0.0022037, -0.0045232, 0.0045000
8: 0.0113923, 0.0141806, 0.0112353, 0.0142732, -0.0028809, 0.0029454
9: 0.0182148, 0.0232298, 0.0179323, 0.0233963, -0.0048739, 0.0050044

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013872, upper bound: 0.0011932
time: 1.45 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014030, upper bound: 0.0014059
time: 1.58 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042082, -0.0040511, -0.0042086, -0.0040464, -0.0001618, 0.0001574
1: -0.0102526, -0.0086629, -0.0102666, -0.0085844, -0.0016682, 0.0016037
2: 0.9641600, 0.9660675, 0.9641432, 0.9661617, -0.0020017, 0.0019243
3: -0.0180443, -0.0039738, -0.0181685, -0.0032789, -0.0120903, 0.0115733
4: -0.0003908, 0.0006793, -0.0004436, 0.0006888, -0.0010796, 0.0011230
5: 0.0168754, 0.0185173, 0.0168220, 0.0185451, -0.0016697, 0.0016953
6: 0.0016800, 0.0036986, 0.0016268, 0.0037246, -0.0020446, 0.0020718
7: -0.0067484, -0.0022157, -0.0069285, -0.0021546, -0.0045938, 0.0047128
8: 0.0113753, 0.0142682, 0.0112324, 0.0142938, -0.0029185, 0.0030358
9: 0.0181841, 0.0233874, 0.0179272, 0.0234333, -0.0049439, 0.0051529

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014164, upper bound: 0.0013969
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014164, upper bound: 0.0014088
time: 1.83 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042069, -0.0040678, -0.0042083, -0.0040496, -0.0001572, 0.0001405
1: -0.0102030, -0.0085989, -0.0102570, -0.0086599, -0.0015431, 0.0016581
2: 0.9642194, 0.9661443, 0.9641545, 0.9660711, -0.0018517, 0.0019898
3: -0.0176058, -0.0034075, -0.0180837, -0.0039474, -0.0110336, 0.0121376
4: -0.0004339, 0.0006460, -0.0003928, 0.0006823, -0.0011162, 0.0010388
5: 0.0168318, 0.0184193, 0.0168733, 0.0185261, -0.0016943, 0.0015459
6: 0.0018677, 0.0037198, 0.0016631, 0.0036996, -0.0018319, 0.0020566
7: -0.0068952, -0.0024310, -0.0067552, -0.0021963, -0.0046989, 0.0043242
8: 0.0112589, 0.0141781, 0.0113698, 0.0142763, -0.0030175, 0.0028082
9: 0.0179747, 0.0232252, 0.0181744, 0.0234020, -0.0051362, 0.0047435

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013739, upper bound: 0.0012016
time: 1.32 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013940, upper bound: 0.0014134
time: 1.71 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042081, -0.0040517, -0.0042086, -0.0040458, -0.0001624, 0.0001569
1: -0.0102509, -0.0085891, -0.0102683, -0.0086585, -0.0015923, 0.0016792
2: 0.9641619, 0.9661561, 0.9641411, 0.9660729, -0.0019109, 0.0020150
3: -0.0180291, -0.0033205, -0.0181838, -0.0039348, -0.0113081, 0.0123448
4: -0.0004405, 0.0006782, -0.0003938, 0.0006900, -0.0011304, 0.0010719
5: 0.0168252, 0.0185139, 0.0168724, 0.0185485, -0.0017234, 0.0016416
6: 0.0016865, 0.0037230, 0.0016203, 0.0037001, -0.0020136, 0.0021028
7: -0.0069177, -0.0022231, -0.0067585, -0.0021471, -0.0047706, 0.0045354
8: 0.0112410, 0.0142651, 0.0113673, 0.0142969, -0.0030559, 0.0028978
9: 0.0179425, 0.0233818, 0.0181697, 0.0234390, -0.0052050, 0.0048896

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014087, upper bound: 0.0014061
time: 2.02 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014087, upper bound: 0.0014165
time: 1.81 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042069, -0.0040678, -0.0042083, -0.0040502, -0.0001567, 0.0001405
1: -0.0102030, -0.0085989, -0.0102553, -0.0085860, -0.0016170, 0.0016564
2: 0.9642194, 0.9661443, 0.9641567, 0.9661599, -0.0019405, 0.0019876
3: -0.0176058, -0.0034075, -0.0180685, -0.0032928, -0.0115823, 0.0119113
4: -0.0004339, 0.0006460, -0.0004426, 0.0006812, -0.0011150, 0.0010886
5: 0.0168318, 0.0184193, 0.0168230, 0.0185227, -0.0016909, 0.0015963
6: 0.0018677, 0.0037198, 0.0016696, 0.0037241, -0.0018564, 0.0020501
7: -0.0068952, -0.0024310, -0.0069249, -0.0022037, -0.0046914, 0.0044939
8: 0.0112589, 0.0141781, 0.0112353, 0.0142732, -0.0030143, 0.0029428
9: 0.0179747, 0.0232252, 0.0179323, 0.0233963, -0.0051064, 0.0049746

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013791, upper bound: 0.0012096
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013952, upper bound: 0.0014141
time: 2.44 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042081, -0.0040517, -0.0042086, -0.0040464, -0.0001618, 0.0001569
1: -0.0102509, -0.0085891, -0.0102666, -0.0085844, -0.0016664, 0.0016775
2: 0.9641619, 0.9661561, 0.9641432, 0.9661617, -0.0019997, 0.0020130
3: -0.0180291, -0.0033205, -0.0181685, -0.0032789, -0.0118595, 0.0121302
4: -0.0004405, 0.0006782, -0.0004436, 0.0006888, -0.0011293, 0.0011218
5: 0.0168252, 0.0185139, 0.0168220, 0.0185451, -0.0017199, 0.0016920
6: 0.0016865, 0.0037230, 0.0016268, 0.0037246, -0.0020381, 0.0020962
7: -0.0069177, -0.0022231, -0.0069285, -0.0021546, -0.0047631, 0.0047054
8: 0.0112410, 0.0142651, 0.0112324, 0.0142938, -0.0030528, 0.0030327
9: 0.0179425, 0.0233818, 0.0179272, 0.0234333, -0.0051753, 0.0051215

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014104, upper bound: 0.0014066
time: 2.05 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014104, upper bound: 0.0014171
time: 2.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.84 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0013739, upper bound: 0.0011900
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0013951, upper bound: 0.0014066
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0013980
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0014095
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0013872, upper bound: 0.0011932
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0014030, upper bound: 0.0014059
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0014164, upper bound: 0.0013969
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0014164, upper bound: 0.0014088
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0013739, upper bound: 0.0012016
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0013940, upper bound: 0.0014134
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0014087, upper bound: 0.0014061
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0014087, upper bound: 0.0014165
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0013791, upper bound: 0.0012096
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0013952, upper bound: 0.0014141
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0014104, upper bound: 0.0014066
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.84
Output dim: 2, lower bound: -0.0014104, upper bound: 0.0014171

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0042063, -0.0040750, -0.0042030, -0.0041165, -0.0000898, 0.0001281
1: -0.0101818, -0.0086728, -0.0100588, -0.0085737, -0.0016081, 0.0013861
2: 0.9642448, 0.9660557, 0.9643924, 0.9661746, -0.0019298, 0.0016633
3: -0.0174182, -0.0040610, -0.0163296, -0.0031842, -0.0114249, 0.0094759
4: -0.0003842, 0.0006317, -0.0004508, 0.0005489, -0.0009331, 0.0010826
5: 0.0168821, 0.0183773, 0.0168147, 0.0181340, -0.0012519, 0.0015627
6: 0.0019480, 0.0036954, 0.0024141, 0.0037281, -0.0017801, 0.0012813
7: -0.0067258, -0.0025232, -0.0069530, -0.0030578, -0.0036680, 0.0044299
8: 0.0113932, 0.0141395, 0.0112129, 0.0139157, -0.0025225, 0.0029266
9: 0.0182164, 0.0231559, 0.0178921, 0.0227533, -0.0042189, 0.0049460

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0011795
time: 1.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0011900
time: 1.96 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042069, -0.0040674, -0.0042081, -0.0040524, -0.0001545, 0.0001407
1: -0.0102044, -0.0086723, -0.0102488, -0.0086605, -0.0015439, 0.0015765
2: 0.9642177, 0.9660563, 0.9641646, 0.9660705, -0.0018528, 0.0018918
3: -0.0176182, -0.0040566, -0.0180106, -0.0039527, -0.0110074, 0.0107701
4: -0.0003845, 0.0006469, -0.0003924, 0.0006768, -0.0010613, 0.0010393
5: 0.0168817, 0.0184221, 0.0168738, 0.0185098, -0.0016281, 0.0015483
6: 0.0018624, 0.0036955, 0.0016944, 0.0036994, -0.0018370, 0.0020011
7: -0.0067270, -0.0024249, -0.0067539, -0.0022321, -0.0044948, 0.0043289
8: 0.0113923, 0.0141806, 0.0113710, 0.0142613, -0.0028690, 0.0028097
9: 0.0182148, 0.0232298, 0.0181764, 0.0233749, -0.0048037, 0.0047429

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011019, upper bound: 0.0013960
time: 1.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011019, upper bound: 0.0014066
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042082, -0.0040511, -0.0042069, -0.0040674, -0.0001408, 0.0001558
1: -0.0102526, -0.0086629, -0.0102044, -0.0086723, -0.0015803, 0.0015415
2: 0.9641600, 0.9660675, 0.9642177, 0.9660563, -0.0018964, 0.0018498
3: -0.0180443, -0.0039738, -0.0176182, -0.0040566, -0.0113156, 0.0109866
4: -0.0003908, 0.0006793, -0.0003845, 0.0006469, -0.0010377, 0.0010638
5: 0.0168754, 0.0185173, 0.0168817, 0.0184221, -0.0015467, 0.0016356
6: 0.0016800, 0.0036986, 0.0018624, 0.0036955, -0.0020155, 0.0018362
7: -0.0067484, -0.0022157, -0.0067270, -0.0024249, -0.0043235, 0.0045113
8: 0.0113753, 0.0142682, 0.0113923, 0.0141806, -0.0028053, 0.0028759
9: 0.0181841, 0.0233874, 0.0182148, 0.0232298, -0.0047351, 0.0048626

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011900, upper bound: 0.0013739
time: 1.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014066, upper bound: 0.0013951
time: 1.89 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042082, -0.0040511, -0.0042082, -0.0040511, -0.0001570, 0.0001570
1: -0.0102526, -0.0086629, -0.0102526, -0.0086629, -0.0015897, 0.0015897
2: 0.9641600, 0.9660675, 0.9641600, 0.9660675, -0.0019075, 0.0019075
3: -0.0180443, -0.0039738, -0.0180443, -0.0039738, -0.0112567, 0.0112567
4: -0.0003908, 0.0006793, -0.0003908, 0.0006793, -0.0010701, 0.0010701
5: 0.0168754, 0.0185173, 0.0168754, 0.0185173, -0.0016419, 0.0016419
6: 0.0016800, 0.0036986, 0.0016800, 0.0036986, -0.0020186, 0.0020186
7: -0.0067484, -0.0022157, -0.0067484, -0.0022157, -0.0045327, 0.0045327
8: 0.0113753, 0.0142682, 0.0113753, 0.0142682, -0.0028929, 0.0028929
9: 0.0181841, 0.0233874, 0.0181841, 0.0233874, -0.0048778, 0.0048778

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011900, upper bound: 0.0013851
time: 1.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014066, upper bound: 0.0013958
time: 1.89 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0042063, -0.0040750, -0.0042030, -0.0041169, -0.0000894, 0.0001280
1: -0.0101818, -0.0086728, -0.0100577, -0.0085031, -0.0016787, 0.0013849
2: 0.9642448, 0.9660557, 0.9643938, 0.9662594, -0.0020146, 0.0016619
3: -0.0174182, -0.0040610, -0.0163193, -0.0025592, -0.0120950, 0.0095229
4: -0.0003842, 0.0006317, -0.0004984, 0.0005481, -0.0009323, 0.0011301
5: 0.0168821, 0.0183773, 0.0167666, 0.0181317, -0.0012496, 0.0016107
6: 0.0019480, 0.0036954, 0.0024185, 0.0037515, -0.0018035, 0.0012769
7: -0.0067258, -0.0025232, -0.0071150, -0.0030629, -0.0036629, 0.0045918
8: 0.0113932, 0.0141395, 0.0110844, 0.0139136, -0.0025203, 0.0030551
9: 0.0182164, 0.0231559, 0.0176611, 0.0227495, -0.0042182, 0.0051782

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011099, upper bound: 0.0011819
time: 1.31 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011099, upper bound: 0.0011932
time: 1.38 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042069, -0.0040674, -0.0042080, -0.0040529, -0.0001540, 0.0001407
1: -0.0102044, -0.0086723, -0.0102471, -0.0085868, -0.0016177, 0.0015748
2: 0.9642177, 0.9660563, 0.9641666, 0.9661590, -0.0019413, 0.0018898
3: -0.0176182, -0.0040566, -0.0179960, -0.0032996, -0.0117913, 0.0108036
4: -0.0003845, 0.0006469, -0.0004421, 0.0006757, -0.0010602, 0.0010890
5: 0.0168817, 0.0184221, 0.0168235, 0.0185065, -0.0016248, 0.0015985
6: 0.0018624, 0.0036955, 0.0017007, 0.0037238, -0.0018614, 0.0019948
7: -0.0067270, -0.0024249, -0.0069231, -0.0022393, -0.0044876, 0.0044982
8: 0.0113923, 0.0141806, 0.0112367, 0.0142583, -0.0028660, 0.0029439
9: 0.0182148, 0.0232298, 0.0179348, 0.0233695, -0.0048021, 0.0050018

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011138, upper bound: 0.0013960
time: 2.05 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011138, upper bound: 0.0014058
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042082, -0.0040511, -0.0042069, -0.0040678, -0.0001404, 0.0001557
1: -0.0102526, -0.0086629, -0.0102030, -0.0085989, -0.0016536, 0.0015401
2: 0.9641600, 0.9660675, 0.9642194, 0.9661443, -0.0019844, 0.0018481
3: -0.0180443, -0.0039738, -0.0176058, -0.0034075, -0.0121118, 0.0110071
4: -0.0003908, 0.0006793, -0.0004339, 0.0006460, -0.0010368, 0.0011132
5: 0.0168754, 0.0185173, 0.0168318, 0.0184193, -0.0015439, 0.0016855
6: 0.0016800, 0.0036986, 0.0018677, 0.0037198, -0.0020398, 0.0018309
7: -0.0067484, -0.0022157, -0.0068952, -0.0024310, -0.0043174, 0.0046795
8: 0.0113753, 0.0142682, 0.0112589, 0.0141781, -0.0028028, 0.0030094
9: 0.0181841, 0.0233874, 0.0179747, 0.0232252, -0.0047337, 0.0051232

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012016, upper bound: 0.0013739
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014133, upper bound: 0.0013940
time: 1.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042082, -0.0040511, -0.0042081, -0.0040517, -0.0001565, 0.0001570
1: -0.0102526, -0.0086629, -0.0102509, -0.0085891, -0.0016634, 0.0015879
2: 0.9641600, 0.9660675, 0.9641619, 0.9661561, -0.0019962, 0.0019056
3: -0.0180443, -0.0039738, -0.0180291, -0.0033205, -0.0120480, 0.0112680
4: -0.0003908, 0.0006793, -0.0004405, 0.0006782, -0.0010690, 0.0011198
5: 0.0168754, 0.0185173, 0.0168252, 0.0185139, -0.0016386, 0.0016922
6: 0.0016800, 0.0036986, 0.0016865, 0.0037230, -0.0020430, 0.0020121
7: -0.0067484, -0.0022157, -0.0069177, -0.0022231, -0.0045253, 0.0047020
8: 0.0113753, 0.0142682, 0.0112410, 0.0142651, -0.0028898, 0.0030272
9: 0.0181841, 0.0233874, 0.0179425, 0.0233818, -0.0048752, 0.0051376

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012016, upper bound: 0.0013850
time: 1.62 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014133, upper bound: 0.0013947
time: 2.06 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0042063, -0.0040754, -0.0042030, -0.0041165, -0.0000898, 0.0001276
1: -0.0101805, -0.0086002, -0.0100588, -0.0085737, -0.0016068, 0.0014587
2: 0.9642465, 0.9661428, 0.9643924, 0.9661746, -0.0019282, 0.0017504
3: -0.0174061, -0.0034182, -0.0163296, -0.0031842, -0.0114507, 0.0102700
4: -0.0004331, 0.0006308, -0.0004508, 0.0005489, -0.0009820, 0.0010816
5: 0.0168327, 0.0183746, 0.0168147, 0.0181340, -0.0013013, 0.0015600
6: 0.0019532, 0.0037194, 0.0024141, 0.0037281, -0.0017749, 0.0013053
7: -0.0068924, -0.0025291, -0.0069530, -0.0030578, -0.0038345, 0.0044239
8: 0.0112611, 0.0141370, 0.0112129, 0.0139157, -0.0026546, 0.0029241
9: 0.0179787, 0.0231514, 0.0178921, 0.0227533, -0.0044762, 0.0049452

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011011, upper bound: 0.0011945
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011011, upper bound: 0.0012016
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042069, -0.0040678, -0.0042081, -0.0040524, -0.0001545, 0.0001403
1: -0.0102030, -0.0085989, -0.0102488, -0.0086605, -0.0015425, 0.0016498
2: 0.9642194, 0.9661443, 0.9641646, 0.9660705, -0.0018511, 0.0019798
3: -0.0176058, -0.0034075, -0.0180106, -0.0039527, -0.0110280, 0.0116213
4: -0.0004339, 0.0006460, -0.0003924, 0.0006768, -0.0011106, 0.0010384
5: 0.0168318, 0.0184193, 0.0168738, 0.0185098, -0.0016780, 0.0015455
6: 0.0018677, 0.0037198, 0.0016944, 0.0036994, -0.0018317, 0.0020254
7: -0.0068952, -0.0024310, -0.0067539, -0.0022321, -0.0046630, 0.0043229
8: 0.0112589, 0.0141781, 0.0113710, 0.0142613, -0.0030024, 0.0028071
9: 0.0179747, 0.0232252, 0.0181764, 0.0233749, -0.0050658, 0.0047415

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011067, upper bound: 0.0014056
time: 2.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011067, upper bound: 0.0014133
time: 1.45 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042081, -0.0040517, -0.0042069, -0.0040674, -0.0001408, 0.0001552
1: -0.0102509, -0.0085891, -0.0102044, -0.0086723, -0.0015786, 0.0016153
2: 0.9641619, 0.9661561, 0.9642177, 0.9660563, -0.0018944, 0.0019384
3: -0.0180291, -0.0033205, -0.0176182, -0.0040566, -0.0113194, 0.0117703
4: -0.0004405, 0.0006782, -0.0003845, 0.0006469, -0.0010874, 0.0010627
5: 0.0168252, 0.0185139, 0.0168817, 0.0184221, -0.0015969, 0.0016322
6: 0.0016865, 0.0037230, 0.0018624, 0.0036955, -0.0020090, 0.0018606
7: -0.0069177, -0.0022231, -0.0067270, -0.0024249, -0.0044928, 0.0045039
8: 0.0112410, 0.0142651, 0.0113923, 0.0141806, -0.0029396, 0.0028728
9: 0.0179425, 0.0233818, 0.0182148, 0.0232298, -0.0049942, 0.0048599

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011932, upper bound: 0.0013872
time: 2.14 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0014031
time: 1.98 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042081, -0.0040517, -0.0042082, -0.0040511, -0.0001570, 0.0001565
1: -0.0102509, -0.0085891, -0.0102526, -0.0086629, -0.0015879, 0.0016634
2: 0.9641619, 0.9661561, 0.9641600, 0.9660675, -0.0019056, 0.0019962
3: -0.0180291, -0.0033205, -0.0180443, -0.0039738, -0.0112680, 0.0120480
4: -0.0004405, 0.0006782, -0.0003908, 0.0006793, -0.0011198, 0.0010690
5: 0.0168252, 0.0185139, 0.0168754, 0.0185173, -0.0016922, 0.0016386
6: 0.0016865, 0.0037230, 0.0016800, 0.0036986, -0.0020121, 0.0020430
7: -0.0069177, -0.0022231, -0.0067484, -0.0022157, -0.0047020, 0.0045253
8: 0.0112410, 0.0142651, 0.0113753, 0.0142682, -0.0030272, 0.0028898
9: 0.0179425, 0.0233818, 0.0181841, 0.0233874, -0.0051376, 0.0048752

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011932, upper bound: 0.0013957
time: 1.88 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0014037
time: 1.84 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0042063, -0.0040754, -0.0042030, -0.0041169, -0.0000894, 0.0001276
1: -0.0101805, -0.0086002, -0.0100577, -0.0085031, -0.0016774, 0.0014575
2: 0.9642465, 0.9661428, 0.9643938, 0.9662594, -0.0020129, 0.0017490
3: -0.0174061, -0.0034182, -0.0163193, -0.0025592, -0.0118420, 0.0100532
4: -0.0004331, 0.0006308, -0.0004984, 0.0005481, -0.0009812, 0.0011292
5: 0.0168327, 0.0183746, 0.0167666, 0.0181317, -0.0012990, 0.0016080
6: 0.0019532, 0.0037194, 0.0024185, 0.0037515, -0.0017983, 0.0013009
7: -0.0068924, -0.0025291, -0.0071150, -0.0030629, -0.0038295, 0.0045859
8: 0.0112611, 0.0141370, 0.0110844, 0.0139136, -0.0026525, 0.0030526
9: 0.0179787, 0.0231514, 0.0176611, 0.0227495, -0.0044481, 0.0051496

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011116, upper bound: 0.0012029
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011116, upper bound: 0.0012096
time: 1.46 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042069, -0.0040678, -0.0042080, -0.0040529, -0.0001539, 0.0001402
1: -0.0102030, -0.0085989, -0.0102471, -0.0085868, -0.0016163, 0.0016482
2: 0.9642194, 0.9661443, 0.9641666, 0.9661590, -0.0019396, 0.0019777
3: -0.0176058, -0.0034075, -0.0179960, -0.0032996, -0.0115752, 0.0113644
4: -0.0004339, 0.0006460, -0.0004421, 0.0006757, -0.0011095, 0.0010881
5: 0.0168318, 0.0184193, 0.0168235, 0.0185065, -0.0016747, 0.0015957
6: 0.0018677, 0.0037198, 0.0017007, 0.0037238, -0.0018561, 0.0020191
7: -0.0068952, -0.0024310, -0.0069231, -0.0022393, -0.0046558, 0.0044921
8: 0.0112589, 0.0141781, 0.0112367, 0.0142583, -0.0029994, 0.0029414
9: 0.0179747, 0.0232252, 0.0179348, 0.0233695, -0.0050338, 0.0049721

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011146, upper bound: 0.0014061
time: 1.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011146, upper bound: 0.0014141
time: 1.57 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042081, -0.0040517, -0.0042069, -0.0040678, -0.0001403, 0.0001552
1: -0.0102509, -0.0085891, -0.0102030, -0.0085989, -0.0016519, 0.0016139
2: 0.9641619, 0.9661561, 0.9642194, 0.9661443, -0.0019824, 0.0019367
3: -0.0180291, -0.0033205, -0.0176058, -0.0034075, -0.0118860, 0.0115547
4: -0.0004405, 0.0006782, -0.0004339, 0.0006460, -0.0010865, 0.0011121
5: 0.0168252, 0.0185139, 0.0168318, 0.0184193, -0.0015941, 0.0016821
6: 0.0016865, 0.0037230, 0.0018677, 0.0037198, -0.0020333, 0.0018553
7: -0.0069177, -0.0022231, -0.0068952, -0.0024310, -0.0044867, 0.0046721
8: 0.0112410, 0.0142651, 0.0112589, 0.0141781, -0.0029371, 0.0030062
9: 0.0179425, 0.0233818, 0.0179747, 0.0232252, -0.0049645, 0.0050932

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012030, upper bound: 0.0013873
time: 1.49 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014074, upper bound: 0.0014035
time: 1.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042081, -0.0040517, -0.0042081, -0.0040517, -0.0001564, 0.0001564
1: -0.0102509, -0.0085891, -0.0102509, -0.0085891, -0.0016617, 0.0016617
2: 0.9641619, 0.9661561, 0.9641619, 0.9661561, -0.0019942, 0.0019942
3: -0.0180291, -0.0033205, -0.0180291, -0.0033205, -0.0118176, 0.0118176
4: -0.0004405, 0.0006782, -0.0004405, 0.0006782, -0.0011187, 0.0011187
5: 0.0168252, 0.0185139, 0.0168252, 0.0185139, -0.0016888, 0.0016888
6: 0.0016865, 0.0037230, 0.0016865, 0.0037230, -0.0020365, 0.0020365
7: -0.0069177, -0.0022231, -0.0069177, -0.0022231, -0.0046946, 0.0046946
8: 0.0112410, 0.0142651, 0.0112410, 0.0142651, -0.0030241, 0.0030241
9: 0.0179425, 0.0233818, 0.0179425, 0.0233818, -0.0051062, 0.0051062

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012030, upper bound: 0.0013959
time: 1.45 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014074, upper bound: 0.0014043
time: 1.59 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.33 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0011795
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0011900
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011019, upper bound: 0.0013960
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011019, upper bound: 0.0014066
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011900, upper bound: 0.0013739
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0014066, upper bound: 0.0013951
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011900, upper bound: 0.0013851
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0014066, upper bound: 0.0013958
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011099, upper bound: 0.0011819
NS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011099, upper bound: 0.0011932
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011138, upper bound: 0.0013960
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011138, upper bound: 0.0014058
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0012016, upper bound: 0.0013739
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0014133, upper bound: 0.0013940
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0012016, upper bound: 0.0013850
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0014133, upper bound: 0.0013947
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011011, upper bound: 0.0011945
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011011, upper bound: 0.0012016
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011067, upper bound: 0.0014056
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011067, upper bound: 0.0014133
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011932, upper bound: 0.0013872
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0014031
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011932, upper bound: 0.0013957
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0014037
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011116, upper bound: 0.0012029
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011116, upper bound: 0.0012096
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011146, upper bound: 0.0014061
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0011146, upper bound: 0.0014141
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0012030, upper bound: 0.0013873
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0014074, upper bound: 0.0014035
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0012030, upper bound: 0.0013959
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 2, lower bound: -0.0014074, upper bound: 0.0014043

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042016, -0.0041339, -0.0042081, -0.0040524, -0.0001492, 0.0000742
1: -0.0100072, -0.0085821, -0.0102488, -0.0086605, -0.0013467, 0.0016667
2: 0.9644544, 0.9661646, 0.9641646, 0.9660705, -0.0016161, 0.0020000
3: -0.0158724, -0.0032582, -0.0180106, -0.0039527, -0.0091617, 0.0119659
4: -0.0004452, 0.0005142, -0.0003924, 0.0006768, -0.0011220, 0.0009066
5: 0.0168204, 0.0180318, 0.0168738, 0.0185098, -0.0016894, 0.0011580
6: 0.0026098, 0.0037254, 0.0016944, 0.0036994, -0.0010896, 0.0020310
7: -0.0069339, -0.0032824, -0.0067539, -0.0022321, -0.0047017, 0.0034715
8: 0.0112281, 0.0138217, 0.0113710, 0.0142613, -0.0030332, 0.0024507
9: 0.0179195, 0.0225842, 0.0181764, 0.0233749, -0.0051385, 0.0040915

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0013744
time: 1.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0013960
time: 1.57 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042067, -0.0040702, -0.0042081, -0.0040524, -0.0001543, 0.0001379
1: -0.0101961, -0.0086726, -0.0102488, -0.0086605, -0.0015355, 0.0015762
2: 0.9642278, 0.9660560, 0.9641646, 0.9660705, -0.0018427, 0.0018914
3: -0.0175440, -0.0040594, -0.0180106, -0.0039527, -0.0104284, 0.0107672
4: -0.0003843, 0.0006413, -0.0003924, 0.0006768, -0.0010611, 0.0010337
5: 0.0168820, 0.0184055, 0.0168738, 0.0185098, -0.0016279, 0.0015317
6: 0.0018942, 0.0036954, 0.0016944, 0.0036994, -0.0018052, 0.0020010
7: -0.0067262, -0.0024613, -0.0067539, -0.0022321, -0.0044941, 0.0042925
8: 0.0113929, 0.0141654, 0.0113710, 0.0142613, -0.0028684, 0.0027944
9: 0.0182158, 0.0232024, 0.0181764, 0.0233749, -0.0048027, 0.0046710

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0013797
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0013847
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042029, -0.0041182, -0.0042063, -0.0040750, -0.0001279, 0.0000881
1: -0.0100536, -0.0085752, -0.0101818, -0.0086728, -0.0013809, 0.0016066
2: 0.9643987, 0.9661729, 0.9642448, 0.9660557, -0.0016570, 0.0019281
3: -0.0162835, -0.0031973, -0.0174182, -0.0040610, -0.0094476, 0.0114116
4: -0.0004499, 0.0005454, -0.0003842, 0.0006317, -0.0010816, 0.0009296
5: 0.0168157, 0.0181237, 0.0168821, 0.0183773, -0.0015617, 0.0012416
6: 0.0024338, 0.0037276, 0.0019480, 0.0036954, -0.0012615, 0.0017796
7: -0.0069497, -0.0030805, -0.0067258, -0.0025232, -0.0044265, 0.0036453
8: 0.0112156, 0.0139062, 0.0113932, 0.0141395, -0.0029239, 0.0025130
9: 0.0178970, 0.0227363, 0.0182164, 0.0231559, -0.0049413, 0.0042025

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011795, upper bound: 0.0010964
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011795, upper bound: 0.0013739
time: 1.22 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042069, -0.0040674, -0.0001406, 0.0001530
1: -0.0102445, -0.0086633, -0.0102044, -0.0086723, -0.0015722, 0.0015411
2: 0.9641697, 0.9660670, 0.9642177, 0.9660563, -0.0018867, 0.0018492
3: -0.0179725, -0.0039775, -0.0176182, -0.0040566, -0.0107457, 0.0109825
4: -0.0003905, 0.0006739, -0.0003845, 0.0006469, -0.0010375, 0.0010584
5: 0.0168757, 0.0185013, 0.0168817, 0.0184221, -0.0015464, 0.0016195
6: 0.0017107, 0.0036985, 0.0018624, 0.0036955, -0.0019848, 0.0018361
7: -0.0067474, -0.0022509, -0.0067270, -0.0024249, -0.0043225, 0.0044761
8: 0.0113760, 0.0142535, 0.0113923, 0.0141806, -0.0028046, 0.0028611
9: 0.0181855, 0.0233608, 0.0182148, 0.0232298, -0.0047337, 0.0047904

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013960, upper bound: 0.0011019
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013960, upper bound: 0.0013951
time: 1.50 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042029, -0.0041182, -0.0042076, -0.0040589, -0.0001440, 0.0000894
1: -0.0100536, -0.0085752, -0.0102297, -0.0086635, -0.0013901, 0.0016545
2: 0.9643987, 0.9661729, 0.9641874, 0.9660669, -0.0016682, 0.0019855
3: -0.0162835, -0.0031973, -0.0178420, -0.0039790, -0.0093904, 0.0116925
4: -0.0004499, 0.0005454, -0.0003904, 0.0006640, -0.0011138, 0.0009358
5: 0.0168157, 0.0181237, 0.0168758, 0.0184721, -0.0016564, 0.0012479
6: 0.0024338, 0.0037276, 0.0017666, 0.0036984, -0.0012646, 0.0019610
7: -0.0069497, -0.0030805, -0.0067471, -0.0023150, -0.0046346, 0.0036666
8: 0.0112156, 0.0139062, 0.0113764, 0.0142266, -0.0030110, 0.0025298
9: 0.0178970, 0.0227363, 0.0181861, 0.0233126, -0.0050838, 0.0042181

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012011, upper bound: 0.0011917
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012011, upper bound: 0.0013851
time: 2.00 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042082, -0.0040511, -0.0001568, 0.0001543
1: -0.0102445, -0.0086633, -0.0102526, -0.0086629, -0.0015815, 0.0015892
2: 0.9641697, 0.9660670, 0.9641600, 0.9660675, -0.0018978, 0.0019070
3: -0.0179725, -0.0039775, -0.0180443, -0.0039738, -0.0106798, 0.0112527
4: -0.0003905, 0.0006739, -0.0003908, 0.0006793, -0.0010699, 0.0010647
5: 0.0168757, 0.0185013, 0.0168754, 0.0185173, -0.0016417, 0.0016259
6: 0.0017107, 0.0036985, 0.0016800, 0.0036986, -0.0019879, 0.0020185
7: -0.0067474, -0.0022509, -0.0067484, -0.0022157, -0.0045318, 0.0044975
8: 0.0113760, 0.0142535, 0.0113753, 0.0142682, -0.0028922, 0.0028782
9: 0.0181855, 0.0233608, 0.0181841, 0.0233874, -0.0048764, 0.0048059

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013972, upper bound: 0.0011975
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013972, upper bound: 0.0013957
time: 1.57 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042016, -0.0041339, -0.0042080, -0.0040529, -0.0001487, 0.0000742
1: -0.0100072, -0.0085821, -0.0102471, -0.0085868, -0.0014204, 0.0016651
2: 0.9644544, 0.9661646, 0.9641666, 0.9661590, -0.0017046, 0.0019980
3: -0.0158724, -0.0032582, -0.0179960, -0.0032996, -0.0099407, 0.0119697
4: -0.0004452, 0.0005142, -0.0004421, 0.0006757, -0.0011209, 0.0009562
5: 0.0168204, 0.0180318, 0.0168235, 0.0185065, -0.0016862, 0.0012082
6: 0.0026098, 0.0037254, 0.0017007, 0.0037238, -0.0011140, 0.0020247
7: -0.0069339, -0.0032824, -0.0069231, -0.0022393, -0.0046945, 0.0036407
8: 0.0112281, 0.0138217, 0.0112367, 0.0142583, -0.0030302, 0.0025850
9: 0.0179195, 0.0225842, 0.0179348, 0.0233695, -0.0051369, 0.0043499

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011099, upper bound: 0.0013744
time: 1.49 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011099, upper bound: 0.0013960
time: 1.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042067, -0.0040702, -0.0042080, -0.0040529, -0.0001537, 0.0001378
1: -0.0101961, -0.0086726, -0.0102471, -0.0085868, -0.0016093, 0.0015745
2: 0.9642278, 0.9660560, 0.9641666, 0.9661590, -0.0019312, 0.0018894
3: -0.0175440, -0.0040594, -0.0179960, -0.0032996, -0.0112683, 0.0108007
4: -0.0003843, 0.0006413, -0.0004421, 0.0006757, -0.0010600, 0.0010834
5: 0.0168820, 0.0184055, 0.0168235, 0.0185065, -0.0016246, 0.0015819
6: 0.0018942, 0.0036954, 0.0017007, 0.0037238, -0.0018297, 0.0019947
7: -0.0067262, -0.0024613, -0.0069231, -0.0022393, -0.0044869, 0.0044618
8: 0.0113929, 0.0141654, 0.0112367, 0.0142583, -0.0028654, 0.0029287
9: 0.0182158, 0.0232024, 0.0179348, 0.0233695, -0.0048011, 0.0049325

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011099, upper bound: 0.0013791
time: 1.47 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011099, upper bound: 0.0013842
time: 1.68 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042029, -0.0041182, -0.0042063, -0.0040754, -0.0001274, 0.0000881
1: -0.0100536, -0.0085752, -0.0101805, -0.0086002, -0.0014535, 0.0016053
2: 0.9643987, 0.9661729, 0.9642465, 0.9661428, -0.0017442, 0.0019264
3: -0.0162835, -0.0031973, -0.0174061, -0.0034182, -0.0102409, 0.0114374
4: -0.0004499, 0.0005454, -0.0004331, 0.0006308, -0.0010807, 0.0009785
5: 0.0168157, 0.0181237, 0.0168327, 0.0183746, -0.0015590, 0.0012910
6: 0.0024338, 0.0037276, 0.0019532, 0.0037194, -0.0012856, 0.0017744
7: -0.0069497, -0.0030805, -0.0068924, -0.0025291, -0.0044205, 0.0038119
8: 0.0112156, 0.0139062, 0.0112611, 0.0141370, -0.0029214, 0.0026451
9: 0.0178970, 0.0227363, 0.0179787, 0.0231514, -0.0049404, 0.0044599

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011945, upper bound: 0.0011011
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011945, upper bound: 0.0013739
time: 1.25 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042069, -0.0040678, -0.0001402, 0.0001530
1: -0.0102445, -0.0086633, -0.0102030, -0.0085989, -0.0016455, 0.0015397
2: 0.9641697, 0.9660670, 0.9642194, 0.9661443, -0.0019746, 0.0018476
3: -0.0179725, -0.0039775, -0.0176058, -0.0034075, -0.0115976, 0.0110031
4: -0.0003905, 0.0006739, -0.0004339, 0.0006460, -0.0010365, 0.0011077
5: 0.0168757, 0.0185013, 0.0168318, 0.0184193, -0.0015436, 0.0016694
6: 0.0017107, 0.0036985, 0.0018677, 0.0037198, -0.0020090, 0.0018308
7: -0.0067474, -0.0022509, -0.0068952, -0.0024310, -0.0043164, 0.0046443
8: 0.0113760, 0.0142535, 0.0112589, 0.0141781, -0.0028020, 0.0029946
9: 0.0181855, 0.0233608, 0.0179747, 0.0232252, -0.0047323, 0.0050526

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014056, upper bound: 0.0011067
time: 1.37 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014056, upper bound: 0.0013940
time: 1.27 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042029, -0.0041182, -0.0042075, -0.0040595, -0.0001434, 0.0000893
1: -0.0100536, -0.0085752, -0.0102278, -0.0085903, -0.0014633, 0.0016526
2: 0.9643987, 0.9661729, 0.9641896, 0.9661547, -0.0017560, 0.0019833
3: -0.0162835, -0.0031973, -0.0178253, -0.0033311, -0.0101793, 0.0117105
4: -0.0004499, 0.0005454, -0.0004397, 0.0006627, -0.0011125, 0.0009851
5: 0.0168157, 0.0181237, 0.0168260, 0.0184684, -0.0016527, 0.0012977
6: 0.0024338, 0.0037276, 0.0017738, 0.0037226, -0.0012888, 0.0019539
7: -0.0069497, -0.0030805, -0.0069150, -0.0023232, -0.0046265, 0.0038345
8: 0.0112156, 0.0139062, 0.0112431, 0.0142232, -0.0030076, 0.0026631
9: 0.0178970, 0.0227363, 0.0179465, 0.0233064, -0.0050812, 0.0044750

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012146, upper bound: 0.0011962
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012146, upper bound: 0.0013850
time: 1.37 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042081, -0.0040517, -0.0001563, 0.0001543
1: -0.0102445, -0.0086633, -0.0102509, -0.0085891, -0.0016553, 0.0015875
2: 0.9641697, 0.9660670, 0.9641619, 0.9661561, -0.0019864, 0.0019050
3: -0.0179725, -0.0039775, -0.0180291, -0.0033205, -0.0115339, 0.0112640
4: -0.0003905, 0.0006739, -0.0004405, 0.0006782, -0.0010687, 0.0011144
5: 0.0168757, 0.0185013, 0.0168252, 0.0185139, -0.0016383, 0.0016761
6: 0.0017107, 0.0036985, 0.0016865, 0.0037230, -0.0020123, 0.0020120
7: -0.0067474, -0.0022509, -0.0069177, -0.0022231, -0.0045244, 0.0046668
8: 0.0113760, 0.0142535, 0.0112410, 0.0142651, -0.0028891, 0.0030125
9: 0.0181855, 0.0233608, 0.0179425, 0.0233818, -0.0048738, 0.0050686

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014068, upper bound: 0.0012030
time: 1.25 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014068, upper bound: 0.0013947
time: 1.47 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042016, -0.0041338, -0.0042081, -0.0040524, -0.0001492, 0.0000743
1: -0.0100073, -0.0085092, -0.0102488, -0.0086605, -0.0013468, 0.0017396
2: 0.9644542, 0.9662521, 0.9641646, 0.9660705, -0.0016162, 0.0020875
3: -0.0158735, -0.0026129, -0.0180106, -0.0039527, -0.0092272, 0.0126641
4: -0.0004943, 0.0005142, -0.0003924, 0.0006768, -0.0011711, 0.0009066
5: 0.0167708, 0.0180320, 0.0168738, 0.0185098, -0.0017390, 0.0011583
6: 0.0026093, 0.0037495, 0.0016944, 0.0036994, -0.0010901, 0.0020551
7: -0.0071011, -0.0032819, -0.0067539, -0.0022321, -0.0048690, 0.0034720
8: 0.0110955, 0.0138219, 0.0113710, 0.0142613, -0.0031658, 0.0024510
9: 0.0176809, 0.0225846, 0.0181764, 0.0233749, -0.0053807, 0.0040953

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011011, upper bound: 0.0013882
time: 1.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011011, upper bound: 0.0014056
time: 1.17 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042066, -0.0040707, -0.0042081, -0.0040524, -0.0001542, 0.0001374
1: -0.0101946, -0.0085997, -0.0102488, -0.0086605, -0.0015340, 0.0016490
2: 0.9642295, 0.9661434, 0.9641646, 0.9660705, -0.0018409, 0.0019789
3: -0.0175310, -0.0034145, -0.0180106, -0.0039527, -0.0104693, 0.0116143
4: -0.0004333, 0.0006403, -0.0003924, 0.0006768, -0.0011101, 0.0010327
5: 0.0168324, 0.0184026, 0.0168738, 0.0185098, -0.0016774, 0.0015288
6: 0.0018997, 0.0037195, 0.0016944, 0.0036994, -0.0017997, 0.0020251
7: -0.0068934, -0.0024677, -0.0067539, -0.0022321, -0.0046612, 0.0042862
8: 0.0112603, 0.0141627, 0.0113710, 0.0142613, -0.0030010, 0.0027918
9: 0.0179773, 0.0231976, 0.0181764, 0.0233749, -0.0050633, 0.0046700

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011011, upper bound: 0.0013878
time: 1.57 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011011, upper bound: 0.0013921
time: 1.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042028, -0.0041190, -0.0042063, -0.0040750, -0.0001279, 0.0000873
1: -0.0100513, -0.0085046, -0.0101818, -0.0086728, -0.0013786, 0.0016772
2: 0.9644014, 0.9662575, 0.9642448, 0.9660557, -0.0016543, 0.0020127
3: -0.0162631, -0.0025728, -0.0174182, -0.0040610, -0.0094781, 0.0120809
4: -0.0004974, 0.0005439, -0.0003842, 0.0006317, -0.0011291, 0.0009280
5: 0.0167677, 0.0181191, 0.0168821, 0.0183773, -0.0016097, 0.0012370
6: 0.0024425, 0.0037510, 0.0019480, 0.0036954, -0.0012528, 0.0018030
7: -0.0071115, -0.0030905, -0.0067258, -0.0025232, -0.0045883, 0.0036353
8: 0.0110872, 0.0139020, 0.0113932, 0.0141395, -0.0030523, 0.0025088
9: 0.0176661, 0.0227287, 0.0182164, 0.0231559, -0.0051732, 0.0041979

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011819, upper bound: 0.0011099
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011819, upper bound: 0.0013872
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040544, -0.0042069, -0.0040674, -0.0001406, 0.0001525
1: -0.0102428, -0.0085899, -0.0102044, -0.0086723, -0.0015705, 0.0016145
2: 0.9641717, 0.9661552, 0.9642177, 0.9660563, -0.0018846, 0.0019375
3: -0.0179578, -0.0033274, -0.0176182, -0.0040566, -0.0107730, 0.0117632
4: -0.0004400, 0.0006728, -0.0003845, 0.0006469, -0.0010869, 0.0010573
5: 0.0168257, 0.0184980, 0.0168817, 0.0184221, -0.0015964, 0.0016162
6: 0.0017170, 0.0037228, 0.0018624, 0.0036955, -0.0019785, 0.0018604
7: -0.0069159, -0.0022581, -0.0067270, -0.0024249, -0.0044910, 0.0044688
8: 0.0112424, 0.0142504, 0.0113923, 0.0141806, -0.0029382, 0.0028581
9: 0.0179451, 0.0233554, 0.0182148, 0.0232298, -0.0049916, 0.0047877

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013959, upper bound: 0.0011138
time: 1.37 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013959, upper bound: 0.0014030
time: 1.37 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042028, -0.0041190, -0.0042076, -0.0040589, -0.0001440, 0.0000886
1: -0.0100513, -0.0085046, -0.0102297, -0.0086635, -0.0013878, 0.0017251
2: 0.9644014, 0.9662575, 0.9641874, 0.9660669, -0.0016655, 0.0020701
3: -0.0162631, -0.0025728, -0.0178420, -0.0039790, -0.0094379, 0.0123747
4: -0.0004974, 0.0005439, -0.0003904, 0.0006640, -0.0011613, 0.0009343
5: 0.0167677, 0.0181191, 0.0168758, 0.0184721, -0.0017044, 0.0012433
6: 0.0024425, 0.0037510, 0.0017666, 0.0036984, -0.0012559, 0.0019844
7: -0.0071115, -0.0030905, -0.0067471, -0.0023150, -0.0047965, 0.0036566
8: 0.0110872, 0.0139020, 0.0113764, 0.0142266, -0.0031394, 0.0025257
9: 0.0176661, 0.0227287, 0.0181861, 0.0233126, -0.0053171, 0.0042149

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012059, upper bound: 0.0012056
time: 1.13 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012059, upper bound: 0.0013957
time: 1.34 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040544, -0.0042082, -0.0040511, -0.0001568, 0.0001538
1: -0.0102428, -0.0085899, -0.0102526, -0.0086629, -0.0015799, 0.0016627
2: 0.9641717, 0.9661552, 0.9641600, 0.9660675, -0.0018958, 0.0019953
3: -0.0179578, -0.0033274, -0.0180443, -0.0039738, -0.0107135, 0.0120409
4: -0.0004400, 0.0006728, -0.0003908, 0.0006793, -0.0011193, 0.0010636
5: 0.0168257, 0.0184980, 0.0168754, 0.0185173, -0.0016916, 0.0016226
6: 0.0017170, 0.0037228, 0.0016800, 0.0036986, -0.0019816, 0.0020428
7: -0.0069159, -0.0022581, -0.0067484, -0.0022157, -0.0047003, 0.0044903
8: 0.0112424, 0.0142504, 0.0113753, 0.0142682, -0.0030258, 0.0028751
9: 0.0179451, 0.0233554, 0.0181841, 0.0233874, -0.0051351, 0.0048034

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013972, upper bound: 0.0012100
time: 1.40 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013972, upper bound: 0.0014037
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042016, -0.0041338, -0.0042080, -0.0040529, -0.0001487, 0.0000742
1: -0.0100073, -0.0085092, -0.0102471, -0.0085868, -0.0014206, 0.0017380
2: 0.9644542, 0.9662521, 0.9641666, 0.9661590, -0.0017048, 0.0020855
3: -0.0158735, -0.0026129, -0.0179960, -0.0032996, -0.0097423, 0.0123920
4: -0.0004943, 0.0005142, -0.0004421, 0.0006757, -0.0011700, 0.0009563
5: 0.0167708, 0.0180320, 0.0168235, 0.0185065, -0.0017358, 0.0012085
6: 0.0026093, 0.0037495, 0.0017007, 0.0037238, -0.0011145, 0.0020488
7: -0.0071011, -0.0032819, -0.0069231, -0.0022393, -0.0048618, 0.0036413
8: 0.0110955, 0.0138219, 0.0112367, 0.0142583, -0.0031628, 0.0025852
9: 0.0176809, 0.0225846, 0.0179348, 0.0233695, -0.0053505, 0.0043251

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011116, upper bound: 0.0011195
time: 1.46 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011116, upper bound: 0.0014061
time: 1.23 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042066, -0.0040707, -0.0042080, -0.0040529, -0.0001537, 0.0001374
1: -0.0101946, -0.0085997, -0.0102471, -0.0085868, -0.0016078, 0.0016474
2: 0.9642295, 0.9661434, 0.9641666, 0.9661590, -0.0019295, 0.0019768
3: -0.0175310, -0.0034145, -0.0179960, -0.0032996, -0.0110226, 0.0113573
4: -0.0004333, 0.0006403, -0.0004421, 0.0006757, -0.0011090, 0.0010824
5: 0.0168324, 0.0184026, 0.0168235, 0.0185065, -0.0016741, 0.0015790
6: 0.0018997, 0.0037195, 0.0017007, 0.0037238, -0.0018241, 0.0020189
7: -0.0068934, -0.0024677, -0.0069231, -0.0022393, -0.0046540, 0.0044554
8: 0.0112603, 0.0141627, 0.0112367, 0.0142583, -0.0029980, 0.0029260
9: 0.0179773, 0.0231976, 0.0179348, 0.0233695, -0.0050313, 0.0049002

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011116, upper bound: 0.0013882
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011116, upper bound: 0.0013925
time: 1.29 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042028, -0.0041190, -0.0042063, -0.0040754, -0.0001274, 0.0000873
1: -0.0100513, -0.0085046, -0.0101805, -0.0086002, -0.0014512, 0.0016758
2: 0.9644014, 0.9662575, 0.9642465, 0.9661428, -0.0017414, 0.0020110
3: -0.0162631, -0.0025728, -0.0174061, -0.0034182, -0.0100196, 0.0118286
4: -0.0004974, 0.0005439, -0.0004331, 0.0006308, -0.0011282, 0.0009769
5: 0.0167677, 0.0181191, 0.0168327, 0.0183746, -0.0016070, 0.0012865
6: 0.0024425, 0.0037510, 0.0019532, 0.0037194, -0.0012769, 0.0017978
7: -0.0071115, -0.0030905, -0.0068924, -0.0025291, -0.0045824, 0.0038019
8: 0.0110872, 0.0139020, 0.0112611, 0.0141370, -0.0030498, 0.0026410
9: 0.0176661, 0.0227287, 0.0179787, 0.0231514, -0.0051446, 0.0044285

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011965, upper bound: 0.0011195
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011965, upper bound: 0.0013873
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040544, -0.0042069, -0.0040678, -0.0001401, 0.0001524
1: -0.0102428, -0.0085899, -0.0102030, -0.0085989, -0.0016438, 0.0016131
2: 0.9641717, 0.9661552, 0.9642194, 0.9661443, -0.0019726, 0.0019358
3: -0.0179578, -0.0033274, -0.0176058, -0.0034075, -0.0113404, 0.0115476
4: -0.0004400, 0.0006728, -0.0004339, 0.0006460, -0.0010860, 0.0011066
5: 0.0168257, 0.0184980, 0.0168318, 0.0184193, -0.0015936, 0.0016661
6: 0.0017170, 0.0037228, 0.0018677, 0.0037198, -0.0020027, 0.0018551
7: -0.0069159, -0.0022581, -0.0068952, -0.0024310, -0.0044849, 0.0046370
8: 0.0112424, 0.0142504, 0.0112589, 0.0141781, -0.0029357, 0.0029916
9: 0.0179451, 0.0233554, 0.0179747, 0.0232252, -0.0049619, 0.0050205

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013997, upper bound: 0.0011229
time: 1.37 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013997, upper bound: 0.0014035
time: 1.66 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042028, -0.0041190, -0.0042075, -0.0040595, -0.0001433, 0.0000885
1: -0.0100513, -0.0085046, -0.0102278, -0.0085903, -0.0014610, 0.0017232
2: 0.9644014, 0.9662575, 0.9641896, 0.9661547, -0.0017533, 0.0020679
3: -0.0162631, -0.0025728, -0.0178253, -0.0033311, -0.0099584, 0.0121048
4: -0.0004974, 0.0005439, -0.0004397, 0.0006627, -0.0011600, 0.0009836
5: 0.0167677, 0.0181191, 0.0168260, 0.0184684, -0.0017007, 0.0012931
6: 0.0024425, 0.0037510, 0.0017738, 0.0037226, -0.0012801, 0.0019772
7: -0.0071115, -0.0030905, -0.0069150, -0.0023232, -0.0047883, 0.0038245
8: 0.0110872, 0.0139020, 0.0112431, 0.0142232, -0.0031360, 0.0026589
9: 0.0176661, 0.0227287, 0.0179465, 0.0233064, -0.0052862, 0.0044433

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012186, upper bound: 0.0012158
time: 1.44 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012186, upper bound: 0.0013959
time: 1.66 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040544, -0.0042081, -0.0040517, -0.0001562, 0.0001537
1: -0.0102428, -0.0085899, -0.0102509, -0.0085891, -0.0016537, 0.0016610
2: 0.9641717, 0.9661552, 0.9641619, 0.9661561, -0.0019844, 0.0019933
3: -0.0179578, -0.0033274, -0.0180291, -0.0033205, -0.0112759, 0.0118105
4: -0.0004400, 0.0006728, -0.0004405, 0.0006782, -0.0011181, 0.0011132
5: 0.0168257, 0.0184980, 0.0168252, 0.0185139, -0.0016882, 0.0016728
6: 0.0017170, 0.0037228, 0.0016865, 0.0037230, -0.0020060, 0.0020363
7: -0.0069159, -0.0022581, -0.0069177, -0.0022231, -0.0046928, 0.0046596
8: 0.0112424, 0.0142504, 0.0112410, 0.0142651, -0.0030227, 0.0030095
9: 0.0179451, 0.0233554, 0.0179425, 0.0233818, -0.0051037, 0.0050355

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014007, upper bound: 0.0012202
time: 1.52 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014007, upper bound: 0.0012202
time: 1.18 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.90 seconds
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0013744
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0013960
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0013797
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0013847
NS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011795, upper bound: 0.0010964
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011795, upper bound: 0.0013739
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0013960, upper bound: 0.0011019
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0013960, upper bound: 0.0013951
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0012011, upper bound: 0.0011917
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0012011, upper bound: 0.0013851
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0013972, upper bound: 0.0011975
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0013972, upper bound: 0.0013957
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011099, upper bound: 0.0013744
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011099, upper bound: 0.0013960
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011099, upper bound: 0.0013791
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011099, upper bound: 0.0013842
NS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011945, upper bound: 0.0011011
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011945, upper bound: 0.0013739
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0014056, upper bound: 0.0011067
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0014056, upper bound: 0.0013940
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0012146, upper bound: 0.0011962
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0012146, upper bound: 0.0013850
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0014068, upper bound: 0.0012030
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0014068, upper bound: 0.0013947
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011011, upper bound: 0.0013882
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011011, upper bound: 0.0014056
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011011, upper bound: 0.0013878
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011011, upper bound: 0.0013921
NS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011819, upper bound: 0.0011099
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011819, upper bound: 0.0013872
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0013959, upper bound: 0.0011138
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0013959, upper bound: 0.0014030
NS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0012059, upper bound: 0.0012056
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0012059, upper bound: 0.0013957
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0013972, upper bound: 0.0012100
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0013972, upper bound: 0.0014037
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011116, upper bound: 0.0011195
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011116, upper bound: 0.0014061
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011116, upper bound: 0.0013882
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011116, upper bound: 0.0013925
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011965, upper bound: 0.0011195
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0011965, upper bound: 0.0013873
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0013997, upper bound: 0.0011229
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0013997, upper bound: 0.0014035
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0012186, upper bound: 0.0012158
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0012186, upper bound: 0.0013959
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0014007, upper bound: 0.0012202
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 2, lower bound: -0.0014007, upper bound: 0.0012202

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0042016, -0.0041339, -0.0042067, -0.0040702, -0.0001314, 0.0000728
1: -0.0100072, -0.0085821, -0.0101961, -0.0086726, -0.0013346, 0.0016140
2: 0.9644544, 0.9661646, 0.9642278, 0.9660560, -0.0016016, 0.0019368
3: -0.0158724, -0.0032582, -0.0175440, -0.0040594, -0.0090181, 0.0114943
4: -0.0004452, 0.0005142, -0.0003843, 0.0006413, -0.0010865, 0.0008984
5: 0.0168204, 0.0180318, 0.0168820, 0.0184055, -0.0015851, 0.0011498
6: 0.0026098, 0.0037254, 0.0018942, 0.0036954, -0.0010856, 0.0018312
7: -0.0069339, -0.0032824, -0.0067262, -0.0024613, -0.0044725, 0.0034438
8: 0.0112281, 0.0138217, 0.0113929, 0.0141654, -0.0029372, 0.0024288
9: 0.0179195, 0.0225842, 0.0182158, 0.0232024, -0.0049646, 0.0040501

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010762, upper bound: 0.0013259
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0009808, upper bound: 0.0012883
time: 1.35 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042016, -0.0041339, -0.0042080, -0.0040539, -0.0001478, 0.0000741
1: -0.0100072, -0.0085821, -0.0102445, -0.0086633, -0.0013438, 0.0016624
2: 0.9644544, 0.9661646, 0.9641697, 0.9660670, -0.0016125, 0.0019949
3: -0.0158724, -0.0032582, -0.0179725, -0.0039775, -0.0091368, 0.0119389
4: -0.0004452, 0.0005142, -0.0003905, 0.0006739, -0.0011191, 0.0009047
5: 0.0168204, 0.0180318, 0.0168757, 0.0185013, -0.0016809, 0.0011561
6: 0.0026098, 0.0037254, 0.0017107, 0.0036985, -0.0010887, 0.0020146
7: -0.0069339, -0.0032824, -0.0067474, -0.0022509, -0.0046830, 0.0034650
8: 0.0112281, 0.0138217, 0.0113760, 0.0142535, -0.0030253, 0.0024456
9: 0.0179195, 0.0225842, 0.0181855, 0.0233608, -0.0051258, 0.0040824

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010762, upper bound: 0.0013833
time: 1.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009808, upper bound: 0.0013740
time: 1.45 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042067, -0.0040702, -0.0042067, -0.0040702, -0.0001365, 0.0001365
1: -0.0101961, -0.0086726, -0.0101961, -0.0086726, -0.0015235, 0.0015235
2: 0.9642278, 0.9660560, 0.9642278, 0.9660560, -0.0018282, 0.0018282
3: -0.0175440, -0.0040594, -0.0175440, -0.0040594, -0.0102962, 0.0102962
4: -0.0003843, 0.0006413, -0.0003843, 0.0006413, -0.0010256, 0.0010256
5: 0.0168820, 0.0184055, 0.0168820, 0.0184055, -0.0015235, 0.0015235
6: 0.0018942, 0.0036954, 0.0018942, 0.0036954, -0.0018012, 0.0018012
7: -0.0067262, -0.0024613, -0.0067262, -0.0024613, -0.0042649, 0.0042649
8: 0.0113929, 0.0141654, 0.0113929, 0.0141654, -0.0027725, 0.0027725
9: 0.0182158, 0.0232024, 0.0182158, 0.0232024, -0.0046300, 0.0046300

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013708, upper bound: 0.0013511
time: 1.48 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013936, upper bound: 0.0013781
time: 1.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042067, -0.0040702, -0.0042080, -0.0040539, -0.0001528, 0.0001378
1: -0.0101961, -0.0086726, -0.0102445, -0.0086633, -0.0015327, 0.0015719
2: 0.9642278, 0.9660560, 0.9641697, 0.9660670, -0.0018392, 0.0018863
3: -0.0175440, -0.0040594, -0.0179725, -0.0039775, -0.0104041, 0.0107428
4: -0.0003843, 0.0006413, -0.0003905, 0.0006739, -0.0010582, 0.0010318
5: 0.0168820, 0.0184055, 0.0168757, 0.0185013, -0.0016193, 0.0015298
6: 0.0018942, 0.0036954, 0.0017107, 0.0036985, -0.0018043, 0.0019847
7: -0.0067262, -0.0024613, -0.0067474, -0.0022509, -0.0044753, 0.0042861
8: 0.0113929, 0.0141654, 0.0113760, 0.0142535, -0.0028606, 0.0027893
9: 0.0182158, 0.0232024, 0.0181855, 0.0233608, -0.0047894, 0.0046618

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013708, upper bound: 0.0013607
time: 1.50 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013936, upper bound: 0.0013832
time: 1.84 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042029, -0.0041182, -0.0042067, -0.0040702, -0.0001327, 0.0000885
1: -0.0100536, -0.0085752, -0.0101961, -0.0086726, -0.0013811, 0.0016209
2: 0.9643987, 0.9661729, 0.9642278, 0.9660560, -0.0016573, 0.0019451
3: -0.0162835, -0.0031973, -0.0175440, -0.0040594, -0.0094499, 0.0115767
4: -0.0004499, 0.0005454, -0.0003843, 0.0006413, -0.0010912, 0.0009297
5: 0.0168157, 0.0181237, 0.0168820, 0.0184055, -0.0015898, 0.0012417
6: 0.0024338, 0.0037276, 0.0018942, 0.0036954, -0.0012616, 0.0018335
7: -0.0069497, -0.0030805, -0.0067262, -0.0024613, -0.0044883, 0.0036457
8: 0.0112156, 0.0139062, 0.0113929, 0.0141654, -0.0029497, 0.0025133
9: 0.0178970, 0.0227363, 0.0182158, 0.0232024, -0.0049889, 0.0042031

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011044, upper bound: 0.0013262
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010700, upper bound: 0.0009692
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042016, -0.0041339, -0.0000741, 0.0001478
1: -0.0102445, -0.0086633, -0.0100072, -0.0085821, -0.0016624, 0.0013438
2: 0.9641697, 0.9660670, 0.9644544, 0.9661646, -0.0019949, 0.0016125
3: -0.0179725, -0.0039775, -0.0158724, -0.0032582, -0.0119389, 0.0091368
4: -0.0003905, 0.0006739, -0.0004452, 0.0005142, -0.0009047, 0.0011191
5: 0.0168757, 0.0185013, 0.0168204, 0.0180318, -0.0011561, 0.0016809
6: 0.0017107, 0.0036985, 0.0026098, 0.0037254, -0.0020146, 0.0010887
7: -0.0067474, -0.0022509, -0.0069339, -0.0032824, -0.0034650, 0.0046830
8: 0.0113760, 0.0142535, 0.0112281, 0.0138217, -0.0024456, 0.0030253
9: 0.0181855, 0.0233608, 0.0179195, 0.0225842, -0.0040824, 0.0051258

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013771, upper bound: 0.0009836
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009808
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042067, -0.0040702, -0.0001378, 0.0001528
1: -0.0102445, -0.0086633, -0.0101961, -0.0086726, -0.0015719, 0.0015327
2: 0.9641697, 0.9660670, 0.9642278, 0.9660560, -0.0018863, 0.0018392
3: -0.0179725, -0.0039775, -0.0175440, -0.0040594, -0.0107428, 0.0104041
4: -0.0003905, 0.0006739, -0.0003843, 0.0006413, -0.0010318, 0.0010582
5: 0.0168757, 0.0185013, 0.0168820, 0.0184055, -0.0015298, 0.0016193
6: 0.0017107, 0.0036985, 0.0018942, 0.0036954, -0.0019847, 0.0018043
7: -0.0067474, -0.0022509, -0.0067262, -0.0024613, -0.0042861, 0.0044753
8: 0.0113760, 0.0142535, 0.0113929, 0.0141654, -0.0027893, 0.0028606
9: 0.0181855, 0.0233608, 0.0182158, 0.0232024, -0.0046618, 0.0047894

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013771, upper bound: 0.0013726
time: 1.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0013728
time: 1.26 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042029, -0.0041182, -0.0042080, -0.0040539, -0.0001490, 0.0000898
1: -0.0100536, -0.0085752, -0.0102445, -0.0086633, -0.0013903, 0.0016693
2: 0.9643987, 0.9661729, 0.9641697, 0.9660670, -0.0016683, 0.0020032
3: -0.0162835, -0.0031973, -0.0179725, -0.0039775, -0.0093926, 0.0118561
4: -0.0004499, 0.0005454, -0.0003905, 0.0006739, -0.0011237, 0.0009359
5: 0.0168157, 0.0181237, 0.0168757, 0.0185013, -0.0016856, 0.0012480
6: 0.0024338, 0.0037276, 0.0017107, 0.0036985, -0.0012647, 0.0020169
7: -0.0069497, -0.0030805, -0.0067474, -0.0022509, -0.0046988, 0.0036670
8: 0.0112156, 0.0139062, 0.0113760, 0.0142535, -0.0030378, 0.0025302
9: 0.0178970, 0.0227363, 0.0181855, 0.0233608, -0.0051336, 0.0042187

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011625, upper bound: 0.0011120
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011112, upper bound: 0.0011048
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042029, -0.0041182, -0.0000898, 0.0001490
1: -0.0102445, -0.0086633, -0.0100536, -0.0085752, -0.0016693, 0.0013903
2: 0.9641697, 0.9660670, 0.9643987, 0.9661729, -0.0020032, 0.0016683
3: -0.0179725, -0.0039775, -0.0162835, -0.0031973, -0.0118561, 0.0093926
4: -0.0003905, 0.0006739, -0.0004499, 0.0005454, -0.0009359, 0.0011237
5: 0.0168757, 0.0185013, 0.0168157, 0.0181237, -0.0012480, 0.0016856
6: 0.0017107, 0.0036985, 0.0024338, 0.0037276, -0.0020169, 0.0012647
7: -0.0067474, -0.0022509, -0.0069497, -0.0030805, -0.0036670, 0.0046988
8: 0.0113760, 0.0142535, 0.0112156, 0.0139062, -0.0025302, 0.0030378
9: 0.0181855, 0.0233608, 0.0178970, 0.0227363, -0.0042187, 0.0051336

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010738, upper bound: 0.0009095
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013953, upper bound: 0.0011954
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042080, -0.0040539, -0.0001541, 0.0001541
1: -0.0102445, -0.0086633, -0.0102445, -0.0086633, -0.0015811, 0.0015811
2: 0.9641697, 0.9660670, 0.9641697, 0.9660670, -0.0018973, 0.0018973
3: -0.0179725, -0.0039775, -0.0179725, -0.0039775, -0.0106759, 0.0106759
4: -0.0003905, 0.0006739, -0.0003905, 0.0006739, -0.0010644, 0.0010644
5: 0.0168757, 0.0185013, 0.0168757, 0.0185013, -0.0016256, 0.0016256
6: 0.0017107, 0.0036985, 0.0017107, 0.0036985, -0.0019877, 0.0019877
7: -0.0067474, -0.0022509, -0.0067474, -0.0022509, -0.0044966, 0.0044966
8: 0.0113760, 0.0142535, 0.0113760, 0.0142535, -0.0028774, 0.0028774
9: 0.0181855, 0.0233608, 0.0181855, 0.0233608, -0.0048045, 0.0048045

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010738, upper bound: 0.0013556
time: 1.39 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013953, upper bound: 0.0013772
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0042016, -0.0041339, -0.0042066, -0.0040707, -0.0001310, 0.0000728
1: -0.0100072, -0.0085821, -0.0101946, -0.0085997, -0.0014075, 0.0016125
2: 0.9644544, 0.9661646, 0.9642295, 0.9661434, -0.0016890, 0.0019351
3: -0.0158724, -0.0032582, -0.0175310, -0.0034145, -0.0098088, 0.0115082
4: -0.0004452, 0.0005142, -0.0004333, 0.0006403, -0.0010855, 0.0009475
5: 0.0168204, 0.0180318, 0.0168324, 0.0184026, -0.0015822, 0.0011994
6: 0.0026098, 0.0037254, 0.0018997, 0.0037195, -0.0011097, 0.0018256
7: -0.0069339, -0.0032824, -0.0068934, -0.0024677, -0.0044662, 0.0036110
8: 0.0112281, 0.0138217, 0.0112603, 0.0141627, -0.0029346, 0.0025614
9: 0.0179195, 0.0225842, 0.0179773, 0.0231976, -0.0049634, 0.0043084

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010882, upper bound: 0.0013259
time: 1.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0009917, upper bound: 0.0012883
time: 1.32 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042016, -0.0041339, -0.0042079, -0.0040544, -0.0001472, 0.0000741
1: -0.0100072, -0.0085821, -0.0102428, -0.0085899, -0.0014173, 0.0016607
2: 0.9644544, 0.9661646, 0.9641717, 0.9661552, -0.0017008, 0.0019929
3: -0.0158724, -0.0032582, -0.0179578, -0.0033274, -0.0099126, 0.0119356
4: -0.0004452, 0.0005142, -0.0004400, 0.0006728, -0.0011180, 0.0009541
5: 0.0168204, 0.0180318, 0.0168257, 0.0184980, -0.0016776, 0.0012061
6: 0.0026098, 0.0037254, 0.0017170, 0.0037228, -0.0011130, 0.0020083
7: -0.0069339, -0.0032824, -0.0069159, -0.0022581, -0.0046757, 0.0036335
8: 0.0112281, 0.0138217, 0.0112424, 0.0142504, -0.0030223, 0.0025793
9: 0.0179195, 0.0225842, 0.0179451, 0.0233554, -0.0051228, 0.0043397

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010882, upper bound: 0.0013259
time: 1.36 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009917, upper bound: 0.0013740
time: 1.80 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042067, -0.0040702, -0.0042066, -0.0040707, -0.0001360, 0.0001364
1: -0.0101961, -0.0086726, -0.0101946, -0.0085997, -0.0015963, 0.0015220
2: 0.9642278, 0.9660560, 0.9642295, 0.9661434, -0.0019156, 0.0018265
3: -0.0175440, -0.0040594, -0.0175310, -0.0034145, -0.0111408, 0.0103371
4: -0.0003843, 0.0006413, -0.0004333, 0.0006403, -0.0010246, 0.0010746
5: 0.0168820, 0.0184055, 0.0168324, 0.0184026, -0.0015206, 0.0015731
6: 0.0018942, 0.0036954, 0.0018997, 0.0037195, -0.0018254, 0.0017957
7: -0.0067262, -0.0024613, -0.0068934, -0.0024677, -0.0042585, 0.0044320
8: 0.0113929, 0.0141654, 0.0112603, 0.0141627, -0.0027698, 0.0029051
9: 0.0182158, 0.0232024, 0.0179773, 0.0231976, -0.0046289, 0.0048902

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013830, upper bound: 0.0013511
time: 1.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014015, upper bound: 0.0013776
time: 1.86 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042067, -0.0040702, -0.0042079, -0.0040544, -0.0001523, 0.0001377
1: -0.0101961, -0.0086726, -0.0102428, -0.0085899, -0.0016062, 0.0015702
2: 0.9642278, 0.9660560, 0.9641717, 0.9661552, -0.0019274, 0.0018843
3: -0.0175440, -0.0040594, -0.0179578, -0.0033274, -0.0112399, 0.0107701
4: -0.0003843, 0.0006413, -0.0004400, 0.0006728, -0.0010570, 0.0010813
5: 0.0168820, 0.0184055, 0.0168257, 0.0184980, -0.0016160, 0.0015798
6: 0.0018942, 0.0036954, 0.0017170, 0.0037228, -0.0018286, 0.0019784
7: -0.0067262, -0.0024613, -0.0069159, -0.0022581, -0.0044681, 0.0044546
8: 0.0113929, 0.0141654, 0.0112424, 0.0142504, -0.0028576, 0.0029230
9: 0.0182158, 0.0232024, 0.0179451, 0.0233554, -0.0047867, 0.0049222

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013830, upper bound: 0.0013607
time: 1.47 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014015, upper bound: 0.0013827
time: 1.63 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042029, -0.0041182, -0.0042066, -0.0040707, -0.0001322, 0.0000884
1: -0.0100536, -0.0085752, -0.0101946, -0.0085997, -0.0014539, 0.0016194
2: 0.9643987, 0.9661729, 0.9642295, 0.9661434, -0.0017447, 0.0019433
3: -0.0162835, -0.0031973, -0.0175310, -0.0034145, -0.0102451, 0.0115905
4: -0.0004499, 0.0005454, -0.0004333, 0.0006403, -0.0010902, 0.0009787
5: 0.0168157, 0.0181237, 0.0168324, 0.0184026, -0.0015869, 0.0012913
6: 0.0024338, 0.0037276, 0.0018997, 0.0037195, -0.0012857, 0.0018279
7: -0.0069497, -0.0030805, -0.0068934, -0.0024677, -0.0044819, 0.0038129
8: 0.0112156, 0.0139062, 0.0112603, 0.0141627, -0.0029471, 0.0026459
9: 0.0178970, 0.0227363, 0.0179773, 0.0231976, -0.0049877, 0.0044612

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011333, upper bound: 0.0013262
time: 1.36 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010932, upper bound: 0.0009882
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042016, -0.0041338, -0.0000741, 0.0001478
1: -0.0102445, -0.0086633, -0.0100073, -0.0085092, -0.0017353, 0.0013440
2: 0.9641697, 0.9660670, 0.9644542, 0.9662521, -0.0020824, 0.0016127
3: -0.0179725, -0.0039775, -0.0158735, -0.0026129, -0.0126371, 0.0092023
4: -0.0003905, 0.0006739, -0.0004943, 0.0005142, -0.0009048, 0.0011682
5: 0.0168757, 0.0185013, 0.0167708, 0.0180320, -0.0011564, 0.0017305
6: 0.0017107, 0.0036985, 0.0026093, 0.0037495, -0.0020388, 0.0010892
7: -0.0067474, -0.0022509, -0.0071011, -0.0032819, -0.0034656, 0.0048502
8: 0.0113760, 0.0142535, 0.0110955, 0.0138219, -0.0024459, 0.0031580
9: 0.0181855, 0.0233608, 0.0176809, 0.0225846, -0.0040862, 0.0053680

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010233, upper bound: 0.0007606
time: 1.16 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014036, upper bound: 0.0011047
time: 1.38 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042066, -0.0040707, -0.0001373, 0.0001528
1: -0.0102445, -0.0086633, -0.0101946, -0.0085997, -0.0016447, 0.0015312
2: 0.9641697, 0.9660670, 0.9642295, 0.9661434, -0.0019737, 0.0018374
3: -0.0179725, -0.0039775, -0.0175310, -0.0034145, -0.0115905, 0.0104450
4: -0.0003905, 0.0006739, -0.0004333, 0.0006403, -0.0010308, 0.0011072
5: 0.0168757, 0.0185013, 0.0168324, 0.0184026, -0.0015269, 0.0016689
6: 0.0017107, 0.0036985, 0.0018997, 0.0037195, -0.0020088, 0.0017987
7: -0.0067474, -0.0022509, -0.0068934, -0.0024677, -0.0042797, 0.0046425
8: 0.0113760, 0.0142535, 0.0112603, 0.0141627, -0.0027867, 0.0029932
9: 0.0181855, 0.0233608, 0.0179773, 0.0231976, -0.0046608, 0.0050500

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010233, upper bound: 0.0013509
time: 1.36 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014036, upper bound: 0.0013755
time: 1.55 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042029, -0.0041182, -0.0042079, -0.0040544, -0.0001485, 0.0000897
1: -0.0100536, -0.0085752, -0.0102428, -0.0085899, -0.0014638, 0.0016676
2: 0.9643987, 0.9661729, 0.9641717, 0.9661552, -0.0017565, 0.0020012
3: -0.0162835, -0.0031973, -0.0179578, -0.0033274, -0.0101835, 0.0118601
4: -0.0004499, 0.0005454, -0.0004400, 0.0006728, -0.0011226, 0.0009854
5: 0.0168157, 0.0181237, 0.0168257, 0.0184980, -0.0016823, 0.0012980
6: 0.0024338, 0.0037276, 0.0017170, 0.0037228, -0.0012890, 0.0020106
7: -0.0069497, -0.0030805, -0.0069159, -0.0022581, -0.0046915, 0.0038355
8: 0.0112156, 0.0139062, 0.0112424, 0.0142504, -0.0030348, 0.0026638
9: 0.0178970, 0.0227363, 0.0179451, 0.0233554, -0.0051310, 0.0044764

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011841, upper bound: 0.0011252
time: 1.62 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011271, upper bound: 0.0011176
time: 1.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042028, -0.0041190, -0.0000890, 0.0001489
1: -0.0102445, -0.0086633, -0.0100513, -0.0085046, -0.0017398, 0.0013880
2: 0.9641697, 0.9660670, 0.9644014, 0.9662575, -0.0020878, 0.0016655
3: -0.0179725, -0.0039775, -0.0162631, -0.0025728, -0.0125383, 0.0094401
4: -0.0003905, 0.0006739, -0.0004974, 0.0005439, -0.0009344, 0.0011712
5: 0.0168757, 0.0185013, 0.0167677, 0.0181191, -0.0012435, 0.0017336
6: 0.0017107, 0.0036985, 0.0024425, 0.0037510, -0.0020403, 0.0012560
7: -0.0067474, -0.0022509, -0.0071115, -0.0030905, -0.0036570, 0.0048606
8: 0.0113760, 0.0142535, 0.0110872, 0.0139020, -0.0025260, 0.0031662
9: 0.0181855, 0.0233608, 0.0176661, 0.0227287, -0.0042155, 0.0053669

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011946, upper bound: 0.0009577
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014048, upper bound: 0.0012010
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042079, -0.0040544, -0.0001536, 0.0001541
1: -0.0102445, -0.0086633, -0.0102428, -0.0085899, -0.0016546, 0.0015794
2: 0.9641697, 0.9660670, 0.9641717, 0.9661552, -0.0019855, 0.0018952
3: -0.0179725, -0.0039775, -0.0179578, -0.0033274, -0.0115268, 0.0107096
4: -0.0003905, 0.0006739, -0.0004400, 0.0006728, -0.0010633, 0.0011138
5: 0.0168757, 0.0185013, 0.0168257, 0.0184980, -0.0016223, 0.0016756
6: 0.0017107, 0.0036985, 0.0017170, 0.0037228, -0.0020120, 0.0019814
7: -0.0067474, -0.0022509, -0.0069159, -0.0022581, -0.0044893, 0.0046650
8: 0.0113760, 0.0142535, 0.0112424, 0.0142504, -0.0028744, 0.0030111
9: 0.0181855, 0.0233608, 0.0179451, 0.0233554, -0.0048020, 0.0050660

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011946, upper bound: 0.0013556
time: 1.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014048, upper bound: 0.0013763
time: 2.05 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0042016, -0.0041338, -0.0042067, -0.0040702, -0.0001314, 0.0000729
1: -0.0100073, -0.0085092, -0.0101961, -0.0086726, -0.0013347, 0.0016869
2: 0.9644542, 0.9662521, 0.9642278, 0.9660560, -0.0016018, 0.0020243
3: -0.0158735, -0.0026129, -0.0175440, -0.0040594, -0.0090836, 0.0121926
4: -0.0004943, 0.0005142, -0.0003843, 0.0006413, -0.0011356, 0.0008985
5: 0.0167708, 0.0180320, 0.0168820, 0.0184055, -0.0016347, 0.0011501
6: 0.0026093, 0.0037495, 0.0018942, 0.0036954, -0.0010861, 0.0018553
7: -0.0071011, -0.0032819, -0.0067262, -0.0024613, -0.0046398, 0.0034444
8: 0.0110955, 0.0138219, 0.0113929, 0.0141654, -0.0030699, 0.0024290
9: 0.0176809, 0.0225846, 0.0182158, 0.0232024, -0.0052069, 0.0040540

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010802, upper bound: 0.0013517
time: 1.30 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010041, upper bound: 0.0013328
time: 1.33 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042016, -0.0041338, -0.0042080, -0.0040539, -0.0001478, 0.0000741
1: -0.0100073, -0.0085092, -0.0102445, -0.0086633, -0.0013440, 0.0017353
2: 0.9644542, 0.9662521, 0.9641697, 0.9660670, -0.0016127, 0.0020824
3: -0.0158735, -0.0026129, -0.0179725, -0.0039775, -0.0092023, 0.0126371
4: -0.0004943, 0.0005142, -0.0003905, 0.0006739, -0.0011682, 0.0009048
5: 0.0167708, 0.0180320, 0.0168757, 0.0185013, -0.0017305, 0.0011564
6: 0.0026093, 0.0037495, 0.0017107, 0.0036985, -0.0010892, 0.0020388
7: -0.0071011, -0.0032819, -0.0067474, -0.0022509, -0.0048502, 0.0034656
8: 0.0110955, 0.0138219, 0.0113760, 0.0142535, -0.0031580, 0.0024459
9: 0.0176809, 0.0225846, 0.0181855, 0.0233608, -0.0053680, 0.0040862

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010802, upper bound: 0.0013517
time: 1.91 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010041, upper bound: 0.0013915
time: 1.30 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042066, -0.0040707, -0.0042067, -0.0040702, -0.0001364, 0.0001360
1: -0.0101946, -0.0085997, -0.0101961, -0.0086726, -0.0015220, 0.0015963
2: 0.9642295, 0.9661434, 0.9642278, 0.9660560, -0.0018265, 0.0019156
3: -0.0175310, -0.0034145, -0.0175440, -0.0040594, -0.0103371, 0.0111408
4: -0.0004333, 0.0006403, -0.0003843, 0.0006413, -0.0010746, 0.0010246
5: 0.0168324, 0.0184026, 0.0168820, 0.0184055, -0.0015731, 0.0015206
6: 0.0018997, 0.0037195, 0.0018942, 0.0036954, -0.0017957, 0.0018254
7: -0.0068934, -0.0024677, -0.0067262, -0.0024613, -0.0044320, 0.0042585
8: 0.0112603, 0.0141627, 0.0113929, 0.0141654, -0.0029051, 0.0027698
9: 0.0179773, 0.0231976, 0.0182158, 0.0232024, -0.0048902, 0.0046289

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013708, upper bound: 0.0013626
time: 1.42 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013924, upper bound: 0.0013863
time: 1.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042066, -0.0040707, -0.0042080, -0.0040539, -0.0001528, 0.0001373
1: -0.0101946, -0.0085997, -0.0102445, -0.0086633, -0.0015312, 0.0016447
2: 0.9642295, 0.9661434, 0.9641697, 0.9660670, -0.0018374, 0.0019737
3: -0.0175310, -0.0034145, -0.0179725, -0.0039775, -0.0104450, 0.0115905
4: -0.0004333, 0.0006403, -0.0003905, 0.0006739, -0.0011072, 0.0010308
5: 0.0168324, 0.0184026, 0.0168757, 0.0185013, -0.0016689, 0.0015269
6: 0.0018997, 0.0037195, 0.0017107, 0.0036985, -0.0017987, 0.0020088
7: -0.0068934, -0.0024677, -0.0067474, -0.0022509, -0.0046425, 0.0042797
8: 0.0112603, 0.0141627, 0.0113760, 0.0142535, -0.0029932, 0.0027867
9: 0.0179773, 0.0231976, 0.0181855, 0.0233608, -0.0050500, 0.0046608

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013708, upper bound: 0.0013711
time: 1.39 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013924, upper bound: 0.0013906
time: 1.63 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042028, -0.0041190, -0.0042067, -0.0040702, -0.0001326, 0.0000877
1: -0.0100513, -0.0085046, -0.0101961, -0.0086726, -0.0013788, 0.0016914
2: 0.9644014, 0.9662575, 0.9642278, 0.9660560, -0.0016546, 0.0020297
3: -0.0162631, -0.0025728, -0.0175440, -0.0040594, -0.0094804, 0.0122460
4: -0.0004974, 0.0005439, -0.0003843, 0.0006413, -0.0011387, 0.0009282
5: 0.0167677, 0.0181191, 0.0168820, 0.0184055, -0.0016378, 0.0012372
6: 0.0024425, 0.0037510, 0.0018942, 0.0036954, -0.0012529, 0.0018568
7: -0.0071115, -0.0030905, -0.0067262, -0.0024613, -0.0046502, 0.0036358
8: 0.0110872, 0.0139020, 0.0113929, 0.0141654, -0.0030781, 0.0025091
9: 0.0176661, 0.0227287, 0.0182158, 0.0232024, -0.0052208, 0.0041985

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011067, upper bound: 0.0013523
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010769, upper bound: 0.0013359
time: 1.27 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040544, -0.0042016, -0.0041339, -0.0000741, 0.0001472
1: -0.0102428, -0.0085899, -0.0100072, -0.0085821, -0.0016607, 0.0014173
2: 0.9641717, 0.9661552, 0.9644544, 0.9661646, -0.0019929, 0.0017008
3: -0.0179578, -0.0033274, -0.0158724, -0.0032582, -0.0119356, 0.0099126
4: -0.0004400, 0.0006728, -0.0004452, 0.0005142, -0.0009541, 0.0011180
5: 0.0168257, 0.0184980, 0.0168204, 0.0180318, -0.0012061, 0.0016776
6: 0.0017170, 0.0037228, 0.0026098, 0.0037254, -0.0020083, 0.0011130
7: -0.0069159, -0.0022581, -0.0069339, -0.0032824, -0.0036335, 0.0046757
8: 0.0112424, 0.0142504, 0.0112281, 0.0138217, -0.0025793, 0.0030223
9: 0.0179451, 0.0233554, 0.0179195, 0.0225842, -0.0043397, 0.0051228

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013771, upper bound: 0.0009952
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009917
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040544, -0.0042067, -0.0040702, -0.0001377, 0.0001523
1: -0.0102428, -0.0085899, -0.0101961, -0.0086726, -0.0015702, 0.0016062
2: 0.9641717, 0.9661552, 0.9642278, 0.9660560, -0.0018843, 0.0019274
3: -0.0179578, -0.0033274, -0.0175440, -0.0040594, -0.0107701, 0.0112399
4: -0.0004400, 0.0006728, -0.0003843, 0.0006413, -0.0010813, 0.0010570
5: 0.0168257, 0.0184980, 0.0168820, 0.0184055, -0.0015798, 0.0016160
6: 0.0017170, 0.0037228, 0.0018942, 0.0036954, -0.0019784, 0.0018286
7: -0.0069159, -0.0022581, -0.0067262, -0.0024613, -0.0044546, 0.0044681
8: 0.0112424, 0.0142504, 0.0113929, 0.0141654, -0.0029230, 0.0028576
9: 0.0179451, 0.0233554, 0.0182158, 0.0232024, -0.0049222, 0.0047867

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013771, upper bound: 0.0013812
time: 1.52 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0013815
time: 1.38 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042028, -0.0041190, -0.0042080, -0.0040539, -0.0001489, 0.0000890
1: -0.0100513, -0.0085046, -0.0102445, -0.0086633, -0.0013880, 0.0017398
2: 0.9644014, 0.9662575, 0.9641697, 0.9660670, -0.0016655, 0.0020878
3: -0.0162631, -0.0025728, -0.0179725, -0.0039775, -0.0094401, 0.0125383
4: -0.0004974, 0.0005439, -0.0003905, 0.0006739, -0.0011712, 0.0009344
5: 0.0167677, 0.0181191, 0.0168757, 0.0185013, -0.0017336, 0.0012435
6: 0.0024425, 0.0037510, 0.0017107, 0.0036985, -0.0012560, 0.0020403
7: -0.0071115, -0.0030905, -0.0067474, -0.0022509, -0.0048606, 0.0036570
8: 0.0110872, 0.0139020, 0.0113760, 0.0142535, -0.0031662, 0.0025260
9: 0.0176661, 0.0227287, 0.0181855, 0.0233608, -0.0053669, 0.0042155

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011664, upper bound: 0.0013864
time: 1.28 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011241, upper bound: 0.0013853
time: 1.53 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040544, -0.0042029, -0.0041182, -0.0000897, 0.0001485
1: -0.0102428, -0.0085899, -0.0100536, -0.0085752, -0.0016676, 0.0014638
2: 0.9641717, 0.9661552, 0.9643987, 0.9661729, -0.0020012, 0.0017565
3: -0.0179578, -0.0033274, -0.0162835, -0.0031973, -0.0118601, 0.0101835
4: -0.0004400, 0.0006728, -0.0004499, 0.0005454, -0.0009854, 0.0011226
5: 0.0168257, 0.0184980, 0.0168157, 0.0181237, -0.0012980, 0.0016823
6: 0.0017170, 0.0037228, 0.0024338, 0.0037276, -0.0020106, 0.0012890
7: -0.0069159, -0.0022581, -0.0069497, -0.0030805, -0.0038355, 0.0046915
8: 0.0112424, 0.0142504, 0.0112156, 0.0139062, -0.0026638, 0.0030348
9: 0.0179451, 0.0233554, 0.0178970, 0.0227363, -0.0044764, 0.0051310

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010982, upper bound: 0.0009584
time: 1.25 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013952, upper bound: 0.0012079
time: 1.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040544, -0.0042080, -0.0040539, -0.0001541, 0.0001536
1: -0.0102428, -0.0085899, -0.0102445, -0.0086633, -0.0015794, 0.0016546
2: 0.9641717, 0.9661552, 0.9641697, 0.9660670, -0.0018952, 0.0019855
3: -0.0179578, -0.0033274, -0.0179725, -0.0039775, -0.0107097, 0.0115268
4: -0.0004400, 0.0006728, -0.0003905, 0.0006739, -0.0011138, 0.0010633
5: 0.0168257, 0.0184980, 0.0168757, 0.0185013, -0.0016756, 0.0016223
6: 0.0017170, 0.0037228, 0.0017107, 0.0036985, -0.0019814, 0.0020120
7: -0.0069159, -0.0022581, -0.0067474, -0.0022509, -0.0046650, 0.0044893
8: 0.0112424, 0.0142504, 0.0113760, 0.0142535, -0.0030111, 0.0028744
9: 0.0179451, 0.0233554, 0.0181855, 0.0233608, -0.0050660, 0.0048020

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010982, upper bound: 0.0013672
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013952, upper bound: 0.0013857
time: 1.72 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042016, -0.0041338, -0.0042079, -0.0040544, -0.0001472, 0.0000741
1: -0.0100073, -0.0085092, -0.0102428, -0.0085899, -0.0014174, 0.0017336
2: 0.9644542, 0.9662521, 0.9641717, 0.9661552, -0.0017010, 0.0020804
3: -0.0158735, -0.0026129, -0.0179578, -0.0033274, -0.0097148, 0.0123671
4: -0.0004943, 0.0005142, -0.0004400, 0.0006728, -0.0011671, 0.0009542
5: 0.0167708, 0.0180320, 0.0168257, 0.0184980, -0.0017272, 0.0012063
6: 0.0026093, 0.0037495, 0.0017170, 0.0037228, -0.0011135, 0.0020325
7: -0.0071011, -0.0032819, -0.0069159, -0.0022581, -0.0048430, 0.0036341
8: 0.0110955, 0.0138219, 0.0112424, 0.0142504, -0.0031550, 0.0025795
9: 0.0176809, 0.0225846, 0.0179451, 0.0233554, -0.0053372, 0.0043150

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010924, upper bound: 0.0013529
time: 1.42 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010183, upper bound: 0.0013931
time: 1.17 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042066, -0.0040707, -0.0042066, -0.0040707, -0.0001360, 0.0001360
1: -0.0101946, -0.0085997, -0.0101946, -0.0085997, -0.0015948, 0.0015948
2: 0.9642295, 0.9661434, 0.9642295, 0.9661434, -0.0019139, 0.0019139
3: -0.0175310, -0.0034145, -0.0175310, -0.0034145, -0.0108866, 0.0108866
4: -0.0004333, 0.0006403, -0.0004333, 0.0006403, -0.0010736, 0.0010736
5: 0.0168324, 0.0184026, 0.0168324, 0.0184026, -0.0015702, 0.0015702
6: 0.0018997, 0.0037195, 0.0018997, 0.0037195, -0.0018198, 0.0018198
7: -0.0068934, -0.0024677, -0.0068934, -0.0024677, -0.0044256, 0.0044256
8: 0.0112603, 0.0141627, 0.0112603, 0.0141627, -0.0029024, 0.0029024
9: 0.0179773, 0.0231976, 0.0179773, 0.0231976, -0.0048582, 0.0048582

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013765, upper bound: 0.0013629
time: 1.49 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013936, upper bound: 0.0013866
time: 1.86 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042066, -0.0040707, -0.0042079, -0.0040544, -0.0001522, 0.0001372
1: -0.0101946, -0.0085997, -0.0102428, -0.0085899, -0.0016047, 0.0016431
2: 0.9642295, 0.9661434, 0.9641717, 0.9661552, -0.0019257, 0.0019717
3: -0.0175310, -0.0034145, -0.0179578, -0.0033274, -0.0109951, 0.0113334
4: -0.0004333, 0.0006403, -0.0004400, 0.0006728, -0.0011061, 0.0010803
5: 0.0168324, 0.0184026, 0.0168257, 0.0184980, -0.0016656, 0.0015769
6: 0.0018997, 0.0037195, 0.0017170, 0.0037228, -0.0018231, 0.0020025
7: -0.0068934, -0.0024677, -0.0069159, -0.0022581, -0.0046352, 0.0044482
8: 0.0112603, 0.0141627, 0.0112424, 0.0142504, -0.0029901, 0.0029203
9: 0.0179773, 0.0231976, 0.0179451, 0.0233554, -0.0050179, 0.0048900

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013765, upper bound: 0.0013718
time: 1.51 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013936, upper bound: 0.0013909
time: 2.09 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042028, -0.0041190, -0.0042066, -0.0040707, -0.0001321, 0.0000876
1: -0.0100513, -0.0085046, -0.0101946, -0.0085997, -0.0014516, 0.0016899
2: 0.9644014, 0.9662575, 0.9642295, 0.9661434, -0.0017420, 0.0020279
3: -0.0162631, -0.0025728, -0.0175310, -0.0034145, -0.0100237, 0.0119839
4: -0.0004974, 0.0005439, -0.0004333, 0.0006403, -0.0011377, 0.0009772
5: 0.0167677, 0.0181191, 0.0168324, 0.0184026, -0.0016349, 0.0012867
6: 0.0024425, 0.0037510, 0.0018997, 0.0037195, -0.0012770, 0.0018513
7: -0.0071115, -0.0030905, -0.0068934, -0.0024677, -0.0046438, 0.0038029
8: 0.0110872, 0.0139020, 0.0112603, 0.0141627, -0.0030755, 0.0026417
9: 0.0176661, 0.0227287, 0.0179773, 0.0231976, -0.0051929, 0.0044298

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011381, upper bound: 0.0010230
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011108, upper bound: 0.0010178
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040544, -0.0042016, -0.0041338, -0.0000741, 0.0001472
1: -0.0102428, -0.0085899, -0.0100073, -0.0085092, -0.0017336, 0.0014174
2: 0.9641717, 0.9661552, 0.9644542, 0.9662521, -0.0020804, 0.0017010
3: -0.0179578, -0.0033274, -0.0158735, -0.0026129, -0.0123671, 0.0097148
4: -0.0004400, 0.0006728, -0.0004943, 0.0005142, -0.0009542, 0.0011671
5: 0.0168257, 0.0184980, 0.0167708, 0.0180320, -0.0012063, 0.0017272
6: 0.0017170, 0.0037228, 0.0026093, 0.0037495, -0.0020325, 0.0011135
7: -0.0069159, -0.0022581, -0.0071011, -0.0032819, -0.0036341, 0.0048430
8: 0.0112424, 0.0142504, 0.0110955, 0.0138219, -0.0025795, 0.0031550
9: 0.0179451, 0.0233554, 0.0176809, 0.0225846, -0.0043150, 0.0053372

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011529, upper bound: 0.0009020
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013976, upper bound: 0.0011208
time: 1.51 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040544, -0.0042066, -0.0040707, -0.0001372, 0.0001522
1: -0.0102428, -0.0085899, -0.0101946, -0.0085997, -0.0016431, 0.0016047
2: 0.9641717, 0.9661552, 0.9642295, 0.9661434, -0.0019717, 0.0019257
3: -0.0179578, -0.0033274, -0.0175310, -0.0034145, -0.0113334, 0.0109951
4: -0.0004400, 0.0006728, -0.0004333, 0.0006403, -0.0010803, 0.0011061
5: 0.0168257, 0.0184980, 0.0168324, 0.0184026, -0.0015769, 0.0016656
6: 0.0017170, 0.0037228, 0.0018997, 0.0037195, -0.0020025, 0.0018231
7: -0.0069159, -0.0022581, -0.0068934, -0.0024677, -0.0044482, 0.0046352
8: 0.0112424, 0.0142504, 0.0112603, 0.0141627, -0.0029203, 0.0029901
9: 0.0179451, 0.0233554, 0.0179773, 0.0231976, -0.0048900, 0.0050179

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011529, upper bound: 0.0013626
time: 1.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013976, upper bound: 0.0013850
time: 1.63 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042028, -0.0041190, -0.0042079, -0.0040544, -0.0001484, 0.0000889
1: -0.0100513, -0.0085046, -0.0102428, -0.0085899, -0.0014615, 0.0017382
2: 0.9644014, 0.9662575, 0.9641717, 0.9661552, -0.0017538, 0.0020857
3: -0.0162631, -0.0025728, -0.0179578, -0.0033274, -0.0099626, 0.0122605
4: -0.0004974, 0.0005439, -0.0004400, 0.0006728, -0.0011701, 0.0009838
5: 0.0167677, 0.0181191, 0.0168257, 0.0184980, -0.0017303, 0.0012934
6: 0.0024425, 0.0037510, 0.0017170, 0.0037228, -0.0012803, 0.0020340
7: -0.0071115, -0.0030905, -0.0069159, -0.0022581, -0.0048534, 0.0038255
8: 0.0110872, 0.0139020, 0.0112424, 0.0142504, -0.0031632, 0.0026596
9: 0.0176661, 0.0227287, 0.0179451, 0.0233554, -0.0053366, 0.0044447

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011868, upper bound: 0.0011471
time: 1.49 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011443, upper bound: 0.0011436
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040544, -0.0042028, -0.0041190, -0.0000889, 0.0001484
1: -0.0102428, -0.0085899, -0.0100513, -0.0085046, -0.0017382, 0.0014615
2: 0.9641717, 0.9661552, 0.9644014, 0.9662575, -0.0020857, 0.0017538
3: -0.0179578, -0.0033274, -0.0162631, -0.0025728, -0.0122605, 0.0099626
4: -0.0004400, 0.0006728, -0.0004974, 0.0005439, -0.0009838, 0.0011701
5: 0.0168257, 0.0184980, 0.0167677, 0.0181191, -0.0012934, 0.0017303
6: 0.0017170, 0.0037228, 0.0024425, 0.0037510, -0.0020340, 0.0012803
7: -0.0069159, -0.0022581, -0.0071115, -0.0030905, -0.0038255, 0.0048534
8: 0.0112424, 0.0142504, 0.0110872, 0.0139020, -0.0026596, 0.0031632
9: 0.0179451, 0.0233554, 0.0176661, 0.0227287, -0.0044447, 0.0053366

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012352, upper bound: 0.0010274
time: 2.19 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013987, upper bound: 0.0012181
time: 1.85 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040544, -0.0042079, -0.0040544, -0.0001535, 0.0001535
1: -0.0102428, -0.0085899, -0.0102428, -0.0085899, -0.0016529, 0.0016529
2: 0.9641717, 0.9661552, 0.9641717, 0.9661552, -0.0019835, 0.0019835
3: -0.0179578, -0.0033274, -0.0179578, -0.0033274, -0.0112689, 0.0112689
4: -0.0004400, 0.0006728, -0.0004400, 0.0006728, -0.0011127, 0.0011127
5: 0.0168257, 0.0184980, 0.0168257, 0.0184980, -0.0016723, 0.0016723
6: 0.0017170, 0.0037228, 0.0017170, 0.0037228, -0.0020057, 0.0020057
7: -0.0069159, -0.0022581, -0.0069159, -0.0022581, -0.0046578, 0.0046578
8: 0.0112424, 0.0142504, 0.0112424, 0.0142504, -0.0030081, 0.0030081
9: 0.0179451, 0.0233554, 0.0179451, 0.0233554, -0.0050329, 0.0050329

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012352, upper bound: 0.0013678
time: 1.50 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013987, upper bound: 0.0013860
time: 1.65 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.43 seconds
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010762, upper bound: 0.0013259
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0009808, upper bound: 0.0012883
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010762, upper bound: 0.0013833
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0009808, upper bound: 0.0013740
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013708, upper bound: 0.0013511
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013936, upper bound: 0.0013781
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013708, upper bound: 0.0013607
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013936, upper bound: 0.0013832
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011044, upper bound: 0.0013262
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010700, upper bound: 0.0009692
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013771, upper bound: 0.0009836
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009808
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013771, upper bound: 0.0013726
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0013728
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011625, upper bound: 0.0011120
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011112, upper bound: 0.0011048
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010738, upper bound: 0.0009095
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013953, upper bound: 0.0011954
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010738, upper bound: 0.0013556
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013953, upper bound: 0.0013772
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010882, upper bound: 0.0013259
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0009917, upper bound: 0.0012883
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010882, upper bound: 0.0013259
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0009917, upper bound: 0.0013740
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013830, upper bound: 0.0013511
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0014015, upper bound: 0.0013776
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013830, upper bound: 0.0013607
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0014015, upper bound: 0.0013827
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011333, upper bound: 0.0013262
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010932, upper bound: 0.0009882
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010233, upper bound: 0.0007606
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0014036, upper bound: 0.0011047
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010233, upper bound: 0.0013509
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0014036, upper bound: 0.0013755
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011841, upper bound: 0.0011252
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011271, upper bound: 0.0011176
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011946, upper bound: 0.0009577
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0014048, upper bound: 0.0012010
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011946, upper bound: 0.0013556
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0014048, upper bound: 0.0013763
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010802, upper bound: 0.0013517
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010041, upper bound: 0.0013328
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010802, upper bound: 0.0013517
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010041, upper bound: 0.0013915
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013708, upper bound: 0.0013626
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013924, upper bound: 0.0013863
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013708, upper bound: 0.0013711
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013924, upper bound: 0.0013906
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011067, upper bound: 0.0013523
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010769, upper bound: 0.0013359
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013771, upper bound: 0.0009952
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009917
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013771, upper bound: 0.0013812
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0013815
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011664, upper bound: 0.0013864
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011241, upper bound: 0.0013853
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010982, upper bound: 0.0009584
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013952, upper bound: 0.0012079
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010982, upper bound: 0.0013672
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013952, upper bound: 0.0013857
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010924, upper bound: 0.0013529
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010183, upper bound: 0.0013931
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013765, upper bound: 0.0013629
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013936, upper bound: 0.0013866
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013765, upper bound: 0.0013718
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013936, upper bound: 0.0013909
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011381, upper bound: 0.0010230
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011108, upper bound: 0.0010178
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011529, upper bound: 0.0009020
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013976, upper bound: 0.0011208
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011529, upper bound: 0.0013626
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013976, upper bound: 0.0013850
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011868, upper bound: 0.0011471
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0011443, upper bound: 0.0011436
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0012352, upper bound: 0.0010274
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013987, upper bound: 0.0012181
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0012352, upper bound: 0.0013678
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013987, upper bound: 0.0013860

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042016, -0.0041343, -0.0042080, -0.0040539, -0.0001477, 0.0000737
1: -0.0100061, -0.0086015, -0.0102445, -0.0086633, -0.0013427, 0.0016429
2: 0.9644557, 0.9661412, 0.9641697, 0.9660670, -0.0016112, 0.0019715
3: -0.0158625, -0.0034305, -0.0179725, -0.0039775, -0.0091268, 0.0117652
4: -0.0004321, 0.0005134, -0.0003905, 0.0006739, -0.0011060, 0.0009039
5: 0.0168336, 0.0180295, 0.0168757, 0.0185013, -0.0016677, 0.0011539
6: 0.0026140, 0.0037189, 0.0017107, 0.0036985, -0.0010844, 0.0020082
7: -0.0068892, -0.0032873, -0.0067474, -0.0022509, -0.0046383, 0.0034602
8: 0.0112636, 0.0138196, 0.0113760, 0.0142535, -0.0029899, 0.0024436
9: 0.0179832, 0.0225806, 0.0181855, 0.0233608, -0.0050622, 0.0040787

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009808, upper bound: 0.0013740
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009808, upper bound: 0.0013740
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042019, -0.0041302, -0.0042080, -0.0040540, -0.0001480, 0.0000778
1: -0.0100182, -0.0086134, -0.0102441, -0.0086705, -0.0013476, 0.0016307
2: 0.9644412, 0.9661269, 0.9641701, 0.9660584, -0.0016173, 0.0019568
3: -0.0159696, -0.0035357, -0.0179697, -0.0040412, -0.0091750, 0.0117502
4: -0.0004241, 0.0005215, -0.0003857, 0.0006737, -0.0010978, 0.0009072
5: 0.0168417, 0.0180535, 0.0168806, 0.0185006, -0.0016589, 0.0011729
6: 0.0025682, 0.0037150, 0.0017120, 0.0036961, -0.0011279, 0.0020030
7: -0.0068619, -0.0032347, -0.0067309, -0.0022523, -0.0046097, 0.0034963
8: 0.0112852, 0.0138417, 0.0113891, 0.0142529, -0.0029677, 0.0024525
9: 0.0180221, 0.0226202, 0.0182091, 0.0233598, -0.0050317, 0.0040954

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009808, upper bound: 0.0013740
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009808, upper bound: 0.0013740
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042083, -0.0040494, -0.0042067, -0.0040703, -0.0001381, 0.0001573
1: -0.0102578, -0.0087884, -0.0101958, -0.0086886, -0.0015692, 0.0014074
2: 0.9641536, 0.9659171, 0.9642281, 0.9660367, -0.0018831, 0.0016890
3: -0.0180907, -0.0050842, -0.0175416, -0.0042010, -0.0106668, 0.0092349
4: -0.0003063, 0.0006829, -0.0003735, 0.0006411, -0.0009474, 0.0010564
5: 0.0169607, 0.0185277, 0.0168928, 0.0184049, -0.0014442, 0.0016349
6: 0.0016601, 0.0036571, 0.0018952, 0.0036901, -0.0020300, 0.0017619
7: -0.0064606, -0.0021929, -0.0066895, -0.0024626, -0.0039981, 0.0044967
8: 0.0116036, 0.0142778, 0.0114220, 0.0141649, -0.0025613, 0.0028558
9: 0.0185948, 0.0234045, 0.0182682, 0.0232015, -0.0042481, 0.0047774

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012100, upper bound: 0.0013227
time: 1.40 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012071, upper bound: 0.0013162
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042067, -0.0040702, -0.0042067, -0.0040702, -0.0001365, 0.0001364
1: -0.0101959, -0.0086844, -0.0101961, -0.0086726, -0.0015233, 0.0015116
2: 0.9642280, 0.9660417, 0.9642278, 0.9660560, -0.0018280, 0.0018139
3: -0.0175424, -0.0041642, -0.0175440, -0.0040594, -0.0102946, 0.0101092
4: -0.0003763, 0.0006412, -0.0003843, 0.0006413, -0.0010176, 0.0010255
5: 0.0168900, 0.0184051, 0.0168820, 0.0184055, -0.0015155, 0.0015232
6: 0.0018949, 0.0036915, 0.0018942, 0.0036954, -0.0018005, 0.0017973
7: -0.0066991, -0.0024622, -0.0067262, -0.0024613, -0.0042377, 0.0042641
8: 0.0114144, 0.0141650, 0.0113929, 0.0141654, -0.0027509, 0.0027722
9: 0.0182546, 0.0232018, 0.0182158, 0.0232024, -0.0045861, 0.0046294

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013686, upper bound: 0.0013552
time: 1.41 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013686, upper bound: 0.0013781
time: 1.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042083, -0.0040494, -0.0042080, -0.0040540, -0.0001544, 0.0001586
1: -0.0102578, -0.0087884, -0.0102442, -0.0086792, -0.0015786, 0.0014558
2: 0.9641536, 0.9659171, 0.9641700, 0.9660481, -0.0018945, 0.0017471
3: -0.0180907, -0.0050842, -0.0179702, -0.0041177, -0.0107768, 0.0096819
4: -0.0003063, 0.0006829, -0.0003798, 0.0006737, -0.0009800, 0.0010627
5: 0.0169607, 0.0185277, 0.0168864, 0.0185007, -0.0015400, 0.0016413
6: 0.0016601, 0.0036571, 0.0017117, 0.0036932, -0.0020331, 0.0019454
7: -0.0064606, -0.0021929, -0.0067111, -0.0022520, -0.0042086, 0.0045182
8: 0.0116036, 0.0142778, 0.0114049, 0.0142530, -0.0026494, 0.0028729
9: 0.0185948, 0.0234045, 0.0182374, 0.0233600, -0.0044075, 0.0048097

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012099, upper bound: 0.0013375
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012071, upper bound: 0.0013347
time: 1.46 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042067, -0.0040702, -0.0042080, -0.0040539, -0.0001528, 0.0001377
1: -0.0101959, -0.0086844, -0.0102445, -0.0086633, -0.0015325, 0.0015600
2: 0.9642280, 0.9660417, 0.9641697, 0.9660670, -0.0018390, 0.0018720
3: -0.0175424, -0.0041642, -0.0179725, -0.0039775, -0.0104025, 0.0105615
4: -0.0003763, 0.0006412, -0.0003905, 0.0006739, -0.0010502, 0.0010317
5: 0.0168900, 0.0184051, 0.0168757, 0.0185013, -0.0016113, 0.0015295
6: 0.0018949, 0.0036915, 0.0017107, 0.0036985, -0.0018036, 0.0019807
7: -0.0066991, -0.0024622, -0.0067474, -0.0022509, -0.0044482, 0.0042853
8: 0.0114144, 0.0141650, 0.0113760, 0.0142535, -0.0028390, 0.0027890
9: 0.0182546, 0.0232018, 0.0181855, 0.0233608, -0.0047457, 0.0046613

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013679, upper bound: 0.0013673
time: 1.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013679, upper bound: 0.0013832
time: 1.52 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040541, -0.0042016, -0.0041339, -0.0000741, 0.0001475
1: -0.0102436, -0.0086824, -0.0100072, -0.0085821, -0.0016615, 0.0013247
2: 0.9641707, 0.9660441, 0.9644544, 0.9661646, -0.0019939, 0.0015897
3: -0.0179648, -0.0041465, -0.0158724, -0.0032582, -0.0119311, 0.0089611
4: -0.0003777, 0.0006733, -0.0004452, 0.0005142, -0.0008918, 0.0011185
5: 0.0168886, 0.0184995, 0.0168204, 0.0180318, -0.0011431, 0.0016792
6: 0.0017140, 0.0036922, 0.0026098, 0.0037254, -0.0020113, 0.0010824
7: -0.0067036, -0.0022547, -0.0069339, -0.0032824, -0.0034212, 0.0046792
8: 0.0114108, 0.0142519, 0.0112281, 0.0138217, -0.0024109, 0.0030237
9: 0.0182480, 0.0233580, 0.0179195, 0.0225842, -0.0040190, 0.0051229

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009808
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009808
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042082, -0.0040506, -0.0042016, -0.0041340, -0.0000742, 0.0001510
1: -0.0102541, -0.0086959, -0.0100068, -0.0085892, -0.0016649, 0.0013109
2: 0.9641582, 0.9660279, 0.9644548, 0.9661561, -0.0019979, 0.0015731
3: -0.0180575, -0.0042658, -0.0158691, -0.0033210, -0.0119656, 0.0089430
4: -0.0003686, 0.0006803, -0.0004405, 0.0005139, -0.0008825, 0.0011208
5: 0.0168978, 0.0185203, 0.0168252, 0.0180310, -0.0011332, 0.0016951
6: 0.0016744, 0.0036877, 0.0026112, 0.0037230, -0.0020487, 0.0010765
7: -0.0066727, -0.0022092, -0.0069176, -0.0032840, -0.0033887, 0.0047084
8: 0.0114353, 0.0142709, 0.0112411, 0.0138210, -0.0023857, 0.0030299
9: 0.0182921, 0.0233923, 0.0179427, 0.0225830, -0.0039858, 0.0051335

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009808
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009808
time: 1.13 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040541, -0.0042067, -0.0040702, -0.0001377, 0.0001525
1: -0.0102436, -0.0086824, -0.0101961, -0.0086726, -0.0015710, 0.0015136
2: 0.9641707, 0.9660441, 0.9642278, 0.9660560, -0.0018853, 0.0018163
3: -0.0179648, -0.0041465, -0.0175440, -0.0040594, -0.0107349, 0.0102317
4: -0.0003777, 0.0006733, -0.0003843, 0.0006413, -0.0010190, 0.0010576
5: 0.0168886, 0.0184995, 0.0168820, 0.0184055, -0.0015168, 0.0016176
6: 0.0017140, 0.0036922, 0.0018942, 0.0036954, -0.0019814, 0.0017980
7: -0.0067036, -0.0022547, -0.0067262, -0.0024613, -0.0042423, 0.0044716
8: 0.0114108, 0.0142519, 0.0113929, 0.0141654, -0.0027546, 0.0028590
9: 0.0182480, 0.0233580, 0.0182158, 0.0232024, -0.0045990, 0.0047865

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014019, upper bound: 0.0013726
time: 1.69 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014019, upper bound: 0.0013726
time: 1.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042082, -0.0040506, -0.0042067, -0.0040703, -0.0001379, 0.0001560
1: -0.0102541, -0.0086959, -0.0101957, -0.0086797, -0.0015743, 0.0014998
2: 0.9641582, 0.9660279, 0.9642281, 0.9660474, -0.0018892, 0.0017998
3: -0.0180575, -0.0042658, -0.0175412, -0.0041227, -0.0107725, 0.0102040
4: -0.0003686, 0.0006803, -0.0003795, 0.0006411, -0.0010097, 0.0010598
5: 0.0168978, 0.0185203, 0.0168868, 0.0184049, -0.0015070, 0.0016334
6: 0.0016744, 0.0036877, 0.0018954, 0.0036930, -0.0020187, 0.0017923
7: -0.0066727, -0.0022092, -0.0067098, -0.0024627, -0.0042100, 0.0045006
8: 0.0114353, 0.0142709, 0.0114059, 0.0141648, -0.0027295, 0.0028650
9: 0.0182921, 0.0233923, 0.0182392, 0.0232014, -0.0045639, 0.0047977

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014019, upper bound: 0.0013728
time: 1.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014019, upper bound: 0.0013728
time: 1.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042029, -0.0041182, -0.0000898, 0.0001490
1: -0.0102443, -0.0086752, -0.0100536, -0.0085752, -0.0016691, 0.0013785
2: 0.9641699, 0.9660529, 0.9643987, 0.9661729, -0.0020030, 0.0016542
3: -0.0179709, -0.0040822, -0.0162835, -0.0031973, -0.0118547, 0.0092161
4: -0.0003826, 0.0006738, -0.0004499, 0.0005454, -0.0009280, 0.0011236
5: 0.0168837, 0.0185009, 0.0168157, 0.0181237, -0.0012400, 0.0016852
6: 0.0017114, 0.0036946, 0.0024338, 0.0037276, -0.0020162, 0.0012608
7: -0.0067203, -0.0022517, -0.0069497, -0.0030805, -0.0036398, 0.0046980
8: 0.0113976, 0.0142531, 0.0112156, 0.0139062, -0.0025086, 0.0030375
9: 0.0182242, 0.0233603, 0.0178970, 0.0227363, -0.0041755, 0.0051330

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013855, upper bound: 0.0011766
time: 1.40 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013820, upper bound: 0.0011137
time: 1.27 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042080, -0.0040539, -0.0001541, 0.0001541
1: -0.0102443, -0.0086752, -0.0102445, -0.0086633, -0.0015809, 0.0015693
2: 0.9641699, 0.9660529, 0.9641697, 0.9660670, -0.0018971, 0.0018832
3: -0.0179709, -0.0040822, -0.0179725, -0.0039775, -0.0106744, 0.0104909
4: -0.0003826, 0.0006738, -0.0003905, 0.0006739, -0.0010564, 0.0010643
5: 0.0168837, 0.0185009, 0.0168757, 0.0185013, -0.0016176, 0.0016253
6: 0.0017114, 0.0036946, 0.0017107, 0.0036985, -0.0019871, 0.0019838
7: -0.0067203, -0.0022517, -0.0067474, -0.0022509, -0.0044694, 0.0044958
8: 0.0113976, 0.0142531, 0.0113760, 0.0142535, -0.0028559, 0.0028771
9: 0.0182242, 0.0233603, 0.0181855, 0.0233608, -0.0047609, 0.0048040

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013825, upper bound: 0.0013619
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013825, upper bound: 0.0013772
time: 1.78 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042019, -0.0041302, -0.0042079, -0.0040546, -0.0001474, 0.0000777
1: -0.0100182, -0.0086134, -0.0102425, -0.0085972, -0.0014210, 0.0016291
2: 0.9644412, 0.9661269, 0.9641720, 0.9661465, -0.0017053, 0.0019549
3: -0.0159696, -0.0035357, -0.0179549, -0.0033919, -0.0099487, 0.0117469
4: -0.0004241, 0.0005215, -0.0004351, 0.0006725, -0.0010967, 0.0009566
5: 0.0168417, 0.0180535, 0.0168306, 0.0184973, -0.0016556, 0.0012229
6: 0.0025682, 0.0037150, 0.0017183, 0.0037204, -0.0011522, 0.0019967
7: -0.0068619, -0.0032347, -0.0068992, -0.0022595, -0.0046024, 0.0036646
8: 0.0112852, 0.0138417, 0.0112556, 0.0142499, -0.0029646, 0.0025860
9: 0.0180221, 0.0226202, 0.0179689, 0.0233544, -0.0050287, 0.0043523

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009917, upper bound: 0.0013740
time: 1.21 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009917, upper bound: 0.0013740
time: 1.20 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042083, -0.0040494, -0.0042066, -0.0040708, -0.0001375, 0.0001573
1: -0.0102578, -0.0087884, -0.0101943, -0.0086159, -0.0016419, 0.0014059
2: 0.9641536, 0.9659171, 0.9642299, 0.9661239, -0.0019703, 0.0016871
3: -0.0180907, -0.0050842, -0.0175284, -0.0035577, -0.0115098, 0.0092756
4: -0.0003063, 0.0006829, -0.0004224, 0.0006401, -0.0009464, 0.0011053
5: 0.0169607, 0.0185277, 0.0168434, 0.0184020, -0.0014413, 0.0016843
6: 0.0016601, 0.0036571, 0.0019009, 0.0037142, -0.0020540, 0.0017562
7: -0.0064606, -0.0021929, -0.0068562, -0.0024690, -0.0039916, 0.0046634
8: 0.0116036, 0.0142778, 0.0112897, 0.0141622, -0.0025586, 0.0029880
9: 0.0185948, 0.0234045, 0.0180303, 0.0231966, -0.0042470, 0.0050378

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012226, upper bound: 0.0013227
time: 1.43 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012189, upper bound: 0.0013162
time: 1.26 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042067, -0.0040702, -0.0042066, -0.0040707, -0.0001360, 0.0001364
1: -0.0101959, -0.0086844, -0.0101946, -0.0085997, -0.0015961, 0.0015101
2: 0.9642280, 0.9660417, 0.9642295, 0.9661434, -0.0019155, 0.0018122
3: -0.0175424, -0.0041642, -0.0175310, -0.0034145, -0.0111393, 0.0101474
4: -0.0003763, 0.0006412, -0.0004333, 0.0006403, -0.0010166, 0.0010745
5: 0.0168900, 0.0184051, 0.0168324, 0.0184026, -0.0015126, 0.0015727
6: 0.0018949, 0.0036915, 0.0018997, 0.0037195, -0.0018247, 0.0017918
7: -0.0066991, -0.0024622, -0.0068934, -0.0024677, -0.0042313, 0.0044312
8: 0.0114144, 0.0141650, 0.0112603, 0.0141627, -0.0027483, 0.0029047
9: 0.0182546, 0.0232018, 0.0179773, 0.0231976, -0.0045847, 0.0048896

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013812, upper bound: 0.0013553
time: 1.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013812, upper bound: 0.0013552
time: 1.32 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042083, -0.0040494, -0.0042079, -0.0040545, -0.0001538, 0.0001586
1: -0.0102578, -0.0087884, -0.0102425, -0.0086058, -0.0016520, 0.0014541
2: 0.9641536, 0.9659171, 0.9641721, 0.9661361, -0.0019825, 0.0017450
3: -0.0180907, -0.0050842, -0.0179552, -0.0034686, -0.0116104, 0.0097087
4: -0.0003063, 0.0006829, -0.0004292, 0.0006726, -0.0009789, 0.0011121
5: 0.0169607, 0.0185277, 0.0168365, 0.0184974, -0.0015367, 0.0016911
6: 0.0016601, 0.0036571, 0.0017181, 0.0037175, -0.0020574, 0.0019390
7: -0.0064606, -0.0021929, -0.0068793, -0.0022594, -0.0042013, 0.0046865
8: 0.0116036, 0.0142778, 0.0112714, 0.0142499, -0.0026463, 0.0030064
9: 0.0185948, 0.0234045, 0.0179973, 0.0233545, -0.0044048, 0.0050700

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012226, upper bound: 0.0013375
time: 1.30 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012189, upper bound: 0.0013347
time: 1.31 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042067, -0.0040702, -0.0042079, -0.0040544, -0.0001523, 0.0001377
1: -0.0101959, -0.0086844, -0.0102428, -0.0085899, -0.0016060, 0.0015584
2: 0.9642280, 0.9660417, 0.9641717, 0.9661552, -0.0019273, 0.0018700
3: -0.0175424, -0.0041642, -0.0179578, -0.0033274, -0.0112384, 0.0105852
4: -0.0003763, 0.0006412, -0.0004400, 0.0006728, -0.0010491, 0.0010811
5: 0.0168900, 0.0184051, 0.0168257, 0.0184980, -0.0016080, 0.0015794
6: 0.0018949, 0.0036915, 0.0017170, 0.0037228, -0.0018279, 0.0019745
7: -0.0066991, -0.0024622, -0.0069159, -0.0022581, -0.0044409, 0.0044538
8: 0.0114144, 0.0141650, 0.0112424, 0.0142504, -0.0028360, 0.0029227
9: 0.0182546, 0.0232018, 0.0179451, 0.0233554, -0.0047433, 0.0049216

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013803, upper bound: 0.0013673
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013803, upper bound: 0.0013827
time: 1.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042016, -0.0041338, -0.0000741, 0.0001477
1: -0.0102443, -0.0086752, -0.0100073, -0.0085092, -0.0017351, 0.0013322
2: 0.9641699, 0.9660529, 0.9644542, 0.9662521, -0.0020822, 0.0015987
3: -0.0179709, -0.0040822, -0.0158735, -0.0026129, -0.0126357, 0.0090357
4: -0.0003826, 0.0006738, -0.0004943, 0.0005142, -0.0008968, 0.0011681
5: 0.0168837, 0.0185009, 0.0167708, 0.0180320, -0.0011483, 0.0017302
6: 0.0017114, 0.0036946, 0.0026093, 0.0037495, -0.0020381, 0.0010852
7: -0.0067203, -0.0022517, -0.0071011, -0.0032819, -0.0034385, 0.0048494
8: 0.0113976, 0.0142531, 0.0110955, 0.0138219, -0.0024243, 0.0031577
9: 0.0182242, 0.0233603, 0.0176809, 0.0225846, -0.0040427, 0.0053675

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013927, upper bound: 0.0010787
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013895, upper bound: 0.0010021
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042066, -0.0040707, -0.0001373, 0.0001527
1: -0.0102443, -0.0086752, -0.0101946, -0.0085997, -0.0016445, 0.0015194
2: 0.9641699, 0.9660529, 0.9642295, 0.9661434, -0.0019736, 0.0018234
3: -0.0179709, -0.0040822, -0.0175310, -0.0034145, -0.0115891, 0.0102547
4: -0.0003826, 0.0006738, -0.0004333, 0.0006403, -0.0010229, 0.0011071
5: 0.0168837, 0.0185009, 0.0168324, 0.0184026, -0.0015189, 0.0016685
6: 0.0017114, 0.0036946, 0.0018997, 0.0037195, -0.0020081, 0.0017948
7: -0.0067203, -0.0022517, -0.0068934, -0.0024677, -0.0042526, 0.0046417
8: 0.0113976, 0.0142531, 0.0112603, 0.0141627, -0.0027651, 0.0029928
9: 0.0182242, 0.0233603, 0.0179773, 0.0231976, -0.0046167, 0.0050494

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013921, upper bound: 0.0013552
time: 1.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013921, upper bound: 0.0013755
time: 1.75 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042028, -0.0041190, -0.0000890, 0.0001489
1: -0.0102443, -0.0086752, -0.0100513, -0.0085046, -0.0017396, 0.0013762
2: 0.9641699, 0.9660529, 0.9644014, 0.9662575, -0.0020876, 0.0016515
3: -0.0179709, -0.0040822, -0.0162631, -0.0025728, -0.0125369, 0.0092568
4: -0.0003826, 0.0006738, -0.0004974, 0.0005439, -0.0009264, 0.0011711
5: 0.0168837, 0.0185009, 0.0167677, 0.0181191, -0.0012354, 0.0017332
6: 0.0017114, 0.0036946, 0.0024425, 0.0037510, -0.0020396, 0.0012521
7: -0.0067203, -0.0022517, -0.0071115, -0.0030905, -0.0036299, 0.0048598
8: 0.0113976, 0.0142531, 0.0110872, 0.0139020, -0.0025045, 0.0031659
9: 0.0182242, 0.0233603, 0.0176661, 0.0227287, -0.0041712, 0.0053663

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013966, upper bound: 0.0011809
time: 1.32 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013948, upper bound: 0.0011291
time: 1.29 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042080, -0.0040539, -0.0042079, -0.0040544, -0.0001535, 0.0001540
1: -0.0102443, -0.0086752, -0.0102428, -0.0085899, -0.0016544, 0.0015676
2: 0.9641699, 0.9660529, 0.9641717, 0.9661552, -0.0019854, 0.0018812
3: -0.0179709, -0.0040822, -0.0179578, -0.0033274, -0.0115254, 0.0105207
4: -0.0003826, 0.0006738, -0.0004400, 0.0006728, -0.0010553, 0.0011137
5: 0.0168837, 0.0185009, 0.0168257, 0.0184980, -0.0016143, 0.0016752
6: 0.0017114, 0.0036946, 0.0017170, 0.0037228, -0.0020114, 0.0019775
7: -0.0067203, -0.0022517, -0.0069159, -0.0022581, -0.0044622, 0.0046643
8: 0.0113976, 0.0142531, 0.0112424, 0.0142504, -0.0028529, 0.0030108
9: 0.0182242, 0.0233603, 0.0179451, 0.0233554, -0.0047583, 0.0050655

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013931, upper bound: 0.0013619
time: 1.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013931, upper bound: 0.0013763
time: 1.54 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042019, -0.0041302, -0.0042080, -0.0040540, -0.0001480, 0.0000777
1: -0.0100180, -0.0085409, -0.0102441, -0.0086705, -0.0013475, 0.0017032
2: 0.9644415, 0.9662139, 0.9641701, 0.9660584, -0.0016170, 0.0020438
3: -0.0159680, -0.0028942, -0.0179697, -0.0040412, -0.0092359, 0.0124509
4: -0.0004729, 0.0005214, -0.0003857, 0.0006737, -0.0011466, 0.0009071
5: 0.0167924, 0.0180531, 0.0168806, 0.0185006, -0.0017083, 0.0011726
6: 0.0025689, 0.0037390, 0.0017120, 0.0036961, -0.0011272, 0.0020270
7: -0.0070282, -0.0032355, -0.0067309, -0.0022523, -0.0047759, 0.0034955
8: 0.0111533, 0.0138413, 0.0113891, 0.0142529, -0.0030996, 0.0024522
9: 0.0177849, 0.0226196, 0.0182091, 0.0233598, -0.0052735, 0.0040976

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010041, upper bound: 0.0013915
time: 1.42 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010041, upper bound: 0.0013915
time: 1.43 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042083, -0.0040499, -0.0042067, -0.0040703, -0.0001380, 0.0001568
1: -0.0102562, -0.0087152, -0.0101958, -0.0086886, -0.0015676, 0.0014806
2: 0.9641556, 0.9660048, 0.9642281, 0.9660367, -0.0018812, 0.0017768
3: -0.0180763, -0.0044366, -0.0175416, -0.0042010, -0.0107087, 0.0100840
4: -0.0003556, 0.0006818, -0.0003735, 0.0006411, -0.0009967, 0.0010553
5: 0.0169109, 0.0185245, 0.0168928, 0.0184049, -0.0014940, 0.0016316
6: 0.0016663, 0.0036813, 0.0018952, 0.0036901, -0.0020238, 0.0017861
7: -0.0066285, -0.0021999, -0.0066895, -0.0024626, -0.0041659, 0.0044896
8: 0.0114704, 0.0142748, 0.0114220, 0.0141649, -0.0026944, 0.0028528
9: 0.0183553, 0.0233992, 0.0182682, 0.0232015, -0.0045090, 0.0047764

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012388, upper bound: 0.0013416
time: 1.26 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012358, upper bound: 0.0013363
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042066, -0.0040707, -0.0042067, -0.0040702, -0.0001364, 0.0001359
1: -0.0101944, -0.0086114, -0.0101961, -0.0086726, -0.0015218, 0.0015846
2: 0.9642297, 0.9661293, 0.9642278, 0.9660560, -0.0018263, 0.0019015
3: -0.0175293, -0.0035182, -0.0175440, -0.0040594, -0.0103354, 0.0109608
4: -0.0004254, 0.0006402, -0.0003843, 0.0006413, -0.0010667, 0.0010245
5: 0.0168403, 0.0184022, 0.0168820, 0.0184055, -0.0015651, 0.0015202
6: 0.0019005, 0.0037156, 0.0018942, 0.0036954, -0.0017949, 0.0018215
7: -0.0068665, -0.0024686, -0.0067262, -0.0024613, -0.0044052, 0.0042577
8: 0.0112816, 0.0141624, 0.0113929, 0.0141654, -0.0028838, 0.0027695
9: 0.0180157, 0.0231970, 0.0182158, 0.0232024, -0.0048470, 0.0046283

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013686, upper bound: 0.0013664
time: 1.40 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013686, upper bound: 0.0013863
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042083, -0.0040499, -0.0042080, -0.0040540, -0.0001543, 0.0001580
1: -0.0102562, -0.0087152, -0.0102442, -0.0086792, -0.0015770, 0.0015290
2: 0.9641556, 0.9660048, 0.9641700, 0.9660481, -0.0018926, 0.0018349
3: -0.0180763, -0.0044366, -0.0179702, -0.0041177, -0.0108187, 0.0105338
4: -0.0003556, 0.0006818, -0.0003798, 0.0006737, -0.0010293, 0.0010616
5: 0.0169109, 0.0185245, 0.0168864, 0.0185007, -0.0015898, 0.0016380
6: 0.0016663, 0.0036813, 0.0017117, 0.0036932, -0.0020270, 0.0019696
7: -0.0066285, -0.0021999, -0.0067111, -0.0022520, -0.0043764, 0.0045112
8: 0.0114704, 0.0142748, 0.0114049, 0.0142530, -0.0027825, 0.0028699
9: 0.0183553, 0.0233992, 0.0182374, 0.0233600, -0.0046688, 0.0048087

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012388, upper bound: 0.0013536
time: 1.34 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012358, upper bound: 0.0013509
time: 1.62 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042066, -0.0040707, -0.0042080, -0.0040539, -0.0001528, 0.0001372
1: -0.0101944, -0.0086114, -0.0102445, -0.0086633, -0.0015310, 0.0016330
2: 0.9642297, 0.9661293, 0.9641697, 0.9660670, -0.0018373, 0.0019596
3: -0.0175293, -0.0035182, -0.0179725, -0.0039775, -0.0104433, 0.0114130
4: -0.0004254, 0.0006402, -0.0003905, 0.0006739, -0.0010993, 0.0010307
5: 0.0168403, 0.0184022, 0.0168757, 0.0185013, -0.0016609, 0.0015265
6: 0.0019005, 0.0037156, 0.0017107, 0.0036985, -0.0017980, 0.0020049
7: -0.0068665, -0.0024686, -0.0067474, -0.0022509, -0.0046156, 0.0042789
8: 0.0112816, 0.0141624, 0.0113760, 0.0142535, -0.0029719, 0.0027863
9: 0.0180157, 0.0231970, 0.0181855, 0.0233608, -0.0050068, 0.0046601

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013679, upper bound: 0.0013767
time: 1.70 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013679, upper bound: 0.0013767
time: 1.78 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040547, -0.0042016, -0.0041339, -0.0000740, 0.0001469
1: -0.0102420, -0.0086087, -0.0100072, -0.0085821, -0.0016599, 0.0013985
2: 0.9641727, 0.9661327, 0.9644544, 0.9661646, -0.0019919, 0.0016783
3: -0.0179505, -0.0034936, -0.0158724, -0.0032582, -0.0119283, 0.0097416
4: -0.0004273, 0.0006722, -0.0004452, 0.0005142, -0.0009415, 0.0011174
5: 0.0168385, 0.0184963, 0.0168204, 0.0180318, -0.0011933, 0.0016760
6: 0.0017202, 0.0037166, 0.0026098, 0.0037254, -0.0020052, 0.0011068
7: -0.0068729, -0.0022617, -0.0069339, -0.0032824, -0.0035905, 0.0046722
8: 0.0112766, 0.0142489, 0.0112281, 0.0138217, -0.0025451, 0.0030208
9: 0.0180066, 0.0233527, 0.0179195, 0.0225842, -0.0042782, 0.0051201

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009917
time: 1.23 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009917
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042082, -0.0040517, -0.0042016, -0.0041340, -0.0000741, 0.0001499
1: -0.0102509, -0.0086221, -0.0100068, -0.0085892, -0.0016618, 0.0013847
2: 0.9641619, 0.9661165, 0.9644548, 0.9661561, -0.0019941, 0.0016617
3: -0.0180299, -0.0036127, -0.0158691, -0.0033210, -0.0119571, 0.0097220
4: -0.0004183, 0.0006782, -0.0004405, 0.0005139, -0.0009322, 0.0011187
5: 0.0168476, 0.0185141, 0.0168252, 0.0180310, -0.0011834, 0.0016889
6: 0.0016862, 0.0037121, 0.0026112, 0.0037230, -0.0020369, 0.0011009
7: -0.0068420, -0.0022227, -0.0069176, -0.0032840, -0.0035580, 0.0046949
8: 0.0113011, 0.0142653, 0.0112411, 0.0138210, -0.0025199, 0.0030242
9: 0.0180506, 0.0233821, 0.0179427, 0.0225830, -0.0042444, 0.0051271

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009917
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009917
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040547, -0.0042067, -0.0040702, -0.0001377, 0.0001520
1: -0.0102420, -0.0086087, -0.0101961, -0.0086726, -0.0015694, 0.0015874
2: 0.9641727, 0.9661327, 0.9642278, 0.9660560, -0.0018833, 0.0019049
3: -0.0179505, -0.0034936, -0.0175440, -0.0040594, -0.0107628, 0.0110732
4: -0.0004273, 0.0006722, -0.0003843, 0.0006413, -0.0010686, 0.0010565
5: 0.0168385, 0.0184963, 0.0168820, 0.0184055, -0.0015670, 0.0016144
6: 0.0017202, 0.0037166, 0.0018942, 0.0036954, -0.0019752, 0.0018224
7: -0.0068729, -0.0022617, -0.0067262, -0.0024613, -0.0044115, 0.0044645
8: 0.0112766, 0.0142489, 0.0113929, 0.0141654, -0.0028888, 0.0028561
9: 0.0180066, 0.0233527, 0.0182158, 0.0232024, -0.0048607, 0.0047840

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014012, upper bound: 0.0013812
time: 1.72 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014012, upper bound: 0.0013812
time: 1.62 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042082, -0.0040517, -0.0042067, -0.0040703, -0.0001379, 0.0001550
1: -0.0102509, -0.0086221, -0.0101957, -0.0086797, -0.0015712, 0.0015736
2: 0.9641619, 0.9661165, 0.9642281, 0.9660474, -0.0018855, 0.0018885
3: -0.0180299, -0.0036127, -0.0175412, -0.0041227, -0.0107877, 0.0110467
4: -0.0004183, 0.0006782, -0.0003795, 0.0006411, -0.0010593, 0.0010577
5: 0.0168476, 0.0185141, 0.0168868, 0.0184049, -0.0015572, 0.0016273
6: 0.0016862, 0.0037121, 0.0018954, 0.0036930, -0.0020069, 0.0018167
7: -0.0068420, -0.0022227, -0.0067098, -0.0024627, -0.0043793, 0.0044871
8: 0.0113011, 0.0142653, 0.0114059, 0.0141648, -0.0028637, 0.0028594
9: 0.0180506, 0.0233821, 0.0182392, 0.0232014, -0.0048261, 0.0047908

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014012, upper bound: 0.0013815
time: 1.75 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014012, upper bound: 0.0013815
time: 2.10 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042028, -0.0041194, -0.0042080, -0.0040539, -0.0001489, 0.0000886
1: -0.0100503, -0.0085237, -0.0102445, -0.0086633, -0.0013869, 0.0017207
2: 0.9644027, 0.9662347, 0.9641697, 0.9660670, -0.0016642, 0.0020650
3: -0.0162538, -0.0027415, -0.0179725, -0.0039775, -0.0094308, 0.0123720
4: -0.0004845, 0.0005432, -0.0003905, 0.0006739, -0.0011584, 0.0009337
5: 0.0167807, 0.0181170, 0.0168757, 0.0185013, -0.0017206, 0.0012414
6: 0.0024465, 0.0037447, 0.0017107, 0.0036985, -0.0012520, 0.0020339
7: -0.0070678, -0.0030950, -0.0067474, -0.0022509, -0.0048169, 0.0036524
8: 0.0111219, 0.0139001, 0.0113760, 0.0142535, -0.0031315, 0.0025241
9: 0.0177285, 0.0227253, 0.0181855, 0.0233608, -0.0053047, 0.0042120

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011374, upper bound: 0.0013853
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011374, upper bound: 0.0013853
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042031, -0.0041155, -0.0042080, -0.0040540, -0.0001491, 0.0000924
1: -0.0100616, -0.0085367, -0.0102441, -0.0086705, -0.0013911, 0.0017075
2: 0.9643892, 0.9662191, 0.9641701, 0.9660584, -0.0016692, 0.0020490
3: -0.0163539, -0.0028562, -0.0179697, -0.0040412, -0.0094703, 0.0123461
4: -0.0004758, 0.0005508, -0.0003857, 0.0006737, -0.0011495, 0.0009364
5: 0.0167895, 0.0181394, 0.0168806, 0.0185006, -0.0017112, 0.0012589
6: 0.0024036, 0.0037404, 0.0017120, 0.0036961, -0.0012924, 0.0020284
7: -0.0070380, -0.0030459, -0.0067309, -0.0022523, -0.0047858, 0.0036851
8: 0.0111455, 0.0139207, 0.0113891, 0.0142529, -0.0031074, 0.0025315
9: 0.0177709, 0.0227623, 0.0182091, 0.0233598, -0.0052708, 0.0042256

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011374, upper bound: 0.0013853
time: 1.54 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011374, upper bound: 0.0013853
time: 1.31 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040545, -0.0042029, -0.0041182, -0.0000897, 0.0001484
1: -0.0102426, -0.0086018, -0.0100536, -0.0085752, -0.0016674, 0.0014519
2: 0.9641718, 0.9661410, 0.9643987, 0.9661729, -0.0020010, 0.0017423
3: -0.0179561, -0.0034327, -0.0162835, -0.0031973, -0.0118585, 0.0100143
4: -0.0004319, 0.0006726, -0.0004499, 0.0005454, -0.0009774, 0.0011225
5: 0.0168338, 0.0184976, 0.0168157, 0.0181237, -0.0012899, 0.0016819
6: 0.0017177, 0.0037188, 0.0024338, 0.0037276, -0.0020099, 0.0012850
7: -0.0068886, -0.0022589, -0.0069497, -0.0030805, -0.0038081, 0.0046907
8: 0.0112640, 0.0142501, 0.0112156, 0.0139062, -0.0026422, 0.0030345
9: 0.0179841, 0.0233548, 0.0178970, 0.0227363, -0.0044335, 0.0051304

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013855, upper bound: 0.0011883
time: 1.55 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013820, upper bound: 0.0011250
time: 1.44 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042096, -0.0040334, -0.0042080, -0.0040540, -0.0001556, 0.0001746
1: -0.0103052, -0.0087045, -0.0102442, -0.0086792, -0.0016260, 0.0015397
2: 0.9640969, 0.9660177, 0.9641700, 0.9660481, -0.0019512, 0.0018477
3: -0.0185098, -0.0043417, -0.0179702, -0.0041177, -0.0110856, 0.0104688
4: -0.0003628, 0.0007148, -0.0003798, 0.0006737, -0.0010365, 0.0010946
5: 0.0169037, 0.0186214, 0.0168864, 0.0185007, -0.0015971, 0.0017350
6: 0.0014807, 0.0036849, 0.0017117, 0.0036932, -0.0022125, 0.0019731
7: -0.0066531, -0.0019870, -0.0067111, -0.0022520, -0.0044010, 0.0047241
8: 0.0114509, 0.0143639, 0.0114049, 0.0142530, -0.0028020, 0.0029591
9: 0.0183202, 0.0235596, 0.0182374, 0.0233600, -0.0046876, 0.0049527

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013713, upper bound: 0.0013560
time: 1.39 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013696, upper bound: 0.0013546
time: 1.55 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040545, -0.0042080, -0.0040539, -0.0001541, 0.0001535
1: -0.0102426, -0.0086018, -0.0102445, -0.0086633, -0.0015793, 0.0016427
2: 0.9641718, 0.9661410, 0.9641697, 0.9660670, -0.0018951, 0.0019713
3: -0.0179561, -0.0034327, -0.0179725, -0.0039775, -0.0107081, 0.0113444
4: -0.0004319, 0.0006726, -0.0003905, 0.0006739, -0.0011058, 0.0010632
5: 0.0168338, 0.0184976, 0.0168757, 0.0185013, -0.0016675, 0.0016220
6: 0.0017177, 0.0037188, 0.0017107, 0.0036985, -0.0019807, 0.0020081
7: -0.0068886, -0.0022589, -0.0067474, -0.0022509, -0.0046377, 0.0044885
8: 0.0112640, 0.0142501, 0.0113760, 0.0142535, -0.0029894, 0.0028741
9: 0.0179841, 0.0233548, 0.0181855, 0.0233608, -0.0050225, 0.0048014

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013824, upper bound: 0.0013728
time: 1.47 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013824, upper bound: 0.0013857
time: 1.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042019, -0.0041302, -0.0042079, -0.0040546, -0.0001474, 0.0000777
1: -0.0100180, -0.0085409, -0.0102425, -0.0085972, -0.0014208, 0.0017015
2: 0.9644415, 0.9662139, 0.9641720, 0.9661465, -0.0017051, 0.0020419
3: -0.0159680, -0.0028942, -0.0179549, -0.0033919, -0.0097436, 0.0121787
4: -0.0004729, 0.0005214, -0.0004351, 0.0006725, -0.0011455, 0.0009565
5: 0.0167924, 0.0180531, 0.0168306, 0.0184973, -0.0017050, 0.0012225
6: 0.0025689, 0.0037390, 0.0017183, 0.0037204, -0.0011515, 0.0020207
7: -0.0070282, -0.0032355, -0.0068992, -0.0022595, -0.0047687, 0.0036638
8: 0.0111533, 0.0138413, 0.0112556, 0.0142499, -0.0030966, 0.0025857
9: 0.0177849, 0.0226196, 0.0179689, 0.0233544, -0.0052424, 0.0043254

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010183, upper bound: 0.0013931
time: 1.46 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010183, upper bound: 0.0013931
time: 1.54 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042083, -0.0040499, -0.0042066, -0.0040708, -0.0001375, 0.0001567
1: -0.0102562, -0.0087152, -0.0101943, -0.0086159, -0.0016403, 0.0014791
2: 0.9641556, 0.9660048, 0.9642299, 0.9661239, -0.0019684, 0.0017749
3: -0.0180763, -0.0044366, -0.0175284, -0.0035577, -0.0112566, 0.0098293
4: -0.0003556, 0.0006818, -0.0004224, 0.0006401, -0.0009957, 0.0011042
5: 0.0169109, 0.0185245, 0.0168434, 0.0184020, -0.0014910, 0.0016811
6: 0.0016663, 0.0036813, 0.0019009, 0.0037142, -0.0020479, 0.0017804
7: -0.0066285, -0.0021999, -0.0068562, -0.0024690, -0.0041594, 0.0046563
8: 0.0114704, 0.0142748, 0.0112897, 0.0141622, -0.0026917, 0.0029851
9: 0.0183553, 0.0233992, 0.0180303, 0.0231966, -0.0044766, 0.0050055

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012566, upper bound: 0.0013423
time: 1.25 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012536, upper bound: 0.0013374
time: 1.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042066, -0.0040707, -0.0042066, -0.0040707, -0.0001360, 0.0001359
1: -0.0101944, -0.0086114, -0.0101946, -0.0085997, -0.0015947, 0.0015831
2: 0.9642297, 0.9661293, 0.9642295, 0.9661434, -0.0019137, 0.0018998
3: -0.0175293, -0.0035182, -0.0175310, -0.0034145, -0.0108852, 0.0107039
4: -0.0004254, 0.0006402, -0.0004333, 0.0006403, -0.0010658, 0.0010735
5: 0.0168403, 0.0184022, 0.0168324, 0.0184026, -0.0015622, 0.0015698
6: 0.0019005, 0.0037156, 0.0018997, 0.0037195, -0.0018191, 0.0018159
7: -0.0068665, -0.0024686, -0.0068934, -0.0024677, -0.0043988, 0.0044248
8: 0.0112816, 0.0141624, 0.0112603, 0.0141627, -0.0028811, 0.0029020
9: 0.0180157, 0.0231970, 0.0179773, 0.0231976, -0.0048146, 0.0048576

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013739, upper bound: 0.0013672
time: 1.47 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013739, upper bound: 0.0013866
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042083, -0.0040499, -0.0042079, -0.0040545, -0.0001538, 0.0001580
1: -0.0102562, -0.0087152, -0.0102425, -0.0086058, -0.0016503, 0.0015273
2: 0.9641556, 0.9660048, 0.9641721, 0.9661361, -0.0019805, 0.0018328
3: -0.0180763, -0.0044366, -0.0179552, -0.0034686, -0.0113653, 0.0102760
4: -0.0003556, 0.0006818, -0.0004292, 0.0006726, -0.0010282, 0.0011110
5: 0.0169109, 0.0185245, 0.0168365, 0.0184974, -0.0015865, 0.0016879
6: 0.0016663, 0.0036813, 0.0017181, 0.0037175, -0.0020512, 0.0019632
7: -0.0066285, -0.0021999, -0.0068793, -0.0022594, -0.0043691, 0.0046794
8: 0.0114704, 0.0142748, 0.0112714, 0.0142499, -0.0027795, 0.0030034
9: 0.0183553, 0.0233992, 0.0179973, 0.0233545, -0.0046363, 0.0050377

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012565, upper bound: 0.0013550
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012536, upper bound: 0.0013526
time: 1.38 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042066, -0.0040707, -0.0042079, -0.0040544, -0.0001522, 0.0001372
1: -0.0101944, -0.0086114, -0.0102428, -0.0085899, -0.0016045, 0.0016314
2: 0.9642297, 0.9661293, 0.9641717, 0.9661552, -0.0019255, 0.0019576
3: -0.0175293, -0.0035182, -0.0179578, -0.0033274, -0.0109937, 0.0111626
4: -0.0004254, 0.0006402, -0.0004400, 0.0006728, -0.0010982, 0.0010801
5: 0.0168403, 0.0184022, 0.0168257, 0.0184980, -0.0016576, 0.0015765
6: 0.0019005, 0.0037156, 0.0017170, 0.0037228, -0.0018223, 0.0019986
7: -0.0068665, -0.0024686, -0.0069159, -0.0022581, -0.0046084, 0.0044474
8: 0.0112816, 0.0141624, 0.0112424, 0.0142504, -0.0029688, 0.0029200
9: 0.0180157, 0.0231970, 0.0179451, 0.0233554, -0.0049750, 0.0048894

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013728, upper bound: 0.0013779
time: 1.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013728, upper bound: 0.0013910
time: 1.91 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042079, -0.0040545, -0.0042016, -0.0041338, -0.0000741, 0.0001471
1: -0.0102426, -0.0086018, -0.0100073, -0.0085092, -0.0017335, 0.0014055
2: 0.9641718, 0.9661410, 0.9644542, 0.9662521, -0.0020803, 0.0016868
3: -0.0179561, -0.0034327, -0.0158735, -0.0026129, -0.0123656, 0.0095454
4: -0.0004319, 0.0006726, -0.0004943, 0.0005142, -0.0009462, 0.0011669
5: 0.0168338, 0.0184976, 0.0167708, 0.0180320, -0.0011982, 0.0017269
6: 0.0017177, 0.0037188, 0.0026093, 0.0037495, -0.0020318, 0.0011095
7: -0.0068886, -0.0022589, -0.0071011, -0.0032819, -0.0036068, 0.0048422
8: 0.0112640, 0.0142501, 0.0110955, 0.0138219, -0.0025579, 0.0031546
9: 0.0179841, 0.0233548, 0.0176809, 0.0225846, -0.0042714, 0.0053366

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013873, upper bound: 0.0010973
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013842, upper bound: 0.0010219
time: 1.33 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.02 + 597.59 = 601.61 seconds
