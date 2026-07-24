## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 7.125826784999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.5828104, 2.9839077, -3.5828104, 2.9839077, -6.5667181, 6.5667181)
1: (-2.8148921, 2.6596637, -2.8148921, 2.6596637, -5.4745560, 5.4745560)
2: (-3.6384213, 2.7152596, -3.6384213, 2.7152596, -6.3536806, 6.3536806)
3: (-4.0147090, 2.3897834, -4.0147090, 2.3897834, -6.4044924, 6.4044924)
4: (-3.9739902, 2.8952332, -3.9739902, 2.8952332, -6.8692236, 6.8692236)
5: (-3.4501100, 2.9820681, -3.4501100, 2.9820681, -6.4321771, 6.4321775)
6: (-3.2314649, 3.2592940, -3.2314649, 3.2592940, -6.4907575, 6.4907589)
7: (-3.3493385, 3.3119178, -3.3493385, 3.3119178, -6.6612563, 6.6612563)
8: (-5.2216048, 3.1262531, -5.2216048, 3.1262531, -8.3478584, 8.3478584)
9: (-3.0501204, 3.2086091, -3.0501204, 3.2086091, -6.2587285, 6.2587290)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.89 + 4.01 = 5.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -7.5008703, upper bound: 7.5008703

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.5003434, upper bound: 7.5003439
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.5003439, upper bound: 7.5003434
time: 2.12 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.71 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.71
Output dim: 8, lower bound: -7.5003434, upper bound: 7.5003439
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.71
Output dim: 8, lower bound: -7.5003439, upper bound: 7.5003434

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.5828104, 2.9839077, -3.5828104, 2.9839077, -6.5667181, 6.5667181
1: -2.8148921, 2.6596637, -2.8148921, 2.6596637, -5.4745560, 5.4745560
2: -3.6384213, 2.7152596, -3.6384213, 2.7152596, -6.3536806, 6.3536806
3: -4.0147090, 2.3897834, -4.0147090, 2.3897834, -6.4044924, 6.4044924
4: -3.9739902, 2.8952332, -3.9739902, 2.8952332, -6.8692236, 6.8692236
5: -3.4501100, 2.9820681, -3.4501100, 2.9820681, -6.4321771, 6.4321775
6: -3.2314649, 3.2592940, -3.2314649, 3.2592940, -6.4907575, 6.4907589
7: -3.3493385, 3.3119178, -3.3493385, 3.3119178, -6.6612563, 6.6612563
8: -5.2216048, 3.1262531, -5.2216048, 3.1262531, -8.3478584, 8.3478584
9: -3.0501204, 3.2086091, -3.0501204, 3.2086091, -6.2587285, 6.2587290

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3934550, upper bound: 7.3934552
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3934550, upper bound: 7.3934552
time: 1.43 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.5828104, 2.9839077, -3.5828104, 2.9839077, -6.5667181, 6.5667181
1: -2.8148921, 2.6596637, -2.8148921, 2.6596637, -5.4745560, 5.4745560
2: -3.6384213, 2.7152596, -3.6384213, 2.7152596, -6.3536806, 6.3536806
3: -4.0147090, 2.3897834, -4.0147090, 2.3897834, -6.4044924, 6.4044924
4: -3.9739902, 2.8952332, -3.9739902, 2.8952332, -6.8692236, 6.8692236
5: -3.4501100, 2.9820681, -3.4501100, 2.9820681, -6.4321771, 6.4321775
6: -3.2314649, 3.2592940, -3.2314649, 3.2592940, -6.4907575, 6.4907589
7: -3.3493385, 3.3119178, -3.3493385, 3.3119178, -6.6612563, 6.6612563
8: -5.2216048, 3.1262531, -5.2216048, 3.1262531, -8.3478584, 8.3478584
9: -3.0501204, 3.2086091, -3.0501204, 3.2086091, -6.2587285, 6.2587290

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3934552, upper bound: 7.3934550
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3934552, upper bound: 7.3934550
time: 1.53 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.88 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.88
Output dim: 8, lower bound: -7.3934550, upper bound: 7.3934552
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.88
Output dim: 8, lower bound: -7.3934550, upper bound: 7.3934552
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.88
Output dim: 8, lower bound: -7.3934552, upper bound: 7.3934550
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.88
Output dim: 8, lower bound: -7.3934552, upper bound: 7.3934550

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5828104, 2.9839077, -3.5828104, 2.9839077, -6.5667181, 6.5667181
1: -2.8148921, 2.6596637, -2.8148921, 2.6596637, -5.4745560, 5.4745560
2: -3.6384213, 2.7152596, -3.6384213, 2.7152596, -6.3536806, 6.3536806
3: -4.0147090, 2.3897834, -4.0147090, 2.3897834, -6.4044924, 6.4044924
4: -3.9739902, 2.8952332, -3.9739902, 2.8952332, -6.8692236, 6.8692236
5: -3.4501100, 2.9820681, -3.4501100, 2.9820681, -6.4321771, 6.4321775
6: -3.2314649, 3.2592940, -3.2314649, 3.2592940, -6.4907575, 6.4907589
7: -3.3493385, 3.3119178, -3.3493385, 3.3119178, -6.6612563, 6.6612563
8: -5.2216048, 3.1262531, -5.2216048, 3.1262531, -8.3478584, 8.3478584
9: -3.0501204, 3.2086091, -3.0501204, 3.2086091, -6.2587285, 6.2587290

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5828104, 2.9839077, -3.5828104, 2.9839077, -6.5667181, 6.5667181
1: -2.8148921, 2.6596637, -2.8148921, 2.6596637, -5.4745560, 5.4745560
2: -3.6384213, 2.7152596, -3.6384213, 2.7152596, -6.3536806, 6.3536806
3: -4.0147090, 2.3897834, -4.0147090, 2.3897834, -6.4044924, 6.4044924
4: -3.9739902, 2.8952332, -3.9739902, 2.8952332, -6.8692236, 6.8692236
5: -3.4501100, 2.9820681, -3.4501100, 2.9820681, -6.4321771, 6.4321775
6: -3.2314649, 3.2592940, -3.2314649, 3.2592940, -6.4907575, 6.4907589
7: -3.3493385, 3.3119178, -3.3493385, 3.3119178, -6.6612563, 6.6612563
8: -5.2216048, 3.1262531, -5.2216048, 3.1262531, -8.3478584, 8.3478584
9: -3.0501204, 3.2086091, -3.0501204, 3.2086091, -6.2587285, 6.2587290

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.5828104, 2.9839077, -3.5828104, 2.9839077, -6.5667181, 6.5667181
1: -2.8148921, 2.6596637, -2.8148921, 2.6596637, -5.4745560, 5.4745560
2: -3.6384213, 2.7152596, -3.6384213, 2.7152596, -6.3536806, 6.3536806
3: -4.0147090, 2.3897834, -4.0147090, 2.3897834, -6.4044924, 6.4044924
4: -3.9739902, 2.8952332, -3.9739902, 2.8952332, -6.8692236, 6.8692236
5: -3.4501100, 2.9820681, -3.4501100, 2.9820681, -6.4321771, 6.4321775
6: -3.2314649, 3.2592940, -3.2314649, 3.2592940, -6.4907575, 6.4907589
7: -3.3493385, 3.3119178, -3.3493385, 3.3119178, -6.6612563, 6.6612563
8: -5.2216048, 3.1262531, -5.2216048, 3.1262531, -8.3478584, 8.3478584
9: -3.0501204, 3.2086091, -3.0501204, 3.2086091, -6.2587285, 6.2587290

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.5828104, 2.9839077, -3.5828104, 2.9839077, -6.5667181, 6.5667181
1: -2.8148921, 2.6596637, -2.8148921, 2.6596637, -5.4745560, 5.4745560
2: -3.6384213, 2.7152596, -3.6384213, 2.7152596, -6.3536806, 6.3536806
3: -4.0147090, 2.3897834, -4.0147090, 2.3897834, -6.4044924, 6.4044924
4: -3.9739902, 2.8952332, -3.9739902, 2.8952332, -6.8692236, 6.8692236
5: -3.4501100, 2.9820681, -3.4501100, 2.9820681, -6.4321771, 6.4321775
6: -3.2314649, 3.2592940, -3.2314649, 3.2592940, -6.4907575, 6.4907589
7: -3.3493385, 3.3119178, -3.3493385, 3.3119178, -6.6612563, 6.6612563
8: -5.2216048, 3.1262531, -5.2216048, 3.1262531, -8.3478584, 8.3478584
9: -3.0501204, 3.2086091, -3.0501204, 3.2086091, -6.2587285, 6.2587290

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.54 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.22 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.22
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.22
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.22
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.22
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.22
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.22
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.22
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.22
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 5.91 + 33.98 = 39.88 seconds
