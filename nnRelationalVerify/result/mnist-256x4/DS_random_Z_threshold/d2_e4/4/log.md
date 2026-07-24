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
execution time: IAR + RelationalAnalysis = 0.80 + 3.91 = 4.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -7.5008703, upper bound: 7.5008703

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 117

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4921971, upper bound: 7.4921971
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4921971, upper bound: 7.4921971
time: 2.27 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.27 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.27
Output dim: 8, lower bound: -7.4921971, upper bound: 7.4921971
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.27
Output dim: 8, lower bound: -7.4921971, upper bound: 7.4921971

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2395531, upper bound: 7.2395531
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2395531, upper bound: 7.2395531
time: 1.11 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4921387, upper bound: 7.4921387
time: 18.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4921387, upper bound: 7.4921387
time: 2.56 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 21.86 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 21.86
Output dim: 8, lower bound: -7.2395531, upper bound: 7.2395531
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 21.86
Output dim: 8, lower bound: -7.2395531, upper bound: 7.2395531
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 21.86
Output dim: 8, lower bound: -7.4921387, upper bound: 7.4921387
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 21.86
Output dim: 8, lower bound: -7.4921387, upper bound: 7.4921387

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253121, upper bound: 7.2253066
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253066, upper bound: 7.2253121
time: 1.51 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 163

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2395505, upper bound: 7.2395531
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2395531, upper bound: 7.2395505
time: 1.59 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4921379, upper bound: 7.4921387
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4921387, upper bound: 7.4921379
time: 2.33 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4854136, upper bound: 7.4854136
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4854136, upper bound: 7.4854136
time: 2.13 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.03 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 8, lower bound: -7.2253121, upper bound: 7.2253066
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 8, lower bound: -7.2253066, upper bound: 7.2253121
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 8, lower bound: -7.2395505, upper bound: 7.2395531
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 8, lower bound: -7.2395531, upper bound: 7.2395505
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 8, lower bound: -7.4921379, upper bound: 7.4921387
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 8, lower bound: -7.4921387, upper bound: 7.4921379
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 8, lower bound: -7.4854136, upper bound: 7.4854136
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 8, lower bound: -7.4854136, upper bound: 7.4854136

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253121, upper bound: 7.2253059
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253097, upper bound: 7.2253066
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253066, upper bound: 7.2253097
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253059, upper bound: 7.2253121
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2395505, upper bound: 7.2395396
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2395401, upper bound: 7.2395531
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1931746, upper bound: 7.1931601
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1931746, upper bound: 7.1931562
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2376622, upper bound: 7.2376630
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2376622, upper bound: 7.2376630
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3721726, upper bound: 7.3721566
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3721726, upper bound: 7.3721566
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4806199, upper bound: 7.4806129
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4806164, upper bound: 7.4806191
time: 2.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3878161, upper bound: 7.3878003
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3878094, upper bound: 7.3878090
time: 1.61 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.17 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.2253121, upper bound: 7.2253059
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.2253097, upper bound: 7.2253066
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.2253066, upper bound: 7.2253097
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.2253059, upper bound: 7.2253121
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.2395505, upper bound: 7.2395396
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.2395401, upper bound: 7.2395531
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.1931746, upper bound: 7.1931601
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.1931746, upper bound: 7.1931562
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.2376622, upper bound: 7.2376630
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.2376622, upper bound: 7.2376630
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.3721726, upper bound: 7.3721566
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.3721726, upper bound: 7.3721566
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.4806199, upper bound: 7.4806129
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.4806164, upper bound: 7.4806191
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.3878161, upper bound: 7.3878003
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 8, lower bound: -7.3878094, upper bound: 7.3878090

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2068499, upper bound: 7.2068539
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2068554, upper bound: 7.2068511
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253097, upper bound: 7.2253016
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253069, upper bound: 7.2253066
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253056, upper bound: 7.2253097
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253066, upper bound: 7.2253085
time: 1.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253059, upper bound: 7.2253121
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253053, upper bound: 7.2253097
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2250930, upper bound: 7.2250915
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2250930, upper bound: 7.2250915
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1191846, upper bound: 7.1192140
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1191857, upper bound: 7.1192131
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0674746, upper bound: 7.0674840
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0674746, upper bound: 7.0674840
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0173097, upper bound: 7.0173086
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0173097, upper bound: 7.0173086
time: 2.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2376622, upper bound: 7.2376592
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2376588, upper bound: 7.2376630
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0714683, upper bound: 7.0714705
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0714683, upper bound: 7.0714705
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3380669, upper bound: 7.3380350
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3380593, upper bound: 7.3380437
time: 2.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3698915, upper bound: 7.3698287
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3698552, upper bound: 7.3698691
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3543638, upper bound: 7.3543543
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3543634, upper bound: 7.3543551
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4783835, upper bound: 7.4783904
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4783865, upper bound: 7.4783904
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3068213, upper bound: 7.3068175
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3068200, upper bound: 7.3068190
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3877999, upper bound: 7.3878086
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3878094, upper bound: 7.3877964
time: 1.82 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.11 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.2068499, upper bound: 7.2068539
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.2068554, upper bound: 7.2068511
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.2253097, upper bound: 7.2253016
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.2253069, upper bound: 7.2253066
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.2253056, upper bound: 7.2253097
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.2253066, upper bound: 7.2253085
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.2253059, upper bound: 7.2253121
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.2253053, upper bound: 7.2253097
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.2250930, upper bound: 7.2250915
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.2250930, upper bound: 7.2250915
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.1191846, upper bound: 7.1192140
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.1191857, upper bound: 7.1192131
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.0674746, upper bound: 7.0674840
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.0674746, upper bound: 7.0674840
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.0173097, upper bound: 7.0173086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.0173097, upper bound: 7.0173086
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.2376622, upper bound: 7.2376592
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.2376588, upper bound: 7.2376630
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.0714683, upper bound: 7.0714705
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.0714683, upper bound: 7.0714705
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.3380669, upper bound: 7.3380350
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.3380593, upper bound: 7.3380437
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.3698915, upper bound: 7.3698287
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.3698552, upper bound: 7.3698691
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.3543638, upper bound: 7.3543543
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.3543634, upper bound: 7.3543551
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.4783835, upper bound: 7.4783904
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.4783865, upper bound: 7.4783904
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.3068213, upper bound: 7.3068175
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.3068200, upper bound: 7.3068190
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.3877999, upper bound: 7.3878086
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 8, lower bound: -7.3878094, upper bound: 7.3877964

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2068499, upper bound: 7.2068519
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2068485, upper bound: 7.2068539
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1583074, upper bound: 7.1582952
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1583013, upper bound: 7.1582983
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253085, upper bound: 7.2253015
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253097, upper bound: 7.2253016
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1061444, upper bound: 7.1061536
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1061446, upper bound: 7.1061510
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 163

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253036, upper bound: 7.2253097
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253056, upper bound: 7.2253045
time: 2.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0565366, upper bound: 7.0565532
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0565366, upper bound: 7.0565532
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0975136, upper bound: 7.0975153
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0975136, upper bound: 7.0975153
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253053, upper bound: 7.2253064
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253020, upper bound: 7.2253097
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1800434, upper bound: 7.1800142
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1800290, upper bound: 7.1800217
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -6.9506900, upper bound: 6.9506974
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -6.9506900, upper bound: 6.9506974
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2376622, upper bound: 7.2376353
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2376476, upper bound: 7.2376592
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1909704, upper bound: 7.1909877
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1909715, upper bound: 7.1909880
time: 3.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3380646, upper bound: 7.3380350
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3380669, upper bound: 7.3380339
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2476183, upper bound: 7.2476069
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2476182, upper bound: 7.2476075
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3698741, upper bound: 7.3698287
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3698915, upper bound: 7.3698225
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3234269, upper bound: 7.3234132
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3234269, upper bound: 7.3234165
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 163

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3543635, upper bound: 7.3543541
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3543638, upper bound: 7.3543543
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2447336, upper bound: 7.2447291
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2447336, upper bound: 7.2447292
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4650264, upper bound: 7.4650338
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4650264, upper bound: 7.4650338
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4739654, upper bound: 7.4739696
time: 2.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4739654, upper bound: 7.4739696
time: 3.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3068202, upper bound: 7.3068175
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3068213, upper bound: 7.3068150
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3067432, upper bound: 7.3067542
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3067501, upper bound: 7.3067407
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3067432, upper bound: 7.3067496
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3067414, upper bound: 7.3067553
time: 1.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3356540, upper bound: 7.3356512
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3356540, upper bound: 7.3356512
time: 1.39 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.54 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.2068499, upper bound: 7.2068519
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.2068485, upper bound: 7.2068539
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.1583074, upper bound: 7.1582952
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.1583013, upper bound: 7.1582983
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.2253085, upper bound: 7.2253015
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.2253097, upper bound: 7.2253016
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.1061444, upper bound: 7.1061536
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.1061446, upper bound: 7.1061510
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.2253036, upper bound: 7.2253097
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.2253056, upper bound: 7.2253045
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.0565366, upper bound: 7.0565532
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.0565366, upper bound: 7.0565532
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.0975136, upper bound: 7.0975153
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.0975136, upper bound: 7.0975153
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.2253053, upper bound: 7.2253064
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.2253020, upper bound: 7.2253097
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.1800434, upper bound: 7.1800142
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.1800290, upper bound: 7.1800217
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 8, lower bound: -6.9506900, upper bound: 6.9506974
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 8, lower bound: -6.9506900, upper bound: 6.9506974
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.2376622, upper bound: 7.2376353
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.2376476, upper bound: 7.2376592
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.1909704, upper bound: 7.1909877
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.1909715, upper bound: 7.1909880
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3380646, upper bound: 7.3380350
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3380669, upper bound: 7.3380339
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.2476183, upper bound: 7.2476069
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.2476182, upper bound: 7.2476075
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3698741, upper bound: 7.3698287
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3698915, upper bound: 7.3698225
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3234269, upper bound: 7.3234132
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3234269, upper bound: 7.3234165
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3543635, upper bound: 7.3543541
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3543638, upper bound: 7.3543543
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.2447336, upper bound: 7.2447291
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.2447336, upper bound: 7.2447292
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.4650264, upper bound: 7.4650338
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.4650264, upper bound: 7.4650338
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.4739654, upper bound: 7.4739696
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.4739654, upper bound: 7.4739696
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3068202, upper bound: 7.3068175
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3068213, upper bound: 7.3068150
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3067432, upper bound: 7.3067542
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3067501, upper bound: 7.3067407
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3067432, upper bound: 7.3067496
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3067414, upper bound: 7.3067553
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3356540, upper bound: 7.3356512
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 8, lower bound: -7.3356540, upper bound: 7.3356512

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0790201, upper bound: 7.0790194
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0790201, upper bound: 7.0790194
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0840334, upper bound: 7.0840403
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0840334, upper bound: 7.0840403
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1394288, upper bound: 7.1394140
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1394288, upper bound: 7.1394140
time: 2.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1107694, upper bound: 7.1107498
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1107659, upper bound: 7.1107515
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1061630, upper bound: 7.1061379
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1061631, upper bound: 7.1061349
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253097, upper bound: 7.2253009
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253039, upper bound: 7.2253016
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0547148, upper bound: 7.0547207
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0547148, upper bound: 7.0547207
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0975119, upper bound: 7.0975147
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0975119, upper bound: 7.0975147
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1032540, upper bound: 7.1032593
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1032540, upper bound: 7.1032593
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1633451, upper bound: 7.1633693
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1633451, upper bound: 7.1633693
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0628190, upper bound: 7.0628030
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0628237, upper bound: 7.0628023
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1800284, upper bound: 7.1800214
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1800290, upper bound: 7.1800217
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2230758, upper bound: 7.2230785
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2230753, upper bound: 7.2230819
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1172899, upper bound: 7.1173104
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1172900, upper bound: 7.1173094
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1909686, upper bound: 7.1909877
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1909704, upper bound: 7.1909819
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1473127, upper bound: 7.1473228
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1473113, upper bound: 7.1473245
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3261401, upper bound: 7.3261356
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3261387, upper bound: 7.3261367
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2521954, upper bound: 7.2521715
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2521954, upper bound: 7.2521763
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1711287, upper bound: 7.1711350
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1711287, upper bound: 7.1711350
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1467321, upper bound: 7.1467228
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1467252, upper bound: 7.1467294
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3698741, upper bound: 7.3698287
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3698716, upper bound: 7.3698285
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2327942, upper bound: 7.2327682
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2327941, upper bound: 7.2327699
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2907827, upper bound: 7.2907704
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2907835, upper bound: 7.2907695
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3234269, upper bound: 7.3234165
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3234269, upper bound: 7.3234165
time: 2.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3543609, upper bound: 7.3543541
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3543635, upper bound: 7.3543509
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3493243, upper bound: 7.3493144
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3493243, upper bound: 7.3493144
time: 1.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0472834, upper bound: 7.0472864
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0472834, upper bound: 7.0472864
time: 1.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1615710, upper bound: 7.1615487
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1615710, upper bound: 7.1615487
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4213345, upper bound: 7.4213469
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4213345, upper bound: 7.4213469
time: 1.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2787488, upper bound: 7.2787372
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2787488, upper bound: 7.2787372
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3253172, upper bound: 7.3253172
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3253172, upper bound: 7.3253172
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4715835, upper bound: 7.4715901
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4715835, upper bound: 7.4715901
time: 2.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2851390, upper bound: 7.2851279
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2851391, upper bound: 7.2851277
time: 1.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3067438, upper bound: 7.3067485
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3067562, upper bound: 7.3067397
time: 1.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3021815, upper bound: 7.3021944
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3021815, upper bound: 7.3021944
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2806341, upper bound: 7.2806237
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2806341, upper bound: 7.2806237
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1919187, upper bound: 7.1919173
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1919221, upper bound: 7.1919173
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2806228, upper bound: 7.2806371
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2806228, upper bound: 7.2806371
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1332081, upper bound: 7.1332102
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1332081, upper bound: 7.1332102
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3339429, upper bound: 7.3339418
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3339429, upper bound: 7.3339418
time: 1.69 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.16 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.0790201, upper bound: 7.0790194
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.0790201, upper bound: 7.0790194
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.0840334, upper bound: 7.0840403
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.0840334, upper bound: 7.0840403
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1394288, upper bound: 7.1394140
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1394288, upper bound: 7.1394140
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1107694, upper bound: 7.1107498
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1107659, upper bound: 7.1107515
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1061630, upper bound: 7.1061379
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1061631, upper bound: 7.1061349
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2253097, upper bound: 7.2253009
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2253039, upper bound: 7.2253016
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.0547148, upper bound: 7.0547207
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.0547148, upper bound: 7.0547207
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.0975119, upper bound: 7.0975147
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.0975119, upper bound: 7.0975147
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1032540, upper bound: 7.1032593
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1032540, upper bound: 7.1032593
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1633451, upper bound: 7.1633693
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1633451, upper bound: 7.1633693
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.0628190, upper bound: 7.0628030
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.0628237, upper bound: 7.0628023
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1800284, upper bound: 7.1800214
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1800290, upper bound: 7.1800217
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2230758, upper bound: 7.2230785
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2230753, upper bound: 7.2230819
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1172899, upper bound: 7.1173104
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1172900, upper bound: 7.1173094
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1909686, upper bound: 7.1909877
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1909704, upper bound: 7.1909819
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1473127, upper bound: 7.1473228
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1473113, upper bound: 7.1473245
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3261401, upper bound: 7.3261356
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3261387, upper bound: 7.3261367
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2521954, upper bound: 7.2521715
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2521954, upper bound: 7.2521763
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1711287, upper bound: 7.1711350
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1711287, upper bound: 7.1711350
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1467321, upper bound: 7.1467228
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1467252, upper bound: 7.1467294
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3698741, upper bound: 7.3698287
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3698716, upper bound: 7.3698285
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2327942, upper bound: 7.2327682
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2327941, upper bound: 7.2327699
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2907827, upper bound: 7.2907704
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2907835, upper bound: 7.2907695
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3234269, upper bound: 7.3234165
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3234269, upper bound: 7.3234165
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3543609, upper bound: 7.3543541
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3543635, upper bound: 7.3543509
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3493243, upper bound: 7.3493144
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3493243, upper bound: 7.3493144
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.0472834, upper bound: 7.0472864
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.0472834, upper bound: 7.0472864
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1615710, upper bound: 7.1615487
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1615710, upper bound: 7.1615487
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.4213345, upper bound: 7.4213469
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.4213345, upper bound: 7.4213469
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2787488, upper bound: 7.2787372
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2787488, upper bound: 7.2787372
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3253172, upper bound: 7.3253172
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3253172, upper bound: 7.3253172
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.4715835, upper bound: 7.4715901
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.4715835, upper bound: 7.4715901
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2851390, upper bound: 7.2851279
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2851391, upper bound: 7.2851277
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3067438, upper bound: 7.3067485
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3067562, upper bound: 7.3067397
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3021815, upper bound: 7.3021944
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3021815, upper bound: 7.3021944
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2806341, upper bound: 7.2806237
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2806341, upper bound: 7.2806237
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1919187, upper bound: 7.1919173
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1919221, upper bound: 7.1919173
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2806228, upper bound: 7.2806371
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.2806228, upper bound: 7.2806371
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1332081, upper bound: 7.1332102
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.1332081, upper bound: 7.1332102
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3339429, upper bound: 7.3339418
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 8, lower bound: -7.3339429, upper bound: 7.3339418

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1394288, upper bound: 7.1394140
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1394288, upper bound: 7.1394140
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1371179, upper bound: 7.1371041
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1371189, upper bound: 7.1371041
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 163

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253045, upper bound: 7.2253009
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2253097, upper bound: 7.2252999
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1759636, upper bound: 7.1759513
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1759628, upper bound: 7.1759565
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1633451, upper bound: 7.1633634
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1633442, upper bound: 7.1633693
time: 1.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0204057, upper bound: 7.0204179
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0204057, upper bound: 7.0204179
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1754064, upper bound: 7.1754066
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1754112, upper bound: 7.1754001
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0105003, upper bound: 7.0105025
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0105003, upper bound: 7.0105025
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0948395, upper bound: 7.0948375
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0948395, upper bound: 7.0948375
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1567796, upper bound: 7.1567778
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1567796, upper bound: 7.1567778
time: 1.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1719111, upper bound: 7.1719241
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1719111, upper bound: 7.1719241
time: 1.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606324, upper bound: 7.0606403
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606324, upper bound: 7.0606403
time: 1.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 163

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1473127, upper bound: 7.1473228
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1473119, upper bound: 7.1473217
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 163

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1473113, upper bound: 7.1473245
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1473109, upper bound: 7.1473222
time: 1.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3261388, upper bound: 7.3261343
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3261401, upper bound: 7.3261356
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3261387, upper bound: 7.3261356
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3261382, upper bound: 7.3261367
time: 1.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2522054, upper bound: 7.2521708
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2522022, upper bound: 7.2521715
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2422291, upper bound: 7.2422386
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2422425, upper bound: 7.2422366
time: 1.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 163

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1711280, upper bound: 7.1711350
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1711287, upper bound: 7.1711324
time: 1.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -6.9149169, upper bound: 6.9149228
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -6.9149169, upper bound: 6.9149228
time: 1.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1467321, upper bound: 7.1467181
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1467283, upper bound: 7.1467228
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1372142, upper bound: 7.1372216
time: 6.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1372082, upper bound: 7.1372322
time: 1.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2777175, upper bound: 7.2777112
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2777313, upper bound: 7.2777003
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3234391, upper bound: 7.3233982
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3234377, upper bound: 7.3233984
time: 2.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2327942, upper bound: 7.2327651
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2327939, upper bound: 7.2327682
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2327873, upper bound: 7.2327699
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2327941, upper bound: 7.2327650
time: 1.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2907827, upper bound: 7.2907701
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2907818, upper bound: 7.2907704
time: 1.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2277894, upper bound: 7.2278033
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2277894, upper bound: 7.2278033
time: 2.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1572012, upper bound: 7.1571884
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1572012, upper bound: 7.1571884
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2672524, upper bound: 7.2672626
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2672524, upper bound: 7.2672626
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -6.9444651, upper bound: 6.9444675
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -6.9444651, upper bound: 6.9444675
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3431406, upper bound: 7.3431328
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3431455, upper bound: 7.3431324
time: 2.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3222837, upper bound: 7.3222812
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3222851, upper bound: 7.3222802
time: 1.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1590238, upper bound: 7.1590206
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1590238, upper bound: 7.1590206
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1615710, upper bound: 7.1615487
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1615710, upper bound: 7.1615473
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0711898, upper bound: 7.0711865
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0711893, upper bound: 7.0711867
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4113002, upper bound: 7.4113037
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4113002, upper bound: 7.4113037
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2461296, upper bound: 7.2461244
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2461296, upper bound: 7.2461244
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2787452, upper bound: 7.2787372
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2787488, upper bound: 7.2787347
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2787452, upper bound: 7.2787372
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2787488, upper bound: 7.2787347
time: 1.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3002905, upper bound: 7.3002959
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3003025, upper bound: 7.3002894
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3122045, upper bound: 7.3122210
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3122217, upper bound: 7.3122046
time: 1.45 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.20 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1394288, upper bound: 7.1394140
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1394288, upper bound: 7.1394140
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1371179, upper bound: 7.1371041
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1371189, upper bound: 7.1371041
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2253045, upper bound: 7.2253009
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2253097, upper bound: 7.2252999
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1759636, upper bound: 7.1759513
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1759628, upper bound: 7.1759565
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1633451, upper bound: 7.1633634
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1633442, upper bound: 7.1633693
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.0204057, upper bound: 7.0204179
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.0204057, upper bound: 7.0204179
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1754064, upper bound: 7.1754066
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1754112, upper bound: 7.1754001
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.0105003, upper bound: 7.0105025
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.0105003, upper bound: 7.0105025
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.0948395, upper bound: 7.0948375
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.0948395, upper bound: 7.0948375
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1567796, upper bound: 7.1567778
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1567796, upper bound: 7.1567778
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1719111, upper bound: 7.1719241
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1719111, upper bound: 7.1719241
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.0606324, upper bound: 7.0606403
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.0606324, upper bound: 7.0606403
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1473127, upper bound: 7.1473228
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1473119, upper bound: 7.1473217
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1473113, upper bound: 7.1473245
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1473109, upper bound: 7.1473222
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.3261388, upper bound: 7.3261343
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.3261401, upper bound: 7.3261356
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.3261387, upper bound: 7.3261356
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.3261382, upper bound: 7.3261367
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2522054, upper bound: 7.2521708
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2522022, upper bound: 7.2521715
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2422291, upper bound: 7.2422386
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2422425, upper bound: 7.2422366
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1711280, upper bound: 7.1711350
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1711287, upper bound: 7.1711324
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.20
Output dim: 8, lower bound: -6.9149169, upper bound: 6.9149228
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.20
Output dim: 8, lower bound: -6.9149169, upper bound: 6.9149228
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1467321, upper bound: 7.1467181
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1467283, upper bound: 7.1467228
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1372142, upper bound: 7.1372216
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1372082, upper bound: 7.1372322
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2777175, upper bound: 7.2777112
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2777313, upper bound: 7.2777003
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.3234391, upper bound: 7.3233982
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.3234377, upper bound: 7.3233984
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2327942, upper bound: 7.2327651
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2327939, upper bound: 7.2327682
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2327873, upper bound: 7.2327699
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2327941, upper bound: 7.2327650
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2907827, upper bound: 7.2907701
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2907818, upper bound: 7.2907704
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2277894, upper bound: 7.2278033
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2277894, upper bound: 7.2278033
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1572012, upper bound: 7.1571884
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1572012, upper bound: 7.1571884
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2672524, upper bound: 7.2672626
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2672524, upper bound: 7.2672626
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.20
Output dim: 8, lower bound: -6.9444651, upper bound: 6.9444675
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.20
Output dim: 8, lower bound: -6.9444651, upper bound: 6.9444675
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.3431406, upper bound: 7.3431328
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.3431455, upper bound: 7.3431324
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.3222837, upper bound: 7.3222812
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.3222851, upper bound: 7.3222802
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1590238, upper bound: 7.1590206
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1590238, upper bound: 7.1590206
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1615710, upper bound: 7.1615487
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.1615710, upper bound: 7.1615473
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.0711898, upper bound: 7.0711865
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.0711893, upper bound: 7.0711867
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.4113002, upper bound: 7.4113037
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.4113002, upper bound: 7.4113037
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2461296, upper bound: 7.2461244
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2461296, upper bound: 7.2461244
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2787452, upper bound: 7.2787372
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2787488, upper bound: 7.2787347
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2787452, upper bound: 7.2787372
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.2787488, upper bound: 7.2787347
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.3002905, upper bound: 7.3002959
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.3003025, upper bound: 7.3002894
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.3122045, upper bound: 7.3122210
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.20
Output dim: 8, lower bound: -7.3122217, upper bound: 7.3122046
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.4715835, upper bound: 7.4715901
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.4715835, upper bound: 7.4715901
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.2851390, upper bound: 7.2851279
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.2851391, upper bound: 7.2851277
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.3067438, upper bound: 7.3067485
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.3067562, upper bound: 7.3067397
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.3021815, upper bound: 7.3021944
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.3021815, upper bound: 7.3021944
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.2806341, upper bound: 7.2806237
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.2806341, upper bound: 7.2806237
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.1919187, upper bound: 7.1919173
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.1919221, upper bound: 7.1919173
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.2806228, upper bound: 7.2806371
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.2806228, upper bound: 7.2806371
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.1332081, upper bound: 7.1332102
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.1332081, upper bound: 7.1332102
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.3339429, upper bound: 7.3339418
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.20
Output dim: 8, lower bound: -7.3339429, upper bound: 7.3339418

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.70 + 596.09 = 600.80 seconds
