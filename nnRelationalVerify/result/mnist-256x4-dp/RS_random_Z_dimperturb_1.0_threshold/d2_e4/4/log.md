## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 7.125826784999999


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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
execution time: IAR + RelationalAnalysis = 1.34 + 4.09 = 5.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -7.5008703, upper bound: 7.5008703

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4991434, upper bound: 7.4991431
time: 2.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4991431, upper bound: 7.4991434
time: 2.77 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 5.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 5.57
Output dim: 8, lower bound: -7.4991434, upper bound: 7.4991431
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 5.57
Output dim: 8, lower bound: -7.4991431, upper bound: 7.4991434

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4941549, upper bound: 7.4941549
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4941549, upper bound: 7.4941549
time: 2.04 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4991431, upper bound: 7.4991415
time: 2.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4991415, upper bound: 7.4991434
time: 2.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.62 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.62
Output dim: 8, lower bound: -7.4941549, upper bound: 7.4941549
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.62
Output dim: 8, lower bound: -7.4941549, upper bound: 7.4941549
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.62
Output dim: 8, lower bound: -7.4991431, upper bound: 7.4991415
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.62
Output dim: 8, lower bound: -7.4991415, upper bound: 7.4991434

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4937648, upper bound: 7.4937650
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4937648, upper bound: 7.4937650
time: 2.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4851482, upper bound: 7.4851482
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4851482, upper bound: 7.4851482
time: 1.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4974396, upper bound: 7.4974399
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4974396, upper bound: 7.4974399
time: 3.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4985105, upper bound: 7.4985126
time: 2.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4985109, upper bound: 7.4985124
time: 2.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.20
Output dim: 8, lower bound: -7.4937648, upper bound: 7.4937650
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.20
Output dim: 8, lower bound: -7.4937648, upper bound: 7.4937650
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.20
Output dim: 8, lower bound: -7.4851482, upper bound: 7.4851482
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.20
Output dim: 8, lower bound: -7.4851482, upper bound: 7.4851482
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.20
Output dim: 8, lower bound: -7.4974396, upper bound: 7.4974399
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.20
Output dim: 8, lower bound: -7.4974396, upper bound: 7.4974399
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.20
Output dim: 8, lower bound: -7.4985105, upper bound: 7.4985126
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.20
Output dim: 8, lower bound: -7.4985109, upper bound: 7.4985124

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4937069, upper bound: 7.4937066
time: 2.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4937062, upper bound: 7.4937072
time: 2.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4928941, upper bound: 7.4928946
time: 2.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4928941, upper bound: 7.4928946
time: 2.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4850592, upper bound: 7.4850593
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4850593, upper bound: 7.4850593
time: 1.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4823720, upper bound: 7.4823591
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4823785, upper bound: 7.4823508
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4964423, upper bound: 7.4964392
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4964423, upper bound: 7.4964392
time: 2.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4974008, upper bound: 7.4974014
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4974016, upper bound: 7.4974006
time: 2.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4931230, upper bound: 7.4931247
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4931242, upper bound: 7.4931236
time: 1.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4984033, upper bound: 7.4984146
time: 2.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4984123, upper bound: 7.4984044
time: 2.45 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4937069, upper bound: 7.4937066
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4937062, upper bound: 7.4937072
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4928941, upper bound: 7.4928946
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4928941, upper bound: 7.4928946
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4850592, upper bound: 7.4850593
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4850593, upper bound: 7.4850593
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4823720, upper bound: 7.4823591
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4823785, upper bound: 7.4823508
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4964423, upper bound: 7.4964392
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4964423, upper bound: 7.4964392
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4974008, upper bound: 7.4974014
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4974016, upper bound: 7.4974006
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4931230, upper bound: 7.4931247
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4931242, upper bound: 7.4931236
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4984033, upper bound: 7.4984146
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.37
Output dim: 8, lower bound: -7.4984123, upper bound: 7.4984044

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4934054, upper bound: 7.4934039
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4934054, upper bound: 7.4934039
time: 2.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4876043, upper bound: 7.4876084
time: 2.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4876069, upper bound: 7.4876059
time: 2.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4927401, upper bound: 7.4927481
time: 2.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4927495, upper bound: 7.4927377
time: 2.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4902993, upper bound: 7.4902968
time: 2.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4902993, upper bound: 7.4902968
time: 2.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4848618, upper bound: 7.4848657
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4848647, upper bound: 7.4848618
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4850593, upper bound: 7.4850593
time: 2.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4850593, upper bound: 7.4850593
time: 2.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4823647, upper bound: 7.4823591
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4823720, upper bound: 7.4823552
time: 2.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4823785, upper bound: 7.4823508
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4823785, upper bound: 7.4823508
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4878000, upper bound: 7.4878038
time: 2.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4878008, upper bound: 7.4878037
time: 2.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4869071, upper bound: 7.4869113
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4869071, upper bound: 7.4869113
time: 2.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4951677, upper bound: 7.4951677
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4951672, upper bound: 7.4951685
time: 2.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2228007, upper bound: 7.2228221
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2228007, upper bound: 7.2228221
time: 1.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4830125, upper bound: 7.4830337
time: 2.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4830125, upper bound: 7.4830337
time: 2.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4907366, upper bound: 7.4907398
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4907366, upper bound: 7.4907398
time: 2.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4943736, upper bound: 7.4944036
time: 2.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4943921, upper bound: 7.4943847
time: 2.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4984115, upper bound: 7.4984044
time: 2.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4984123, upper bound: 7.4984033
time: 2.15 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4934054, upper bound: 7.4934039
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4934054, upper bound: 7.4934039
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4876043, upper bound: 7.4876084
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4876069, upper bound: 7.4876059
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4927401, upper bound: 7.4927481
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4927495, upper bound: 7.4927377
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4902993, upper bound: 7.4902968
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4902993, upper bound: 7.4902968
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4848618, upper bound: 7.4848657
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4848647, upper bound: 7.4848618
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4850593, upper bound: 7.4850593
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4850593, upper bound: 7.4850593
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4823647, upper bound: 7.4823591
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4823720, upper bound: 7.4823552
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4823785, upper bound: 7.4823508
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4823785, upper bound: 7.4823508
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4878000, upper bound: 7.4878038
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4878008, upper bound: 7.4878037
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4869071, upper bound: 7.4869113
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4869071, upper bound: 7.4869113
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4951677, upper bound: 7.4951677
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4951672, upper bound: 7.4951685
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.2228007, upper bound: 7.2228221
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.2228007, upper bound: 7.2228221
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4830125, upper bound: 7.4830337
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4830125, upper bound: 7.4830337
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4907366, upper bound: 7.4907398
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4907366, upper bound: 7.4907398
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4943736, upper bound: 7.4944036
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4943921, upper bound: 7.4943847
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4984115, upper bound: 7.4984044
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.12
Output dim: 8, lower bound: -7.4984123, upper bound: 7.4984033

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4908549, upper bound: 7.4908496
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4908549, upper bound: 7.4908496
time: 2.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4934054, upper bound: 7.4934034
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4934037, upper bound: 7.4934039
time: 2.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3497535, upper bound: 7.3497126
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3497535, upper bound: 7.3497126
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4842669, upper bound: 7.4842493
time: 2.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4842579, upper bound: 7.4842559
time: 2.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4920890, upper bound: 7.4920960
time: 3.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4920891, upper bound: 7.4920955
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4921681, upper bound: 7.4921592
time: 2.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4921681, upper bound: 7.4921592
time: 3.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4902993, upper bound: 7.4902968
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4902993, upper bound: 7.4902968
time: 2.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4396270, upper bound: 7.4395550
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4396270, upper bound: 7.4395550
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3805092, upper bound: 7.3804854
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3805092, upper bound: 7.3804854
time: 1.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4805850, upper bound: 7.4805768
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4805921, upper bound: 7.4805653
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3121514, upper bound: 7.3121340
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3121514, upper bound: 7.3121340
time: 1.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1329630, upper bound: 7.1329482
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1329630, upper bound: 7.1329482
time: 2.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4807474, upper bound: 7.4807162
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4807474, upper bound: 7.4807162
time: 2.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4823720, upper bound: 7.4823552
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4823720, upper bound: 7.4823552
time: 2.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4807577, upper bound: 7.4807034
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4807577, upper bound: 7.4807034
time: 1.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4817327, upper bound: 7.4817055
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4817337, upper bound: 7.4817053
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4706836, upper bound: 7.4707007
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4706836, upper bound: 7.4707007
time: 1.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4183489, upper bound: 7.4183727
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4183628, upper bound: 7.4183644
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4868477, upper bound: 7.4868522
time: 2.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4868481, upper bound: 7.4868519
time: 2.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4869070, upper bound: 7.4869113
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4869071, upper bound: 7.4869113
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4949770, upper bound: 7.4949769
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4949782, upper bound: 7.4949737
time: 2.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4947089, upper bound: 7.4947119
time: 2.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4947089, upper bound: 7.4947119
time: 9.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2227933, upper bound: 7.2228221
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2228007, upper bound: 7.2228163
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2228007, upper bound: 7.2228221
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2228007, upper bound: 7.2228221
time: 1.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4665477, upper bound: 7.4665730
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4665477, upper bound: 7.4665730
time: 2.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4830078, upper bound: 7.4830337
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4830126, upper bound: 7.4830285
time: 2.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2840982, upper bound: 7.2840987
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2840982, upper bound: 7.2840987
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0195539, upper bound: 7.0195628
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0195539, upper bound: 7.0195628
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4930165, upper bound: 7.4930441
time: 3.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4930165, upper bound: 7.4930441
time: 3.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4943920, upper bound: 7.4943847
time: 2.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4943921, upper bound: 7.4943847
time: 2.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4908746, upper bound: 7.4908723
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4908743, upper bound: 7.4908723
time: 2.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4908800, upper bound: 7.4908668
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4908802, upper bound: 7.4908668
time: 2.83 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 6.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4908549, upper bound: 7.4908496
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4908549, upper bound: 7.4908496
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4934054, upper bound: 7.4934034
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4934037, upper bound: 7.4934039
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.3497535, upper bound: 7.3497126
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.3497535, upper bound: 7.3497126
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4842669, upper bound: 7.4842493
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4842579, upper bound: 7.4842559
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4920890, upper bound: 7.4920960
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4920891, upper bound: 7.4920955
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4921681, upper bound: 7.4921592
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4921681, upper bound: 7.4921592
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4902993, upper bound: 7.4902968
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4902993, upper bound: 7.4902968
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4396270, upper bound: 7.4395550
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4396270, upper bound: 7.4395550
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.3805092, upper bound: 7.3804854
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.3805092, upper bound: 7.3804854
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4805850, upper bound: 7.4805768
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4805921, upper bound: 7.4805653
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.3121514, upper bound: 7.3121340
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.3121514, upper bound: 7.3121340
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.1329630, upper bound: 7.1329482
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.1329630, upper bound: 7.1329482
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4807474, upper bound: 7.4807162
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4807474, upper bound: 7.4807162
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4823720, upper bound: 7.4823552
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4823720, upper bound: 7.4823552
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4807577, upper bound: 7.4807034
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4807577, upper bound: 7.4807034
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4817327, upper bound: 7.4817055
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4817337, upper bound: 7.4817053
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4706836, upper bound: 7.4707007
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4706836, upper bound: 7.4707007
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4183489, upper bound: 7.4183727
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4183628, upper bound: 7.4183644
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4868477, upper bound: 7.4868522
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4868481, upper bound: 7.4868519
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4869070, upper bound: 7.4869113
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4869071, upper bound: 7.4869113
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4949770, upper bound: 7.4949769
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4949782, upper bound: 7.4949737
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4947089, upper bound: 7.4947119
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4947089, upper bound: 7.4947119
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.2227933, upper bound: 7.2228221
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.2228007, upper bound: 7.2228163
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.2228007, upper bound: 7.2228221
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.2228007, upper bound: 7.2228221
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4665477, upper bound: 7.4665730
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4665477, upper bound: 7.4665730
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4830078, upper bound: 7.4830337
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4830126, upper bound: 7.4830285
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.2840982, upper bound: 7.2840987
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.2840982, upper bound: 7.2840987
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.0195539, upper bound: 7.0195628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.0195539, upper bound: 7.0195628
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4930165, upper bound: 7.4930441
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4930165, upper bound: 7.4930441
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4943920, upper bound: 7.4943847
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4943921, upper bound: 7.4943847
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4908746, upper bound: 7.4908723
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4908743, upper bound: 7.4908723
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4908800, upper bound: 7.4908668
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.61
Output dim: 8, lower bound: -7.4908802, upper bound: 7.4908668

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4904613, upper bound: 7.4904576
time: 21.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4904613, upper bound: 7.4904577
time: 2.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4901470, upper bound: 7.4901395
time: 9.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4901449, upper bound: 7.4901406
time: 2.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4820186, upper bound: 7.4819970
time: 2.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4820186, upper bound: 7.4819970
time: 2.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2589704, upper bound: 7.2589383
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2589704, upper bound: 7.2589383
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3497534, upper bound: 7.3497126
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3497535, upper bound: 7.3497100
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3336935, upper bound: 7.3336546
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3336965, upper bound: 7.3336509
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4530388, upper bound: 7.4530110
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4530324, upper bound: 7.4530111
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4400629, upper bound: 7.4400085
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4400629, upper bound: 7.4400085
time: 2.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4920310, upper bound: 7.4920374
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4920292, upper bound: 7.4920376
time: 2.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4128442, upper bound: 7.4128308
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4128442, upper bound: 7.4128308
time: 2.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4702348, upper bound: 7.4701539
time: 2.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4702348, upper bound: 7.4701539
time: 1.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4862887, upper bound: 7.4862713
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4862888, upper bound: 7.4862713
time: 1.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4871965, upper bound: 7.4871931
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4871952, upper bound: 7.4871937
time: 1.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3713421, upper bound: 7.3713201
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3713421, upper bound: 7.3713201
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4396270, upper bound: 7.4395525
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4396221, upper bound: 7.4395550
time: 2.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2962471, upper bound: 7.2961958
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.2962471, upper bound: 7.2961958
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1592645, upper bound: 7.1592701
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.1592645, upper bound: 7.1592701
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3800166, upper bound: 7.3800073
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3800166, upper bound: 7.3800073
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4436088, upper bound: 7.4435850
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4436078, upper bound: 7.4435864
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4805920, upper bound: 7.4805653
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4805921, upper bound: 7.4805653
time: 1.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3121501, upper bound: 7.3121340
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3121514, upper bound: 7.3121328
time: 1.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3077282, upper bound: 7.3077145
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3077278, upper bound: 7.3077148
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -6.9743906, upper bound: 6.9743821
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -6.9743908, upper bound: 6.9743835
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -6.9743906, upper bound: 6.9743821
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -6.9743908, upper bound: 6.9743835
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4807473, upper bound: 7.4807162
time: 2.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4807474, upper bound: 7.4807112
time: 2.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4066290, upper bound: 7.4066294
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4066292, upper bound: 7.4066192
time: 2.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4331217, upper bound: 7.4330759
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4331217, upper bound: 7.4330759
time: 2.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1125171, upper bound: 7.1124603
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.1125171, upper bound: 7.1124603
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4807522, upper bound: 7.4807034
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4807522, upper bound: 7.4807034
time: 2.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4703825, upper bound: 7.4703601
time: 2.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4703934, upper bound: 7.4703505
time: 2.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4817292, upper bound: 7.4817055
time: 2.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4817327, upper bound: 7.4816976
time: 2.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3775503, upper bound: 7.3775171
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3775533, upper bound: 7.3775171
time: 1.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4706836, upper bound: 7.4706989
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4706817, upper bound: 7.4707007
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4704725, upper bound: 7.4704976
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.4704804, upper bound: 7.4704906
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3266476, upper bound: 7.3266833
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3266476, upper bound: 7.3266833
time: 1.63 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4904613, upper bound: 7.4904576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4904613, upper bound: 7.4904577
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4901470, upper bound: 7.4901395
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4901449, upper bound: 7.4901406
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4820186, upper bound: 7.4819970
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4820186, upper bound: 7.4819970
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.2589704, upper bound: 7.2589383
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.2589704, upper bound: 7.2589383
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3497534, upper bound: 7.3497126
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3497535, upper bound: 7.3497100
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3336935, upper bound: 7.3336546
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3336965, upper bound: 7.3336509
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4530388, upper bound: 7.4530110
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4530324, upper bound: 7.4530111
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4400629, upper bound: 7.4400085
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4400629, upper bound: 7.4400085
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4920310, upper bound: 7.4920374
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4920292, upper bound: 7.4920376
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4128442, upper bound: 7.4128308
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4128442, upper bound: 7.4128308
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4702348, upper bound: 7.4701539
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4702348, upper bound: 7.4701539
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4862887, upper bound: 7.4862713
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4862888, upper bound: 7.4862713
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4871965, upper bound: 7.4871931
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4871952, upper bound: 7.4871937
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3713421, upper bound: 7.3713201
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3713421, upper bound: 7.3713201
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4396270, upper bound: 7.4395525
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4396221, upper bound: 7.4395550
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.2962471, upper bound: 7.2961958
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.2962471, upper bound: 7.2961958
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.1592645, upper bound: 7.1592701
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.1592645, upper bound: 7.1592701
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3800166, upper bound: 7.3800073
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3800166, upper bound: 7.3800073
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4436088, upper bound: 7.4435850
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4436078, upper bound: 7.4435864
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4805920, upper bound: 7.4805653
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4805921, upper bound: 7.4805653
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3121501, upper bound: 7.3121340
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3121514, upper bound: 7.3121328
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3077282, upper bound: 7.3077145
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3077278, upper bound: 7.3077148
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.65
Output dim: 8, lower bound: -6.9743906, upper bound: 6.9743821
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.65
Output dim: 8, lower bound: -6.9743908, upper bound: 6.9743835
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.65
Output dim: 8, lower bound: -6.9743906, upper bound: 6.9743821
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.65
Output dim: 8, lower bound: -6.9743908, upper bound: 6.9743835
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4807473, upper bound: 7.4807162
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4807474, upper bound: 7.4807112
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4066290, upper bound: 7.4066294
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4066292, upper bound: 7.4066192
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4331217, upper bound: 7.4330759
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4331217, upper bound: 7.4330759
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.1125171, upper bound: 7.1124603
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.1125171, upper bound: 7.1124603
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4807522, upper bound: 7.4807034
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4807522, upper bound: 7.4807034
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4703825, upper bound: 7.4703601
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4703934, upper bound: 7.4703505
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4817292, upper bound: 7.4817055
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4817327, upper bound: 7.4816976
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3775503, upper bound: 7.3775171
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3775533, upper bound: 7.3775171
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4706836, upper bound: 7.4706989
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4706817, upper bound: 7.4707007
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4704725, upper bound: 7.4704976
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.4704804, upper bound: 7.4704906
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3266476, upper bound: 7.3266833
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.65
Output dim: 8, lower bound: -7.3266476, upper bound: 7.3266833
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4183628, upper bound: 7.4183644
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4868477, upper bound: 7.4868522
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4868481, upper bound: 7.4868519
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4869070, upper bound: 7.4869113
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4869071, upper bound: 7.4869113
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4949770, upper bound: 7.4949769
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4949782, upper bound: 7.4949737
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4947089, upper bound: 7.4947119
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4947089, upper bound: 7.4947119
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.2227933, upper bound: 7.2228221
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.2228007, upper bound: 7.2228163
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.2228007, upper bound: 7.2228221
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.2228007, upper bound: 7.2228221
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4665477, upper bound: 7.4665730
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4665477, upper bound: 7.4665730
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4830078, upper bound: 7.4830337
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4830126, upper bound: 7.4830285
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.2840982, upper bound: 7.2840987
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.2840982, upper bound: 7.2840987
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4930165, upper bound: 7.4930441
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4930165, upper bound: 7.4930441
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4943920, upper bound: 7.4943847
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4943921, upper bound: 7.4943847
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4908746, upper bound: 7.4908723
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4908743, upper bound: 7.4908723
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4908800, upper bound: 7.4908668
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 8, lower bound: -7.4908802, upper bound: 7.4908668

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 5.44 + 597.81 = 603.24 seconds
