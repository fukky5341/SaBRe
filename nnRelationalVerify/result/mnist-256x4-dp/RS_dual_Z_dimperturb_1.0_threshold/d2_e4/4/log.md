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
execution time: IAR + RelationalAnalysis = 1.33 + 4.08 = 5.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -7.5008703, upper bound: 7.5008703

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.5003434, upper bound: 7.5003439
time: 2.46 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.5003439, upper bound: 7.5003434
time: 2.12 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.71 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 4.71
Output dim: 8, lower bound: -7.5003434, upper bound: 7.5003439
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 4.71
Output dim: 8, lower bound: -7.5003439, upper bound: 7.5003434

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3934550, upper bound: 7.3934552
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3934550, upper bound: 7.3934552
time: 1.44 seconds

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3934552, upper bound: 7.3934550
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -7.3934552, upper bound: 7.3934550
time: 1.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.47 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 8, lower bound: -7.3934550, upper bound: 7.3934552
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 8, lower bound: -7.3934550, upper bound: 7.3934552
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 8, lower bound: -7.3934552, upper bound: 7.3934550
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 8, lower bound: -7.3934552, upper bound: 7.3934550

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.50 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.49 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.54 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
time: 1.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.54 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 4.54
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 4.54
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 4.54
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 4.54
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 4.54
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 4.54
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 4.54
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 4.54
Output dim: 8, lower bound: -7.0606195, upper bound: 7.0606195

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 5.40 + 31.03 = 36.43 seconds
