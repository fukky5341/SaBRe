## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.001357455


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

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
execution time: IAR + RelationalAnalysis = 1.19 + 2.59 = 3.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0014289, upper bound: 0.0014289

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014208, upper bound: 0.0014129
time: 1.95 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014216, upper bound: 0.0014215
time: 1.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.60 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.60
Output dim: 2, lower bound: -0.0014208, upper bound: 0.0014129
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.60
Output dim: 2, lower bound: -0.0014216, upper bound: 0.0014215

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014129
time: 2.05 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014129
time: 1.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014208
time: 1.66 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014215
time: 1.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.77 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.77
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014129
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.77
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014129
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.77
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014208
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.77
Output dim: 2, lower bound: -0.0014129, upper bound: 0.0014215

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013980, upper bound: 0.0014087
time: 1.67 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0014088
time: 1.71 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013980, upper bound: 0.0014088
time: 1.73 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0014088
time: 1.73 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014088, upper bound: 0.0014061
time: 1.85 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014088, upper bound: 0.0014164
time: 2.01 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013969, upper bound: 0.0014171
time: 1.66 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014088, upper bound: 0.0014172
time: 1.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.66 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 2, lower bound: -0.0013980, upper bound: 0.0014087
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0014088
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 2, lower bound: -0.0013980, upper bound: 0.0014088
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0014088
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 2, lower bound: -0.0014088, upper bound: 0.0014061
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 2, lower bound: -0.0014088, upper bound: 0.0014164
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 2, lower bound: -0.0013969, upper bound: 0.0014171
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 2, lower bound: -0.0014088, upper bound: 0.0014172

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011019, upper bound: 0.0013960
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013951, upper bound: 0.0014066
time: 1.87 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0013980
time: 1.53 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0014095
time: 1.74 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013872, upper bound: 0.0011932
time: 1.53 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014030, upper bound: 0.0014059
time: 1.69 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014068, upper bound: 0.0012114
time: 2.08 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014134, upper bound: 0.0014059
time: 1.62 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0042083, -0.0040502, -0.0042069, -0.0040674, -0.0001409, 0.0001567
1: -0.0102553, -0.0085860, -0.0102044, -0.0086723, -0.0015830, 0.0016184
2: 0.9641567, 0.9661599, 0.9642177, 0.9660563, -0.0018997, 0.0019422
3: -0.0180685, -0.0032928, -0.0176182, -0.0040566, -0.0113516, 0.0117984
4: -0.0004426, 0.0006812, -0.0003845, 0.0006469, -0.0010895, 0.0010657
5: 0.0168230, 0.0185227, 0.0168817, 0.0184221, -0.0015990, 0.0016410
6: 0.0016696, 0.0037241, 0.0018624, 0.0036955, -0.0020259, 0.0018617
7: -0.0069249, -0.0022037, -0.0067270, -0.0024249, -0.0045000, 0.0045232
8: 0.0112353, 0.0142732, 0.0113923, 0.0141806, -0.0029454, 0.0028809
9: 0.0179323, 0.0233963, 0.0182148, 0.0232298, -0.0050044, 0.0048739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011932, upper bound: 0.0013872
time: 1.56 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0014031
time: 1.63 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0042086, -0.0040464, -0.0042082, -0.0040511, -0.0001574, 0.0001618
1: -0.0102666, -0.0085844, -0.0102526, -0.0086629, -0.0016037, 0.0016682
2: 0.9641432, 0.9661617, 0.9641600, 0.9660675, -0.0019243, 0.0020017
3: -0.0181685, -0.0032789, -0.0180443, -0.0039738, -0.0115733, 0.0120903
4: -0.0004436, 0.0006888, -0.0003908, 0.0006793, -0.0011230, 0.0010796
5: 0.0168220, 0.0185451, 0.0168754, 0.0185173, -0.0016953, 0.0016697
6: 0.0016268, 0.0037246, 0.0016800, 0.0036986, -0.0020718, 0.0020446
7: -0.0069285, -0.0021546, -0.0067484, -0.0022157, -0.0047128, 0.0045938
8: 0.0112324, 0.0142938, 0.0113753, 0.0142682, -0.0030358, 0.0029185
9: 0.0179272, 0.0234333, 0.0181841, 0.0233874, -0.0051529, 0.0049439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012114, upper bound: 0.0014068
time: 1.73 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014059, upper bound: 0.0014134
time: 1.79 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011146, upper bound: 0.0014061
time: 1.33 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013952, upper bound: 0.0014141
time: 1.68 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014104, upper bound: 0.0014066
time: 1.88 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014104, upper bound: 0.0014171
time: 2.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.10 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0011019, upper bound: 0.0013960
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0013951, upper bound: 0.0014066
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0013980
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0014095, upper bound: 0.0014095
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0013872, upper bound: 0.0011932
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0014030, upper bound: 0.0014059
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0014068, upper bound: 0.0012114
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0014134, upper bound: 0.0014059
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0011932, upper bound: 0.0013872
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0014031
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0012114, upper bound: 0.0014068
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0014059, upper bound: 0.0014134
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0011146, upper bound: 0.0014061
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0013952, upper bound: 0.0014141
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0014104, upper bound: 0.0014066
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 2, lower bound: -0.0014104, upper bound: 0.0014171

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0042016, -0.0041339, -0.0042077, -0.0040573, -0.0001443, 0.0000738
1: -0.0100072, -0.0085821, -0.0102343, -0.0086608, -0.0013464, 0.0016522
2: 0.9644544, 0.9661646, 0.9641820, 0.9660702, -0.0016158, 0.0019826
3: -0.0158724, -0.0032582, -0.0178822, -0.0039549, -0.0091589, 0.0118022
4: -0.0004452, 0.0005142, -0.0003922, 0.0006670, -0.0011122, 0.0009064
5: 0.0168204, 0.0180318, 0.0168739, 0.0184811, -0.0016607, 0.0011578
6: 0.0026098, 0.0037254, 0.0017494, 0.0036993, -0.0010895, 0.0019760
7: -0.0069339, -0.0032824, -0.0067533, -0.0022952, -0.0046387, 0.0034709
8: 0.0112281, 0.0138217, 0.0113714, 0.0142349, -0.0030068, 0.0024503
9: 0.0179195, 0.0225842, 0.0181772, 0.0233275, -0.0050892, 0.0040907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011019, upper bound: 0.0013744
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011019, upper bound: 0.0013960
time: 1.48 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0042067, -0.0040702, -0.0042083, -0.0040496, -0.0001571, 0.0001381
1: -0.0101961, -0.0086726, -0.0102570, -0.0086599, -0.0015361, 0.0015844
2: 0.9642278, 0.9660560, 0.9641545, 0.9660711, -0.0018433, 0.0019014
3: -0.0175440, -0.0040594, -0.0180837, -0.0039474, -0.0104341, 0.0113372
4: -0.0003843, 0.0006413, -0.0003928, 0.0006823, -0.0010666, 0.0010341
5: 0.0168820, 0.0184055, 0.0168733, 0.0185261, -0.0016442, 0.0015321
6: 0.0018942, 0.0036954, 0.0016631, 0.0036996, -0.0018054, 0.0020323
7: -0.0067262, -0.0024613, -0.0067552, -0.0021963, -0.0045299, 0.0042939
8: 0.0113929, 0.0141654, 0.0113698, 0.0142763, -0.0028835, 0.0027955
9: 0.0182158, 0.0232024, 0.0181744, 0.0234020, -0.0048745, 0.0046730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013950, upper bound: 0.0014000
time: 1.77 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013950, upper bound: 0.0014066
time: 1.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013960, upper bound: 0.0011019
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014066, upper bound: 0.0013951
time: 1.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011900, upper bound: 0.0013851
time: 1.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014066, upper bound: 0.0013958
time: 1.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013523, upper bound: 0.0011524
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013359, upper bound: 0.0011112
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014030, upper bound: 0.0013991
time: 1.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014030, upper bound: 0.0014058
time: 1.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0042076, -0.0040589, -0.0042033, -0.0041132, -0.0000944, 0.0001444
1: -0.0102297, -0.0086635, -0.0100685, -0.0085018, -0.0017279, 0.0014050
2: 0.9641874, 0.9660669, 0.9643808, 0.9662609, -0.0020735, 0.0016861
3: -0.0178420, -0.0039790, -0.0164153, -0.0025481, -0.0124010, 0.0097390
4: -0.0003904, 0.0006640, -0.0004992, 0.0005554, -0.0009458, 0.0011632
5: 0.0168758, 0.0184721, 0.0167658, 0.0181531, -0.0012774, 0.0017063
6: 0.0017666, 0.0036984, 0.0023774, 0.0037519, -0.0019853, 0.0013210
7: -0.0067471, -0.0023150, -0.0071179, -0.0030157, -0.0037313, 0.0048029
8: 0.0113764, 0.0142266, 0.0110822, 0.0139333, -0.0025569, 0.0031445
9: 0.0181861, 0.0233126, 0.0176569, 0.0227850, -0.0042862, 0.0053264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013986, upper bound: 0.0011883
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013968, upper bound: 0.0011374
time: 1.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042082, -0.0040511, -0.0042084, -0.0040491, -0.0001591, 0.0001572
1: -0.0102526, -0.0086629, -0.0102585, -0.0085852, -0.0016673, 0.0015956
2: 0.9641600, 0.9660675, 0.9641528, 0.9661608, -0.0020008, 0.0019147
3: -0.0180443, -0.0039738, -0.0180968, -0.0032862, -0.0120829, 0.0110148
4: -0.0003908, 0.0006793, -0.0004431, 0.0006833, -0.0010741, 0.0011224
5: 0.0168754, 0.0185173, 0.0168225, 0.0185291, -0.0016537, 0.0016948
6: 0.0016800, 0.0036986, 0.0016575, 0.0037243, -0.0020443, 0.0020411
7: -0.0067484, -0.0022157, -0.0069266, -0.0021898, -0.0045586, 0.0047110
8: 0.0113753, 0.0142682, 0.0112339, 0.0142790, -0.0029037, 0.0030343
9: 0.0181841, 0.0233874, 0.0179298, 0.0234068, -0.0048717, 0.0051503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014133, upper bound: 0.0013939
time: 1.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014133, upper bound: 0.0013946
time: 1.98 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0042030, -0.0041169, -0.0042063, -0.0040750, -0.0001280, 0.0000894
1: -0.0100577, -0.0085031, -0.0101818, -0.0086728, -0.0013849, 0.0016787
2: 0.9643938, 0.9662594, 0.9642448, 0.9660557, -0.0016619, 0.0020146
3: -0.0163193, -0.0025592, -0.0174182, -0.0040610, -0.0095229, 0.0120950
4: -0.0004984, 0.0005481, -0.0003842, 0.0006317, -0.0011301, 0.0009323
5: 0.0167666, 0.0181317, 0.0168821, 0.0183773, -0.0016107, 0.0012496
6: 0.0024185, 0.0037515, 0.0019480, 0.0036954, -0.0012769, 0.0018035
7: -0.0071150, -0.0030629, -0.0067258, -0.0025232, -0.0045918, 0.0036629
8: 0.0110844, 0.0139136, 0.0113932, 0.0141395, -0.0030551, 0.0025203
9: 0.0176611, 0.0227495, 0.0182164, 0.0231559, -0.0051782, 0.0042182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011524, upper bound: 0.0013523
time: 1.30 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011112, upper bound: 0.0013359
time: 1.46 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0042080, -0.0040529, -0.0042069, -0.0040674, -0.0001407, 0.0001540
1: -0.0102471, -0.0085868, -0.0102044, -0.0086723, -0.0015748, 0.0016177
2: 0.9641666, 0.9661590, 0.9642177, 0.9660563, -0.0018898, 0.0019413
3: -0.0179960, -0.0032996, -0.0176182, -0.0040566, -0.0108036, 0.0117913
4: -0.0004421, 0.0006757, -0.0003845, 0.0006469, -0.0010890, 0.0010602
5: 0.0168235, 0.0185065, 0.0168817, 0.0184221, -0.0015985, 0.0016248
6: 0.0017007, 0.0037238, 0.0018624, 0.0036955, -0.0019948, 0.0018614
7: -0.0069231, -0.0022393, -0.0067270, -0.0024249, -0.0044982, 0.0044876
8: 0.0112367, 0.0142583, 0.0113923, 0.0141806, -0.0029439, 0.0028660
9: 0.0179348, 0.0233695, 0.0182148, 0.0232298, -0.0050018, 0.0048021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013991, upper bound: 0.0014030
time: 1.99 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013991, upper bound: 0.0014030
time: 2.06 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0042033, -0.0041132, -0.0042076, -0.0040589, -0.0001444, 0.0000944
1: -0.0100685, -0.0085018, -0.0102297, -0.0086635, -0.0014050, 0.0017279
2: 0.9643808, 0.9662609, 0.9641874, 0.9660669, -0.0016861, 0.0020735
3: -0.0164153, -0.0025481, -0.0178420, -0.0039790, -0.0097390, 0.0124010
4: -0.0004992, 0.0005554, -0.0003904, 0.0006640, -0.0011632, 0.0009458
5: 0.0167658, 0.0181531, 0.0168758, 0.0184721, -0.0017063, 0.0012774
6: 0.0023774, 0.0037519, 0.0017666, 0.0036984, -0.0013210, 0.0019853
7: -0.0071179, -0.0030157, -0.0067471, -0.0023150, -0.0048029, 0.0037313
8: 0.0110822, 0.0139333, 0.0113764, 0.0142266, -0.0031445, 0.0025569
9: 0.0176569, 0.0227850, 0.0181861, 0.0233126, -0.0053264, 0.0042862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011883, upper bound: 0.0013986
time: 1.31 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011374, upper bound: 0.0013968
time: 1.97 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0042084, -0.0040491, -0.0042082, -0.0040511, -0.0001572, 0.0001591
1: -0.0102585, -0.0085852, -0.0102526, -0.0086629, -0.0015956, 0.0016673
2: 0.9641528, 0.9661608, 0.9641600, 0.9660675, -0.0019147, 0.0020008
3: -0.0180968, -0.0032862, -0.0180443, -0.0039738, -0.0110148, 0.0120829
4: -0.0004431, 0.0006833, -0.0003908, 0.0006793, -0.0011224, 0.0010741
5: 0.0168225, 0.0185291, 0.0168754, 0.0185173, -0.0016948, 0.0016537
6: 0.0016575, 0.0037243, 0.0016800, 0.0036986, -0.0020411, 0.0020443
7: -0.0069266, -0.0021898, -0.0067484, -0.0022157, -0.0047110, 0.0045586
8: 0.0112339, 0.0142790, 0.0113753, 0.0142682, -0.0030343, 0.0029037
9: 0.0179298, 0.0234068, 0.0181841, 0.0233874, -0.0051503, 0.0048717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013939, upper bound: 0.0014134
time: 1.68 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013939, upper bound: 0.0014037
time: 1.78 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0042016, -0.0041338, -0.0042077, -0.0040579, -0.0001438, 0.0000738
1: -0.0100073, -0.0085092, -0.0102325, -0.0085872, -0.0014202, 0.0017234
2: 0.9644542, 0.9662521, 0.9641840, 0.9661584, -0.0017042, 0.0020680
3: -0.0158735, -0.0026129, -0.0178671, -0.0033031, -0.0097382, 0.0122364
4: -0.0004943, 0.0005142, -0.0004418, 0.0006659, -0.0011602, 0.0009560
5: 0.0167708, 0.0180320, 0.0168238, 0.0184777, -0.0017069, 0.0012082
6: 0.0026093, 0.0037495, 0.0017559, 0.0037237, -0.0011144, 0.0019936
7: -0.0071011, -0.0032819, -0.0069222, -0.0023027, -0.0047984, 0.0036404
8: 0.0110955, 0.0138219, 0.0112374, 0.0142318, -0.0031363, 0.0025845
9: 0.0176809, 0.0225846, 0.0179361, 0.0233219, -0.0053005, 0.0043238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011146, upper bound: 0.0013883
time: 1.65 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011146, upper bound: 0.0014061
time: 1.60 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0042066, -0.0040707, -0.0042083, -0.0040502, -0.0001564, 0.0001376
1: -0.0101946, -0.0085997, -0.0102553, -0.0085860, -0.0016086, 0.0016556
2: 0.9642295, 0.9661434, 0.9641567, 0.9661599, -0.0019304, 0.0019867
3: -0.0175310, -0.0034145, -0.0180685, -0.0032928, -0.0110297, 0.0119042
4: -0.0004333, 0.0006403, -0.0004426, 0.0006812, -0.0011145, 0.0010829
5: 0.0168324, 0.0184026, 0.0168230, 0.0185227, -0.0016904, 0.0015796
6: 0.0018997, 0.0037195, 0.0016696, 0.0037241, -0.0018244, 0.0020499
7: -0.0068934, -0.0024677, -0.0069249, -0.0022037, -0.0046896, 0.0044572
8: 0.0112603, 0.0141627, 0.0112353, 0.0142732, -0.0030129, 0.0029274
9: 0.0179773, 0.0231976, 0.0179323, 0.0233963, -0.0051039, 0.0049028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013953, upper bound: 0.0014078
time: 1.93 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013953, upper bound: 0.0014141
time: 1.85 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013997, upper bound: 0.0011229
time: 1.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014075, upper bound: 0.0014035
time: 1.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012030, upper bound: 0.0013959
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014074, upper bound: 0.0014043
time: 1.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.32 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0011019, upper bound: 0.0013744
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0011019, upper bound: 0.0013960
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0013950, upper bound: 0.0014000
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0013950, upper bound: 0.0014066
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0013960, upper bound: 0.0011019
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0014066, upper bound: 0.0013951
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0011900, upper bound: 0.0013851
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0014066, upper bound: 0.0013958
IS_A1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0013523, upper bound: 0.0011524
IS_A1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0013359, upper bound: 0.0011112
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0014030, upper bound: 0.0013991
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0014030, upper bound: 0.0014058
IS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0013986, upper bound: 0.0011883
IS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0013968, upper bound: 0.0011374
IS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0014133, upper bound: 0.0013939
IS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0014133, upper bound: 0.0013946
IS_A2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0011524, upper bound: 0.0013523
IS_A2_B1_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0011112, upper bound: 0.0013359
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0013991, upper bound: 0.0014030
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0013991, upper bound: 0.0014030
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0011883, upper bound: 0.0013986
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0011374, upper bound: 0.0013968
IS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0013939, upper bound: 0.0014134
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0013939, upper bound: 0.0014037
IS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0011146, upper bound: 0.0013883
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0011146, upper bound: 0.0014061
IS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0013953, upper bound: 0.0014078
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0013953, upper bound: 0.0014141
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0013997, upper bound: 0.0011229
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0014075, upper bound: 0.0014035
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0012030, upper bound: 0.0013959
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 2, lower bound: -0.0014074, upper bound: 0.0014043

## BFS IS instance: IS_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0042016, -0.0041339, -0.0042063, -0.0040750, -0.0001267, 0.0000724
1: -0.0100072, -0.0085821, -0.0101818, -0.0086728, -0.0013344, 0.0015998
2: 0.9644544, 0.9661646, 0.9642448, 0.9660557, -0.0016013, 0.0019198
3: -0.0158724, -0.0032582, -0.0174182, -0.0040610, -0.0090159, 0.0113293
4: -0.0004452, 0.0005142, -0.0003842, 0.0006317, -0.0010770, 0.0008983
5: 0.0168204, 0.0180318, 0.0168821, 0.0183773, -0.0015570, 0.0011497
6: 0.0026098, 0.0037254, 0.0019480, 0.0036954, -0.0010856, 0.0017773
7: -0.0069339, -0.0032824, -0.0067258, -0.0025232, -0.0044107, 0.0034434
8: 0.0112281, 0.0138217, 0.0113932, 0.0141395, -0.0029114, 0.0024285
9: 0.0179195, 0.0225842, 0.0182164, 0.0231559, -0.0049170, 0.0040495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010762, upper bound: 0.0013259
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0009808, upper bound: 0.0012883
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0042016, -0.0041339, -0.0042076, -0.0040589, -0.0001428, 0.0000737
1: -0.0100072, -0.0085821, -0.0102297, -0.0086635, -0.0013437, 0.0016476
2: 0.9644544, 0.9661646, 0.9641874, 0.9660669, -0.0016125, 0.0019772
3: -0.0158724, -0.0032582, -0.0178420, -0.0039790, -0.0091345, 0.0117785
4: -0.0004452, 0.0005142, -0.0003904, 0.0006640, -0.0011092, 0.0009046
5: 0.0168204, 0.0180318, 0.0168758, 0.0184721, -0.0016517, 0.0011560
6: 0.0026098, 0.0037254, 0.0017666, 0.0036984, -0.0010886, 0.0019588
7: -0.0069339, -0.0032824, -0.0067471, -0.0023150, -0.0046189, 0.0034647
8: 0.0112281, 0.0138217, 0.0113764, 0.0142266, -0.0029985, 0.0024453
9: 0.0179195, 0.0225842, 0.0181861, 0.0233126, -0.0050758, 0.0040818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010762, upper bound: 0.0013833
time: 1.30 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009808, upper bound: 0.0013740
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0042067, -0.0040702, -0.0042069, -0.0040674, -0.0001393, 0.0001367
1: -0.0101961, -0.0086726, -0.0102044, -0.0086723, -0.0015238, 0.0015318
2: 0.9642278, 0.9660560, 0.9642177, 0.9660563, -0.0018286, 0.0018383
3: -0.0175440, -0.0040594, -0.0176182, -0.0040566, -0.0102991, 0.0108639
4: -0.0003843, 0.0006413, -0.0003845, 0.0006469, -0.0010312, 0.0010258
5: 0.0168820, 0.0184055, 0.0168817, 0.0184221, -0.0015401, 0.0015237
6: 0.0018942, 0.0036954, 0.0018624, 0.0036955, -0.0018014, 0.0018330
7: -0.0067262, -0.0024613, -0.0067270, -0.0024249, -0.0043013, 0.0042656
8: 0.0113929, 0.0141654, 0.0113923, 0.0141806, -0.0027877, 0.0027731
9: 0.0182158, 0.0232024, 0.0182148, 0.0232298, -0.0047015, 0.0046310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013739, upper bound: 0.0011020
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013739, upper bound: 0.0011020
time: 1.36 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042067, -0.0040702, -0.0042082, -0.0040511, -0.0001555, 0.0001380
1: -0.0101961, -0.0086726, -0.0102526, -0.0086629, -0.0015331, 0.0015800
2: 0.9642278, 0.9660560, 0.9641600, 0.9660675, -0.0018397, 0.0018960
3: -0.0175440, -0.0040594, -0.0180443, -0.0039738, -0.0104081, 0.0113124
4: -0.0003843, 0.0006413, -0.0003908, 0.0006793, -0.0010636, 0.0010321
5: 0.0168820, 0.0184055, 0.0168754, 0.0185173, -0.0016354, 0.0015301
6: 0.0018942, 0.0036954, 0.0016800, 0.0036986, -0.0018044, 0.0020154
7: -0.0067262, -0.0024613, -0.0067484, -0.0022157, -0.0045106, 0.0042871
8: 0.0113929, 0.0141654, 0.0113753, 0.0142682, -0.0028753, 0.0027901
9: 0.0182158, 0.0232024, 0.0181841, 0.0233874, -0.0048616, 0.0046632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013902, upper bound: 0.0014021
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013901, upper bound: 0.0014019
time: 2.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0042076, -0.0040589, -0.0042016, -0.0041339, -0.0000737, 0.0001428
1: -0.0102297, -0.0086635, -0.0100072, -0.0085821, -0.0016476, 0.0013437
2: 0.9641874, 0.9660669, 0.9644544, 0.9661646, -0.0019772, 0.0016125
3: -0.0178420, -0.0039790, -0.0158724, -0.0032582, -0.0117785, 0.0091345
4: -0.0003904, 0.0006640, -0.0004452, 0.0005142, -0.0009046, 0.0011092
5: 0.0168758, 0.0184721, 0.0168204, 0.0180318, -0.0011560, 0.0016517
6: 0.0017666, 0.0036984, 0.0026098, 0.0037254, -0.0019588, 0.0010886
7: -0.0067471, -0.0023150, -0.0069339, -0.0032824, -0.0034647, 0.0046189
8: 0.0113764, 0.0142266, 0.0112281, 0.0138217, -0.0024453, 0.0029985
9: 0.0181861, 0.0233126, 0.0179195, 0.0225842, -0.0040818, 0.0050758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013833, upper bound: 0.0010765
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009808
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0042082, -0.0040511, -0.0042067, -0.0040702, -0.0001380, 0.0001555
1: -0.0102526, -0.0086629, -0.0101961, -0.0086726, -0.0015800, 0.0015331
2: 0.9641600, 0.9660675, 0.9642278, 0.9660560, -0.0018960, 0.0018397
3: -0.0180443, -0.0039738, -0.0175440, -0.0040594, -0.0113124, 0.0104081
4: -0.0003908, 0.0006793, -0.0003843, 0.0006413, -0.0010321, 0.0010636
5: 0.0168754, 0.0185173, 0.0168820, 0.0184055, -0.0015301, 0.0016354
6: 0.0016800, 0.0036986, 0.0018942, 0.0036954, -0.0020154, 0.0018044
7: -0.0067484, -0.0022157, -0.0067262, -0.0024613, -0.0042871, 0.0045106
8: 0.0113753, 0.0142682, 0.0113929, 0.0141654, -0.0027901, 0.0028753
9: 0.0181841, 0.0233874, 0.0182158, 0.0232024, -0.0046632, 0.0048616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014021, upper bound: 0.0013902
time: 1.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014019, upper bound: 0.0013901
time: 1.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011836, upper bound: 0.0013746
time: 1.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011221, upper bound: 0.0013714
time: 1.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013972, upper bound: 0.0011975
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013972, upper bound: 0.0013957
time: 1.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0042069, -0.0040674, -0.0042066, -0.0040707, -0.0001362, 0.0001393
1: -0.0102044, -0.0086723, -0.0101946, -0.0085997, -0.0016047, 0.0015223
2: 0.9642177, 0.9660563, 0.9642295, 0.9661434, -0.0019257, 0.0018268
3: -0.0176182, -0.0040566, -0.0175310, -0.0034145, -0.0116595, 0.0103400
4: -0.0003845, 0.0006469, -0.0004333, 0.0006403, -0.0010248, 0.0010803
5: 0.0168817, 0.0184221, 0.0168324, 0.0184026, -0.0015208, 0.0015897
6: 0.0018624, 0.0036955, 0.0018997, 0.0037195, -0.0018571, 0.0017958
7: -0.0067270, -0.0024249, -0.0068934, -0.0024677, -0.0042592, 0.0044684
8: 0.0113923, 0.0141806, 0.0112603, 0.0141627, -0.0027704, 0.0029203
9: 0.0182148, 0.0232298, 0.0179773, 0.0231976, -0.0046299, 0.0049603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013981, upper bound: 0.0013938
time: 1.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013982, upper bound: 0.0013936
time: 1.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0042069, -0.0040674, -0.0042079, -0.0040544, -0.0001525, 0.0001406
1: -0.0102044, -0.0086723, -0.0102428, -0.0085899, -0.0016145, 0.0015705
2: 0.9642177, 0.9660563, 0.9641717, 0.9661552, -0.0019375, 0.0018846
3: -0.0176182, -0.0040566, -0.0179578, -0.0033274, -0.0117632, 0.0107730
4: -0.0003845, 0.0006469, -0.0004400, 0.0006728, -0.0010573, 0.0010869
5: 0.0168817, 0.0184221, 0.0168257, 0.0184980, -0.0016162, 0.0015964
6: 0.0018624, 0.0036955, 0.0017170, 0.0037228, -0.0018604, 0.0019785
7: -0.0067270, -0.0024249, -0.0069159, -0.0022581, -0.0044688, 0.0044910
8: 0.0113923, 0.0141806, 0.0112424, 0.0142504, -0.0028581, 0.0029382
9: 0.0182148, 0.0232298, 0.0179451, 0.0233554, -0.0047877, 0.0049916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013981, upper bound: 0.0014012
time: 1.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013982, upper bound: 0.0014012
time: 1.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0042076, -0.0040589, -0.0042033, -0.0041135, -0.0000940, 0.0001444
1: -0.0102297, -0.0086635, -0.0100675, -0.0085209, -0.0017088, 0.0014040
2: 0.9641874, 0.9660669, 0.9643821, 0.9662380, -0.0020506, 0.0016848
3: -0.0178420, -0.0039790, -0.0164061, -0.0027171, -0.0122348, 0.0097298
4: -0.0003904, 0.0006640, -0.0004864, 0.0005548, -0.0009451, 0.0011503
5: 0.0168758, 0.0184721, 0.0167788, 0.0181511, -0.0012753, 0.0016933
6: 0.0017666, 0.0036984, 0.0023813, 0.0037456, -0.0019790, 0.0013171
7: -0.0067471, -0.0023150, -0.0070741, -0.0030202, -0.0037268, 0.0047591
8: 0.0113764, 0.0142266, 0.0111169, 0.0139314, -0.0025551, 0.0031097
9: 0.0181861, 0.0233126, 0.0177194, 0.0227816, -0.0042828, 0.0052642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013948, upper bound: 0.0010807
time: 1.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013948, upper bound: 0.0011828
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0042076, -0.0040590, -0.0042036, -0.0041097, -0.0000979, 0.0001446
1: -0.0102294, -0.0086707, -0.0100788, -0.0085334, -0.0016960, 0.0014081
2: 0.9641877, 0.9660582, 0.9643685, 0.9662230, -0.0020353, 0.0016897
3: -0.0178392, -0.0040427, -0.0165065, -0.0028273, -0.0122142, 0.0097696
4: -0.0003856, 0.0006637, -0.0004780, 0.0005624, -0.0009479, 0.0011417
5: 0.0168807, 0.0184715, 0.0167872, 0.0181735, -0.0012929, 0.0016842
6: 0.0017678, 0.0036960, 0.0023383, 0.0037415, -0.0019737, 0.0013577
7: -0.0067305, -0.0023164, -0.0070455, -0.0029709, -0.0037596, 0.0047292
8: 0.0113895, 0.0142261, 0.0111396, 0.0139521, -0.0025626, 0.0030865
9: 0.0182096, 0.0233116, 0.0177602, 0.0228187, -0.0042962, 0.0052320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013915, upper bound: 0.0010041
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013915, upper bound: 0.0011311
time: 1.45 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0042082, -0.0040511, -0.0042066, -0.0040707, -0.0001375, 0.0001555
1: -0.0102526, -0.0086629, -0.0101946, -0.0085997, -0.0016528, 0.0015317
2: 0.9641600, 0.9660675, 0.9642295, 0.9661434, -0.0019835, 0.0018380
3: -0.0180443, -0.0039738, -0.0175310, -0.0034145, -0.0121047, 0.0104490
4: -0.0003908, 0.0006793, -0.0004333, 0.0006403, -0.0010311, 0.0011127
5: 0.0168754, 0.0185173, 0.0168324, 0.0184026, -0.0015272, 0.0016849
6: 0.0016800, 0.0036986, 0.0018997, 0.0037195, -0.0020395, 0.0017989
7: -0.0067484, -0.0022157, -0.0068934, -0.0024677, -0.0042807, 0.0046777
8: 0.0113753, 0.0142682, 0.0112603, 0.0141627, -0.0027874, 0.0030079
9: 0.0181841, 0.0233874, 0.0179773, 0.0231976, -0.0046621, 0.0051206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014089, upper bound: 0.0013891
time: 1.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014088, upper bound: 0.0013891
time: 2.32 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0042082, -0.0040511, -0.0042079, -0.0040544, -0.0001538, 0.0001568
1: -0.0102526, -0.0086629, -0.0102428, -0.0085899, -0.0016627, 0.0015799
2: 0.9641600, 0.9660675, 0.9641717, 0.9661552, -0.0019953, 0.0018958
3: -0.0180443, -0.0039738, -0.0179578, -0.0033274, -0.0120409, 0.0107135
4: -0.0003908, 0.0006793, -0.0004400, 0.0006728, -0.0010636, 0.0011193
5: 0.0168754, 0.0185173, 0.0168257, 0.0184980, -0.0016226, 0.0016916
6: 0.0016800, 0.0036986, 0.0017170, 0.0037228, -0.0020428, 0.0019816
7: -0.0067484, -0.0022157, -0.0069159, -0.0022581, -0.0044903, 0.0047003
8: 0.0113753, 0.0142682, 0.0112424, 0.0142504, -0.0028751, 0.0030258
9: 0.0181841, 0.0233874, 0.0179451, 0.0233554, -0.0048034, 0.0051351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014089, upper bound: 0.0013903
time: 2.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014087, upper bound: 0.0013902
time: 2.10 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0042066, -0.0040707, -0.0042069, -0.0040674, -0.0001393, 0.0001362
1: -0.0101946, -0.0085997, -0.0102044, -0.0086723, -0.0015223, 0.0016047
2: 0.9642295, 0.9661434, 0.9642177, 0.9660563, -0.0018268, 0.0019257
3: -0.0175310, -0.0034145, -0.0176182, -0.0040566, -0.0103400, 0.0116595
4: -0.0004333, 0.0006403, -0.0003845, 0.0006469, -0.0010803, 0.0010248
5: 0.0168324, 0.0184026, 0.0168817, 0.0184221, -0.0015897, 0.0015208
6: 0.0018997, 0.0037195, 0.0018624, 0.0036955, -0.0017958, 0.0018571
7: -0.0068934, -0.0024677, -0.0067270, -0.0024249, -0.0044684, 0.0042592
8: 0.0112603, 0.0141627, 0.0113923, 0.0141806, -0.0029203, 0.0027704
9: 0.0179773, 0.0231976, 0.0182148, 0.0232298, -0.0049603, 0.0046299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013937, upper bound: 0.0013981
time: 1.69 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013936, upper bound: 0.0013982
time: 1.78 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013937, upper bound: 0.0013981
time: 1.88 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013936, upper bound: 0.0013982
time: 2.19 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0042033, -0.0041135, -0.0042076, -0.0040589, -0.0001444, 0.0000940
1: -0.0100675, -0.0085209, -0.0102297, -0.0086635, -0.0014040, 0.0017088
2: 0.9643821, 0.9662380, 0.9641874, 0.9660669, -0.0016848, 0.0020506
3: -0.0164061, -0.0027171, -0.0178420, -0.0039790, -0.0097298, 0.0122348
4: -0.0004864, 0.0005548, -0.0003904, 0.0006640, -0.0011503, 0.0009451
5: 0.0167788, 0.0181511, 0.0168758, 0.0184721, -0.0016933, 0.0012753
6: 0.0023813, 0.0037456, 0.0017666, 0.0036984, -0.0013171, 0.0019790
7: -0.0070741, -0.0030202, -0.0067471, -0.0023150, -0.0047591, 0.0037268
8: 0.0111169, 0.0139314, 0.0113764, 0.0142266, -0.0031097, 0.0025551
9: 0.0177194, 0.0227816, 0.0181861, 0.0233126, -0.0052642, 0.0042828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010802, upper bound: 0.0013948
time: 1.44 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010802, upper bound: 0.0013864
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0042036, -0.0041097, -0.0042076, -0.0040590, -0.0001446, 0.0000979
1: -0.0100788, -0.0085334, -0.0102294, -0.0086707, -0.0014081, 0.0016960
2: 0.9643685, 0.9662230, 0.9641877, 0.9660582, -0.0016897, 0.0020353
3: -0.0165065, -0.0028273, -0.0178392, -0.0040427, -0.0097696, 0.0122142
4: -0.0004780, 0.0005624, -0.0003856, 0.0006637, -0.0011417, 0.0009479
5: 0.0167872, 0.0181735, 0.0168807, 0.0184715, -0.0016842, 0.0012929
6: 0.0023383, 0.0037415, 0.0017678, 0.0036960, -0.0013577, 0.0019737
7: -0.0070455, -0.0029709, -0.0067305, -0.0023164, -0.0047292, 0.0037596
8: 0.0111396, 0.0139521, 0.0113895, 0.0142261, -0.0030865, 0.0025626
9: 0.0177602, 0.0228187, 0.0182096, 0.0233116, -0.0052320, 0.0042962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010041, upper bound: 0.0013915
time: 1.20 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010041, upper bound: 0.0013853
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0042066, -0.0040707, -0.0042082, -0.0040511, -0.0001555, 0.0001375
1: -0.0101946, -0.0085997, -0.0102526, -0.0086629, -0.0015317, 0.0016528
2: 0.9642295, 0.9661434, 0.9641600, 0.9660675, -0.0018380, 0.0019835
3: -0.0175310, -0.0034145, -0.0180443, -0.0039738, -0.0104490, 0.0121047
4: -0.0004333, 0.0006403, -0.0003908, 0.0006793, -0.0011127, 0.0010311
5: 0.0168324, 0.0184026, 0.0168754, 0.0185173, -0.0016849, 0.0015272
6: 0.0018997, 0.0037195, 0.0016800, 0.0036986, -0.0017989, 0.0020395
7: -0.0068934, -0.0024677, -0.0067484, -0.0022157, -0.0046777, 0.0042807
8: 0.0112603, 0.0141627, 0.0113753, 0.0142682, -0.0030079, 0.0027874
9: 0.0179773, 0.0231976, 0.0181841, 0.0233874, -0.0051206, 0.0046621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013891, upper bound: 0.0014090
time: 1.82 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013891, upper bound: 0.0014088
time: 1.78 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013891, upper bound: 0.0013991
time: 2.15 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013891, upper bound: 0.0013993
time: 2.05 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0042016, -0.0041338, -0.0042063, -0.0040754, -0.0001262, 0.0000724
1: -0.0100073, -0.0085092, -0.0101805, -0.0086002, -0.0014072, 0.0016713
2: 0.9644542, 0.9662521, 0.9642465, 0.9661428, -0.0016886, 0.0020056
3: -0.0158735, -0.0026129, -0.0174061, -0.0034182, -0.0095948, 0.0117621
4: -0.0004943, 0.0005142, -0.0004331, 0.0006308, -0.0011251, 0.0009473
5: 0.0167708, 0.0180320, 0.0168327, 0.0183746, -0.0016039, 0.0011993
6: 0.0026093, 0.0037495, 0.0019532, 0.0037194, -0.0011101, 0.0017963
7: -0.0071011, -0.0032819, -0.0068924, -0.0025291, -0.0045720, 0.0036105
8: 0.0110955, 0.0138219, 0.0112611, 0.0141370, -0.0030415, 0.0025608
9: 0.0176809, 0.0225846, 0.0179787, 0.0231514, -0.0051287, 0.0042822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010924, upper bound: 0.0013529
time: 1.51 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010183, upper bound: 0.0013344
time: 1.53 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0042016, -0.0041338, -0.0042075, -0.0040595, -0.0001422, 0.0000737
1: -0.0100073, -0.0085092, -0.0102278, -0.0085903, -0.0014170, 0.0017187
2: 0.9644542, 0.9662521, 0.9641896, 0.9661547, -0.0017005, 0.0020625
3: -0.0158735, -0.0026129, -0.0178253, -0.0033311, -0.0097104, 0.0122109
4: -0.0004943, 0.0005142, -0.0004397, 0.0006627, -0.0011570, 0.0009539
5: 0.0167708, 0.0180320, 0.0168260, 0.0184684, -0.0016976, 0.0012060
6: 0.0026093, 0.0037495, 0.0017738, 0.0037226, -0.0011133, 0.0019757
7: -0.0071011, -0.0032819, -0.0069150, -0.0023232, -0.0047779, 0.0036331
8: 0.0110955, 0.0138219, 0.0112431, 0.0142232, -0.0031277, 0.0025788
9: 0.0176809, 0.0225846, 0.0179465, 0.0233064, -0.0052867, 0.0043136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010924, upper bound: 0.0013529
time: 1.60 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010183, upper bound: 0.0013931
time: 1.46 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0042066, -0.0040707, -0.0042069, -0.0040678, -0.0001388, 0.0001362
1: -0.0101946, -0.0085997, -0.0102030, -0.0085989, -0.0015956, 0.0016033
2: 0.9642295, 0.9661434, 0.9642194, 0.9661443, -0.0019148, 0.0019240
3: -0.0175310, -0.0034145, -0.0176058, -0.0034075, -0.0108937, 0.0114318
4: -0.0004333, 0.0006403, -0.0004339, 0.0006460, -0.0010793, 0.0010742
5: 0.0168324, 0.0184026, 0.0168318, 0.0184193, -0.0015869, 0.0015707
6: 0.0018997, 0.0037195, 0.0018677, 0.0037198, -0.0018201, 0.0018518
7: -0.0068934, -0.0024677, -0.0068952, -0.0024310, -0.0044623, 0.0044274
8: 0.0112603, 0.0141627, 0.0112589, 0.0141781, -0.0029178, 0.0029038
9: 0.0179773, 0.0231976, 0.0179747, 0.0232252, -0.0049305, 0.0048608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A2_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013791, upper bound: 0.0011229
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013791, upper bound: 0.0013882
time: 1.81 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042066, -0.0040707, -0.0042081, -0.0040517, -0.0001549, 0.0001375
1: -0.0101946, -0.0085997, -0.0102509, -0.0085891, -0.0016055, 0.0016511
2: 0.9642295, 0.9661434, 0.9641619, 0.9661561, -0.0019266, 0.0019815
3: -0.0175310, -0.0034145, -0.0180291, -0.0033205, -0.0110023, 0.0118789
4: -0.0004333, 0.0006403, -0.0004405, 0.0006782, -0.0011115, 0.0010808
5: 0.0168324, 0.0184026, 0.0168252, 0.0185139, -0.0016815, 0.0015774
6: 0.0018997, 0.0037195, 0.0016865, 0.0037230, -0.0018233, 0.0020330
7: -0.0068934, -0.0024677, -0.0069177, -0.0022231, -0.0046703, 0.0044500
8: 0.0112603, 0.0141627, 0.0112410, 0.0142651, -0.0030048, 0.0029217
9: 0.0179773, 0.0231976, 0.0179425, 0.0233818, -0.0050906, 0.0048925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013904, upper bound: 0.0014096
time: 1.75 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013904, upper bound: 0.0014093
time: 1.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0042075, -0.0040595, -0.0042016, -0.0041338, -0.0000737, 0.0001422
1: -0.0102278, -0.0085903, -0.0100073, -0.0085092, -0.0017187, 0.0014170
2: 0.9641896, 0.9661547, 0.9644542, 0.9662521, -0.0020625, 0.0017005
3: -0.0178253, -0.0033311, -0.0158735, -0.0026129, -0.0122109, 0.0097104
4: -0.0004397, 0.0006627, -0.0004943, 0.0005142, -0.0009539, 0.0011570
5: 0.0168260, 0.0184684, 0.0167708, 0.0180320, -0.0012060, 0.0016976
6: 0.0017738, 0.0037226, 0.0026093, 0.0037495, -0.0019757, 0.0011133
7: -0.0069150, -0.0023232, -0.0071011, -0.0032819, -0.0036331, 0.0047779
8: 0.0112431, 0.0142232, 0.0110955, 0.0138219, -0.0025788, 0.0031277
9: 0.0179465, 0.0233064, 0.0176809, 0.0225846, -0.0043136, 0.0052867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013893, upper bound: 0.0010993
time: 1.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013862, upper bound: 0.0010239
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0042081, -0.0040517, -0.0042066, -0.0040707, -0.0001375, 0.0001549
1: -0.0102509, -0.0085891, -0.0101946, -0.0085997, -0.0016511, 0.0016055
2: 0.9641619, 0.9661561, 0.9642295, 0.9661434, -0.0019815, 0.0019266
3: -0.0180291, -0.0033205, -0.0175310, -0.0034145, -0.0118789, 0.0110023
4: -0.0004405, 0.0006782, -0.0004333, 0.0006403, -0.0010808, 0.0011115
5: 0.0168252, 0.0185139, 0.0168324, 0.0184026, -0.0015774, 0.0016815
6: 0.0016865, 0.0037230, 0.0018997, 0.0037195, -0.0020330, 0.0018233
7: -0.0069177, -0.0022231, -0.0068934, -0.0024677, -0.0044500, 0.0046703
8: 0.0112410, 0.0142651, 0.0112603, 0.0141627, -0.0029217, 0.0030048
9: 0.0179425, 0.0233818, 0.0179773, 0.0231976, -0.0048925, 0.0050906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014028, upper bound: 0.0013986
time: 2.13 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014027, upper bound: 0.0013986
time: 1.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012005, upper bound: 0.0013870
time: 1.58 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011502, upper bound: 0.0013861
time: 1.50 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014007, upper bound: 0.0012202
time: 1.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014007, upper bound: 0.0012202
time: 1.26 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.13 seconds
IS_A1_B1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0010762, upper bound: 0.0013259
IS_A1_B1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0009808, upper bound: 0.0012883
IS_A1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0010762, upper bound: 0.0013833
IS_A1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0009808, upper bound: 0.0013740
IS_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013739, upper bound: 0.0011020
IS_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013739, upper bound: 0.0011020
IS_A1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013902, upper bound: 0.0014021
IS_A1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013901, upper bound: 0.0014019
IS_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013833, upper bound: 0.0010765
IS_A1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009808
IS_A1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0014021, upper bound: 0.0013902
IS_A1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0014019, upper bound: 0.0013901
IS_A1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0011836, upper bound: 0.0013746
IS_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0011221, upper bound: 0.0013714
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013972, upper bound: 0.0011975
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013972, upper bound: 0.0013957
IS_A1_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013981, upper bound: 0.0013938
IS_A1_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013982, upper bound: 0.0013936
IS_A1_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013981, upper bound: 0.0014012
IS_A1_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013982, upper bound: 0.0014012
IS_A1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013948, upper bound: 0.0010807
IS_A1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013948, upper bound: 0.0011828
IS_A1_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013915, upper bound: 0.0010041
IS_A1_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013915, upper bound: 0.0011311
IS_A1_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0014089, upper bound: 0.0013891
IS_A1_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0014088, upper bound: 0.0013891
IS_A1_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0014089, upper bound: 0.0013903
IS_A1_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0014087, upper bound: 0.0013902
IS_A2_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013937, upper bound: 0.0013981
IS_A2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013936, upper bound: 0.0013982
IS_A2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013937, upper bound: 0.0013981
IS_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013936, upper bound: 0.0013982
IS_A2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0010802, upper bound: 0.0013948
IS_A2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0010802, upper bound: 0.0013864
IS_A2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0010041, upper bound: 0.0013915
IS_A2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0010041, upper bound: 0.0013853
IS_A2_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013891, upper bound: 0.0014090
IS_A2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013891, upper bound: 0.0014088
IS_A2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013891, upper bound: 0.0013991
IS_A2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013891, upper bound: 0.0013993
IS_A2_B2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0010924, upper bound: 0.0013529
IS_A2_B2_A1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0010183, upper bound: 0.0013344
IS_A2_B2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0010924, upper bound: 0.0013529
IS_A2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0010183, upper bound: 0.0013931
IS_A2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013791, upper bound: 0.0011229
IS_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013791, upper bound: 0.0013882
IS_A2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013904, upper bound: 0.0014096
IS_A2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013904, upper bound: 0.0014093
IS_A2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013893, upper bound: 0.0010993
IS_A2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0013862, upper bound: 0.0010239
IS_A2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0014028, upper bound: 0.0013986
IS_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0014027, upper bound: 0.0013986
IS_A2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0012005, upper bound: 0.0013870
IS_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0011502, upper bound: 0.0013861
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0014007, upper bound: 0.0012202
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 2, lower bound: -0.0014007, upper bound: 0.0012202

## BFS IS instance: IS_A1_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0042016, -0.0041343, -0.0042076, -0.0040589, -0.0001428, 0.0000733
1: -0.0100061, -0.0086015, -0.0102297, -0.0086635, -0.0013426, 0.0016282
2: 0.9644557, 0.9661412, 0.9641874, 0.9660669, -0.0016112, 0.0019538
3: -0.0158625, -0.0034305, -0.0178420, -0.0039790, -0.0091245, 0.0116048
4: -0.0004321, 0.0005134, -0.0003904, 0.0006640, -0.0010961, 0.0009038
5: 0.0168336, 0.0180295, 0.0168758, 0.0184721, -0.0016385, 0.0011538
6: 0.0026140, 0.0037189, 0.0017666, 0.0036984, -0.0010844, 0.0019523
7: -0.0068892, -0.0032873, -0.0067471, -0.0023150, -0.0045742, 0.0034598
8: 0.0112636, 0.0138196, 0.0113764, 0.0142266, -0.0029631, 0.0024433
9: 0.0179832, 0.0225806, 0.0181861, 0.0233126, -0.0050122, 0.0040781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010475, upper bound: 0.0010963
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010475, upper bound: 0.0013833
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0042019, -0.0041302, -0.0042076, -0.0040590, -0.0001430, 0.0000774
1: -0.0100182, -0.0086134, -0.0102294, -0.0086707, -0.0013475, 0.0016160
2: 0.9644412, 0.9661269, 0.9641877, 0.9660582, -0.0016170, 0.0019392
3: -0.0159696, -0.0035357, -0.0178392, -0.0040427, -0.0091728, 0.0115898
4: -0.0004241, 0.0005215, -0.0003856, 0.0006637, -0.0010879, 0.0009071
5: 0.0168417, 0.0180535, 0.0168807, 0.0184715, -0.0016298, 0.0011728
6: 0.0025682, 0.0037150, 0.0017678, 0.0036960, -0.0011278, 0.0019472
7: -0.0068619, -0.0032347, -0.0067305, -0.0023164, -0.0045456, 0.0034959
8: 0.0112852, 0.0138417, 0.0113895, 0.0142261, -0.0029408, 0.0024522
9: 0.0180221, 0.0226202, 0.0182096, 0.0233116, -0.0049817, 0.0040948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0009692, upper bound: 0.0010700
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0009692, upper bound: 0.0010700
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0042067, -0.0040702, -0.0042016, -0.0041339, -0.0000728, 0.0001314
1: -0.0101961, -0.0086726, -0.0100072, -0.0085821, -0.0016140, 0.0013346
2: 0.9642278, 0.9660560, 0.9644544, 0.9661646, -0.0019368, 0.0016016
3: -0.0175440, -0.0040594, -0.0158724, -0.0032582, -0.0114943, 0.0090181
4: -0.0003843, 0.0006413, -0.0004452, 0.0005142, -0.0008984, 0.0010865
5: 0.0168820, 0.0184055, 0.0168204, 0.0180318, -0.0011498, 0.0015851
6: 0.0018942, 0.0036954, 0.0026098, 0.0037254, -0.0018312, 0.0010856
7: -0.0067262, -0.0024613, -0.0069339, -0.0032824, -0.0034438, 0.0044725
8: 0.0113929, 0.0141654, 0.0112281, 0.0138217, -0.0024288, 0.0029372
9: 0.0182158, 0.0232024, 0.0179195, 0.0225842, -0.0040501, 0.0049646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013259, upper bound: 0.0010762
time: 1.43 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012883, upper bound: 0.0009808
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013069, upper bound: 0.0013742
time: 1.64 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012883, upper bound: 0.0013743
time: 1.69 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0042067, -0.0040705, -0.0042082, -0.0040511, -0.0001555, 0.0001377
1: -0.0101952, -0.0086921, -0.0102526, -0.0086629, -0.0015323, 0.0015605
2: 0.9642289, 0.9660325, 0.9641600, 0.9660675, -0.0018386, 0.0018725
3: -0.0175363, -0.0042319, -0.0180443, -0.0039738, -0.0104002, 0.0111374
4: -0.0003712, 0.0006407, -0.0003908, 0.0006793, -0.0010505, 0.0010315
5: 0.0168952, 0.0184037, 0.0168754, 0.0185173, -0.0016221, 0.0015284
6: 0.0018975, 0.0036890, 0.0016800, 0.0036986, -0.0018011, 0.0020089
7: -0.0066815, -0.0024652, -0.0067484, -0.0022157, -0.0044658, 0.0042833
8: 0.0114284, 0.0141638, 0.0113753, 0.0142682, -0.0028399, 0.0027885
9: 0.0182796, 0.0231995, 0.0181841, 0.0233874, -0.0047975, 0.0046603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013146, upper bound: 0.0011085
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013146, upper bound: 0.0013801
time: 1.52 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0042070, -0.0040668, -0.0042082, -0.0040512, -0.0001557, 0.0001413
1: -0.0102060, -0.0087047, -0.0102523, -0.0086701, -0.0015359, 0.0015475
2: 0.9642158, 0.9660175, 0.9641603, 0.9660589, -0.0018430, 0.0018572
3: -0.0176323, -0.0043437, -0.0180415, -0.0040375, -0.0104427, 0.0111219
4: -0.0003627, 0.0006480, -0.0003859, 0.0006791, -0.0010418, 0.0010340
5: 0.0169038, 0.0184252, 0.0168803, 0.0185167, -0.0016129, 0.0015449
6: 0.0018564, 0.0036848, 0.0016812, 0.0036962, -0.0018399, 0.0020036
7: -0.0066525, -0.0024180, -0.0067319, -0.0022170, -0.0044355, 0.0043139
8: 0.0114513, 0.0141835, 0.0113884, 0.0142677, -0.0028163, 0.0027951
9: 0.0183209, 0.0232351, 0.0182077, 0.0233864, -0.0047667, 0.0046727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012950, upper bound: 0.0011003
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012950, upper bound: 0.0013801
time: 1.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0042076, -0.0040589, -0.0042016, -0.0041343, -0.0000733, 0.0001428
1: -0.0102297, -0.0086635, -0.0100061, -0.0086015, -0.0016282, 0.0013426
2: 0.9641874, 0.9660669, 0.9644557, 0.9661412, -0.0019538, 0.0016112
3: -0.0178420, -0.0039790, -0.0158625, -0.0034305, -0.0116048, 0.0091245
4: -0.0003904, 0.0006640, -0.0004321, 0.0005134, -0.0009038, 0.0010961
5: 0.0168758, 0.0184721, 0.0168336, 0.0180295, -0.0011538, 0.0016385
6: 0.0017666, 0.0036984, 0.0026140, 0.0037189, -0.0019523, 0.0010844
7: -0.0067471, -0.0023150, -0.0068892, -0.0032873, -0.0034598, 0.0045742
8: 0.0113764, 0.0142266, 0.0112636, 0.0138196, -0.0024433, 0.0029631
9: 0.0181861, 0.0233126, 0.0179832, 0.0225806, -0.0040781, 0.0050122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010963, upper bound: 0.0010475
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010963, upper bound: 0.0010765
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0042076, -0.0040590, -0.0042019, -0.0041302, -0.0000774, 0.0001430
1: -0.0102294, -0.0086707, -0.0100182, -0.0086134, -0.0016160, 0.0013475
2: 0.9641877, 0.9660582, 0.9644412, 0.9661269, -0.0019392, 0.0016170
3: -0.0178392, -0.0040427, -0.0159696, -0.0035357, -0.0115898, 0.0091728
4: -0.0003856, 0.0006637, -0.0004241, 0.0005215, -0.0009071, 0.0010879
5: 0.0168807, 0.0184715, 0.0168417, 0.0180535, -0.0011728, 0.0016298
6: 0.0017678, 0.0036960, 0.0025682, 0.0037150, -0.0019472, 0.0011278
7: -0.0067305, -0.0023164, -0.0068619, -0.0032347, -0.0034959, 0.0045456
8: 0.0113895, 0.0142261, 0.0112852, 0.0138417, -0.0024522, 0.0029408
9: 0.0182096, 0.0233116, 0.0180221, 0.0226202, -0.0040948, 0.0049817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010700, upper bound: 0.0009692
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010700, upper bound: 0.0009808
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0042082, -0.0040511, -0.0042067, -0.0040705, -0.0001377, 0.0001555
1: -0.0102526, -0.0086629, -0.0101952, -0.0086921, -0.0015605, 0.0015323
2: 0.9641600, 0.9660675, 0.9642289, 0.9660325, -0.0018725, 0.0018386
3: -0.0180443, -0.0039738, -0.0175363, -0.0042319, -0.0111374, 0.0104002
4: -0.0003908, 0.0006793, -0.0003712, 0.0006407, -0.0010315, 0.0010505
5: 0.0168754, 0.0185173, 0.0168952, 0.0184037, -0.0015284, 0.0016221
6: 0.0016800, 0.0036986, 0.0018975, 0.0036890, -0.0020089, 0.0018011
7: -0.0067484, -0.0022157, -0.0066815, -0.0024652, -0.0042833, 0.0044658
8: 0.0113753, 0.0142682, 0.0114284, 0.0141638, -0.0027885, 0.0028399
9: 0.0181841, 0.0233874, 0.0182796, 0.0231995, -0.0046603, 0.0047975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010963, upper bound: 0.0013146
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010963, upper bound: 0.0013729
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0042082, -0.0040512, -0.0042070, -0.0040668, -0.0001413, 0.0001557
1: -0.0102523, -0.0086701, -0.0102060, -0.0087047, -0.0015475, 0.0015359
2: 0.9641603, 0.9660589, 0.9642158, 0.9660175, -0.0018572, 0.0018430
3: -0.0180415, -0.0040375, -0.0176323, -0.0043437, -0.0111219, 0.0104427
4: -0.0003859, 0.0006791, -0.0003627, 0.0006480, -0.0010340, 0.0010418
5: 0.0168803, 0.0185167, 0.0169038, 0.0184252, -0.0015449, 0.0016129
6: 0.0016812, 0.0036962, 0.0018564, 0.0036848, -0.0020036, 0.0018399
7: -0.0067319, -0.0022170, -0.0066525, -0.0024180, -0.0043139, 0.0044355
8: 0.0113884, 0.0142677, 0.0114513, 0.0141835, -0.0027951, 0.0028163
9: 0.0182077, 0.0233864, 0.0183209, 0.0232351, -0.0046727, 0.0047667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010700, upper bound: 0.0012950
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010700, upper bound: 0.0013728
time: 1.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0042029, -0.0041186, -0.0042076, -0.0040589, -0.0001440, 0.0000890
1: -0.0100525, -0.0085948, -0.0102297, -0.0086635, -0.0013890, 0.0016349
2: 0.9644000, 0.9661493, 0.9641874, 0.9660669, -0.0016669, 0.0019619
3: -0.0162738, -0.0033707, -0.0178420, -0.0039790, -0.0093804, 0.0115174
4: -0.0004367, 0.0005447, -0.0003904, 0.0006640, -0.0011006, 0.0009351
5: 0.0168290, 0.0181215, 0.0168758, 0.0184721, -0.0016431, 0.0012457
6: 0.0024380, 0.0037212, 0.0017666, 0.0036984, -0.0012605, 0.0019546
7: -0.0069047, -0.0030852, -0.0067471, -0.0023150, -0.0045897, 0.0036618
8: 0.0112513, 0.0139042, 0.0113764, 0.0142266, -0.0029753, 0.0025278
9: 0.0179611, 0.0227327, 0.0181861, 0.0233126, -0.0050193, 0.0042145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011625, upper bound: 0.0011120
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011625, upper bound: 0.0011120
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0042032, -0.0041148, -0.0042076, -0.0040590, -0.0001442, 0.0000928
1: -0.0100639, -0.0086066, -0.0102294, -0.0086707, -0.0013932, 0.0016228
2: 0.9643864, 0.9661351, 0.9641877, 0.9660582, -0.0016718, 0.0019474
3: -0.0163744, -0.0034752, -0.0178392, -0.0040427, -0.0094232, 0.0114971
4: -0.0004287, 0.0005523, -0.0003856, 0.0006637, -0.0010925, 0.0009379
5: 0.0168370, 0.0181440, 0.0168807, 0.0184715, -0.0016344, 0.0012633
6: 0.0023949, 0.0037173, 0.0017678, 0.0036960, -0.0013011, 0.0019495
7: -0.0068776, -0.0030358, -0.0067305, -0.0023164, -0.0045613, 0.0036947
8: 0.0112728, 0.0139249, 0.0113895, 0.0142261, -0.0029533, 0.0025354
9: 0.0179998, 0.0227699, 0.0182096, 0.0233116, -0.0049893, 0.0042285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011112, upper bound: 0.0011048
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011112, upper bound: 0.0011048
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013875, upper bound: 0.0011785
time: 1.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013840, upper bound: 0.0011157
time: 1.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013850, upper bound: 0.0013739
time: 1.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013840, upper bound: 0.0013742
time: 1.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0042069, -0.0040674, -0.0042066, -0.0040709, -0.0001360, 0.0001393
1: -0.0102044, -0.0086723, -0.0101938, -0.0086185, -0.0015859, 0.0015215
2: 0.9642177, 0.9660563, 0.9642305, 0.9661208, -0.0019031, 0.0018258
3: -0.0176182, -0.0040566, -0.0175237, -0.0035809, -0.0114892, 0.0103327
4: -0.0003845, 0.0006469, -0.0004207, 0.0006398, -0.0010243, 0.0010676
5: 0.0168817, 0.0184221, 0.0168452, 0.0184009, -0.0015192, 0.0015769
6: 0.0018624, 0.0036955, 0.0019028, 0.0037133, -0.0018509, 0.0017927
7: -0.0067270, -0.0024249, -0.0068502, -0.0024713, -0.0042557, 0.0044253
8: 0.0113923, 0.0141806, 0.0112945, 0.0141612, -0.0027689, 0.0028861
9: 0.0182148, 0.0232298, 0.0180389, 0.0231949, -0.0046272, 0.0048979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013618, upper bound: 0.0013494
time: 1.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014006, upper bound: 0.0013922
time: 1.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0042069, -0.0040675, -0.0042069, -0.0040677, -0.0001392, 0.0001394
1: -0.0102041, -0.0086794, -0.0102034, -0.0086311, -0.0015730, 0.0015240
2: 0.9642181, 0.9660478, 0.9642189, 0.9661056, -0.0018875, 0.0018290
3: -0.0176154, -0.0041199, -0.0176095, -0.0036924, -0.0114740, 0.0103673
4: -0.0003797, 0.0006467, -0.0004122, 0.0006463, -0.0010260, 0.0010589
5: 0.0168866, 0.0184214, 0.0168537, 0.0184201, -0.0015335, 0.0015677
6: 0.0018636, 0.0036931, 0.0018661, 0.0037091, -0.0018455, 0.0018270
7: -0.0067105, -0.0024263, -0.0068213, -0.0024292, -0.0042814, 0.0043950
8: 0.0114053, 0.0141800, 0.0113174, 0.0141788, -0.0027735, 0.0028626
9: 0.0182382, 0.0232288, 0.0180801, 0.0232266, -0.0046364, 0.0048673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013530, upper bound: 0.0012358
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014005, upper bound: 0.0013921
time: 1.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0042069, -0.0040674, -0.0042079, -0.0040547, -0.0001522, 0.0001406
1: -0.0102044, -0.0086723, -0.0102420, -0.0086087, -0.0015958, 0.0015697
2: 0.9642177, 0.9660563, 0.9641727, 0.9661327, -0.0019150, 0.0018837
3: -0.0176182, -0.0040566, -0.0179505, -0.0034936, -0.0115923, 0.0107657
4: -0.0003845, 0.0006469, -0.0004273, 0.0006722, -0.0010567, 0.0010743
5: 0.0168817, 0.0184221, 0.0168385, 0.0184963, -0.0016146, 0.0015836
6: 0.0018624, 0.0036955, 0.0017202, 0.0037166, -0.0018542, 0.0019754
7: -0.0067270, -0.0024249, -0.0068729, -0.0022617, -0.0044653, 0.0044479
8: 0.0113923, 0.0141806, 0.0112766, 0.0142489, -0.0028566, 0.0029041
9: 0.0182148, 0.0232298, 0.0180066, 0.0233527, -0.0047850, 0.0049302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009906, upper bound: 0.0013771
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009906, upper bound: 0.0013797
time: 1.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0042069, -0.0040675, -0.0042082, -0.0040517, -0.0001552, 0.0001407
1: -0.0102041, -0.0086794, -0.0102509, -0.0086221, -0.0015820, 0.0015715
2: 0.9642181, 0.9660478, 0.9641619, 0.9661165, -0.0018984, 0.0018859
3: -0.0176154, -0.0041199, -0.0180299, -0.0036127, -0.0115731, 0.0107906
4: -0.0003797, 0.0006467, -0.0004183, 0.0006782, -0.0010579, 0.0010650
5: 0.0168866, 0.0184214, 0.0168476, 0.0185141, -0.0016275, 0.0015738
6: 0.0018636, 0.0036931, 0.0016862, 0.0037121, -0.0018485, 0.0020070
7: -0.0067105, -0.0024263, -0.0068420, -0.0022227, -0.0044878, 0.0044157
8: 0.0114053, 0.0141800, 0.0113011, 0.0142653, -0.0028599, 0.0028790
9: 0.0182382, 0.0232288, 0.0180506, 0.0233821, -0.0047919, 0.0048965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009851, upper bound: 0.0013740
time: 1.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009851, upper bound: 0.0013797
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0042076, -0.0040589, -0.0042016, -0.0041342, -0.0000734, 0.0001428
1: -0.0102297, -0.0086635, -0.0100063, -0.0085282, -0.0017016, 0.0013427
2: 0.9641874, 0.9660669, 0.9644555, 0.9662293, -0.0020419, 0.0016114
3: -0.0178420, -0.0039790, -0.0158640, -0.0027810, -0.0123063, 0.0091906
4: -0.0003904, 0.0006640, -0.0004815, 0.0005135, -0.0009039, 0.0011455
5: 0.0168758, 0.0184721, 0.0167837, 0.0180299, -0.0011541, 0.0016884
6: 0.0017666, 0.0036984, 0.0026134, 0.0037432, -0.0019766, 0.0010850
7: -0.0067471, -0.0023150, -0.0070575, -0.0032865, -0.0034605, 0.0047425
8: 0.0113764, 0.0142266, 0.0111300, 0.0138200, -0.0024436, 0.0030966
9: 0.0181861, 0.0233126, 0.0177430, 0.0225811, -0.0040821, 0.0052558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013873, upper bound: 0.0010596
time: 1.46 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013912, upper bound: 0.0010663
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0042076, -0.0040589, -0.0042028, -0.0041194, -0.0000882, 0.0001439
1: -0.0102297, -0.0086635, -0.0100503, -0.0085237, -0.0017060, 0.0013868
2: 0.9641874, 0.9660669, 0.9644027, 0.9662347, -0.0020473, 0.0016642
3: -0.0178420, -0.0039790, -0.0162538, -0.0027415, -0.0122084, 0.0094286
4: -0.0003904, 0.0006640, -0.0004845, 0.0005432, -0.0009336, 0.0011485
5: 0.0168758, 0.0184721, 0.0167807, 0.0181170, -0.0012413, 0.0016914
6: 0.0017666, 0.0036984, 0.0024465, 0.0037447, -0.0019781, 0.0012519
7: -0.0067471, -0.0023150, -0.0070678, -0.0030950, -0.0036520, 0.0047527
8: 0.0113764, 0.0142266, 0.0111219, 0.0139001, -0.0025238, 0.0031047
9: 0.0181861, 0.0233126, 0.0177285, 0.0227253, -0.0042114, 0.0052550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013873, upper bound: 0.0011605
time: 1.49 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013912, upper bound: 0.0011687
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0042076, -0.0040590, -0.0042019, -0.0041302, -0.0000774, 0.0001430
1: -0.0102294, -0.0086707, -0.0100180, -0.0085409, -0.0016885, 0.0013473
2: 0.9641877, 0.9660582, 0.9644415, 0.9662139, -0.0020262, 0.0016167
3: -0.0178392, -0.0040427, -0.0159680, -0.0028942, -0.0122905, 0.0092336
4: -0.0003856, 0.0006637, -0.0004729, 0.0005214, -0.0009070, 0.0011367
5: 0.0168807, 0.0184715, 0.0167924, 0.0180531, -0.0011725, 0.0016791
6: 0.0017678, 0.0036960, 0.0025689, 0.0037390, -0.0019712, 0.0011272
7: -0.0067305, -0.0023164, -0.0070282, -0.0032355, -0.0034951, 0.0047118
8: 0.0113895, 0.0142261, 0.0111533, 0.0138413, -0.0024519, 0.0030728
9: 0.0182096, 0.0233116, 0.0177849, 0.0226196, -0.0040970, 0.0052235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013817, upper bound: 0.0009847
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013875, upper bound: 0.0009895
time: 1.44 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0042076, -0.0040590, -0.0042031, -0.0041155, -0.0000920, 0.0001441
1: -0.0102294, -0.0086707, -0.0100616, -0.0085367, -0.0016927, 0.0013909
2: 0.9641877, 0.9660582, 0.9643892, 0.9662191, -0.0020313, 0.0016690
3: -0.0178392, -0.0040427, -0.0163539, -0.0028562, -0.0121825, 0.0094680
4: -0.0003856, 0.0006637, -0.0004758, 0.0005508, -0.0009363, 0.0011395
5: 0.0168807, 0.0184715, 0.0167895, 0.0181394, -0.0012587, 0.0016820
6: 0.0017678, 0.0036960, 0.0024036, 0.0037404, -0.0019726, 0.0012924
7: -0.0067305, -0.0023164, -0.0070380, -0.0030459, -0.0036847, 0.0047217
8: 0.0113895, 0.0142261, 0.0111455, 0.0139207, -0.0025312, 0.0030806
9: 0.0182096, 0.0233116, 0.0177709, 0.0227623, -0.0042250, 0.0052211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013817, upper bound: 0.0009847
time: 1.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013875, upper bound: 0.0011168
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0042082, -0.0040511, -0.0042066, -0.0040709, -0.0001373, 0.0001555
1: -0.0102526, -0.0086629, -0.0101938, -0.0086185, -0.0016340, 0.0015309
2: 0.9641600, 0.9660675, 0.9642305, 0.9661208, -0.0019608, 0.0018370
3: -0.0180443, -0.0039738, -0.0175237, -0.0035809, -0.0119344, 0.0104418
4: -0.0003908, 0.0006793, -0.0004207, 0.0006398, -0.0010305, 0.0011000
5: 0.0168754, 0.0185173, 0.0168452, 0.0184009, -0.0015256, 0.0016721
6: 0.0016800, 0.0036986, 0.0019028, 0.0037133, -0.0020333, 0.0017958
7: -0.0067484, -0.0022157, -0.0068502, -0.0024713, -0.0042771, 0.0046346
8: 0.0113753, 0.0142682, 0.0112945, 0.0141612, -0.0027859, 0.0029737
9: 0.0181841, 0.0233874, 0.0180389, 0.0231949, -0.0046594, 0.0050582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013788, upper bound: 0.0013493
time: 1.49 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014074, upper bound: 0.0013876
time: 1.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0042082, -0.0040512, -0.0042069, -0.0040677, -0.0001405, 0.0001556
1: -0.0102523, -0.0086701, -0.0102034, -0.0086311, -0.0016211, 0.0015333
2: 0.9641603, 0.9660589, 0.9642189, 0.9661056, -0.0019454, 0.0018400
3: -0.0180415, -0.0040375, -0.0176095, -0.0036924, -0.0119194, 0.0104767
4: -0.0003859, 0.0006791, -0.0004122, 0.0006463, -0.0010322, 0.0010913
5: 0.0168803, 0.0185167, 0.0168537, 0.0184201, -0.0015398, 0.0016630
6: 0.0016812, 0.0036962, 0.0018661, 0.0037091, -0.0020279, 0.0018301
7: -0.0067319, -0.0022170, -0.0068213, -0.0024292, -0.0043027, 0.0046044
8: 0.0113884, 0.0142677, 0.0113174, 0.0141788, -0.0027904, 0.0029502
9: 0.0182077, 0.0233864, 0.0180801, 0.0232266, -0.0046685, 0.0050277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013716, upper bound: 0.0012358
time: 1.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014072, upper bound: 0.0013876
time: 1.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0042082, -0.0040511, -0.0042079, -0.0040547, -0.0001535, 0.0001568
1: -0.0102526, -0.0086629, -0.0102420, -0.0086087, -0.0016439, 0.0015791
2: 0.9641600, 0.9660675, 0.9641727, 0.9661327, -0.0019727, 0.0018948
3: -0.0180443, -0.0039738, -0.0179505, -0.0034936, -0.0118709, 0.0107062
4: -0.0003908, 0.0006793, -0.0004273, 0.0006722, -0.0010630, 0.0011067
5: 0.0168754, 0.0185173, 0.0168385, 0.0184963, -0.0016210, 0.0016789
6: 0.0016800, 0.0036986, 0.0017202, 0.0037166, -0.0020365, 0.0019785
7: -0.0067484, -0.0022157, -0.0068729, -0.0022617, -0.0044867, 0.0046572
8: 0.0113753, 0.0142682, 0.0112766, 0.0142489, -0.0028737, 0.0029917
9: 0.0181841, 0.0233874, 0.0180066, 0.0233527, -0.0048007, 0.0050737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013820, upper bound: 0.0013658
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014075, upper bound: 0.0013888
time: 1.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0042082, -0.0040512, -0.0042082, -0.0040517, -0.0001565, 0.0001569
1: -0.0102523, -0.0086701, -0.0102509, -0.0086221, -0.0016301, 0.0015808
2: 0.9641603, 0.9660589, 0.9641619, 0.9661165, -0.0019563, 0.0018969
3: -0.0180415, -0.0040375, -0.0180299, -0.0036127, -0.0118465, 0.0107331
4: -0.0003859, 0.0006791, -0.0004183, 0.0006782, -0.0010642, 0.0010974
5: 0.0168803, 0.0185167, 0.0168476, 0.0185141, -0.0016338, 0.0016691
6: 0.0016812, 0.0036962, 0.0016862, 0.0037121, -0.0020309, 0.0020101
7: -0.0067319, -0.0022170, -0.0068420, -0.0022227, -0.0045092, 0.0046250
8: 0.0113884, 0.0142677, 0.0113011, 0.0142653, -0.0028769, 0.0029666
9: 0.0182077, 0.0233864, 0.0180506, 0.0233821, -0.0048072, 0.0050395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B2_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011271, upper bound: 0.0011176
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011271, upper bound: 0.0013734
time: 1.66 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0042066, -0.0040709, -0.0042069, -0.0040674, -0.0001393, 0.0001360
1: -0.0101938, -0.0086185, -0.0102044, -0.0086723, -0.0015215, 0.0015859
2: 0.9642305, 0.9661208, 0.9642177, 0.9660563, -0.0018258, 0.0019031
3: -0.0175237, -0.0035809, -0.0176182, -0.0040566, -0.0103327, 0.0114892
4: -0.0004207, 0.0006398, -0.0003845, 0.0006469, -0.0010676, 0.0010243
5: 0.0168452, 0.0184009, 0.0168817, 0.0184221, -0.0015769, 0.0015192
6: 0.0019028, 0.0037133, 0.0018624, 0.0036955, -0.0017927, 0.0018509
7: -0.0068502, -0.0024713, -0.0067270, -0.0024249, -0.0044253, 0.0042557
8: 0.0112945, 0.0141612, 0.0113923, 0.0141806, -0.0028861, 0.0027689
9: 0.0180389, 0.0231949, 0.0182148, 0.0232298, -0.0048979, 0.0046272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B1_A2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013494, upper bound: 0.0013618
time: 1.26 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013922, upper bound: 0.0014006
time: 2.42 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0042069, -0.0040677, -0.0042069, -0.0040675, -0.0001394, 0.0001392
1: -0.0102034, -0.0086311, -0.0102041, -0.0086794, -0.0015240, 0.0015730
2: 0.9642189, 0.9661056, 0.9642181, 0.9660478, -0.0018290, 0.0018875
3: -0.0176095, -0.0036924, -0.0176154, -0.0041199, -0.0103673, 0.0114740
4: -0.0004122, 0.0006463, -0.0003797, 0.0006467, -0.0010589, 0.0010260
5: 0.0168537, 0.0184201, 0.0168866, 0.0184214, -0.0015677, 0.0015335
6: 0.0018661, 0.0037091, 0.0018636, 0.0036931, -0.0018270, 0.0018455
7: -0.0068213, -0.0024292, -0.0067105, -0.0024263, -0.0043950, 0.0042814
8: 0.0113174, 0.0141788, 0.0114053, 0.0141800, -0.0028626, 0.0027735
9: 0.0180801, 0.0232266, 0.0182382, 0.0232288, -0.0048673, 0.0046364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B1_A2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012358, upper bound: 0.0013530
time: 1.64 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013921, upper bound: 0.0014004
time: 1.66 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0042079, -0.0040547, -0.0042069, -0.0040674, -0.0001406, 0.0001522
1: -0.0102420, -0.0086087, -0.0102044, -0.0086723, -0.0015697, 0.0015958
2: 0.9641727, 0.9661327, 0.9642177, 0.9660563, -0.0018837, 0.0019150
3: -0.0179505, -0.0034936, -0.0176182, -0.0040566, -0.0107657, 0.0115923
4: -0.0004273, 0.0006722, -0.0003845, 0.0006469, -0.0010743, 0.0010567
5: 0.0168385, 0.0184963, 0.0168817, 0.0184221, -0.0015836, 0.0016146
6: 0.0017202, 0.0037166, 0.0018624, 0.0036955, -0.0019754, 0.0018542
7: -0.0068729, -0.0022617, -0.0067270, -0.0024249, -0.0044479, 0.0044653
8: 0.0112766, 0.0142489, 0.0113923, 0.0141806, -0.0029041, 0.0028566
9: 0.0180066, 0.0233527, 0.0182148, 0.0232298, -0.0049302, 0.0047850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013771, upper bound: 0.0009952
time: 1.15 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013771, upper bound: 0.0009952
time: 1.32 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0042082, -0.0040517, -0.0042069, -0.0040675, -0.0001407, 0.0001552
1: -0.0102509, -0.0086221, -0.0102041, -0.0086794, -0.0015715, 0.0015820
2: 0.9641619, 0.9661165, 0.9642181, 0.9660478, -0.0018859, 0.0018984
3: -0.0180299, -0.0036127, -0.0176154, -0.0041199, -0.0107906, 0.0115731
4: -0.0004183, 0.0006782, -0.0003797, 0.0006467, -0.0010650, 0.0010579
5: 0.0168476, 0.0185141, 0.0168866, 0.0184214, -0.0015738, 0.0016275
6: 0.0016862, 0.0037121, 0.0018636, 0.0036931, -0.0020070, 0.0018485
7: -0.0068420, -0.0022227, -0.0067105, -0.0024263, -0.0044157, 0.0044878
8: 0.0113011, 0.0142653, 0.0114053, 0.0141800, -0.0028790, 0.0028599
9: 0.0180506, 0.0233821, 0.0182382, 0.0232288, -0.0048965, 0.0047919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009917
time: 1.27 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0013815
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0042016, -0.0041342, -0.0042076, -0.0040589, -0.0001428, 0.0000734
1: -0.0100063, -0.0085282, -0.0102297, -0.0086635, -0.0013427, 0.0017016
2: 0.9644555, 0.9662293, 0.9641874, 0.9660669, -0.0016114, 0.0020419
3: -0.0158640, -0.0027810, -0.0178420, -0.0039790, -0.0091906, 0.0123063
4: -0.0004815, 0.0005135, -0.0003904, 0.0006640, -0.0011455, 0.0009039
5: 0.0167837, 0.0180299, 0.0168758, 0.0184721, -0.0016884, 0.0011541
6: 0.0026134, 0.0037432, 0.0017666, 0.0036984, -0.0010850, 0.0019766
7: -0.0070575, -0.0032865, -0.0067471, -0.0023150, -0.0047425, 0.0034605
8: 0.0111300, 0.0138200, 0.0113764, 0.0142266, -0.0030966, 0.0024436
9: 0.0177430, 0.0225811, 0.0181861, 0.0233126, -0.0052558, 0.0040821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010594, upper bound: 0.0013307
time: 1.30 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010659, upper bound: 0.0013912
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0042028, -0.0041194, -0.0042076, -0.0040589, -0.0001439, 0.0000882
1: -0.0100503, -0.0085237, -0.0102297, -0.0086635, -0.0013868, 0.0017060
2: 0.9644027, 0.9662347, 0.9641874, 0.9660669, -0.0016642, 0.0020473
3: -0.0162538, -0.0027415, -0.0178420, -0.0039790, -0.0094286, 0.0122084
4: -0.0004845, 0.0005432, -0.0003904, 0.0006640, -0.0011485, 0.0009336
5: 0.0167807, 0.0181170, 0.0168758, 0.0184721, -0.0016914, 0.0012413
6: 0.0024465, 0.0037447, 0.0017666, 0.0036984, -0.0012519, 0.0019781
7: -0.0070678, -0.0030950, -0.0067471, -0.0023150, -0.0047527, 0.0036520
8: 0.0111219, 0.0139001, 0.0113764, 0.0142266, -0.0031047, 0.0025238
9: 0.0177285, 0.0227253, 0.0181861, 0.0233126, -0.0052550, 0.0042114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010594, upper bound: 0.0013807
time: 1.62 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010659, upper bound: 0.0013830
time: 1.41 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0042019, -0.0041302, -0.0042076, -0.0040590, -0.0001430, 0.0000774
1: -0.0100180, -0.0085409, -0.0102294, -0.0086707, -0.0013473, 0.0016885
2: 0.9644415, 0.9662139, 0.9641877, 0.9660582, -0.0016167, 0.0020262
3: -0.0159680, -0.0028942, -0.0178392, -0.0040427, -0.0092336, 0.0122905
4: -0.0004729, 0.0005214, -0.0003856, 0.0006637, -0.0011367, 0.0009070
5: 0.0167924, 0.0180531, 0.0168807, 0.0184715, -0.0016791, 0.0011725
6: 0.0025689, 0.0037390, 0.0017678, 0.0036960, -0.0011272, 0.0019712
7: -0.0070282, -0.0032355, -0.0067305, -0.0023164, -0.0047118, 0.0034951
8: 0.0111533, 0.0138413, 0.0113895, 0.0142261, -0.0030728, 0.0024519
9: 0.0177849, 0.0226196, 0.0182096, 0.0233116, -0.0052235, 0.0040970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009847, upper bound: 0.0013817
time: 1.47 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009895, upper bound: 0.0013875
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0042031, -0.0041155, -0.0042076, -0.0040590, -0.0001441, 0.0000920
1: -0.0100616, -0.0085367, -0.0102294, -0.0086707, -0.0013909, 0.0016927
2: 0.9643892, 0.9662191, 0.9641877, 0.9660582, -0.0016690, 0.0020313
3: -0.0163539, -0.0028562, -0.0178392, -0.0040427, -0.0094680, 0.0121825
4: -0.0004758, 0.0005508, -0.0003856, 0.0006637, -0.0011395, 0.0009363
5: 0.0167895, 0.0181394, 0.0168807, 0.0184715, -0.0016820, 0.0012587
6: 0.0024036, 0.0037404, 0.0017678, 0.0036960, -0.0012924, 0.0019726
7: -0.0070380, -0.0030459, -0.0067305, -0.0023164, -0.0047217, 0.0036847
8: 0.0111455, 0.0139207, 0.0113895, 0.0142261, -0.0030806, 0.0025312
9: 0.0177709, 0.0227623, 0.0182096, 0.0233116, -0.0052211, 0.0042250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009847, upper bound: 0.0013789
time: 1.57 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0009895, upper bound: 0.0013819
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0042066, -0.0040709, -0.0042082, -0.0040511, -0.0001555, 0.0001373
1: -0.0101938, -0.0086185, -0.0102526, -0.0086629, -0.0015309, 0.0016340
2: 0.9642305, 0.9661208, 0.9641600, 0.9660675, -0.0018370, 0.0019608
3: -0.0175237, -0.0035809, -0.0180443, -0.0039738, -0.0104418, 0.0119344
4: -0.0004207, 0.0006398, -0.0003908, 0.0006793, -0.0011000, 0.0010305
5: 0.0168452, 0.0184009, 0.0168754, 0.0185173, -0.0016721, 0.0015256
6: 0.0019028, 0.0037133, 0.0016800, 0.0036986, -0.0017958, 0.0020333
7: -0.0068502, -0.0024713, -0.0067484, -0.0022157, -0.0046346, 0.0042771
8: 0.0112945, 0.0141612, 0.0113753, 0.0142682, -0.0029737, 0.0027859
9: 0.0180389, 0.0231949, 0.0181841, 0.0233874, -0.0050582, 0.0046594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013493, upper bound: 0.0013788
time: 1.38 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013876, upper bound: 0.0014073
time: 1.88 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0042069, -0.0040677, -0.0042082, -0.0040512, -0.0001556, 0.0001405
1: -0.0102034, -0.0086311, -0.0102523, -0.0086701, -0.0015333, 0.0016211
2: 0.9642189, 0.9661056, 0.9641603, 0.9660589, -0.0018400, 0.0019454
3: -0.0176095, -0.0036924, -0.0180415, -0.0040375, -0.0104767, 0.0119194
4: -0.0004122, 0.0006463, -0.0003859, 0.0006791, -0.0010913, 0.0010322
5: 0.0168537, 0.0184201, 0.0168803, 0.0185167, -0.0016630, 0.0015398
6: 0.0018661, 0.0037091, 0.0016812, 0.0036962, -0.0018301, 0.0020279
7: -0.0068213, -0.0024292, -0.0067319, -0.0022170, -0.0046044, 0.0043027
8: 0.0113174, 0.0141788, 0.0113884, 0.0142677, -0.0029502, 0.0027904
9: 0.0180801, 0.0232266, 0.0182077, 0.0233864, -0.0050277, 0.0046685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012358, upper bound: 0.0013716
time: 1.90 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013876, upper bound: 0.0014072
time: 1.83 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0042079, -0.0040547, -0.0042082, -0.0040511, -0.0001568, 0.0001535
1: -0.0102420, -0.0086087, -0.0102526, -0.0086629, -0.0015791, 0.0016439
2: 0.9641727, 0.9661327, 0.9641600, 0.9660675, -0.0018948, 0.0019727
3: -0.0179505, -0.0034936, -0.0180443, -0.0039738, -0.0107062, 0.0118709
4: -0.0004273, 0.0006722, -0.0003908, 0.0006793, -0.0011067, 0.0010630
5: 0.0168385, 0.0184963, 0.0168754, 0.0185173, -0.0016789, 0.0016210
6: 0.0017202, 0.0037166, 0.0016800, 0.0036986, -0.0019785, 0.0020365
7: -0.0068729, -0.0022617, -0.0067484, -0.0022157, -0.0046572, 0.0044867
8: 0.0112766, 0.0142489, 0.0113753, 0.0142682, -0.0029917, 0.0028737
9: 0.0180066, 0.0233527, 0.0181841, 0.0233874, -0.0050737, 0.0048007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013741, upper bound: 0.0013728
time: 1.60 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013998, upper bound: 0.0013976
time: 1.86 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0042082, -0.0040517, -0.0042082, -0.0040512, -0.0001569, 0.0001565
1: -0.0102509, -0.0086221, -0.0102523, -0.0086701, -0.0015808, 0.0016301
2: 0.9641619, 0.9661165, 0.9641603, 0.9660589, -0.0018969, 0.0019563
3: -0.0180299, -0.0036127, -0.0180415, -0.0040375, -0.0107331, 0.0118465
4: -0.0004183, 0.0006782, -0.0003859, 0.0006791, -0.0010974, 0.0010642
5: 0.0168476, 0.0185141, 0.0168803, 0.0185167, -0.0016691, 0.0016338
6: 0.0016862, 0.0037121, 0.0016812, 0.0036962, -0.0020101, 0.0020309
7: -0.0068420, -0.0022227, -0.0067319, -0.0022170, -0.0046250, 0.0045092
8: 0.0113011, 0.0142653, 0.0113884, 0.0142677, -0.0029666, 0.0028769
9: 0.0180506, 0.0233821, 0.0182077, 0.0233864, -0.0050395, 0.0048072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A2_B1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013840, upper bound: 0.0011270
time: 1.42 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013840, upper bound: 0.0013830
time: 1.51 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0042019, -0.0041302, -0.0042075, -0.0040596, -0.0001424, 0.0000773
1: -0.0100180, -0.0085409, -0.0102275, -0.0085976, -0.0014204, 0.0016866
2: 0.9644415, 0.9662139, 0.9641901, 0.9661460, -0.0017045, 0.0020239
3: -0.0159680, -0.0028942, -0.0178226, -0.0033956, -0.0097393, 0.0120226
4: -0.0004729, 0.0005214, -0.0004348, 0.0006625, -0.0011354, 0.0009562
5: 0.0167924, 0.0180531, 0.0168309, 0.0184678, -0.0016754, 0.0012222
6: 0.0025689, 0.0037390, 0.0017749, 0.0037202, -0.0011514, 0.0019641
7: -0.0070282, -0.0032355, -0.0068982, -0.0023245, -0.0047037, 0.0036628
8: 0.0111533, 0.0138413, 0.0112564, 0.0142226, -0.0030693, 0.0025849
9: 0.0177849, 0.0226196, 0.0179703, 0.0233054, -0.0051919, 0.0043240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A2_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010120, upper bound: 0.0011163
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010120, upper bound: 0.0011164
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0042066, -0.0040707, -0.0042016, -0.0041338, -0.0000728, 0.0001310
1: -0.0101946, -0.0085997, -0.0100073, -0.0085092, -0.0016854, 0.0014076
2: 0.9642295, 0.9661434, 0.9644542, 0.9662521, -0.0020226, 0.0016892
3: -0.0175310, -0.0034145, -0.0158735, -0.0026129, -0.0119173, 0.0095990
4: -0.0004333, 0.0006403, -0.0004943, 0.0005142, -0.0009476, 0.0011346
5: 0.0168324, 0.0184026, 0.0167708, 0.0180320, -0.0011996, 0.0016318
6: 0.0018997, 0.0037195, 0.0026093, 0.0037495, -0.0018498, 0.0011102
7: -0.0068934, -0.0024677, -0.0071011, -0.0032819, -0.0036115, 0.0046334
8: 0.0112603, 0.0141627, 0.0110955, 0.0138219, -0.0025616, 0.0030672
9: 0.0179773, 0.0231976, 0.0176809, 0.0225846, -0.0042836, 0.0051770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013413, upper bound: 0.0010992
time: 1.37 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013173, upper bound: 0.0010239
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013413, upper bound: 0.0013830
time: 1.73 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013173, upper bound: 0.0013829
time: 2.01 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0042066, -0.0040709, -0.0042081, -0.0040517, -0.0001549, 0.0001372
1: -0.0101938, -0.0086185, -0.0102509, -0.0085891, -0.0016046, 0.0016323
2: 0.9642305, 0.9661208, 0.9641619, 0.9661561, -0.0019256, 0.0019588
3: -0.0175237, -0.0035809, -0.0180291, -0.0033205, -0.0109947, 0.0117040
4: -0.0004207, 0.0006398, -0.0004405, 0.0006782, -0.0010989, 0.0010802
5: 0.0168452, 0.0184009, 0.0168252, 0.0185139, -0.0016688, 0.0015758
6: 0.0019028, 0.0037133, 0.0016865, 0.0037230, -0.0018202, 0.0020268
7: -0.0068502, -0.0024713, -0.0069177, -0.0022231, -0.0046271, 0.0044464
8: 0.0112945, 0.0141612, 0.0112410, 0.0142651, -0.0029706, 0.0029202
9: 0.0180389, 0.0231949, 0.0179425, 0.0233818, -0.0050281, 0.0048898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A2_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013371, upper bound: 0.0011414
time: 1.95 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013371, upper bound: 0.0013879
time: 1.59 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0042069, -0.0040677, -0.0042081, -0.0040518, -0.0001551, 0.0001404
1: -0.0102034, -0.0086311, -0.0102505, -0.0085964, -0.0016071, 0.0016194
2: 0.9642189, 0.9661056, 0.9641624, 0.9661474, -0.0019285, 0.0019432
3: -0.0176095, -0.0036924, -0.0180264, -0.0033850, -0.0110279, 0.0116909
4: -0.0004122, 0.0006463, -0.0004356, 0.0006780, -0.0010902, 0.0010819
5: 0.0168537, 0.0184201, 0.0168301, 0.0185133, -0.0016596, 0.0015900
6: 0.0018661, 0.0037091, 0.0016877, 0.0037206, -0.0018545, 0.0020214
7: -0.0068213, -0.0024292, -0.0069010, -0.0022244, -0.0045969, 0.0044718
8: 0.0113174, 0.0141788, 0.0112542, 0.0142645, -0.0029471, 0.0029246
9: 0.0180801, 0.0232266, 0.0179664, 0.0233808, -0.0049976, 0.0048983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013206, upper bound: 0.0011334
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013206, upper bound: 0.0013881
time: 1.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0042075, -0.0040595, -0.0042016, -0.0041342, -0.0000733, 0.0001421
1: -0.0102278, -0.0085903, -0.0100063, -0.0085282, -0.0016997, 0.0014159
2: 0.9641896, 0.9661547, 0.9644555, 0.9662293, -0.0020397, 0.0016991
3: -0.0178253, -0.0033311, -0.0158640, -0.0027810, -0.0120407, 0.0097006
4: -0.0004397, 0.0006627, -0.0004815, 0.0005135, -0.0009532, 0.0011442
5: 0.0168260, 0.0184684, 0.0167837, 0.0180299, -0.0012039, 0.0016847
6: 0.0017738, 0.0037226, 0.0026134, 0.0037432, -0.0019695, 0.0011093
7: -0.0069150, -0.0023232, -0.0070575, -0.0032865, -0.0036285, 0.0047344
8: 0.0112431, 0.0142232, 0.0111300, 0.0138200, -0.0025768, 0.0030932
9: 0.0179465, 0.0233064, 0.0177430, 0.0225811, -0.0043101, 0.0052246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011274, upper bound: 0.0010844
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011274, upper bound: 0.0010993
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0042075, -0.0040596, -0.0042019, -0.0041302, -0.0000773, 0.0001424
1: -0.0102275, -0.0085976, -0.0100180, -0.0085409, -0.0016866, 0.0014204
2: 0.9641901, 0.9661460, 0.9644415, 0.9662139, -0.0020239, 0.0017045
3: -0.0178226, -0.0033956, -0.0159680, -0.0028942, -0.0120226, 0.0097393
4: -0.0004348, 0.0006625, -0.0004729, 0.0005214, -0.0009562, 0.0011354
5: 0.0168309, 0.0184678, 0.0167924, 0.0180531, -0.0012222, 0.0016754
6: 0.0017749, 0.0037202, 0.0025689, 0.0037390, -0.0019641, 0.0011514
7: -0.0068982, -0.0023245, -0.0070282, -0.0032355, -0.0036628, 0.0047037
8: 0.0112564, 0.0142226, 0.0111533, 0.0138413, -0.0025849, 0.0030693
9: 0.0179703, 0.0233054, 0.0177849, 0.0226196, -0.0043240, 0.0051919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011108, upper bound: 0.0010178
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011108, upper bound: 0.0010239
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0042081, -0.0040517, -0.0042066, -0.0040709, -0.0001372, 0.0001549
1: -0.0102509, -0.0085891, -0.0101938, -0.0086185, -0.0016323, 0.0016046
2: 0.9641619, 0.9661561, 0.9642305, 0.9661208, -0.0019588, 0.0019256
3: -0.0180291, -0.0033205, -0.0175237, -0.0035809, -0.0117040, 0.0109947
4: -0.0004405, 0.0006782, -0.0004207, 0.0006398, -0.0010802, 0.0010989
5: 0.0168252, 0.0185139, 0.0168452, 0.0184009, -0.0015758, 0.0016688
6: 0.0016865, 0.0037230, 0.0019028, 0.0037133, -0.0020268, 0.0018202
7: -0.0069177, -0.0022231, -0.0068502, -0.0024713, -0.0044464, 0.0046271
8: 0.0112410, 0.0142651, 0.0112945, 0.0141612, -0.0029202, 0.0029706
9: 0.0179425, 0.0233818, 0.0180389, 0.0231949, -0.0048898, 0.0050281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011274, upper bound: 0.0010844
time: 1.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011274, upper bound: 0.0013818
time: 1.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0042081, -0.0040518, -0.0042069, -0.0040677, -0.0001404, 0.0001551
1: -0.0102505, -0.0085964, -0.0102034, -0.0086311, -0.0016194, 0.0016071
2: 0.9641624, 0.9661474, 0.9642189, 0.9661056, -0.0019432, 0.0019285
3: -0.0180264, -0.0033850, -0.0176095, -0.0036924, -0.0116909, 0.0110279
4: -0.0004356, 0.0006780, -0.0004122, 0.0006463, -0.0010819, 0.0010902
5: 0.0168301, 0.0185133, 0.0168537, 0.0184201, -0.0015900, 0.0016596
6: 0.0016877, 0.0037206, 0.0018661, 0.0037091, -0.0020214, 0.0018545
7: -0.0069010, -0.0022244, -0.0068213, -0.0024292, -0.0044718, 0.0045969
8: 0.0112542, 0.0142645, 0.0113174, 0.0141788, -0.0029246, 0.0029471
9: 0.0179664, 0.0233808, 0.0180801, 0.0232266, -0.0048983, 0.0049976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011108, upper bound: 0.0013371
time: 1.21 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0011108, upper bound: 0.0013816
time: 1.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0042028, -0.0041194, -0.0042075, -0.0040595, -0.0001433, 0.0000882
1: -0.0100503, -0.0085237, -0.0102278, -0.0085903, -0.0014600, 0.0017041
2: 0.9644027, 0.9662347, 0.9641896, 0.9661547, -0.0017520, 0.0020451
3: -0.0162538, -0.0027415, -0.0178253, -0.0033311, -0.0099488, 0.0119355
4: -0.0004845, 0.0005432, -0.0004397, 0.0006627, -0.0011472, 0.0009828
5: 0.0167807, 0.0181170, 0.0168260, 0.0184684, -0.0016877, 0.0012911
6: 0.0024465, 0.0037447, 0.0017738, 0.0037226, -0.0012762, 0.0019709
7: -0.0070678, -0.0030950, -0.0069150, -0.0023232, -0.0047446, 0.0038199
8: 0.0111219, 0.0139001, 0.0112431, 0.0142232, -0.0031013, 0.0026570
9: 0.0177285, 0.0227253, 0.0179465, 0.0233064, -0.0052242, 0.0044399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A2_B2_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011868, upper bound: 0.0011471
time: 1.27 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011868, upper bound: 0.0011471
time: 1.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0042031, -0.0041155, -0.0042075, -0.0040596, -0.0001435, 0.0000920
1: -0.0100616, -0.0085367, -0.0102275, -0.0085976, -0.0014640, 0.0016909
2: 0.9643892, 0.9662191, 0.9641901, 0.9661460, -0.0017568, 0.0020290
3: -0.0163539, -0.0028562, -0.0178226, -0.0033956, -0.0099861, 0.0119148
4: -0.0004758, 0.0005508, -0.0004348, 0.0006625, -0.0011383, 0.0009856
5: 0.0167895, 0.0181394, 0.0168309, 0.0184678, -0.0016783, 0.0013085
6: 0.0024036, 0.0037404, 0.0017749, 0.0037202, -0.0013166, 0.0019655
7: -0.0070380, -0.0030459, -0.0068982, -0.0023245, -0.0047135, 0.0038524
8: 0.0111455, 0.0139207, 0.0112564, 0.0142226, -0.0030771, 0.0026643
9: 0.0177709, 0.0227623, 0.0179703, 0.0233054, -0.0051908, 0.0044529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A2_B2_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011443, upper bound: 0.0011436
time: 1.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011443, upper bound: 0.0011436
time: 1.50 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013931, upper bound: 0.0012008
time: 1.40 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013918, upper bound: 0.0011499
time: 1.48 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013925, upper bound: 0.0013829
time: 1.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013918, upper bound: 0.0013833
time: 3.18 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.17 seconds
IS_A1_B1_A1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010475, upper bound: 0.0010963
IS_A1_B1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010475, upper bound: 0.0013833
IS_A1_B1_A1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0009692, upper bound: 0.0010700
IS_A1_B1_A1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0009692, upper bound: 0.0010700
IS_A1_B1_A1_A2_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013259, upper bound: 0.0010762
IS_A1_B1_A1_A2_B1_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0012883, upper bound: 0.0009808
IS_A1_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013069, upper bound: 0.0013742
IS_A1_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0012883, upper bound: 0.0013743
IS_A1_B1_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013146, upper bound: 0.0011085
IS_A1_B1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013146, upper bound: 0.0013801
IS_A1_B1_A1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0012950, upper bound: 0.0011003
IS_A1_B1_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0012950, upper bound: 0.0013801
IS_A1_B1_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010963, upper bound: 0.0010475
IS_A1_B1_A2_B1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010963, upper bound: 0.0010765
IS_A1_B1_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010700, upper bound: 0.0009692
IS_A1_B1_A2_B1_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010700, upper bound: 0.0009808
IS_A1_B1_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010963, upper bound: 0.0013146
IS_A1_B1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010963, upper bound: 0.0013729
IS_A1_B1_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010700, upper bound: 0.0012950
IS_A1_B1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010700, upper bound: 0.0013728
IS_A1_B1_A2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011625, upper bound: 0.0011120
IS_A1_B1_A2_B2_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011625, upper bound: 0.0011120
IS_A1_B1_A2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011112, upper bound: 0.0011048
IS_A1_B1_A2_B2_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011112, upper bound: 0.0011048
IS_A1_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013875, upper bound: 0.0011785
IS_A1_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013840, upper bound: 0.0011157
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013850, upper bound: 0.0013739
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013840, upper bound: 0.0013742
IS_A1_B2_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013618, upper bound: 0.0013494
IS_A1_B2_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0014006, upper bound: 0.0013922
IS_A1_B2_A1_B2_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013530, upper bound: 0.0012358
IS_A1_B2_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0014005, upper bound: 0.0013921
IS_A1_B2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0009906, upper bound: 0.0013771
IS_A1_B2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0009906, upper bound: 0.0013797
IS_A1_B2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0009851, upper bound: 0.0013740
IS_A1_B2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0009851, upper bound: 0.0013797
IS_A1_B2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013873, upper bound: 0.0010596
IS_A1_B2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013912, upper bound: 0.0010663
IS_A1_B2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013873, upper bound: 0.0011605
IS_A1_B2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013912, upper bound: 0.0011687
IS_A1_B2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013817, upper bound: 0.0009847
IS_A1_B2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013875, upper bound: 0.0009895
IS_A1_B2_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013817, upper bound: 0.0009847
IS_A1_B2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013875, upper bound: 0.0011168
IS_A1_B2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013788, upper bound: 0.0013493
IS_A1_B2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0014074, upper bound: 0.0013876
IS_A1_B2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013716, upper bound: 0.0012358
IS_A1_B2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0014072, upper bound: 0.0013876
IS_A1_B2_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013820, upper bound: 0.0013658
IS_A1_B2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0014075, upper bound: 0.0013888
IS_A1_B2_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011271, upper bound: 0.0011176
IS_A1_B2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011271, upper bound: 0.0013734
IS_A2_B1_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013494, upper bound: 0.0013618
IS_A2_B1_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013922, upper bound: 0.0014006
IS_A2_B1_B1_A2_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0012358, upper bound: 0.0013530
IS_A2_B1_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013921, upper bound: 0.0014004
IS_A2_B1_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013771, upper bound: 0.0009952
IS_A2_B1_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013771, upper bound: 0.0009952
IS_A2_B1_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0009917
IS_A2_B1_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013740, upper bound: 0.0013815
IS_A2_B1_B2_A1_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010594, upper bound: 0.0013307
IS_A2_B1_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010659, upper bound: 0.0013912
IS_A2_B1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010594, upper bound: 0.0013807
IS_A2_B1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010659, upper bound: 0.0013830
IS_A2_B1_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0009847, upper bound: 0.0013817
IS_A2_B1_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0009895, upper bound: 0.0013875
IS_A2_B1_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0009847, upper bound: 0.0013789
IS_A2_B1_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0009895, upper bound: 0.0013819
IS_A2_B1_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013493, upper bound: 0.0013788
IS_A2_B1_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013876, upper bound: 0.0014073
IS_A2_B1_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0012358, upper bound: 0.0013716
IS_A2_B1_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013876, upper bound: 0.0014072
IS_A2_B1_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013741, upper bound: 0.0013728
IS_A2_B1_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013998, upper bound: 0.0013976
IS_A2_B1_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013840, upper bound: 0.0011270
IS_A2_B1_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013840, upper bound: 0.0013830
IS_A2_B2_A1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010120, upper bound: 0.0011163
IS_A2_B2_A1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0010120, upper bound: 0.0011164
IS_A2_B2_A1_A2_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013413, upper bound: 0.0010992
IS_A2_B2_A1_A2_B1_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013173, upper bound: 0.0010239
IS_A2_B2_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013413, upper bound: 0.0013830
IS_A2_B2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013173, upper bound: 0.0013829
IS_A2_B2_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013371, upper bound: 0.0011414
IS_A2_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013371, upper bound: 0.0013879
IS_A2_B2_A1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013206, upper bound: 0.0011334
IS_A2_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013206, upper bound: 0.0013881
IS_A2_B2_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011274, upper bound: 0.0010844
IS_A2_B2_A2_B1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011274, upper bound: 0.0010993
IS_A2_B2_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011108, upper bound: 0.0010178
IS_A2_B2_A2_B1_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011108, upper bound: 0.0010239
IS_A2_B2_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011274, upper bound: 0.0010844
IS_A2_B2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011274, upper bound: 0.0013818
IS_A2_B2_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011108, upper bound: 0.0013371
IS_A2_B2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011108, upper bound: 0.0013816
IS_A2_B2_A2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011868, upper bound: 0.0011471
IS_A2_B2_A2_B2_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011868, upper bound: 0.0011471
IS_A2_B2_A2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011443, upper bound: 0.0011436
IS_A2_B2_A2_B2_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0011443, upper bound: 0.0011436
IS_A2_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013931, upper bound: 0.0012008
IS_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013918, upper bound: 0.0011499
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013925, upper bound: 0.0013829
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.17
Output dim: 2, lower bound: -0.0013918, upper bound: 0.0013833

## BFS IS instance: IS_A1_B1_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010243, upper bound: 0.0010708
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010329, upper bound: 0.0010823
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0042067, -0.0040705, -0.0042067, -0.0040702, -0.0001365, 0.0001362
1: -0.0101952, -0.0086921, -0.0101961, -0.0086726, -0.0015226, 0.0015040
2: 0.9642289, 0.9660325, 0.9642278, 0.9660560, -0.0018271, 0.0018047
3: -0.0175363, -0.0042319, -0.0175440, -0.0040594, -0.0102883, 0.0101223
4: -0.0003712, 0.0006407, -0.0003843, 0.0006413, -0.0010125, 0.0010250
5: 0.0168952, 0.0184037, 0.0168820, 0.0184055, -0.0015103, 0.0015218
6: 0.0018975, 0.0036890, 0.0018942, 0.0036954, -0.0017979, 0.0017948
7: -0.0066815, -0.0024652, -0.0067262, -0.0024613, -0.0042202, 0.0042611
8: 0.0114284, 0.0141638, 0.0113929, 0.0141654, -0.0027370, 0.0027709
9: 0.0182796, 0.0231995, 0.0182158, 0.0232024, -0.0045665, 0.0046271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013404, upper bound: 0.0011986
time: 1.40 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013929, upper bound: 0.0013728
time: 1.86 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0042070, -0.0040668, -0.0042067, -0.0040703, -0.0001367, 0.0001398
1: -0.0102060, -0.0087047, -0.0101957, -0.0086797, -0.0015263, 0.0014910
2: 0.9642158, 0.9660175, 0.9642281, 0.9660474, -0.0018316, 0.0017894
3: -0.0176323, -0.0043437, -0.0175412, -0.0041227, -0.0103304, 0.0100964
4: -0.0003627, 0.0006480, -0.0003795, 0.0006411, -0.0010037, 0.0010275
5: 0.0169038, 0.0184252, 0.0168868, 0.0184049, -0.0015010, 0.0015384
6: 0.0018564, 0.0036848, 0.0018954, 0.0036930, -0.0018367, 0.0017894
7: -0.0066525, -0.0024180, -0.0067098, -0.0024627, -0.0041898, 0.0042918
8: 0.0114513, 0.0141835, 0.0114059, 0.0141648, -0.0027134, 0.0027776
9: 0.0183209, 0.0232351, 0.0182392, 0.0232014, -0.0045333, 0.0046396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013349, upper bound: 0.0011953
time: 1.60 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013928, upper bound: 0.0013728
time: 1.69 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0042067, -0.0040705, -0.0042080, -0.0040539, -0.0001528, 0.0001375
1: -0.0101952, -0.0086921, -0.0102445, -0.0086633, -0.0015318, 0.0015524
2: 0.9642289, 0.9660325, 0.9641697, 0.9660670, -0.0018381, 0.0018628
3: -0.0175363, -0.0042319, -0.0179725, -0.0039775, -0.0103962, 0.0105690
4: -0.0003712, 0.0006407, -0.0003905, 0.0006739, -0.0010450, 0.0010312
5: 0.0168952, 0.0184037, 0.0168757, 0.0185013, -0.0016060, 0.0015281
6: 0.0018975, 0.0036890, 0.0017107, 0.0036985, -0.0018010, 0.0019782
7: -0.0066815, -0.0024652, -0.0067474, -0.0022509, -0.0044306, 0.0042823
8: 0.0114284, 0.0141638, 0.0113760, 0.0142535, -0.0028251, 0.0027877
9: 0.0182796, 0.0231995, 0.0181855, 0.0233608, -0.0047259, 0.0046590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_A2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013146, upper bound: 0.0013801
time: 1.79 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013146, upper bound: 0.0013801
time: 1.53 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042070, -0.0040668, -0.0042080, -0.0040540, -0.0001530, 0.0001411
1: -0.0102060, -0.0087047, -0.0102441, -0.0086705, -0.0015355, 0.0015394
2: 0.9642158, 0.9660175, 0.9641701, 0.9660584, -0.0018426, 0.0018474
3: -0.0176323, -0.0043437, -0.0179697, -0.0040412, -0.0104386, 0.0105431
4: -0.0003627, 0.0006480, -0.0003857, 0.0006737, -0.0010363, 0.0010337
5: 0.0169038, 0.0184252, 0.0168806, 0.0185006, -0.0015968, 0.0015447
6: 0.0018564, 0.0036848, 0.0017120, 0.0036961, -0.0018397, 0.0019728
7: -0.0066525, -0.0024180, -0.0067309, -0.0022523, -0.0044003, 0.0043130
8: 0.0114513, 0.0141835, 0.0113891, 0.0142529, -0.0028015, 0.0027944
9: 0.0183209, 0.0232351, 0.0182091, 0.0233598, -0.0046928, 0.0046713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012950, upper bound: 0.0013801
time: 1.99 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0012950, upper bound: 0.0013801
time: 1.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0042080, -0.0040539, -0.0042067, -0.0040705, -0.0001375, 0.0001528
1: -0.0102445, -0.0086633, -0.0101952, -0.0086921, -0.0015524, 0.0015318
2: 0.9641697, 0.9660670, 0.9642289, 0.9660325, -0.0018628, 0.0018381
3: -0.0179725, -0.0039775, -0.0175363, -0.0042319, -0.0105690, 0.0103962
4: -0.0003905, 0.0006739, -0.0003712, 0.0006407, -0.0010312, 0.0010450
5: 0.0168757, 0.0185013, 0.0168952, 0.0184037, -0.0015281, 0.0016060
6: 0.0017107, 0.0036985, 0.0018975, 0.0036890, -0.0019782, 0.0018010
7: -0.0067474, -0.0022509, -0.0066815, -0.0024652, -0.0042823, 0.0044306
8: 0.0113760, 0.0142535, 0.0114284, 0.0141638, -0.0027877, 0.0028251
9: 0.0181855, 0.0233608, 0.0182796, 0.0231995, -0.0046590, 0.0047259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010963, upper bound: 0.0010475
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010963, upper bound: 0.0013729
time: 1.45 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0042080, -0.0040540, -0.0042070, -0.0040668, -0.0001411, 0.0001530
1: -0.0102441, -0.0086705, -0.0102060, -0.0087047, -0.0015394, 0.0015355
2: 0.9641701, 0.9660584, 0.9642158, 0.9660175, -0.0018474, 0.0018426
3: -0.0179697, -0.0040412, -0.0176323, -0.0043437, -0.0105431, 0.0104386
4: -0.0003857, 0.0006737, -0.0003627, 0.0006480, -0.0010337, 0.0010363
5: 0.0168806, 0.0185006, 0.0169038, 0.0184252, -0.0015447, 0.0015968
6: 0.0017120, 0.0036961, 0.0018564, 0.0036848, -0.0019728, 0.0018397
7: -0.0067309, -0.0022523, -0.0066525, -0.0024180, -0.0043130, 0.0044003
8: 0.0113891, 0.0142529, 0.0114513, 0.0141835, -0.0027944, 0.0028015
9: 0.0182091, 0.0233598, 0.0183209, 0.0232351, -0.0046713, 0.0046928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010700, upper bound: 0.0013726
time: 1.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0010700, upper bound: 0.0013728
time: 1.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0042080, -0.0040539, -0.0042029, -0.0041186, -0.0000894, 0.0001490
1: -0.0102445, -0.0086633, -0.0100525, -0.0085948, -0.0016497, 0.0013892
2: 0.9641697, 0.9660670, 0.9644000, 0.9661493, -0.0019796, 0.0016670
3: -0.0179725, -0.0039775, -0.0162738, -0.0033707, -0.0116811, 0.0093826
4: -0.0003905, 0.0006739, -0.0004367, 0.0005447, -0.0009352, 0.0011106
5: 0.0168757, 0.0185013, 0.0168290, 0.0181215, -0.0012458, 0.0016723
6: 0.0017107, 0.0036985, 0.0024380, 0.0037212, -0.0020104, 0.0012605
7: -0.0067474, -0.0022509, -0.0069047, -0.0030852, -0.0036622, 0.0046538
8: 0.0113760, 0.0142535, 0.0112513, 0.0139042, -0.0025282, 0.0030022
9: 0.0181855, 0.0233608, 0.0179611, 0.0227327, -0.0042151, 0.0050691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.78 + 597.57 = 601.35 seconds
