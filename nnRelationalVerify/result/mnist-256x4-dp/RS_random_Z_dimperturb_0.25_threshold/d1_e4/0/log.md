## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 8.477e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0007041, 0.0007041)
1: (-0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000936, 0.0000936)
2: (0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0008617, 0.0008617)
3: (1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002346, 0.0002346)
4: (-0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001269, 0.0001269)
5: (0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0005356, 0.0005356)
6: (-0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000429, 0.0000429)
7: (-0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0014350, 0.0014350)
8: (-0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0012823, 0.0012823)
9: (-0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005806, 0.0005806)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.38 + 1.37 = 2.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0001217, upper bound: 0.0001217

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001163, upper bound: 0.0001159
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001159, upper bound: 0.0001163
time: 0.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.10 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 3, lower bound: -0.0001163, upper bound: 0.0001159
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 3, lower bound: -0.0001159, upper bound: 0.0001163

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006879, 0.0006860
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000933, 0.0000933
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0008431, 0.0008418
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002262, 0.0002258
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001244, 0.0001245
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0005234, 0.0005220
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000412, 0.0000409
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013861, 0.0013900
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0012597, 0.0012603
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005726, 0.0005723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001019, upper bound: 0.0001119
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001120, upper bound: 0.0001006
time: 0.62 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006860, 0.0006879
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000933, 0.0000933
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0008418, 0.0008431
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002258, 0.0002262
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001245, 0.0001244
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0005220, 0.0005234
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000409, 0.0000412
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013900, 0.0013861
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0012603, 0.0012597
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005723, 0.0005726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001152, upper bound: 0.0001156
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001153, upper bound: 0.0001157
time: 0.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 3, lower bound: -0.0001019, upper bound: 0.0001119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 3, lower bound: -0.0001120, upper bound: 0.0001006
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 3, lower bound: -0.0001152, upper bound: 0.0001156
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 3, lower bound: -0.0001153, upper bound: 0.0001157

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006729, 0.0006494
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000862, 0.0000816
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0008227, 0.0007930
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002093, 0.0002127
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001164, 0.0001210
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0005118, 0.0004938
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000410, 0.0000406
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013341, 0.0013710
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011718, 0.0012199
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005509, 0.0005280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000963, upper bound: 0.0001083
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000961, upper bound: 0.0001082
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006513, 0.0006707
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000816, 0.0000862
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007943, 0.0008207
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002131, 0.0002089
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001208, 0.0001165
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004952, 0.0005102
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000408, 0.0000407
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013672, 0.0013380
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0012178, 0.0011723
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005283, 0.0005503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001084, upper bound: 0.0000957
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001084, upper bound: 0.0000959
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006764, 0.0006765
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000891, 0.0000888
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0008286, 0.0008276
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002273, 0.0002277
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001219, 0.0001222
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0005146, 0.0005146
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000410, 0.0000412
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013766, 0.0013748
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0012316, 0.0012345
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005597, 0.0005582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001000, upper bound: 0.0001114
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001112, upper bound: 0.0001010
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006750, 0.0006783
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000889, 0.0000891
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0008267, 0.0008299
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002273, 0.0002277
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001222, 0.0001219
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0005135, 0.0005159
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000410, 0.0000413
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013787, 0.0013725
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0012351, 0.0012311
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005580, 0.0005599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000997, upper bound: 0.0001115
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001112, upper bound: 0.0001014
time: 0.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.44 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0000963, upper bound: 0.0001083
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0000961, upper bound: 0.0001082
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0001084, upper bound: 0.0000957
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0001084, upper bound: 0.0000959
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0001000, upper bound: 0.0001114
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0001112, upper bound: 0.0001010
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0000997, upper bound: 0.0001115
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -0.0001112, upper bound: 0.0001014

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006580, 0.0006384
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000841, 0.0000804
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0008041, 0.0007788
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002091, 0.0002123
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001142, 0.0001182
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0005004, 0.0004854
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000404, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013134, 0.0013422
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011497, 0.0011914
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005377, 0.0005180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000957, upper bound: 0.0001076
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000955, upper bound: 0.0001076
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006729, 0.0006344
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000862, 0.0000795
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0008227, 0.0007744
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002089, 0.0002127
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001136, 0.0001210
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0005118, 0.0004824
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000410, 0.0000400
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013054, 0.0013710
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011433, 0.0012199
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005509, 0.0005148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000957, upper bound: 0.0001042
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000931, upper bound: 0.0001078
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006363, 0.0006590
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000795, 0.0000843
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007757, 0.0008057
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002129, 0.0002085
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001184, 0.0001137
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004838, 0.0005012
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000402, 0.0000400
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013395, 0.0013093
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011941, 0.0011439
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005151, 0.0005391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001080, upper bound: 0.0000926
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001043, upper bound: 0.0000953
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006513, 0.0006557
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000816, 0.0000841
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007943, 0.0008020
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002127, 0.0002089
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001180, 0.0001165
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004952, 0.0004988
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000408, 0.0000401
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013385, 0.0013380
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011894, 0.0011723
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005283, 0.0005371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001078, upper bound: 0.0000951
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001078, upper bound: 0.0000952
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006606, 0.0006394
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000815, 0.0000767
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0008066, 0.0007780
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002101, 0.0002144
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001138, 0.0001184
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0005024, 0.0004860
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000408, 0.0000409
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013243, 0.0013555
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011422, 0.0011911
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005363, 0.0005126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000997, upper bound: 0.0001073
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000968, upper bound: 0.0001111
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006393, 0.0006613
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000770, 0.0000814
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007790, 0.0008066
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002141, 0.0002106
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001183, 0.0001140
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004860, 0.0005028
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000410
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013579, 0.0013224
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011896, 0.0011451
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005141, 0.0005352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001076, upper bound: 0.0000954
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001076, upper bound: 0.0000955
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006591, 0.0006411
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000814, 0.0000769
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0008046, 0.0007802
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002102, 0.0002144
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001141, 0.0001180
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0005012, 0.0004874
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000408, 0.0000409
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013263, 0.0013540
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011456, 0.0011878
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005347, 0.0005143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0001078
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000950, upper bound: 0.0001078
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006379, 0.0006628
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000767, 0.0000815
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007771, 0.0008086
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002140, 0.0002106
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001186, 0.0001137
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004849, 0.0005040
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000411
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013593, 0.0013201
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011932, 0.0011416
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005123, 0.0005369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001076, upper bound: 0.0000955
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001076, upper bound: 0.0000957
time: 0.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0000957, upper bound: 0.0001076
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0000955, upper bound: 0.0001076
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0000957, upper bound: 0.0001042
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0000931, upper bound: 0.0001078
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0001080, upper bound: 0.0000926
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0001043, upper bound: 0.0000953
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0001078, upper bound: 0.0000951
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0001078, upper bound: 0.0000952
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0000997, upper bound: 0.0001073
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0000968, upper bound: 0.0001111
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0001076, upper bound: 0.0000954
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0001076, upper bound: 0.0000955
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0001078
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0000950, upper bound: 0.0001078
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0001076, upper bound: 0.0000955
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 3, lower bound: -0.0001076, upper bound: 0.0000957

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006471, 0.0006261
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000792, 0.0000751
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007890, 0.0007620
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002104, 0.0002136
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001114, 0.0001156
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004920, 0.0004759
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000404, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012987, 0.0013299
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011175, 0.0011627
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005229, 0.0005014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000952, upper bound: 0.0001035
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000927, upper bound: 0.0001071
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006456, 0.0006276
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000791, 0.0000755
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007869, 0.0007637
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002104, 0.0002137
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001117, 0.0001153
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004908, 0.0004770
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000404, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013011, 0.0013286
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011209, 0.0011591
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005212, 0.0005032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000950, upper bound: 0.0001035
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000926, upper bound: 0.0001071
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006618, 0.0006245
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000843, 0.0000779
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0008085, 0.0007619
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002078, 0.0002112
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001117, 0.0001189
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0005033, 0.0004749
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012879, 0.0013524
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011241, 0.0011979
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005406, 0.0005058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0001035
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000949, upper bound: 0.0001035
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006631, 0.0006234
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000846, 0.0000776
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0008102, 0.0007600
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002074, 0.0002116
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001114, 0.0001191
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0005043, 0.0004740
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012861, 0.0013535
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011211, 0.0012007
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005418, 0.0005044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000925, upper bound: 0.0001072
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000924, upper bound: 0.0001071
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006250, 0.0006492
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000776, 0.0000826
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007614, 0.0007932
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002117, 0.0002071
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001166, 0.0001115
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004751, 0.0004937
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000399, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013221, 0.0012913
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011749, 0.0011217
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005046, 0.0005301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001073, upper bound: 0.0000919
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001074, upper bound: 0.0000919
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006264, 0.0006482
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000779, 0.0000824
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007632, 0.0007920
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002113, 0.0002074
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001164, 0.0001118
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004762, 0.0004929
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000399, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013206, 0.0012918
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011725, 0.0011247
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005060, 0.0005288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001037, upper bound: 0.0000946
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001037, upper bound: 0.0000946
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006411, 0.0006434
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000769, 0.0000791
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007802, 0.0007849
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002140, 0.0002102
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001151, 0.0001141
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004874, 0.0004892
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000409, 0.0000402
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013246, 0.0013263
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011573, 0.0011456
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005143, 0.0005207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001073, upper bound: 0.0000921
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001037, upper bound: 0.0000947
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006394, 0.0006449
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000767, 0.0000792
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007780, 0.0007869
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002140, 0.0002101
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001154, 0.0001138
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004860, 0.0004904
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000409, 0.0000402
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013262, 0.0013243
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011607, 0.0011422
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005126, 0.0005223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001074, upper bound: 0.0000921
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001037, upper bound: 0.0000948
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006489, 0.0006290
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000793, 0.0000748
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007916, 0.0007648
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002090, 0.0002128
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001117, 0.0001161
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004934, 0.0004780
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000405, 0.0000406
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013062, 0.0013362
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011210, 0.0011673
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005251, 0.0005026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000948, upper bound: 0.0001037
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000946, upper bound: 0.0001037
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006502, 0.0006276
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000796, 0.0000745
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007934, 0.0007629
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002087, 0.0002132
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001114, 0.0001163
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004944, 0.0004770
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000405, 0.0000406
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013061, 0.0013375
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011180, 0.0011699
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005263, 0.0005012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000921, upper bound: 0.0001074
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000920, upper bound: 0.0001074
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006235, 0.0006475
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000746, 0.0000792
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007593, 0.0007895
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002137, 0.0002102
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001157, 0.0001110
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004740, 0.0004923
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000400, 0.0000403
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013307, 0.0012931
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011631, 0.0011146
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005000, 0.0005230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001071, upper bound: 0.0000924
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001035, upper bound: 0.0000949
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006393, 0.0006456
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000770, 0.0000791
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007790, 0.0007869
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002137, 0.0002106
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001153, 0.0001140
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004860, 0.0004908
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000404
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013286, 0.0013224
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011591, 0.0011451
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005141, 0.0005212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001071, upper bound: 0.0000926
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001035, upper bound: 0.0000950
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006434, 0.0006278
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000791, 0.0000754
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007849, 0.0007637
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002100, 0.0002140
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001116, 0.0001151
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004892, 0.0004771
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000402, 0.0000402
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013047, 0.0013246
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011209, 0.0011573
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005207, 0.0005031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000947, upper bound: 0.0001037
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000920, upper bound: 0.0001073
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006591, 0.0006254
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000814, 0.0000746
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0008046, 0.0007606
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002098, 0.0002144
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001111, 0.0001180
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0005012, 0.0004753
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000408, 0.0000403
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012970, 0.0013540
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011152, 0.0011878
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005347, 0.0005003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000946, upper bound: 0.0001037
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000919, upper bound: 0.0001073
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006221, 0.0006489
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000744, 0.0000793
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007574, 0.0007913
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002136, 0.0002102
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001160, 0.0001107
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004729, 0.0004934
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000400, 0.0000403
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013321, 0.0012908
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011666, 0.0011111
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004983, 0.0005248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001072, upper bound: 0.0000925
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001035, upper bound: 0.0000951
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006379, 0.0006471
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000767, 0.0000792
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007771, 0.0007890
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002136, 0.0002106
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001156, 0.0001137
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004849, 0.0004920
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000404
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013299, 0.0013201
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011627, 0.0011416
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005123, 0.0005229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001072, upper bound: 0.0000927
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001035, upper bound: 0.0000952
time: 0.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000952, upper bound: 0.0001035
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000927, upper bound: 0.0001071
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000950, upper bound: 0.0001035
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000926, upper bound: 0.0001071
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0001035
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000949, upper bound: 0.0001035
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000925, upper bound: 0.0001072
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000924, upper bound: 0.0001071
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001073, upper bound: 0.0000919
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001074, upper bound: 0.0000919
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001037, upper bound: 0.0000946
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001037, upper bound: 0.0000946
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001073, upper bound: 0.0000921
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001037, upper bound: 0.0000947
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001074, upper bound: 0.0000921
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001037, upper bound: 0.0000948
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000948, upper bound: 0.0001037
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000946, upper bound: 0.0001037
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000921, upper bound: 0.0001074
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000920, upper bound: 0.0001074
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001071, upper bound: 0.0000924
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001035, upper bound: 0.0000949
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001071, upper bound: 0.0000926
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001035, upper bound: 0.0000950
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000947, upper bound: 0.0001037
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000920, upper bound: 0.0001073
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000946, upper bound: 0.0001037
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0000919, upper bound: 0.0001073
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001072, upper bound: 0.0000925
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001035, upper bound: 0.0000951
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001072, upper bound: 0.0000927
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 3, lower bound: -0.0001035, upper bound: 0.0000952

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006354, 0.0006157
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000770, 0.0000732
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007740, 0.0007487
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002092, 0.0002121
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001093, 0.0001133
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004830, 0.0004679
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000402, 0.0000396
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012807, 0.0013108
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010963, 0.0011388
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005117, 0.0004914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000785, upper bound: 0.0000864
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000785, upper bound: 0.0000883
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006366, 0.0006149
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000773, 0.0000729
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007757, 0.0007474
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002088, 0.0002124
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001091, 0.0001136
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004840, 0.0004673
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000401, 0.0000396
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012800, 0.0013119
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010942, 0.0011415
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005129, 0.0004902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000886, upper bound: 0.0001037
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000889, upper bound: 0.0001036
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006338, 0.0006171
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000769, 0.0000735
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007720, 0.0007505
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002092, 0.0002122
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001096, 0.0001130
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004818, 0.0004690
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000401, 0.0000396
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012830, 0.0013098
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010998, 0.0011353
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005101, 0.0004932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000784, upper bound: 0.0000863
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000784, upper bound: 0.0000883
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006351, 0.0006164
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000772, 0.0000732
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007737, 0.0007493
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002089, 0.0002125
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001094, 0.0001133
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004828, 0.0004685
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000401, 0.0000396
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012824, 0.0013105
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010976, 0.0011379
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005112, 0.0004920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000763, upper bound: 0.0000898
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000761, upper bound: 0.0000915
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006511, 0.0006117
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000794, 0.0000725
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007937, 0.0007442
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002090, 0.0002125
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001087, 0.0001163
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004950, 0.0004649
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000408, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012727, 0.0013401
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010900, 0.0011692
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005257, 0.0004883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000785, upper bound: 0.0000864
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000785, upper bound: 0.0000883
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006496, 0.0006131
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000792, 0.0000727
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007916, 0.0007460
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002090, 0.0002126
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001090, 0.0001160
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004938, 0.0004660
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000408, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012750, 0.0013392
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010934, 0.0011658
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005241, 0.0004900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000784, upper bound: 0.0000863
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000784, upper bound: 0.0000883
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006524, 0.0006106
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000796, 0.0000722
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007954, 0.0007425
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002086, 0.0002128
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001084, 0.0001166
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004960, 0.0004641
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012714, 0.0013412
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010870, 0.0011720
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005269, 0.0004869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000762, upper bound: 0.0000899
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000762, upper bound: 0.0000916
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006509, 0.0006119
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000795, 0.0000724
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007933, 0.0007441
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002087, 0.0002129
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001087, 0.0001162
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004949, 0.0004651
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012733, 0.0013399
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010904, 0.0011684
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005252, 0.0004886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000847, upper bound: 0.0001030
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000882, upper bound: 0.0001021
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006135, 0.0006364
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000724, 0.0000773
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007455, 0.0007757
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002130, 0.0002083
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001136, 0.0001088
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004663, 0.0004838
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000400, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013074, 0.0012785
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011407, 0.0010910
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004888, 0.0005126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000915, upper bound: 0.0000757
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000901, upper bound: 0.0000761
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006119, 0.0006377
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000722, 0.0000774
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007432, 0.0007773
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002130, 0.0002082
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001138, 0.0001085
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004650, 0.0004848
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000400, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013092, 0.0012768
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011442, 0.0010875
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004872, 0.0005143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001036, upper bound: 0.0000886
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001038, upper bound: 0.0000883
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006150, 0.0006356
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000727, 0.0000770
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007473, 0.0007744
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002126, 0.0002086
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001134, 0.0001091
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004674, 0.0004832
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000400, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013060, 0.0012789
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011388, 0.0010940
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004903, 0.0005114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000986, upper bound: 0.0000903
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000995, upper bound: 0.0000866
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006132, 0.0006367
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000725, 0.0000772
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007451, 0.0007761
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002126, 0.0002086
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001136, 0.0001087
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004660, 0.0004841
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000400, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013077, 0.0012769
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011418, 0.0010905
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004885, 0.0005130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000987, upper bound: 0.0000904
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000995, upper bound: 0.0000866
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006293, 0.0006329
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000747, 0.0000772
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007652, 0.0007717
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002129, 0.0002087
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001130, 0.0001118
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004783, 0.0004812
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000406, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013066, 0.0013078
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011361, 0.0011214
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005029, 0.0005107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000915, upper bound: 0.0000758
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000901, upper bound: 0.0000761
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006307, 0.0006319
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000750, 0.0000769
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007670, 0.0007700
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002125, 0.0002090
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001128, 0.0001120
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004794, 0.0004804
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000406, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013054, 0.0013083
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011336, 0.0011244
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005043, 0.0005095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000986, upper bound: 0.0000904
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000995, upper bound: 0.0000866
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006276, 0.0006344
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000745, 0.0000773
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007629, 0.0007737
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002128, 0.0002087
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001134, 0.0001114
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004770, 0.0004824
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000406, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013081, 0.0013061
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011395, 0.0011180
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005012, 0.0005123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001025, upper bound: 0.0000878
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001032, upper bound: 0.0000839
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006290, 0.0006332
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000748, 0.0000770
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007648, 0.0007719
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002124, 0.0002090
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001131, 0.0001117
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004780, 0.0004814
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000406, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013068, 0.0013062
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011368, 0.0011210
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005026, 0.0005110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000987, upper bound: 0.0000906
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000995, upper bound: 0.0000866
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006332, 0.0006164
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000770, 0.0000732
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007719, 0.0007489
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002087, 0.0002124
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001093, 0.0001131
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004814, 0.0004684
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000399, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012853, 0.0013068
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010964, 0.0011368
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005110, 0.0004913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000913, upper bound: 0.0001000
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000914, upper bound: 0.0000998
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006489, 0.0006132
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000793, 0.0000725
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007916, 0.0007451
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002086, 0.0002128
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001087, 0.0001161
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004934, 0.0004660
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000405, 0.0000400
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012769, 0.0013362
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010905, 0.0011673
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005251, 0.0004885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000784, upper bound: 0.0000868
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000783, upper bound: 0.0000883
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006344, 0.0006156
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000773, 0.0000729
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007737, 0.0007479
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002084, 0.0002128
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001091, 0.0001134
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004824, 0.0004678
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000399, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012839, 0.0013081
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010938, 0.0011395
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005123, 0.0004900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000761, upper bound: 0.0000902
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000758, upper bound: 0.0000916
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006502, 0.0006119
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000796, 0.0000722
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007934, 0.0007432
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002082, 0.0002132
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001085, 0.0001163
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004944, 0.0004650
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000405, 0.0000400
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012768, 0.0013375
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010875, 0.0011699
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005263, 0.0004872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0001038
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000886, upper bound: 0.0001036
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006119, 0.0006371
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000724, 0.0000773
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007441, 0.0007762
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002125, 0.0002087
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001136, 0.0001087
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004651, 0.0004843
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000397, 0.0000400
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013127, 0.0012733
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011419, 0.0010904
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004886, 0.0005130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001036, upper bound: 0.0000888
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001037, upper bound: 0.0000885
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006131, 0.0006363
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000727, 0.0000770
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007460, 0.0007751
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002122, 0.0002090
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001135, 0.0001090
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004660, 0.0004837
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000397, 0.0000400
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013116, 0.0012750
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011396, 0.0010934
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004900, 0.0005118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000984, upper bound: 0.0000908
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000993, upper bound: 0.0000872
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006277, 0.0006351
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000747, 0.0000772
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007638, 0.0007737
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002125, 0.0002091
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001133, 0.0001117
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004771, 0.0004828
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000404, 0.0000401
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013105, 0.0013026
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011379, 0.0011209
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005027, 0.0005112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000915, upper bound: 0.0000761
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000898, upper bound: 0.0000763
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006288, 0.0006338
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000750, 0.0000769
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007657, 0.0007720
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002122, 0.0002095
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001130, 0.0001120
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004780, 0.0004818
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000404, 0.0000401
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013098, 0.0013044
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011353, 0.0011239
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005041, 0.0005101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000984, upper bound: 0.0000908
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000993, upper bound: 0.0000872
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006319, 0.0006173
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000769, 0.0000735
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007700, 0.0007505
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002088, 0.0002125
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001096, 0.0001128
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004804, 0.0004691
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000399, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012867, 0.0013054
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010997, 0.0011336
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005095, 0.0004931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000911, upper bound: 0.0001000
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000912, upper bound: 0.0000997
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006329, 0.0006166
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000772, 0.0000732
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007717, 0.0007493
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002085, 0.0002129
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001094, 0.0001130
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004812, 0.0004686
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000399, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012852, 0.0013066
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010968, 0.0011361
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005107, 0.0004918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000885, upper bound: 0.0001037
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000887, upper bound: 0.0001036
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006476, 0.0006150
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000792, 0.0000727
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007897, 0.0007473
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002086, 0.0002129
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001091, 0.0001157
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004924, 0.0004674
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000405, 0.0000400
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012789, 0.0013347
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010940, 0.0011641
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005236, 0.0004903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000911, upper bound: 0.0001000
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000912, upper bound: 0.0000997
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006487, 0.0006135
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000795, 0.0000724
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007914, 0.0007455
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002083, 0.0002133
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001088, 0.0001160
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004932, 0.0004663
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000405, 0.0000400
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012785, 0.0013359
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010910, 0.0011666
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005247, 0.0004888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000839, upper bound: 0.0001032
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000876, upper bound: 0.0001024
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006106, 0.0006384
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000722, 0.0000774
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007425, 0.0007781
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002124, 0.0002086
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001140, 0.0001084
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004641, 0.0004854
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000397, 0.0000400
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013141, 0.0012714
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011454, 0.0010870
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004869, 0.0005148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001022, upper bound: 0.0000882
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001030, upper bound: 0.0000846
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006117, 0.0006375
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000725, 0.0000772
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007442, 0.0007769
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002121, 0.0002090
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001137, 0.0001087
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004649, 0.0004847
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000397, 0.0000400
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013129, 0.0012727
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011428, 0.0010900
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004883, 0.0005135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0000785
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000864, upper bound: 0.0000785
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006264, 0.0006366
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000745, 0.0000773
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007622, 0.0007757
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002124, 0.0002090
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001136, 0.0001114
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004761, 0.0004840
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000404, 0.0000401
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013119, 0.0013008
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011415, 0.0011175
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005009, 0.0005129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000916, upper bound: 0.0000762
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000899, upper bound: 0.0000763
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006274, 0.0006354
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000748, 0.0000770
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007638, 0.0007740
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002121, 0.0002094
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001133, 0.0001117
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004769, 0.0004830
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000403, 0.0000402
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013108, 0.0013021
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011388, 0.0011204
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005023, 0.0005117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000998, upper bound: 0.0000916
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000999, upper bound: 0.0000914
time: 0.66 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000785, upper bound: 0.0000864
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000785, upper bound: 0.0000883
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000886, upper bound: 0.0001037
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000889, upper bound: 0.0001036
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000784, upper bound: 0.0000863
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000784, upper bound: 0.0000883
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000763, upper bound: 0.0000898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000761, upper bound: 0.0000915
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000785, upper bound: 0.0000864
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000785, upper bound: 0.0000883
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000784, upper bound: 0.0000863
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000784, upper bound: 0.0000883
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000762, upper bound: 0.0000899
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000762, upper bound: 0.0000916
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000847, upper bound: 0.0001030
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000882, upper bound: 0.0001021
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000915, upper bound: 0.0000757
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000901, upper bound: 0.0000761
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0001036, upper bound: 0.0000886
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0001038, upper bound: 0.0000883
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000986, upper bound: 0.0000903
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000995, upper bound: 0.0000866
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000987, upper bound: 0.0000904
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000995, upper bound: 0.0000866
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000915, upper bound: 0.0000758
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000901, upper bound: 0.0000761
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000986, upper bound: 0.0000904
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000995, upper bound: 0.0000866
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0001025, upper bound: 0.0000878
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0001032, upper bound: 0.0000839
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000987, upper bound: 0.0000906
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000995, upper bound: 0.0000866
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000913, upper bound: 0.0001000
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000914, upper bound: 0.0000998
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000784, upper bound: 0.0000868
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000783, upper bound: 0.0000883
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000761, upper bound: 0.0000902
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000758, upper bound: 0.0000916
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0001038
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000886, upper bound: 0.0001036
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0001036, upper bound: 0.0000888
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0001037, upper bound: 0.0000885
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000984, upper bound: 0.0000908
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000993, upper bound: 0.0000872
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000915, upper bound: 0.0000761
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000898, upper bound: 0.0000763
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000984, upper bound: 0.0000908
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000993, upper bound: 0.0000872
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000911, upper bound: 0.0001000
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000912, upper bound: 0.0000997
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000885, upper bound: 0.0001037
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000887, upper bound: 0.0001036
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000911, upper bound: 0.0001000
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000912, upper bound: 0.0000997
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000839, upper bound: 0.0001032
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000876, upper bound: 0.0001024
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0001022, upper bound: 0.0000882
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0001030, upper bound: 0.0000846
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0000785
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000864, upper bound: 0.0000785
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000916, upper bound: 0.0000762
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000899, upper bound: 0.0000763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000998, upper bound: 0.0000916
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 3, lower bound: -0.0000999, upper bound: 0.0000914

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006017, 0.0005851
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000753, 0.0000715
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007387, 0.0007166
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001838, 0.0001878
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001056, 0.0001092
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004579, 0.0004451
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000347, 0.0000354
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011861, 0.0012043
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010664, 0.0011049
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005015, 0.0004824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000713, upper bound: 0.0000823
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000746, upper bound: 0.0000799
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006048, 0.0005854
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000755, 0.0000714
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007420, 0.0007159
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001849, 0.0001887
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001054, 0.0001096
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004602, 0.0004453
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000359, 0.0000346
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011885, 0.0012162
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010650, 0.0011088
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005027, 0.0004820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000850
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000752, upper bound: 0.0000849
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006276, 0.0006030
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000747, 0.0000696
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007637, 0.0007316
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002029, 0.0002071
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001065, 0.0001116
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004770, 0.0004582
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000401, 0.0000396
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012644, 0.0012993
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010664, 0.0011204
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005024, 0.0004767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000729, upper bound: 0.0000865
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000724, upper bound: 0.0000884
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006258, 0.0006059
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000740, 0.0000703
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007611, 0.0007353
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002035, 0.0002065
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001072, 0.0001112
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004756, 0.0004603
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000401, 0.0000396
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012674, 0.0012981
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010731, 0.0011154
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005000, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000730, upper bound: 0.0000858
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000727, upper bound: 0.0000883
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005998, 0.0005865
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000751, 0.0000718
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007364, 0.0007184
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001839, 0.0001879
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001059, 0.0001088
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004565, 0.0004462
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000347, 0.0000354
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011885, 0.0012021
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010698, 0.0011012
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004999, 0.0004842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000713, upper bound: 0.0000823
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000745, upper bound: 0.0000796
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006032, 0.0005868
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000754, 0.0000718
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007399, 0.0007179
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001849, 0.0001884
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001058, 0.0001093
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004590, 0.0004463
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000359, 0.0000346
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011907, 0.0012153
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010685, 0.0011054
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005011, 0.0004838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000749, upper bound: 0.0000850
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000751, upper bound: 0.0000849
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006008, 0.0005859
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000754, 0.0000715
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007376, 0.0007172
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001831, 0.0001882
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001057, 0.0001090
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004572, 0.0004457
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000347, 0.0000354
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011878, 0.0012023
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010677, 0.0011035
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005009, 0.0004830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000858
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000722, upper bound: 0.0000830
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006045, 0.0005863
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000757, 0.0000715
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007416, 0.0007170
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001846, 0.0001893
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001056, 0.0001095
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004600, 0.0004459
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000358, 0.0000346
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011901, 0.0012160
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010665, 0.0011080
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005022, 0.0004827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000875
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000721, upper bound: 0.0000871
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006182, 0.0005811
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000776, 0.0000708
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007592, 0.0007121
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001838, 0.0001882
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001050, 0.0001122
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004705, 0.0004421
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000354, 0.0000355
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011782, 0.0012359
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010600, 0.0011361
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005158, 0.0004793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000828
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000751, upper bound: 0.0000820
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006205, 0.0005811
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000778, 0.0000707
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007617, 0.0007114
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001847, 0.0001888
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001048, 0.0001126
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004722, 0.0004421
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000365, 0.0000347
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011761, 0.0012456
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010584, 0.0011393
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005167, 0.0004788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000708, upper bound: 0.0000843
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000745, upper bound: 0.0000838
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006163, 0.0005825
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000775, 0.0000710
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007568, 0.0007139
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001838, 0.0001883
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001053, 0.0001119
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004691, 0.0004432
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000354, 0.0000355
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011804, 0.0012337
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010635, 0.0011324
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005142, 0.0004811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000713, upper bound: 0.0000823
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000744, upper bound: 0.0000795
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006190, 0.0005830
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000777, 0.0000710
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007596, 0.0007137
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001848, 0.0001886
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001051, 0.0001123
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004710, 0.0004435
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000365, 0.0000347
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011786, 0.0012446
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010622, 0.0011359
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005152, 0.0004806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000749, upper bound: 0.0000850
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000751, upper bound: 0.0000849
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006191, 0.0005800
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000779, 0.0000705
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007605, 0.0007104
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001828, 0.0001885
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001047, 0.0001125
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004712, 0.0004413
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000355, 0.0000355
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011768, 0.0012362
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010571, 0.0011384
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005169, 0.0004779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000729, upper bound: 0.0000865
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000729, upper bound: 0.0000857
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006218, 0.0005805
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000781, 0.0000704
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007633, 0.0007101
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001844, 0.0001895
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001046, 0.0001129
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004732, 0.0004415
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000365, 0.0000347
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011756, 0.0012467
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010559, 0.0011420
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005180, 0.0004775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000876
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000721, upper bound: 0.0000872
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003791, 0.0003083
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000695, 0.0000613
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004762, 0.0003912
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001213, 0.0001294
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000604, 0.0000726
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002894, 0.0002357
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000173, 0.0000149
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005497, 0.0006939
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006296, 0.0007508
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003520, 0.0002974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000812, upper bound: 0.0000995
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000812, upper bound: 0.0000994
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003471, 0.0003335
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000681, 0.0000624
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004400, 0.0004194
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001250, 0.0001248
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000640, 0.0000678
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002653, 0.0002546
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000161, 0.0000152
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0006102, 0.0006175
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006633, 0.0007064
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003335, 0.0003110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000842, upper bound: 0.0000987
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000846, upper bound: 0.0000987
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005836, 0.0006058
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000706, 0.0000758
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007140, 0.0007436
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001892, 0.0001840
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001099, 0.0001051
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004439, 0.0004610
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000347, 0.0000355
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012128, 0.0011821
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011108, 0.0010602
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004795, 0.0005036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000871, upper bound: 0.0000715
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000875, upper bound: 0.0000676
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005830, 0.0006022
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000707, 0.0000755
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007135, 0.0007396
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001888, 0.0001827
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001094, 0.0001051
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004435, 0.0004583
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000357, 0.0000346
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012027, 0.0011839
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011070, 0.0010610
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004799, 0.0005025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000837, upper bound: 0.0000720
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000862, upper bound: 0.0000686
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006028, 0.0006260
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000696, 0.0000742
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007312, 0.0007619
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002070, 0.0002029
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001114, 0.0001065
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004580, 0.0004759
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000399, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012937, 0.0012642
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011177, 0.0010664
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004767, 0.0005012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0000724
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000857, upper bound: 0.0000729
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006003, 0.0006286
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000688, 0.0000749
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007278, 0.0007653
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002076, 0.0002023
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001119, 0.0001059
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004561, 0.0004779
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000399, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012966, 0.0012617
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011230, 0.0010598
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004734, 0.0005039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000884, upper bound: 0.0000722
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000866, upper bound: 0.0000728
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003387, 0.0003320
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000628, 0.0000658
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004256, 0.0004215
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001248, 0.0001250
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000650, 0.0000648
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002586, 0.0002538
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000157, 0.0000149
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005824, 0.0006226
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006780, 0.0006707
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003140, 0.0003202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0000870
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0000868
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003113, 0.0003596
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000616, 0.0000671
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0003944, 0.0004526
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001290, 0.0001196
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000691, 0.0000607
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002380, 0.0002746
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000151, 0.0000159
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0006596, 0.0005553
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0007163, 0.0006332
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0002991, 0.0003361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000843, upper bound: 0.0000700
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000827, upper bound: 0.0000711
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003377, 0.0003331
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000626, 0.0000660
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004241, 0.0004232
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001248, 0.0001249
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000653, 0.0000645
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002578, 0.0002547
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000157, 0.0000149
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005841, 0.0006214
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006810, 0.0006679
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003126, 0.0003218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0000871
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0000870
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003096, 0.0003606
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000613, 0.0000672
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0003922, 0.0004537
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001289, 0.0001196
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000693, 0.0000604
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002366, 0.0002753
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000151, 0.0000159
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0006608, 0.0005533
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0007186, 0.0006297
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0002973, 0.0003375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000843, upper bound: 0.0000699
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000827, upper bound: 0.0000711
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006001, 0.0006024
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000730, 0.0000757
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007345, 0.0007396
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001892, 0.0001844
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001093, 0.0001082
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004565, 0.0004584
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000354, 0.0000356
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012120, 0.0012137
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011062, 0.0010914
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004938, 0.0005017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000871, upper bound: 0.0000716
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000875, upper bound: 0.0000676
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005987, 0.0005995
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000730, 0.0000754
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007332, 0.0007363
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001886, 0.0001829
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001089, 0.0001080
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004555, 0.0004563
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000364, 0.0000347
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011976, 0.0012132
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011024, 0.0010915
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004939, 0.0005005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000837, upper bound: 0.0000720
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000862, upper bound: 0.0000686
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003617, 0.0003282
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000651, 0.0000658
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004533, 0.0004171
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001261, 0.0001255
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000644, 0.0000689
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002760, 0.0002510
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000172, 0.0000150
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005818, 0.0006716
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006728, 0.0007108
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003318, 0.0003184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000837, upper bound: 0.0000743
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000807, upper bound: 0.0000743
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003269, 0.0003499
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000636, 0.0000668
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004137, 0.0004416
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001288, 0.0001197
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000677, 0.0000636
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002498, 0.0002673
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000160, 0.0000152
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0006306, 0.0005859
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0007027, 0.0006624
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003125, 0.0003306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000843, upper bound: 0.0000700
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000827, upper bound: 0.0000711
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003608, 0.0003308
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000645, 0.0000661
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004514, 0.0004208
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001267, 0.0001251
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000650, 0.0000684
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002753, 0.0002530
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000174, 0.0000150
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005845, 0.0006744
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006787, 0.0007049
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003286, 0.0003211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000990, upper bound: 0.0000844
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000990, upper bound: 0.0000842
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003238, 0.0003525
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000631, 0.0000673
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004096, 0.0004452
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001291, 0.0001191
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000683, 0.0000630
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002474, 0.0002693
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000160, 0.0000151
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0006322, 0.0005837
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0007096, 0.0006560
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003094, 0.0003341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000876, upper bound: 0.0000675
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000862, upper bound: 0.0000685
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003606, 0.0003296
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000649, 0.0000659
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004518, 0.0004191
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001260, 0.0001254
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000647, 0.0000686
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002752, 0.0002520
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000172, 0.0000150
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005832, 0.0006704
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006760, 0.0007079
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003304, 0.0003199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0000872
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0000870
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003252, 0.0003510
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000634, 0.0000669
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004114, 0.0004430
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001288, 0.0001197
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000679, 0.0000633
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002485, 0.0002681
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000159, 0.0000152
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0006316, 0.0005838
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0007058, 0.0006589
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003108, 0.0003319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000843, upper bound: 0.0000699
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000827, upper bound: 0.0000711
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006241, 0.0006043
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000744, 0.0000698
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007599, 0.0007331
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002027, 0.0002071
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001068, 0.0001111
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004744, 0.0004591
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000398, 0.0000398
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012708, 0.0012943
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010689, 0.0011157
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005006, 0.0004778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000833, upper bound: 0.0000958
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000870, upper bound: 0.0000952
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006218, 0.0006073
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000738, 0.0000706
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007570, 0.0007369
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002034, 0.0002065
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001074, 0.0001107
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004726, 0.0004614
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000399, 0.0000398
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012728, 0.0012913
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010753, 0.0011103
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004979, 0.0004809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000831, upper bound: 0.0000956
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000872, upper bound: 0.0000951
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006168, 0.0005826
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000776, 0.0000708
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007576, 0.0007130
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001836, 0.0001885
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001050, 0.0001121
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004694, 0.0004432
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000354, 0.0000357
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011823, 0.0012310
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010606, 0.0011351
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005154, 0.0004796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000711, upper bound: 0.0000827
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000743, upper bound: 0.0000810
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006184, 0.0005827
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000778, 0.0000707
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007596, 0.0007129
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001843, 0.0001887
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001049, 0.0001124
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004706, 0.0004432
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000362, 0.0000347
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011802, 0.0012416
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010589, 0.0011373
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005161, 0.0004790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000749, upper bound: 0.0000850
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000849
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006013, 0.0005851
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000755, 0.0000711
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007385, 0.0007158
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001828, 0.0001885
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001054, 0.0001092
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004576, 0.0004450
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000347, 0.0000356
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011893, 0.0011999
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010639, 0.0011062
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005022, 0.0004811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000685, upper bound: 0.0000862
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000838
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006038, 0.0005856
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000759, 0.0000711
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007416, 0.0007157
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001841, 0.0001893
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001053, 0.0001097
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004596, 0.0004454
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000356, 0.0000346
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011906, 0.0012136
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010628, 0.0011095
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005033, 0.0004808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000723, upper bound: 0.0000884
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000725, upper bound: 0.0000883
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006409, 0.0006003
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000771, 0.0000688
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007811, 0.0007278
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002023, 0.0002079
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001059, 0.0001143
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004873, 0.0004561
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000405, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012617, 0.0013245
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010598, 0.0011486
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005158, 0.0004734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000806, upper bound: 0.0000996
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000840, upper bound: 0.0000990
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006386, 0.0006028
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000765, 0.0000696
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007780, 0.0007312
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002029, 0.0002072
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001065, 0.0001138
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004855, 0.0004580
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000405, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012642, 0.0013215
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010664, 0.0011431
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005130, 0.0004767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000805, upper bound: 0.0000995
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000843, upper bound: 0.0000990
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006029, 0.0006256
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000698, 0.0000740
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007321, 0.0007612
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002066, 0.0002033
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001112, 0.0001068
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004581, 0.0004755
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000397, 0.0000400
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012982, 0.0012607
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011158, 0.0010693
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004782, 0.0005001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000987, upper bound: 0.0000846
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000994, upper bound: 0.0000812
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005997, 0.0006280
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000692, 0.0000747
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007282, 0.0007642
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002072, 0.0002027
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001117, 0.0001061
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004557, 0.0004774
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000397, 0.0000400
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0013001, 0.0012566
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011208, 0.0010625
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004749, 0.0005026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0000723
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000864, upper bound: 0.0000729
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003347, 0.0003327
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000628, 0.0000658
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004211, 0.0004222
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001242, 0.0001254
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000651, 0.0000643
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002556, 0.0002543
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000151, 0.0000152
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005880, 0.0006100
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006788, 0.0006671
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003132, 0.0003206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000836, upper bound: 0.0000744
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000795, upper bound: 0.0000744
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003094, 0.0003636
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000616, 0.0000671
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0003931, 0.0004568
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001285, 0.0001206
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000696, 0.0000607
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002366, 0.0002776
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000149, 0.0000167
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0006722, 0.0005514
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0007206, 0.0006326
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0002989, 0.0003375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000955, upper bound: 0.0000839
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000957, upper bound: 0.0000839
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005987, 0.0006045
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000730, 0.0000757
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007330, 0.0007416
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001893, 0.0001848
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001095, 0.0001080
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004554, 0.0004600
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000354, 0.0000358
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012160, 0.0012098
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011080, 0.0010908
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004936, 0.0005022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0000727
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0000723
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005971, 0.0006008
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000730, 0.0000754
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007317, 0.0007376
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001882, 0.0001831
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001090, 0.0001080
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004543, 0.0004572
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000361, 0.0000347
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012023, 0.0012080
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011035, 0.0010909
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004937, 0.0005009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000830, upper bound: 0.0000722
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000858, upper bound: 0.0000689
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003577, 0.0003302
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000652, 0.0000658
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004488, 0.0004191
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001255, 0.0001259
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000647, 0.0000684
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002730, 0.0002524
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000166, 0.0000153
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005862, 0.0006591
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006745, 0.0007072
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003311, 0.0003189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000950, upper bound: 0.0000873
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000950, upper bound: 0.0000870
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003250, 0.0003545
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000636, 0.0000668
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004124, 0.0004464
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001285, 0.0001207
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000682, 0.0000635
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002485, 0.0002707
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000157, 0.0000159
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0006474, 0.0005820
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0007064, 0.0006618
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003123, 0.0003317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000955, upper bound: 0.0000838
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000957, upper bound: 0.0000839
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006228, 0.0006054
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000743, 0.0000702
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007580, 0.0007350
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002028, 0.0002071
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001071, 0.0001108
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004734, 0.0004600
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000398, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012724, 0.0012928
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010728, 0.0011125
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004991, 0.0004798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000830
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000748, upper bound: 0.0000850
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006202, 0.0006083
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000737, 0.0000710
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007549, 0.0007384
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002034, 0.0002066
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001076, 0.0001103
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004715, 0.0004622
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000398, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012741, 0.0012898
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010785, 0.0011069
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004963, 0.0004826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000832, upper bound: 0.0000955
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000870, upper bound: 0.0000951
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006239, 0.0006049
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000746, 0.0000700
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007596, 0.0007339
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002026, 0.0002075
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001069, 0.0001111
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004743, 0.0004596
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000398, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012712, 0.0012940
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010704, 0.0011150
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005002, 0.0004786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000807, upper bound: 0.0000996
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000842, upper bound: 0.0000989
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006214, 0.0006075
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000739, 0.0000707
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007564, 0.0007373
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002031, 0.0002069
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001074, 0.0001105
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004724, 0.0004616
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000398, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012726, 0.0012908
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010757, 0.0011091
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004973, 0.0004813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000729, upper bound: 0.0000858
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000725, upper bound: 0.0000883
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006384, 0.0006032
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000768, 0.0000694
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007775, 0.0007317
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002026, 0.0002076
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001065, 0.0001138
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004853, 0.0004583
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000405, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012637, 0.0013218
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010661, 0.0011427
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005131, 0.0004764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000830
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000748, upper bound: 0.0000850
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006359, 0.0006059
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000761, 0.0000701
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007744, 0.0007353
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002033, 0.0002070
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001071, 0.0001133
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004834, 0.0004604
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000405, 0.0000399
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012664, 0.0013188
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010728, 0.0011373
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005104, 0.0004798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000819
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000849
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003744, 0.0003099
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000695, 0.0000613
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004712, 0.0003927
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001203, 0.0001297
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000604, 0.0000720
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002859, 0.0002369
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000167, 0.0000151
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005549, 0.0006801
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006301, 0.0007471
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003507, 0.0002976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000862
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000875
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003449, 0.0003387
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000681, 0.0000624
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004381, 0.0004248
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001247, 0.0001255
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000646, 0.0000676
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002637, 0.0002585
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000158, 0.0000159
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0006264, 0.0006135
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006677, 0.0007045
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003329, 0.0003122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000840, upper bound: 0.0000989
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000842, upper bound: 0.0000989
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003323, 0.0003348
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000621, 0.0000663
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004179, 0.0004252
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001249, 0.0001250
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000656, 0.0000638
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002537, 0.0002560
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000152, 0.0000152
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005904, 0.0006090
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006845, 0.0006610
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003096, 0.0003236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000988, upper bound: 0.0000847
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000988, upper bound: 0.0000844
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003070, 0.0003654
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000610, 0.0000676
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0003896, 0.0004591
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001288, 0.0001200
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000700, 0.0000601
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002347, 0.0002789
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000149, 0.0000166
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0006738, 0.0005478
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0007256, 0.0006262
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0002957, 0.0003403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000876, upper bound: 0.0000684
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000859, upper bound: 0.0000688
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005811, 0.0006069
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000707, 0.0000757
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007114, 0.0007448
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001886, 0.0001847
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001100, 0.0001048
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004421, 0.0004619
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000347, 0.0000358
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012183, 0.0011761
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011128, 0.0010584
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004788, 0.0005046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000837, upper bound: 0.0000745
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000843, upper bound: 0.0000708
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005811, 0.0006041
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000708, 0.0000754
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007121, 0.0007414
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001879, 0.0001838
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001096, 0.0001050
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004421, 0.0004597
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000355, 0.0000346
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012082, 0.0011782
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011093, 0.0010600
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004793, 0.0005036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000820, upper bound: 0.0000751
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000828, upper bound: 0.0000750
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005970, 0.0006060
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000727, 0.0000759
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007306, 0.0007436
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001894, 0.0001847
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001099, 0.0001076
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004541, 0.0004612
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000354, 0.0000359
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012173, 0.0012072
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011116, 0.0010871
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004918, 0.0005039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0000727
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000884, upper bound: 0.0000724
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005958, 0.0006026
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000727, 0.0000755
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007301, 0.0007400
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001881, 0.0001830
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001094, 0.0001077
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004533, 0.0004586
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000361, 0.0000347
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012046, 0.0012062
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011072, 0.0010875
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004920, 0.0005027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000832, upper bound: 0.0000722
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000859, upper bound: 0.0000688
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006182, 0.0006247
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000723, 0.0000738
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007516, 0.0007597
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002063, 0.0002041
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001110, 0.0001097
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004698, 0.0004748
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000403, 0.0000401
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012973, 0.0012891
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011132, 0.0010991
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004919, 0.0004990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000849, upper bound: 0.0000752
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000820, upper bound: 0.0000752
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006151, 0.0006263
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000715, 0.0000745
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007475, 0.0007620
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002067, 0.0002033
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001114, 0.0001090
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004675, 0.0004761
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000403, 0.0000401
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012982, 0.0012847
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011176, 0.0010919
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004884, 0.0005012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0000871
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000958, upper bound: 0.0000839
time: 0.66 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 6.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000713, upper bound: 0.0000823
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000746, upper bound: 0.0000799
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000850
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000752, upper bound: 0.0000849
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000729, upper bound: 0.0000865
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000724, upper bound: 0.0000884
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000730, upper bound: 0.0000858
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000727, upper bound: 0.0000883
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000713, upper bound: 0.0000823
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000745, upper bound: 0.0000796
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000749, upper bound: 0.0000850
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000751, upper bound: 0.0000849
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000858
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000722, upper bound: 0.0000830
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000875
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000721, upper bound: 0.0000871
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000828
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000751, upper bound: 0.0000820
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000708, upper bound: 0.0000843
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000745, upper bound: 0.0000838
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000713, upper bound: 0.0000823
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000744, upper bound: 0.0000795
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000749, upper bound: 0.0000850
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000751, upper bound: 0.0000849
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000729, upper bound: 0.0000865
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000729, upper bound: 0.0000857
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000876
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000721, upper bound: 0.0000872
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000812, upper bound: 0.0000995
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000812, upper bound: 0.0000994
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000842, upper bound: 0.0000987
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000846, upper bound: 0.0000987
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000871, upper bound: 0.0000715
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000875, upper bound: 0.0000676
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000837, upper bound: 0.0000720
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000862, upper bound: 0.0000686
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0000724
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000857, upper bound: 0.0000729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000884, upper bound: 0.0000722
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000866, upper bound: 0.0000728
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0000870
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0000868
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000843, upper bound: 0.0000700
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000827, upper bound: 0.0000711
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0000871
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0000870
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000843, upper bound: 0.0000699
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000827, upper bound: 0.0000711
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000871, upper bound: 0.0000716
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000875, upper bound: 0.0000676
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000837, upper bound: 0.0000720
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000862, upper bound: 0.0000686
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000837, upper bound: 0.0000743
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000807, upper bound: 0.0000743
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000843, upper bound: 0.0000700
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000827, upper bound: 0.0000711
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000990, upper bound: 0.0000844
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000990, upper bound: 0.0000842
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000876, upper bound: 0.0000675
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000862, upper bound: 0.0000685
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0000872
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0000870
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000843, upper bound: 0.0000699
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000827, upper bound: 0.0000711
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000833, upper bound: 0.0000958
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000870, upper bound: 0.0000952
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000831, upper bound: 0.0000956
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000872, upper bound: 0.0000951
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000711, upper bound: 0.0000827
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000743, upper bound: 0.0000810
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000749, upper bound: 0.0000850
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000849
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000685, upper bound: 0.0000862
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000838
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000723, upper bound: 0.0000884
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000725, upper bound: 0.0000883
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000806, upper bound: 0.0000996
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000840, upper bound: 0.0000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000805, upper bound: 0.0000995
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000843, upper bound: 0.0000990
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000987, upper bound: 0.0000846
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000994, upper bound: 0.0000812
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0000723
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000864, upper bound: 0.0000729
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000836, upper bound: 0.0000744
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000795, upper bound: 0.0000744
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000955, upper bound: 0.0000839
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000957, upper bound: 0.0000839
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0000727
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0000723
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000830, upper bound: 0.0000722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000858, upper bound: 0.0000689
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000950, upper bound: 0.0000873
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000950, upper bound: 0.0000870
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000955, upper bound: 0.0000838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000957, upper bound: 0.0000839
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000830
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000748, upper bound: 0.0000850
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000832, upper bound: 0.0000955
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000870, upper bound: 0.0000951
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000807, upper bound: 0.0000996
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000842, upper bound: 0.0000989
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000729, upper bound: 0.0000858
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000725, upper bound: 0.0000883
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000830
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000748, upper bound: 0.0000850
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000819
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000849
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000862
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000875
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000840, upper bound: 0.0000989
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000842, upper bound: 0.0000989
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000988, upper bound: 0.0000847
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000988, upper bound: 0.0000844
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000876, upper bound: 0.0000684
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000859, upper bound: 0.0000688
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000837, upper bound: 0.0000745
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000843, upper bound: 0.0000708
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000820, upper bound: 0.0000751
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000828, upper bound: 0.0000750
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0000727
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000884, upper bound: 0.0000724
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000832, upper bound: 0.0000722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000859, upper bound: 0.0000688
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000849, upper bound: 0.0000752
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000820, upper bound: 0.0000752
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000951, upper bound: 0.0000871
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.63
Output dim: 3, lower bound: -0.0000958, upper bound: 0.0000839

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006263, 0.0006037
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000745, 0.0000699
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007620, 0.0007327
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002031, 0.0002067
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001067, 0.0001114
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004761, 0.0004587
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000401, 0.0000396
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012648, 0.0012982
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010687, 0.0011176
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005012, 0.0004778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000675, upper bound: 0.0000810
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000710, upper bound: 0.0000806
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006247, 0.0006066
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000738, 0.0000706
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007597, 0.0007367
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002038, 0.0002063
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001074, 0.0001110
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004748, 0.0004609
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000401, 0.0000396
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012681, 0.0012973
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010752, 0.0011132
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004990, 0.0004809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000675, upper bound: 0.0000809
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000712, upper bound: 0.0000806
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006026, 0.0005844
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000755, 0.0000711
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007400, 0.0007153
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001829, 0.0001881
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001054, 0.0001094
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004586, 0.0004445
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000347, 0.0000354
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011854, 0.0012046
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010643, 0.0011072
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005027, 0.0004812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000825
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000688, upper bound: 0.0000798
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006060, 0.0005849
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000759, 0.0000711
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007436, 0.0007152
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001846, 0.0001894
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001053, 0.0001099
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004612, 0.0004448
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000359, 0.0000346
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011879, 0.0012173
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010632, 0.0011116
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005039, 0.0004809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000648, upper bound: 0.0000844
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000683, upper bound: 0.0000840
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006026, 0.0005844
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000755, 0.0000711
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007400, 0.0007153
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001829, 0.0001881
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001054, 0.0001094
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004586, 0.0004445
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000347, 0.0000354
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011854, 0.0012046
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010643, 0.0011072
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005027, 0.0004812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000818
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000796
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006060, 0.0005849
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000759, 0.0000711
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007436, 0.0007152
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001846, 0.0001894
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001053, 0.0001099
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004612, 0.0004448
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000359, 0.0000346
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0011879, 0.0012173
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010632, 0.0011116
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005039, 0.0004809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000649, upper bound: 0.0000844
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000841
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006248, 0.0006055
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000743, 0.0000702
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007599, 0.0007349
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002031, 0.0002068
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001071, 0.0001110
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004749, 0.0004601
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000401, 0.0000396
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012674, 0.0012973
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010728, 0.0011142
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004996, 0.0004798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000675, upper bound: 0.0000810
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000709, upper bound: 0.0000805
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006230, 0.0006081
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000737, 0.0000710
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007575, 0.0007384
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002039, 0.0002063
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001077, 0.0001106
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004735, 0.0004620
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000401, 0.0000396
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012705, 0.0012962
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010786, 0.0011096
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004973, 0.0004827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000675, upper bound: 0.0000809
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000711, upper bound: 0.0000806
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003561, 0.0003128
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000672, 0.0000621
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004485, 0.0003964
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001200, 0.0001288
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000611, 0.0000685
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002720, 0.0002391
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000157, 0.0000148
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005588, 0.0006448
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006368, 0.0007107
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003341, 0.0003008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000824
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000818
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003561, 0.0003128
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000672, 0.0000621
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004485, 0.0003964
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001200, 0.0001288
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000611, 0.0000685
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002720, 0.0002391
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000157, 0.0000148
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005588, 0.0006448
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006368, 0.0007107
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003341, 0.0003008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000649, upper bound: 0.0000844
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000650, upper bound: 0.0000843
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003315, 0.0003509
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000661, 0.0000633
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004208, 0.0004392
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001252, 0.0001251
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000665, 0.0000649
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002534, 0.0002678
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000153, 0.0000159
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0006503, 0.0005869
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006865, 0.0006771
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003200, 0.0003204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000682, upper bound: 0.0000840
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000840
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006403, 0.0006009
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000768, 0.0000694
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007794, 0.0007298
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002030, 0.0002072
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001064, 0.0001140
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004867, 0.0004566
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012579, 0.0013262
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010651, 0.0011445
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005136, 0.0004761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000675, upper bound: 0.0000810
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000709, upper bound: 0.0000805
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006387, 0.0006040
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000761, 0.0000701
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007771, 0.0007340
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002037, 0.0002067
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001070, 0.0001136
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004855, 0.0004590
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012625, 0.0013253
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010723, 0.0011400
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005114, 0.0004796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000675, upper bound: 0.0000809
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000711, upper bound: 0.0000806
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006431, 0.0005983
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000772, 0.0000688
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007831, 0.0007264
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002026, 0.0002075
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001058, 0.0001146
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004889, 0.0004546
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012549, 0.0013283
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010590, 0.0011506
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005164, 0.0004731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000825
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000688, upper bound: 0.0000798
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006414, 0.0006015
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000765, 0.0000696
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007806, 0.0007305
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002033, 0.0002069
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001064, 0.0001141
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004876, 0.0004571
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012588, 0.0013271
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010659, 0.0011458
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005141, 0.0004764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000818
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000796
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003801, 0.0003070
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000697, 0.0000610
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004777, 0.0003896
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001213, 0.0001293
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000601, 0.0000728
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002902, 0.0002347
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000173, 0.0000149
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005478, 0.0006950
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006262, 0.0007536
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003532, 0.0002957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000648, upper bound: 0.0000844
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000649, upper bound: 0.0000844
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003486, 0.0003323
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000682, 0.0000621
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004421, 0.0004179
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001250, 0.0001249
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000638, 0.0000681
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002665, 0.0002537
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000161, 0.0000152
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0006090, 0.0006188
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006610, 0.0007099
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003352, 0.0003096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000683, upper bound: 0.0000840
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000841
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006417, 0.0005997
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000771, 0.0000692
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007811, 0.0007282
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002027, 0.0002076
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001061, 0.0001142
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004878, 0.0004557
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012566, 0.0013269
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010625, 0.0011471
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005147, 0.0004749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000824
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000649, upper bound: 0.0000844
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006398, 0.0006029
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000763, 0.0000698
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007784, 0.0007321
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002033, 0.0002069
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001068, 0.0001138
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004863, 0.0004581
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012607, 0.0013257
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010693, 0.0011421
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005124, 0.0004782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000818
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000650, upper bound: 0.0000843
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006417, 0.0005997
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000771, 0.0000692
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007811, 0.0007282
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002027, 0.0002076
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001061, 0.0001142
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004878, 0.0004557
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012566, 0.0013269
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010625, 0.0011471
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005147, 0.0004749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000797
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000682, upper bound: 0.0000840
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006398, 0.0006029
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000763, 0.0000698
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007784, 0.0007321
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002033, 0.0002069
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001068, 0.0001138
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004863, 0.0004581
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000407, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012607, 0.0013257
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0010693, 0.0011421
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0005124, 0.0004782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000796
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000685, upper bound: 0.0000840
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003387, 0.0003328
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000624, 0.0000661
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004248, 0.0004228
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001255, 0.0001247
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000652, 0.0000646
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002585, 0.0002544
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000159, 0.0000149
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005838, 0.0006264
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006799, 0.0006677
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003122, 0.0003214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000840, upper bound: 0.0000682
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000840, upper bound: 0.0000680
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003099, 0.0003606
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000613, 0.0000675
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0003927, 0.0004538
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001294, 0.0001190
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000693, 0.0000604
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002369, 0.0002754
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000151, 0.0000158
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0006622, 0.0005549
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0007193, 0.0006301
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0002976, 0.0003377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000843, upper bound: 0.0000644
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000844, upper bound: 0.0000644
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003099, 0.0003606
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000613, 0.0000675
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0003927, 0.0004538
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001294, 0.0001190
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000693, 0.0000604
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002369, 0.0002754
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000151, 0.0000158
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0006622, 0.0005549
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0007193, 0.0006301
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0002976, 0.0003377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000818, upper bound: 0.0000653
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000825, upper bound: 0.0000655
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005819, 0.0006071
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000704, 0.0000760
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007118, 0.0007453
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001893, 0.0001840
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001101, 0.0001047
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004426, 0.0004620
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000347, 0.0000355
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012146, 0.0011798
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011142, 0.0010564
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004778, 0.0005054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000841, upper bound: 0.0000683
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000844, upper bound: 0.0000643
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005813, 0.0006037
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000705, 0.0000757
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007112, 0.0007416
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001887, 0.0001827
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001097, 0.0001048
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004421, 0.0004595
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000357, 0.0000346
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012047, 0.0011822
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011108, 0.0010576
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004782, 0.0005044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000798, upper bound: 0.0000688
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000818, upper bound: 0.0000651
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005819, 0.0006071
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000704, 0.0000760
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007118, 0.0007453
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001893, 0.0001840
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001101, 0.0001047
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004426, 0.0004620
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000347, 0.0000355
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012146, 0.0011798
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011142, 0.0010564
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004778, 0.0005054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000841, upper bound: 0.0000680
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000844, upper bound: 0.0000644
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0005813, 0.0006037
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000705, 0.0000757
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007112, 0.0007416
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001887, 0.0001827
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001097, 0.0001048
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004421, 0.0004595
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000357, 0.0000346
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012047, 0.0011822
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011108, 0.0010576
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004782, 0.0005044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 241

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000801, upper bound: 0.0000687
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000827, upper bound: 0.0000654
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006059, 0.0006240
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000701, 0.0000737
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007353, 0.0007590
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002067, 0.0002033
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001109, 0.0001071
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004604, 0.0004743
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000399, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012904, 0.0012664
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011119, 0.0010728
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004798, 0.0004984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 233

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000806, upper bound: 0.0000709
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000764, upper bound: 0.0000710
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006032, 0.0006265
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000694, 0.0000744
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007317, 0.0007624
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002073, 0.0002026
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001114, 0.0001065
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004583, 0.0004762
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000399, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012934, 0.0012637
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011176, 0.0010661
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004764, 0.0005009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000806, upper bound: 0.0000708
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000769, upper bound: 0.0000710
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006041, 0.0006252
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000699, 0.0000740
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007330, 0.0007608
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002067, 0.0002032
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001112, 0.0001068
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004590, 0.0004752
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000399, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012923, 0.0012643
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011154, 0.0010694
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004781, 0.0005001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000807, upper bound: 0.0000710
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000765, upper bound: 0.0000711
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0006014, 0.0006276
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000690, 0.0000746
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0007293, 0.0007641
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0002072, 0.0002026
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0001117, 0.0001062
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0004569, 0.0004771
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000399, 0.0000397
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0012951, 0.0012615
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0011207, 0.0010623
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0004746, 0.0005026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000807, upper bound: 0.0000709
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000771, upper bound: 0.0000710
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037824, -0.0024505, -0.0037824, -0.0024505, -0.0003616, 0.0003293
1: -0.0045857, -0.0043286, -0.0045857, -0.0043286, -0.0000647, 0.0000660
2: 0.0097497, 0.0114527, 0.0097497, 0.0114527, -0.0004525, 0.0004188
3: 1.0086287, 1.0090333, 1.0086287, 1.0090333, -0.0001268, 0.0001252
4: -0.0034825, -0.0032176, -0.0034825, -0.0032176, -0.0000647, 0.0000686
5: 0.0010557, 0.0020750, 0.0010557, 0.0020750, -0.0002759, 0.0002518
6: -0.0025321, -0.0024842, -0.0025321, -0.0024842, -0.0000174, 0.0000150
7: -0.0094425, -0.0071791, -0.0094425, -0.0071791, -0.0005830, 0.0006755
8: -0.0052352, -0.0024575, -0.0052352, -0.0024575, -0.0006753, 0.0007077
9: -0.0029656, -0.0016425, -0.0029656, -0.0016425, -0.0003300, 0.0003195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 2.75 + 598.70 = 601.45 seconds
