## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00086954


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0009399, 0.0009399)
1: (0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008953, 0.0008953)
2: (-0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0022408, 0.0022408)
3: (-0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0019445, 0.0019445)
4: (0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001653, 0.0001653)
5: (-0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0027382, 0.0027382)
6: (0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0018450, 0.0018450)
7: (0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0008095, 0.0008095)
8: (0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0008301, 0.0008301)
9: (-0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0018824, 0.0018824)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.60 + 1.35 = 2.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0010983, upper bound: 0.0010983

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010844, upper bound: 0.0010501
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010501, upper bound: 0.0010844
time: 0.49 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.00 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 6, lower bound: -0.0010844, upper bound: 0.0010501
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 6, lower bound: -0.0010501, upper bound: 0.0010844

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0009393, 0.0009396
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008949, 0.0008951
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0022395, 0.0022402
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0019440, 0.0019434
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001652, 0.0001653
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0027375, 0.0027367
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0018445, 0.0018440
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0008093, 0.0008091
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0008297, 0.0008299
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0018814, 0.0018819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009803, upper bound: 0.0009798
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009803, upper bound: 0.0009798
time: 0.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0009396, 0.0009399
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008951, 0.0008953
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0022402, 0.0022408
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0019445, 0.0019440
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001653, 0.0001653
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0027382, 0.0027375
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0018450, 0.0018445
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0008095, 0.0008093
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0008299, 0.0008301
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0018819, 0.0018824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009798, upper bound: 0.0009803
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009798, upper bound: 0.0009803
time: 0.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 6, lower bound: -0.0009803, upper bound: 0.0009798
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 6, lower bound: -0.0009803, upper bound: 0.0009798
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 6, lower bound: -0.0009798, upper bound: 0.0009803
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 6, lower bound: -0.0009798, upper bound: 0.0009803

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0009298, 0.0009308
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008858, 0.0008867
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0022169, 0.0022191
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0019257, 0.0019237
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001636, 0.0001637
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0027118, 0.0027090
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0018271, 0.0018253
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0008017, 0.0008009
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0008213, 0.0008221
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0018623, 0.0018642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009424, upper bound: 0.0009233
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009233, upper bound: 0.0009424
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0009305, 0.0009396
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008864, 0.0008951
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0022185, 0.0022402
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0019440, 0.0019251
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001637, 0.0001653
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0027375, 0.0027110
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0018445, 0.0018266
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0008093, 0.0008015
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0008219, 0.0008299
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0018637, 0.0018819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009424, upper bound: 0.0009233
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009233, upper bound: 0.0009424
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0009301, 0.0009310
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008860, 0.0008869
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0022175, 0.0022197
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0019262, 0.0019243
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001636, 0.0001638
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0027125, 0.0027098
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0018276, 0.0018258
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0008019, 0.0008011
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0008215, 0.0008223
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0018629, 0.0018647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009424, upper bound: 0.0009233
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009233, upper bound: 0.0009424
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0009308, 0.0009399
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008867, 0.0008953
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0022191, 0.0022408
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0019445, 0.0019257
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001637, 0.0001653
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0027382, 0.0027118
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0018450, 0.0018271
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0008095, 0.0008017
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0008221, 0.0008301
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0018642, 0.0018824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009424, upper bound: 0.0009233
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009233, upper bound: 0.0009424
time: 0.48 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 6, lower bound: -0.0009424, upper bound: 0.0009233
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 6, lower bound: -0.0009233, upper bound: 0.0009424
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 6, lower bound: -0.0009424, upper bound: 0.0009233
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 6, lower bound: -0.0009233, upper bound: 0.0009424
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 6, lower bound: -0.0009424, upper bound: 0.0009233
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 6, lower bound: -0.0009233, upper bound: 0.0009424
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 6, lower bound: -0.0009424, upper bound: 0.0009233
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 6, lower bound: -0.0009233, upper bound: 0.0009424

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008482, 0.0008666
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008080, 0.0008256
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020223, 0.0020662
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017930, 0.0017549
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001492, 0.0001525
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025249, 0.0024712
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017012, 0.0016651
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007465, 0.0007306
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007492, 0.0007654
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016989, 0.0017358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 3.69 seconds

### Candidate
type: RSZ, layer: 3, pos: 130

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009408, upper bound: 0.0009144
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009315, upper bound: 0.0009217
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008667, 0.0008492
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008256, 0.0008089
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020662, 0.0020245
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017568, 0.0017930
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001525, 0.0001494
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024740, 0.0025249
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016669, 0.0017013
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007314, 0.0007465
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007655, 0.0007500
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017358, 0.0017008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 3.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008766, upper bound: 0.0008973
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008769, upper bound: 0.0008890
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008489, 0.0008775
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008087, 0.0008359
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020239, 0.0020920
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0018154, 0.0017562
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001493, 0.0001544
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025565, 0.0024732
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017225, 0.0016664
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007558, 0.0007312
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007498, 0.0007750
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017002, 0.0017575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 3.73 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009385, upper bound: 0.0009190
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009384, upper bound: 0.0009195
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008664, 0.0008600
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008253, 0.0008193
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020656, 0.0020504
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017792, 0.0017924
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001524, 0.0001513
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025055, 0.0025241
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016882, 0.0017007
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007408, 0.0007463
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007652, 0.0007596
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017352, 0.0017224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160

Time for candidate selection: 3.69 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009195, upper bound: 0.0009384
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009190, upper bound: 0.0009385
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008485, 0.0008669
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008083, 0.0008259
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020229, 0.0020669
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017935, 0.0017554
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001493, 0.0001525
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025257, 0.0024720
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017018, 0.0016656
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007467, 0.0007308
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007494, 0.0007657
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016994, 0.0017363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 3.71 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009089, upper bound: 0.0009127
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009316, upper bound: 0.0008845
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008669, 0.0008494
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008259, 0.0008092
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020669, 0.0020252
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017574, 0.0017936
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001525, 0.0001494
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024748, 0.0025257
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016674, 0.0017018
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007317, 0.0007467
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007657, 0.0007502
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017363, 0.0017013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 160

Time for candidate selection: 3.69 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006513, upper bound: 0.0006635
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006513, upper bound: 0.0006635
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008492, 0.0008778
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008089, 0.0008362
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020245, 0.0020927
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0018160, 0.0017568
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001494, 0.0001544
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025573, 0.0024740
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017230, 0.0016669
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007561, 0.0007314
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007500, 0.0007753
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017008, 0.0017580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 130

Time for candidate selection: 3.64 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006635, upper bound: 0.0006513
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006635, upper bound: 0.0006513
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008666, 0.0008603
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008256, 0.0008195
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020662, 0.0020510
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017798, 0.0017930
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001525, 0.0001513
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025063, 0.0025249
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016887, 0.0017012
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007410, 0.0007465
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007654, 0.0007598
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017358, 0.0017230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 3.68 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006513, upper bound: 0.0006635
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006513, upper bound: 0.0006635
time: 0.50 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0009408, upper bound: 0.0009144
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0009315, upper bound: 0.0009217
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0008766, upper bound: 0.0008973
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0008769, upper bound: 0.0008890
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0009385, upper bound: 0.0009190
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0009384, upper bound: 0.0009195
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0009195, upper bound: 0.0009384
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0009190, upper bound: 0.0009385
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0009089, upper bound: 0.0009127
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0009316, upper bound: 0.0008845
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0006513, upper bound: 0.0006635
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0006513, upper bound: 0.0006635
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0006635, upper bound: 0.0006513
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0006635, upper bound: 0.0006513
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0006513, upper bound: 0.0006635
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.24
Output dim: 6, lower bound: -0.0006513, upper bound: 0.0006635

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006509, 0.0006822
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006201, 0.0006499
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0015519, 0.0016264
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0014113, 0.0013467
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001145, 0.0001200
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0019874, 0.0018964
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0013391, 0.0012778
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005876, 0.0005607
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0005749, 0.0006025
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013037, 0.0013663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008875, upper bound: 0.0008686
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008957, upper bound: 0.0008683
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006637, 0.0006651
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006323, 0.0006336
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0015824, 0.0015856
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0013760, 0.0013732
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001168, 0.0001170
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0019376, 0.0019338
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0013055, 0.0013029
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005729, 0.0005717
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0005862, 0.0005874
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013294, 0.0013320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009273, upper bound: 0.0009175
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009276, upper bound: 0.0009180
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008361, 0.0008187
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0007965, 0.0007799
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0019935, 0.0019519
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0016938, 0.0017299
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001471, 0.0001440
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0023853, 0.0024360
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016071, 0.0016414
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007052, 0.0007202
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007385, 0.0007231
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016747, 0.0016398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 130

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008751, upper bound: 0.0008872
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008683, upper bound: 0.0008957
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008362, 0.0008164
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0007966, 0.0007777
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0019936, 0.0019464
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0016891, 0.0017300
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001471, 0.0001436
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0023786, 0.0024362
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016026, 0.0016415
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007032, 0.0007203
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007386, 0.0007211
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016748, 0.0016352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 130

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008754, upper bound: 0.0008785
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008686, upper bound: 0.0008875
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008514, 0.0008813
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008111, 0.0008395
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020299, 0.0021011
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0018232, 0.0017615
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001498, 0.0001550
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025675, 0.0024805
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017299, 0.0016713
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007591, 0.0007334
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007520, 0.0007784
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017053, 0.0017650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 130

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009370, upper bound: 0.0009105
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009273, upper bound: 0.0009175
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008521, 0.0008801
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008118, 0.0008384
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020316, 0.0020982
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0018207, 0.0017629
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001499, 0.0001548
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025640, 0.0024826
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017276, 0.0016727
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007580, 0.0007340
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007526, 0.0007773
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017067, 0.0017626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008852, upper bound: 0.0008732
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008934, upper bound: 0.0008728
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008689, 0.0008638
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008278, 0.0008229
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020716, 0.0020595
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017871, 0.0017977
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001529, 0.0001520
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025167, 0.0025315
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016957, 0.0017057
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007440, 0.0007484
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007675, 0.0007630
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017403, 0.0017301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008807, upper bound: 0.0009276
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009089, upper bound: 0.0009033
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008696, 0.0008626
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008284, 0.0008217
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020732, 0.0020565
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017846, 0.0017990
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001530, 0.0001517
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025130, 0.0025334
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016932, 0.0017070
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007430, 0.0007490
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007680, 0.0007619
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017416, 0.0017276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008807, upper bound: 0.0009278
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009083, upper bound: 0.0009051
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008439, 0.0008614
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008039, 0.0008206
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020119, 0.0020536
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017820, 0.0017459
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001485, 0.0001515
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025095, 0.0024585
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016908, 0.0016565
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007419, 0.0007269
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007453, 0.0007608
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016901, 0.0017252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008622, upper bound: 0.0008663
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008620, upper bound: 0.0008660
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008428, 0.0008623
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008029, 0.0008214
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020094, 0.0020558
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017840, 0.0017437
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001483, 0.0001517
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025122, 0.0024554
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016927, 0.0016544
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007427, 0.0007259
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007444, 0.0007616
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016880, 0.0017270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009278, upper bound: 0.0008807
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009276, upper bound: 0.0008807
time: 0.55 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0008875, upper bound: 0.0008686
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0008957, upper bound: 0.0008683
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0009273, upper bound: 0.0009175
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0009276, upper bound: 0.0009180
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0008751, upper bound: 0.0008872
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0008683, upper bound: 0.0008957
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0008754, upper bound: 0.0008785
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0008686, upper bound: 0.0008875
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0009370, upper bound: 0.0009105
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0009273, upper bound: 0.0009175
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0008852, upper bound: 0.0008732
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0008934, upper bound: 0.0008728
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0008807, upper bound: 0.0009276
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0009089, upper bound: 0.0009033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0008807, upper bound: 0.0009278
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0009083, upper bound: 0.0009051
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0008622, upper bound: 0.0008663
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0008620, upper bound: 0.0008660
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0009278, upper bound: 0.0008807
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 6, lower bound: -0.0009276, upper bound: 0.0008807

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008151, 0.0008362
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0007765, 0.0007966
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0019434, 0.0019936
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017300, 0.0016864
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001434, 0.0001471
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024362, 0.0023748
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016415, 0.0016001
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007203, 0.0007021
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007199, 0.0007386
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016326, 0.0016748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008835, upper bound: 0.0008647
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008836, upper bound: 0.0008640
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008178, 0.0008363
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0007790, 0.0007967
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0019497, 0.0019939
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017303, 0.0016919
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001439, 0.0001471
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024366, 0.0023825
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016417, 0.0016053
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007204, 0.0007044
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007223, 0.0007387
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016379, 0.0016751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008626, upper bound: 0.0008575
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008850, upper bound: 0.0008181
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008507, 0.0008697
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008104, 0.0008285
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020283, 0.0020734
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017993, 0.0017601
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001497, 0.0001530
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025337, 0.0024786
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017072, 0.0016700
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007491, 0.0007328
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007514, 0.0007681
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017039, 0.0017418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008895, upper bound: 0.0009068
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009165, upper bound: 0.0008792
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008513, 0.0008692
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008110, 0.0008280
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020296, 0.0020722
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017982, 0.0017612
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001498, 0.0001529
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025323, 0.0024802
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017062, 0.0016711
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007487, 0.0007333
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007519, 0.0007677
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017050, 0.0017408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008746, upper bound: 0.0008717
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008834, upper bound: 0.0008713
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006624, 0.0006647
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006311, 0.0006332
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0015793, 0.0015847
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0013752, 0.0013705
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001165, 0.0001169
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0019365, 0.0019300
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0013048, 0.0013004
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005725, 0.0005706
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0005851, 0.0005871
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013268, 0.0013313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 160

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006023, upper bound: 0.0006073
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006023, upper bound: 0.0006073
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006822, 0.0006542
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006499, 0.0006232
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0016264, 0.0015596
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0013534, 0.0014113
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001200, 0.0001151
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0019059, 0.0019875
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0012842, 0.0013391
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005635, 0.0005876
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0006025, 0.0005778
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013663, 0.0013102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005993, upper bound: 0.0006151
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005993, upper bound: 0.0006151
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006624, 0.0006647
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006311, 0.0006332
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0015793, 0.0015847
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0013752, 0.0013705
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001165, 0.0001169
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0019365, 0.0019300
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0013048, 0.0013004
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005725, 0.0005706
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0005851, 0.0005871
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013268, 0.0013313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008257, upper bound: 0.0008680
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008648, upper bound: 0.0008410
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006822, 0.0006542
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006499, 0.0006232
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0016264, 0.0015596
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0013534, 0.0014113
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001200, 0.0001151
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0019059, 0.0019875
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0012842, 0.0013391
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005635, 0.0005876
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0006025, 0.0005778
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013663, 0.0013102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008640, upper bound: 0.0008836
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008647, upper bound: 0.0008834
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006516, 0.0006920
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006207, 0.0006592
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0015534, 0.0016499
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0014317, 0.0013480
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001146, 0.0001217
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0020161, 0.0018983
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0013584, 0.0012790
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005961, 0.0005612
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0005755, 0.0006112
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013050, 0.0013860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006582, upper bound: 0.0006443
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006582, upper bound: 0.0006443
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006644, 0.0006706
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006329, 0.0006389
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0015840, 0.0015989
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0013874, 0.0013746
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001169, 0.0001180
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0019538, 0.0019357
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0013164, 0.0013042
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005776, 0.0005723
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0005868, 0.0005923
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013307, 0.0013432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008743, upper bound: 0.0008712
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008834, upper bound: 0.0008707
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008161, 0.0008468
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0007775, 0.0008067
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0019457, 0.0020190
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017520, 0.0016884
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001436, 0.0001490
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024672, 0.0023777
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016623, 0.0016020
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007294, 0.0007030
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007208, 0.0007479
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016345, 0.0016961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 130

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008836, upper bound: 0.0008640
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008746, upper bound: 0.0008717
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008184, 0.0008477
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0007797, 0.0008075
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0019513, 0.0020210
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017537, 0.0016933
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001440, 0.0001491
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024696, 0.0023845
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016640, 0.0016066
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007301, 0.0007050
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007229, 0.0007487
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016392, 0.0016978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008557, upper bound: 0.0008623
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008825, upper bound: 0.0008328
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008618, 0.0008540
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008209, 0.0008136
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020546, 0.0020361
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017668, 0.0017829
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001516, 0.0001502
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024881, 0.0025107
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016764, 0.0016916
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007356, 0.0007423
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007611, 0.0007543
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017260, 0.0017105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006101, upper bound: 0.0006496
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006125, upper bound: 0.0006496
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008609, 0.0008551
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008201, 0.0008146
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020524, 0.0020386
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017691, 0.0017810
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001514, 0.0001504
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024912, 0.0025080
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016785, 0.0016899
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007365, 0.0007415
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007603, 0.0007552
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017242, 0.0017126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 130

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009074, upper bound: 0.0008841
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008981, upper bound: 0.0009017
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008618, 0.0008540
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008209, 0.0008136
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020546, 0.0020361
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017668, 0.0017829
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001516, 0.0001502
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024881, 0.0025107
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016764, 0.0016916
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007356, 0.0007423
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007611, 0.0007543
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017260, 0.0017105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 130

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008792, upper bound: 0.0009165
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008628, upper bound: 0.0009262
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008609, 0.0008551
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008201, 0.0008146
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020524, 0.0020386
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017691, 0.0017810
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001514, 0.0001504
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024912, 0.0025080
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016785, 0.0016899
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007365, 0.0007415
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007603, 0.0007552
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017242, 0.0017126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 130

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009068, upper bound: 0.0008838
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008996, upper bound: 0.0009036
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008510, 0.0008696
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008107, 0.0008284
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020289, 0.0020732
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017990, 0.0017606
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001497, 0.0001530
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025334, 0.0024794
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017070, 0.0016705
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007490, 0.0007330
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007516, 0.0007680
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017045, 0.0017416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006497, upper bound: 0.0006128
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006497, upper bound: 0.0006101
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008514, 0.0008695
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008111, 0.0008283
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020299, 0.0020729
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017988, 0.0017615
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001498, 0.0001530
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025331, 0.0024805
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017067, 0.0016713
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007489, 0.0007334
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007520, 0.0007679
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017053, 0.0017414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006496, upper bound: 0.0006125
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006496, upper bound: 0.0006101
time: 0.54 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008835, upper bound: 0.0008647
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008836, upper bound: 0.0008640
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008626, upper bound: 0.0008575
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008850, upper bound: 0.0008181
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008895, upper bound: 0.0009068
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0009165, upper bound: 0.0008792
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008746, upper bound: 0.0008717
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008834, upper bound: 0.0008713
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0006023, upper bound: 0.0006073
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0006023, upper bound: 0.0006073
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0005993, upper bound: 0.0006151
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0005993, upper bound: 0.0006151
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008257, upper bound: 0.0008680
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008648, upper bound: 0.0008410
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008640, upper bound: 0.0008836
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008647, upper bound: 0.0008834
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0006582, upper bound: 0.0006443
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0006582, upper bound: 0.0006443
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008743, upper bound: 0.0008712
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008834, upper bound: 0.0008707
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008836, upper bound: 0.0008640
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008746, upper bound: 0.0008717
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008557, upper bound: 0.0008623
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008825, upper bound: 0.0008328
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0006101, upper bound: 0.0006496
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0006125, upper bound: 0.0006496
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0009074, upper bound: 0.0008841
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008981, upper bound: 0.0009017
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008792, upper bound: 0.0009165
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008628, upper bound: 0.0009262
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0009068, upper bound: 0.0008838
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0008996, upper bound: 0.0009036
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0006497, upper bound: 0.0006128
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0006497, upper bound: 0.0006101
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0006496, upper bound: 0.0006125
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 6, lower bound: -0.0006496, upper bound: 0.0006101

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008507, 0.0008697
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008104, 0.0008285
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020283, 0.0020734
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017993, 0.0017601
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001497, 0.0001530
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025337, 0.0024786
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017072, 0.0016700
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007491, 0.0007328
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007514, 0.0007681
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017039, 0.0017418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008589, upper bound: 0.0008539
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008733, upper bound: 0.0008090
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008513, 0.0008692
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008110, 0.0008280
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020296, 0.0020722
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017982, 0.0017612
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001498, 0.0001529
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025323, 0.0024802
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017062, 0.0016711
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007487, 0.0007333
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007519, 0.0007677
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017050, 0.0017408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 130

### Candidate
type: RSZ, layer: 3, pos: 160

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006090, upper bound: 0.0005910
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006090, upper bound: 0.0005910
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008426, 0.0008620
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008026, 0.0008212
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020088, 0.0020552
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017834, 0.0017431
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001482, 0.0001516
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025114, 0.0024547
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016922, 0.0016539
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007425, 0.0007257
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007442, 0.0007614
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016875, 0.0017265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008812, upper bound: 0.0008143
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008810, upper bound: 0.0008130
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008436, 0.0008611
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008036, 0.0008203
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020113, 0.0020530
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017815, 0.0017453
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001484, 0.0001515
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025088, 0.0024578
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016904, 0.0016560
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007417, 0.0007266
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007451, 0.0007606
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016896, 0.0017247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008440, upper bound: 0.0008605
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008451, upper bound: 0.0008602
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008426, 0.0008620
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008026, 0.0008212
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020088, 0.0020552
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017834, 0.0017431
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001482, 0.0001516
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025114, 0.0024547
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016922, 0.0016539
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007425, 0.0007257
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007442, 0.0007614
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016875, 0.0017265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006386, upper bound: 0.0006113
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006386, upper bound: 0.0006085
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008151, 0.0008362
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0007765, 0.0007966
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0019434, 0.0019936
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017300, 0.0016864
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001434, 0.0001471
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024362, 0.0023748
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016415, 0.0016001
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007203, 0.0007021
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007199, 0.0007386
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016326, 0.0016748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008440, upper bound: 0.0008610
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008641, upper bound: 0.0008219
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008178, 0.0008363
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0007790, 0.0007967
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0019497, 0.0019939
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017303, 0.0016919
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001439, 0.0001471
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024366, 0.0023825
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016417, 0.0016053
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007204, 0.0007044
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007223, 0.0007387
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016379, 0.0016751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008452, upper bound: 0.0008608
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008724, upper bound: 0.0008313
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008692, 0.0008522
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008280, 0.0008119
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020723, 0.0020318
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017632, 0.0017982
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001529, 0.0001499
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024829, 0.0025323
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016729, 0.0017062
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007341, 0.0007487
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007677, 0.0007527
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017408, 0.0017069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 130

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005910, upper bound: 0.0006090
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005910, upper bound: 0.0006090
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008699, 0.0008517
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008287, 0.0008113
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020741, 0.0020306
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017620, 0.0017998
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001530, 0.0001498
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024813, 0.0025345
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016719, 0.0017077
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007336, 0.0007493
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007684, 0.0007522
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017424, 0.0017058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005968, upper bound: 0.0006090
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005968, upper bound: 0.0006090
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008161, 0.0008468
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0007775, 0.0008067
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0019457, 0.0020190
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017520, 0.0016884
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001436, 0.0001490
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024672, 0.0023777
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016623, 0.0016020
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007294, 0.0007030
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007208, 0.0007479
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016345, 0.0016961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005990, upper bound: 0.0005990
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005990, upper bound: 0.0005990
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008184, 0.0008477
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0007797, 0.0008075
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0019513, 0.0020210
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017537, 0.0016933
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001440, 0.0001491
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024696, 0.0023845
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016640, 0.0016066
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007301, 0.0007050
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007229, 0.0007487
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016392, 0.0016978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006015, upper bound: 0.0005986
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006015, upper bound: 0.0005986
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006516, 0.0006920
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006207, 0.0006592
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0015534, 0.0016499
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0014317, 0.0013480
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001146, 0.0001217
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0020161, 0.0018983
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0013584, 0.0012790
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005961, 0.0005612
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0005755, 0.0006112
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013050, 0.0013860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 160

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006090, upper bound: 0.0005910
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006090, upper bound: 0.0005910
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006644, 0.0006706
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006329, 0.0006389
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0015840, 0.0015989
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0013874, 0.0013746
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001169, 0.0001180
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0019538, 0.0019357
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0013164, 0.0013042
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005776, 0.0005723
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0005868, 0.0005923
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013307, 0.0013432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006011, upper bound: 0.0005975
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006011, upper bound: 0.0005975
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008433, 0.0008726
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008034, 0.0008312
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020106, 0.0020803
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0018052, 0.0017447
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001484, 0.0001535
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025421, 0.0024570
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017128, 0.0016555
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007516, 0.0007264
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007449, 0.0007707
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016891, 0.0017476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006027, upper bound: 0.0005648
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006027, upper bound: 0.0005609
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006628, 0.0006745
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006314, 0.0006426
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0015803, 0.0016082
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0013955, 0.0013714
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001166, 0.0001187
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0019652, 0.0019312
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0013241, 0.0013012
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005810, 0.0005709
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0005854, 0.0005958
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013276, 0.0013510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008608, upper bound: 0.0008377
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008610, upper bound: 0.0008371
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006819, 0.0006597
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006496, 0.0006285
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0016258, 0.0015729
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0013649, 0.0014108
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001200, 0.0001161
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0019221, 0.0019867
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0012950, 0.0013386
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005683, 0.0005874
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0006023, 0.0005827
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013658, 0.0013213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006258, upper bound: 0.0006342
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006258, upper bound: 0.0006341
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006628, 0.0006745
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006314, 0.0006426
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0015803, 0.0016082
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0013955, 0.0013714
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001166, 0.0001187
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0019652, 0.0019312
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0013241, 0.0013012
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005810, 0.0005709
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0005854, 0.0005958
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013276, 0.0013510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008313, upper bound: 0.0008724
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008216, upper bound: 0.0008638
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006819, 0.0006597
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006496, 0.0006285
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0016258, 0.0015729
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0013649, 0.0014108
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001200, 0.0001161
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0019221, 0.0019867
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0012950, 0.0013386
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005683, 0.0005874
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0006023, 0.0005827
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013658, 0.0013213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008150, upper bound: 0.0008812
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008100, upper bound: 0.0008732
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006628, 0.0006745
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006314, 0.0006426
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0015803, 0.0016082
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0013955, 0.0013714
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001166, 0.0001187
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0019652, 0.0019312
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0013241, 0.0013012
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005810, 0.0005709
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0005854, 0.0005958
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013276, 0.0013510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008602, upper bound: 0.0008376
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008605, upper bound: 0.0008368
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0006819, 0.0006597
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0006496, 0.0006285
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0016258, 0.0015729
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0013649, 0.0014108
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001200, 0.0001161
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0019221, 0.0019867
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0012950, 0.0013386
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0005683, 0.0005874
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0006023, 0.0005827
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0013658, 0.0013213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 140

### Candidate
type: RSZ, layer: 3, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008537, upper bound: 0.0008566
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0008539, upper bound: 0.0008568
time: 0.53 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008589, upper bound: 0.0008539
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008733, upper bound: 0.0008090
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0006090, upper bound: 0.0005910
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0006090, upper bound: 0.0005910
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008812, upper bound: 0.0008143
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008810, upper bound: 0.0008130
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008440, upper bound: 0.0008605
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008451, upper bound: 0.0008602
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0006386, upper bound: 0.0006113
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0006386, upper bound: 0.0006085
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008440, upper bound: 0.0008610
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008641, upper bound: 0.0008219
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008452, upper bound: 0.0008608
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008724, upper bound: 0.0008313
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0005910, upper bound: 0.0006090
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0005910, upper bound: 0.0006090
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0005968, upper bound: 0.0006090
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0005968, upper bound: 0.0006090
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0005990, upper bound: 0.0005990
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0005990, upper bound: 0.0005990
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0006015, upper bound: 0.0005986
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0006015, upper bound: 0.0005986
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0006090, upper bound: 0.0005910
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0006090, upper bound: 0.0005910
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0006011, upper bound: 0.0005975
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0006011, upper bound: 0.0005975
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0006027, upper bound: 0.0005648
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0006027, upper bound: 0.0005609
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008608, upper bound: 0.0008377
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008610, upper bound: 0.0008371
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0006258, upper bound: 0.0006342
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0006258, upper bound: 0.0006341
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008313, upper bound: 0.0008724
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008216, upper bound: 0.0008638
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008150, upper bound: 0.0008812
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008100, upper bound: 0.0008732
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008602, upper bound: 0.0008376
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008605, upper bound: 0.0008368
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008537, upper bound: 0.0008566
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 6, lower bound: -0.0008539, upper bound: 0.0008568

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008426, 0.0008620
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008026, 0.0008212
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020088, 0.0020552
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017834, 0.0017431
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001482, 0.0001516
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025114, 0.0024547
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016922, 0.0016539
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007425, 0.0007257
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007442, 0.0007614
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016875, 0.0017265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005989, upper bound: 0.0005540
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005989, upper bound: 0.0005399
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008507, 0.0008697
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008104, 0.0008285
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020283, 0.0020734
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017993, 0.0017601
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001497, 0.0001530
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025337, 0.0024786
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017072, 0.0016700
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007491, 0.0007328
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007514, 0.0007681
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017039, 0.0017418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Candidate
type: RSZ, layer: 3, pos: 130

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006012, upper bound: 0.0005577
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006012, upper bound: 0.0005526
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008513, 0.0008692
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008110, 0.0008280
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020296, 0.0020722
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017982, 0.0017612
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001498, 0.0001529
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025323, 0.0024802
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0017062, 0.0016711
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007487, 0.0007333
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007519, 0.0007677
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0017050, 0.0017408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006011, upper bound: 0.0005564
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006011, upper bound: 0.0005526
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008426, 0.0008620
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0008026, 0.0008212
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0020088, 0.0020552
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017834, 0.0017431
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001482, 0.0001516
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0025114, 0.0024547
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016922, 0.0016539
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007425, 0.0007257
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007442, 0.0007614
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016875, 0.0017265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005935, upper bound: 0.0005633
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005935, upper bound: 0.0005594
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008361, 0.0008293
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0007964, 0.0007901
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0019933, 0.0019773
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017158, 0.0017297
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001471, 0.0001459
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024163, 0.0024358
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016280, 0.0016412
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007144, 0.0007201
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007384, 0.0007325
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016745, 0.0016611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 140

### Candidate
type: RSZ, layer: 3, pos: 130

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005592, upper bound: 0.0005912
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005634, upper bound: 0.0005912
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008361, 0.0008293
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0007964, 0.0007901
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0019933, 0.0019773
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017158, 0.0017297
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001471, 0.0001459
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024163, 0.0024358
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016280, 0.0016412
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007144, 0.0007201
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007384, 0.0007325
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016745, 0.0016611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 130

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005526, upper bound: 0.0006012
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005577, upper bound: 0.0006012
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0060260, 0.0080430, 0.0060260, 0.0080430, -0.0008359, 0.0008277
1: 0.0019056, 0.0038270, 0.0019056, 0.0038270, -0.0007963, 0.0007885
2: -0.0204511, -0.0156424, -0.0204511, -0.0156424, -0.0019930, 0.0019735
3: -0.0008017, 0.0033712, -0.0008017, 0.0033712, -0.0017125, 0.0017295
4: 0.0153649, 0.0157197, 0.0153649, 0.0157197, -0.0001471, 0.0001456
5: -0.0023783, 0.0034980, -0.0023783, 0.0034980, -0.0024116, 0.0024354
6: 0.9958501, 0.9998096, 0.9958501, 0.9998096, -0.0016249, 0.0016410
7: 0.0152719, 0.0170093, 0.0152719, 0.0170093, -0.0007130, 0.0007200
8: 0.0037965, 0.0055780, 0.0037965, 0.0055780, -0.0007383, 0.0007311
9: -0.0231384, -0.0190987, -0.0231384, -0.0190987, -0.0016743, 0.0016579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 130
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005399, upper bound: 0.0005990
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005540, upper bound: 0.0005990
time: 0.55 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 6, lower bound: -0.0005989, upper bound: 0.0005540
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 6, lower bound: -0.0005989, upper bound: 0.0005399
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 6, lower bound: -0.0006012, upper bound: 0.0005577
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 6, lower bound: -0.0006012, upper bound: 0.0005526
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 6, lower bound: -0.0006011, upper bound: 0.0005564
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 6, lower bound: -0.0006011, upper bound: 0.0005526
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 6, lower bound: -0.0005935, upper bound: 0.0005633
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 6, lower bound: -0.0005935, upper bound: 0.0005594
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 6, lower bound: -0.0005592, upper bound: 0.0005912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 6, lower bound: -0.0005634, upper bound: 0.0005912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 6, lower bound: -0.0005526, upper bound: 0.0006012
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 6, lower bound: -0.0005577, upper bound: 0.0006012
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 6, lower bound: -0.0005399, upper bound: 0.0005990
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 6, lower bound: -0.0005540, upper bound: 0.0005990

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.95 + 215.40 = 218.35 seconds
