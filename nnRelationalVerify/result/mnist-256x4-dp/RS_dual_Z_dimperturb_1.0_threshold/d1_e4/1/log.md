## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0302838


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107326, 0.0107326)
1: (0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176)
2: (0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544)
3: (-0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004)
4: (-0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596)
5: (0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762)
6: (-0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913)
7: (-0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638)
8: (0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195)
9: (-0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.20 + 2.83 = 4.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0356280, upper bound: 0.0356280

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0334691, upper bound: 0.0334691
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0334691, upper bound: 0.0334691
time: 2.17 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.16 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 4.16
Output dim: 8, lower bound: -0.0334691, upper bound: 0.0334691
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 4.16
Output dim: 8, lower bound: -0.0334691, upper bound: 0.0334691

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107032, 0.0107072
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0333050, upper bound: 0.0332985
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0332985, upper bound: 0.0333050
time: 1.83 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107072, 0.0107032
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0333050, upper bound: 0.0332985
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0332985, upper bound: 0.0333050
time: 1.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.87 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.87
Output dim: 8, lower bound: -0.0333050, upper bound: 0.0332985
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.87
Output dim: 8, lower bound: -0.0332985, upper bound: 0.0333050
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.87
Output dim: 8, lower bound: -0.0333050, upper bound: 0.0332985
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.87
Output dim: 8, lower bound: -0.0332985, upper bound: 0.0333050

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107023, 0.0107077
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107034, 0.0107063
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816
time: 1.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107063, 0.0107034
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107077, 0.0107023
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816
time: 1.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.81 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.81
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.81
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.81
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.81
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.81
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.81
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.81
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.81
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106775, 0.0106879
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
time: 1.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106732, 0.0106829
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
time: 1.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106786, 0.0106861
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106751, 0.0106815
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106815, 0.0106751
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106861, 0.0106786
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
time: 1.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106829, 0.0106732
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106879, 0.0106775
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545
time: 1.53 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106200, 0.0106178
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106075, 0.0106275
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106163, 0.0106129
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106031, 0.0106233
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106213, 0.0106160
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106086, 0.0106257
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823
time: 3.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106183, 0.0106114
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
time: 1.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106050, 0.0106224
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823
time: 3.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106224, 0.0106050
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106114, 0.0106183
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
time: 1.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106257, 0.0106086
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106160, 0.0106213
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
time: 1.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106233, 0.0106031
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106128, 0.0106163
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823
time: 3.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106275, 0.0106075
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106178, 0.0106200
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823
time: 3.99 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.97
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105672, 0.0105628
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105612, 0.0105651
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105547, 0.0105730
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105483, 0.0105748
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 2.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105635, 0.0105575
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105599, 0.0105601
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105504, 0.0105669
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105466, 0.0105705
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105686, 0.0105602
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105631, 0.0105633
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105558, 0.0105707
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105504, 0.0105730
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 1.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105655, 0.0105555
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105624, 0.0105586
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105522, 0.0105653
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105493, 0.0105696
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 1.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105696, 0.0105493
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105653, 0.0105522
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105586, 0.0105624
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105555, 0.0105655
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 2.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105730, 0.0105504
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105707, 0.0105558
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105633, 0.0105631
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105602, 0.0105686
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 2.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105705, 0.0105466
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105669, 0.0105504
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105601, 0.0105599
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105575, 0.0105635
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 1.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105748, 0.0105483
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105730, 0.0105547
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105651, 0.0105612
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105628, 0.0105672
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 1.89 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.20
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105577, 0.0105617
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105579, 0.0105615
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105512, 0.0105696
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105512, 0.0105695
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105563, 0.0105566
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105564, 0.0105566
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105468, 0.0105636
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105470, 0.0105634
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105596, 0.0105599
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105598, 0.0105598
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105523, 0.0105674
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105523, 0.0105672
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105589, 0.0105552
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105590, 0.0105551
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105487, 0.0105619
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105489, 0.0105618
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105618, 0.0105489
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105619, 0.0105487
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105551, 0.0105590
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105552, 0.0105589
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105672, 0.0105523
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105674, 0.0105523
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105598, 0.0105598
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105599, 0.0105596
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105634, 0.0105470
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105636, 0.0105468
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105566, 0.0105564
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105566, 0.0105563
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105695, 0.0105512
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105696, 0.0105512
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105615, 0.0105579
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105617, 0.0105577
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.77 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 4.03 + 464.33 = 468.35 seconds
