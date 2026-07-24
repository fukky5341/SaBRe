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
execution time: IAR + RelationalAnalysis = 1.24 + 2.86 = 4.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0356280, upper bound: 0.0356280

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0343658, upper bound: 0.0343658
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0343658, upper bound: 0.0343658
time: 1.92 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.92 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.92
Output dim: 8, lower bound: -0.0343658, upper bound: 0.0343658
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.92
Output dim: 8, lower bound: -0.0343658, upper bound: 0.0343658

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107273, 0.0107274
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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0339132, upper bound: 0.0339189
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0339189, upper bound: 0.0339132
time: 2.26 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107274, 0.0107273
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0341535, upper bound: 0.0342240
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0342240, upper bound: 0.0341535
time: 1.89 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.55
Output dim: 8, lower bound: -0.0339132, upper bound: 0.0339189
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.55
Output dim: 8, lower bound: -0.0339189, upper bound: 0.0339132
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.55
Output dim: 8, lower bound: -0.0341535, upper bound: 0.0342240
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.55
Output dim: 8, lower bound: -0.0342240, upper bound: 0.0341535

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107205, 0.0107184
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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0333746, upper bound: 0.0333819
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0333746, upper bound: 0.0333819
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107185, 0.0107206
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
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311841, upper bound: 0.0311798
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311841, upper bound: 0.0311798
time: 1.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106091, 0.0106140
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0227376, 0.0227929
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
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0314498, upper bound: 0.0315298
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0314498, upper bound: 0.0315298
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106142, 0.0106090
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0227957, 0.0227364
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
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0341888, upper bound: 0.0341191
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0341891, upper bound: 0.0341191
time: 1.91 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.68 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 8, lower bound: -0.0333746, upper bound: 0.0333819
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 8, lower bound: -0.0333746, upper bound: 0.0333819
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 8, lower bound: -0.0311841, upper bound: 0.0311798
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 8, lower bound: -0.0311841, upper bound: 0.0311798
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 8, lower bound: -0.0314498, upper bound: 0.0315298
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 8, lower bound: -0.0314498, upper bound: 0.0315298
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 8, lower bound: -0.0341888, upper bound: 0.0341191
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.68
Output dim: 8, lower bound: -0.0341891, upper bound: 0.0341191

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107170, 0.0107150
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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309513, upper bound: 0.0309534
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309550, upper bound: 0.0309501
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107170, 0.0107149
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
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0328614, upper bound: 0.0328978
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0328933, upper bound: 0.0328616
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106890, 0.0106948
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
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301766, upper bound: 0.0302878
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0302951, upper bound: 0.0301733
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106936, 0.0106911
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
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295237, upper bound: 0.0296255
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296437, upper bound: 0.0295230
time: 1.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105797, 0.0105915
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0223734, 0.0225081
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
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0297698, upper bound: 0.0299637
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298814, upper bound: 0.0298482
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105834, 0.0105846
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0224155, 0.0224286
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
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0314460, upper bound: 0.0315298
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0314498, upper bound: 0.0315238
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106093, 0.0106008
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0228799, 0.0227834
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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0340474, upper bound: 0.0338999
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0339643, upper bound: 0.0339778
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106060, 0.0106056
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0228427, 0.0228379
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
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0331973, upper bound: 0.0332834
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0333422, upper bound: 0.0331237
time: 1.87 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0309513, upper bound: 0.0309534
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0309550, upper bound: 0.0309501
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0328614, upper bound: 0.0328978
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0328933, upper bound: 0.0328616
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0301766, upper bound: 0.0302878
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0302951, upper bound: 0.0301733
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0295237, upper bound: 0.0296255
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0296437, upper bound: 0.0295230
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0297698, upper bound: 0.0299637
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0298814, upper bound: 0.0298482
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0314460, upper bound: 0.0315298
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0314498, upper bound: 0.0315238
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0340474, upper bound: 0.0338999
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0339643, upper bound: 0.0339778
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0331973, upper bound: 0.0332834
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0333422, upper bound: 0.0331237

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106839, 0.0106872
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
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0307685, upper bound: 0.0308218
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308205, upper bound: 0.0307723
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106865, 0.0106819
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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271343, upper bound: 0.0271344
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271343, upper bound: 0.0271344
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107120, 0.0107098
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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0327191, upper bound: 0.0326656
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0326329, upper bound: 0.0327565
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107119, 0.0107095
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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304388, upper bound: 0.0304090
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304409, upper bound: 0.0304080
time: 1.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106480, 0.0106572
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
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296219, upper bound: 0.0297623
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296496, upper bound: 0.0297367
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106514, 0.0106538
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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301762, upper bound: 0.0299470
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301090, upper bound: 0.0300489
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105683, 0.0105716
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0223352, 0.0223719
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
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309610, upper bound: 0.0310481
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309656, upper bound: 0.0310441
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105703, 0.0105695
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0223570, 0.0223483
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
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308848, upper bound: 0.0309837
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309105, upper bound: 0.0309556
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105533, 0.0105307
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0226103, 0.0223520
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
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0330606, upper bound: 0.0330668
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0332013, upper bound: 0.0328893
time: 1.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105391, 0.0105432
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0224485, 0.0224948
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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0319784, upper bound: 0.0319803
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0319784, upper bound: 0.0319803
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105642, 0.0105647
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0223465, 0.0223523
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
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0278239, upper bound: 0.0278818
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0278239, upper bound: 0.0278818
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105654, 0.0105637
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0223603, 0.0223417
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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0282540, upper bound: 0.0282002
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0282540, upper bound: 0.0282002
time: 1.10 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0307685, upper bound: 0.0308218
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0308205, upper bound: 0.0307723
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0271343, upper bound: 0.0271344
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0271343, upper bound: 0.0271344
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0327191, upper bound: 0.0326656
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0326329, upper bound: 0.0327565
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0304388, upper bound: 0.0304090
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0304409, upper bound: 0.0304080
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0296219, upper bound: 0.0297623
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0296496, upper bound: 0.0297367
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0301762, upper bound: 0.0299470
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0301090, upper bound: 0.0300489
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0309610, upper bound: 0.0310481
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0309656, upper bound: 0.0310441
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0308848, upper bound: 0.0309837
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0309105, upper bound: 0.0309556
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0330606, upper bound: 0.0330668
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0332013, upper bound: 0.0328893
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0319784, upper bound: 0.0319803
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0319784, upper bound: 0.0319803
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0278239, upper bound: 0.0278818
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0278239, upper bound: 0.0278818
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0282540, upper bound: 0.0282002
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.32
Output dim: 8, lower bound: -0.0282540, upper bound: 0.0282002

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105651, 0.0105718
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0223987, 0.0224749
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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306029, upper bound: 0.0306512
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305979, upper bound: 0.0306547
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105701, 0.0105684
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0224552, 0.0224356
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301338, upper bound: 0.0300806
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301338, upper bound: 0.0300806
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106545, 0.0106399
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
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0320698, upper bound: 0.0320150
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0320698, upper bound: 0.0320150
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106421, 0.0106518
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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301758, upper bound: 0.0303013
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301768, upper bound: 0.0302986
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106788, 0.0106815
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
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304388, upper bound: 0.0304090
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304388, upper bound: 0.0304090
time: 1.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106814, 0.0106764
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
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304409, upper bound: 0.0304080
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304409, upper bound: 0.0304080
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105629, 0.0105644
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0224313, 0.0224482
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
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293108, upper bound: 0.0295116
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294040, upper bound: 0.0293909
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105614, 0.0105661
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0224135, 0.0224679
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
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304167, upper bound: 0.0305198
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304431, upper bound: 0.0304916
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105654, 0.0105643
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0223042, 0.0222925
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298641, upper bound: 0.0300987
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300072, upper bound: 0.0299706
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105651, 0.0105642
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0223012, 0.0222911
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
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301880, upper bound: 0.0302319
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301880, upper bound: 0.0302319
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105112, 0.0104905
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0221029, 0.0218664
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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0307636, upper bound: 0.0306723
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0307643, upper bound: 0.0306686
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105097, 0.0104886
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0220860, 0.0218445
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
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0316612, upper bound: 0.0314952
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0317962, upper bound: 0.0313513
time: 1.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105145, 0.0105148
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0220934, 0.0220975
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
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0314488, upper bound: 0.0314706
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0314488, upper bound: 0.0314706
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105119, 0.0105185
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0220636, 0.0221397
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
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313953, upper bound: 0.0314265
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0314278, upper bound: 0.0313896
time: 1.48 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0306029, upper bound: 0.0306512
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0305979, upper bound: 0.0306547
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0301338, upper bound: 0.0300806
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0301338, upper bound: 0.0300806
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0320698, upper bound: 0.0320150
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0320698, upper bound: 0.0320150
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0301758, upper bound: 0.0303013
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0301768, upper bound: 0.0302986
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0304388, upper bound: 0.0304090
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0304388, upper bound: 0.0304090
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0304409, upper bound: 0.0304080
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0304409, upper bound: 0.0304080
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0293108, upper bound: 0.0295116
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0294040, upper bound: 0.0293909
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0304167, upper bound: 0.0305198
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0304431, upper bound: 0.0304916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0298641, upper bound: 0.0300987
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0300072, upper bound: 0.0299706
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0301880, upper bound: 0.0302319
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0301880, upper bound: 0.0302319
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0307636, upper bound: 0.0306723
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0307643, upper bound: 0.0306686
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0316612, upper bound: 0.0314952
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0317962, upper bound: 0.0313513
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0314488, upper bound: 0.0314706
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0314488, upper bound: 0.0314706
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0313953, upper bound: 0.0314265
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.17
Output dim: 8, lower bound: -0.0314278, upper bound: 0.0313896

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105654, 0.0105737
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0225033, 0.0225979
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305715, upper bound: 0.0306183
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305696, upper bound: 0.0306199
time: 1.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105665, 0.0105721
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0225159, 0.0225795
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304653, upper bound: 0.0305206
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304639, upper bound: 0.0305243
time: 3.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106477, 0.0106314
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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305163, upper bound: 0.0305745
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306325, upper bound: 0.0304629
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106459, 0.0106327
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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0260121, upper bound: 0.0260151
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0260121, upper bound: 0.0260151
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106086, 0.0106235
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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0288151, upper bound: 0.0289896
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0288765, upper bound: 0.0289411
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106111, 0.0106182
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
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300071, upper bound: 0.0301280
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300054, upper bound: 0.0301321
time: 1.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106644, 0.0106692
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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302732, upper bound: 0.0302384
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302679, upper bound: 0.0302410
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106658, 0.0106670
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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302730, upper bound: 0.0302384
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302680, upper bound: 0.0302413
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106669, 0.0106639
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
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303099, upper bound: 0.0302740
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303068, upper bound: 0.0302766
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106684, 0.0106619
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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302754, upper bound: 0.0302374
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302702, upper bound: 0.0302401
time: 1.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105561, 0.0105610
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0223434, 0.0223994
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293395, upper bound: 0.0295760
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294582, upper bound: 0.0294111
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105562, 0.0105608
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0223450, 0.0223971
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
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304072, upper bound: 0.0304572
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304092, upper bound: 0.0304531
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104772, 0.0104614
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0217337, 0.0215524
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306017, upper bound: 0.0305140
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305974, upper bound: 0.0305158
time: 2.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104755, 0.0104565
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0217138, 0.0214972
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271302, upper bound: 0.0270220
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271302, upper bound: 0.0270220
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104568, 0.0104292
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0214314, 0.0211166
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0310438, upper bound: 0.0308747
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0310462, upper bound: 0.0308728
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104514, 0.0104357
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0213695, 0.0211900
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311782, upper bound: 0.0307286
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311853, upper bound: 0.0307286
time: 1.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105109, 0.0105114
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0220433, 0.0220481
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
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304817, upper bound: 0.0306505
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306280, upper bound: 0.0305024
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105111, 0.0105113
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0220457, 0.0220475
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308595, upper bound: 0.0309121
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308932, upper bound: 0.0308845
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105069, 0.0105135
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0220103, 0.0220864
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298161, upper bound: 0.0299745
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299463, upper bound: 0.0298447
time: 2.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105069, 0.0105136
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0220103, 0.0220875
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298550, upper bound: 0.0299463
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299744, upper bound: 0.0298132
time: 1.55 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0305715, upper bound: 0.0306183
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0305696, upper bound: 0.0306199
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0304653, upper bound: 0.0305206
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0304639, upper bound: 0.0305243
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0305163, upper bound: 0.0305745
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0306325, upper bound: 0.0304629
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0260121, upper bound: 0.0260151
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0260121, upper bound: 0.0260151
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0288151, upper bound: 0.0289896
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0288765, upper bound: 0.0289411
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0300071, upper bound: 0.0301280
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0300054, upper bound: 0.0301321
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0302732, upper bound: 0.0302384
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0302679, upper bound: 0.0302410
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0302730, upper bound: 0.0302384
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0302680, upper bound: 0.0302413
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0303099, upper bound: 0.0302740
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0303068, upper bound: 0.0302766
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0302754, upper bound: 0.0302374
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0302702, upper bound: 0.0302401
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0293395, upper bound: 0.0295760
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0294582, upper bound: 0.0294111
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0304072, upper bound: 0.0304572
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0304092, upper bound: 0.0304531
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0306017, upper bound: 0.0305140
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0305974, upper bound: 0.0305158
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0271302, upper bound: 0.0270220
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0271302, upper bound: 0.0270220
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0310438, upper bound: 0.0308747
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0310462, upper bound: 0.0308728
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0311782, upper bound: 0.0307286
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0311853, upper bound: 0.0307286
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0304817, upper bound: 0.0306505
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0306280, upper bound: 0.0305024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0308595, upper bound: 0.0309121
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0308932, upper bound: 0.0308845
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0298161, upper bound: 0.0299745
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0299463, upper bound: 0.0298447
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0298550, upper bound: 0.0299463
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.41
Output dim: 8, lower bound: -0.0299744, upper bound: 0.0298132

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105612, 0.0105647
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0225008, 0.0225409
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305715, upper bound: 0.0306183
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305715, upper bound: 0.0306156
time: 1.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105564, 0.0105657
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0224463, 0.0225524
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304323, upper bound: 0.0303852
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303263, upper bound: 0.0304830
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105452, 0.0105502
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0222694, 0.0223262
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0291155, upper bound: 0.0292186
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0291557, upper bound: 0.0291686
time: 1.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105446, 0.0105512
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0222625, 0.0223371
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0297769, upper bound: 0.0298350
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0297769, upper bound: 0.0298350
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105943, 0.0105765
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303582, upper bound: 0.0304034
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303428, upper bound: 0.0304057
time: 1.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105884, 0.0105780
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306002, upper bound: 0.0304284
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306012, upper bound: 0.0304323
time: 4.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106441, 0.0106420
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301290, upper bound: 0.0301448
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301807, upper bound: 0.0300963
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106450, 0.0106428
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301268, upper bound: 0.0301475
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301776, upper bound: 0.0300968
time: 2.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105493, 0.0105518
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0222797, 0.0223084
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0297042, upper bound: 0.0297593
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0297042, upper bound: 0.0297593
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105472, 0.0105547
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0222563, 0.0223419
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0302886, upper bound: 0.0302562
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0302147, upper bound: 0.0303325
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104762, 0.0104618
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0218421, 0.0216773
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295757, upper bound: 0.0296397
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0297162, upper bound: 0.0294766
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104768, 0.0104603
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0218496, 0.0216608
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301324, upper bound: 0.0300561
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301320, upper bound: 0.0300546
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104499, 0.0104207
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0213627, 0.0210292
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305956, upper bound: 0.0304324
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305966, upper bound: 0.0304108
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104483, 0.0104218
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0213440, 0.0210420
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300671, upper bound: 0.0300524
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302340, upper bound: 0.0298960
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104442, 0.0104271
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0212970, 0.0211026
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0291007, upper bound: 0.0286390
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0291007, upper bound: 0.0286390
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104429, 0.0104288
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0212821, 0.0211219
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311852, upper bound: 0.0307286
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311853, upper bound: 0.0307283
time: 2.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104689, 0.0104709
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0215381, 0.0215614
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303208, upper bound: 0.0304845
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303100, upper bound: 0.0304981
time: 3.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104694, 0.0104693
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0215433, 0.0215429
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300286, upper bound: 0.0299348
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300636, upper bound: 0.0299145
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105058, 0.0105063
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0219889, 0.0219942
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292576, upper bound: 0.0292863
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292576, upper bound: 0.0292863
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105061, 0.0105068
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0219924, 0.0219995
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0262249, upper bound: 0.0262417
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0262249, upper bound: 0.0262417
time: 1.20 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0305715, upper bound: 0.0306183
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0305715, upper bound: 0.0306156
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0304323, upper bound: 0.0303852
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0303263, upper bound: 0.0304830
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0291155, upper bound: 0.0292186
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0291557, upper bound: 0.0291686
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0297769, upper bound: 0.0298350
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0297769, upper bound: 0.0298350
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0303582, upper bound: 0.0304034
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0303428, upper bound: 0.0304057
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0306002, upper bound: 0.0304284
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0306012, upper bound: 0.0304323
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0301290, upper bound: 0.0301448
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0301807, upper bound: 0.0300963
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0301268, upper bound: 0.0301475
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0301776, upper bound: 0.0300968
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0297042, upper bound: 0.0297593
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0297042, upper bound: 0.0297593
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0302886, upper bound: 0.0302562
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0302147, upper bound: 0.0303325
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0295757, upper bound: 0.0296397
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0297162, upper bound: 0.0294766
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0301324, upper bound: 0.0300561
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0301320, upper bound: 0.0300546
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0305956, upper bound: 0.0304324
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0305966, upper bound: 0.0304108
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0300671, upper bound: 0.0300524
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0302340, upper bound: 0.0298960
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0291007, upper bound: 0.0286390
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0291007, upper bound: 0.0286390
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0311852, upper bound: 0.0307286
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0311853, upper bound: 0.0307283
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0303208, upper bound: 0.0304845
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0303100, upper bound: 0.0304981
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0300286, upper bound: 0.0299348
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0300636, upper bound: 0.0299145
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0292576, upper bound: 0.0292863
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0292576, upper bound: 0.0292863
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0262249, upper bound: 0.0262417
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 8, lower bound: -0.0262249, upper bound: 0.0262417

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105478, 0.0105541
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0224182, 0.0224892
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0268304, upper bound: 0.0268684
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0268304, upper bound: 0.0268684
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105495, 0.0105513
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0224370, 0.0224583
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0268318, upper bound: 0.0268675
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0268318, upper bound: 0.0268675
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104979, 0.0104946
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0220872, 0.0220506
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294512, upper bound: 0.0295421
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295841, upper bound: 0.0294027
time: 1.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104854, 0.0105070
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0219444, 0.0221918
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293471, upper bound: 0.0296270
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294899, upper bound: 0.0295027
time: 1.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105938, 0.0105788
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0240893, upper bound: 0.0241130
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0240893, upper bound: 0.0241130
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105951, 0.0105761
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303120, upper bound: 0.0303745
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303113, upper bound: 0.0303737
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105852, 0.0105710
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304360, upper bound: 0.0302575
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304296, upper bound: 0.0302669
time: 1.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105814, 0.0105753
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306012, upper bound: 0.0304323
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306006, upper bound: 0.0304298
time: 1.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104852, 0.0104841
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0218271, 0.0218150
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286207, upper bound: 0.0287023
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0287404, upper bound: 0.0285997
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104766, 0.0104985
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0217294, 0.0219793
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0253631, upper bound: 0.0254758
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0253631, upper bound: 0.0254758
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104422, 0.0104162
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0213135, 0.0210161
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0285837, upper bound: 0.0283951
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0285837, upper bound: 0.0283951
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104377, 0.0104130
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0212618, 0.0209800
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305912, upper bound: 0.0304088
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305966, upper bound: 0.0304108
time: 1.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104280, 0.0104154
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0211732, 0.0210293
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0288264, upper bound: 0.0285009
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0288288, upper bound: 0.0285008
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104296, 0.0104139
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0211919, 0.0210130
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0310168, upper bound: 0.0305577
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0310146, upper bound: 0.0305623
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104678, 0.0104717
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0216406, 0.0216853
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303208, upper bound: 0.0304845
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303205, upper bound: 0.0304845
time: 1.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104688, 0.0104698
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0216523, 0.0216639
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300386, upper bound: 0.0302238
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300386, upper bound: 0.0302238
time: 1.89 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 6.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0268304, upper bound: 0.0268684
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0268304, upper bound: 0.0268684
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0268318, upper bound: 0.0268675
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0268318, upper bound: 0.0268675
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0294512, upper bound: 0.0295421
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0295841, upper bound: 0.0294027
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0293471, upper bound: 0.0296270
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0294899, upper bound: 0.0295027
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0240893, upper bound: 0.0241130
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0240893, upper bound: 0.0241130
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0303120, upper bound: 0.0303745
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0303113, upper bound: 0.0303737
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0304360, upper bound: 0.0302575
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0304296, upper bound: 0.0302669
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0306012, upper bound: 0.0304323
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0306006, upper bound: 0.0304298
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0286207, upper bound: 0.0287023
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0287404, upper bound: 0.0285997
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0253631, upper bound: 0.0254758
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0253631, upper bound: 0.0254758
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0285837, upper bound: 0.0283951
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0285837, upper bound: 0.0283951
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0305912, upper bound: 0.0304088
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0305966, upper bound: 0.0304108
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0288264, upper bound: 0.0285009
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0288288, upper bound: 0.0285008
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0310168, upper bound: 0.0305577
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0310146, upper bound: 0.0305623
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0303208, upper bound: 0.0304845
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0303205, upper bound: 0.0304845
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0300386, upper bound: 0.0302238
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.82
Output dim: 8, lower bound: -0.0300386, upper bound: 0.0302238

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105926, 0.0105690
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301434, upper bound: 0.0302427
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301818, upper bound: 0.0301975
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105881, 0.0105720
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301771, upper bound: 0.0302332
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301762, upper bound: 0.0302332
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105847, 0.0105724
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0280682, upper bound: 0.0279623
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0280719, upper bound: 0.0279623
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105864, 0.0105705
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295590, upper bound: 0.0295153
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296748, upper bound: 0.0294084
time: 2.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105667, 0.0105629
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304372, upper bound: 0.0302608
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304307, upper bound: 0.0302710
time: 1.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105685, 0.0105606
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0285626, upper bound: 0.0284157
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0285626, upper bound: 0.0284157
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104242, 0.0104011
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0211944, 0.0209303
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0248338, upper bound: 0.0247405
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0248338, upper bound: 0.0247405
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104266, 0.0103995
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0212211, 0.0209126
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0283358, upper bound: 0.0281414
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0283362, upper bound: 0.0281404
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104288, 0.0104140
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0213177, 0.0211493
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286648, upper bound: 0.0283301
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286728, upper bound: 0.0283301
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104304, 0.0104131
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0213364, 0.0211388
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305583, upper bound: 0.0301180
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305679, upper bound: 0.0301151
time: 1.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104534, 0.0104596
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0215648, 0.0216364
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292262, upper bound: 0.0294734
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293034, upper bound: 0.0293646
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104554, 0.0104573
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0215875, 0.0216095
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0261623, upper bound: 0.0262120
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0261623, upper bound: 0.0262120
time: 1.23 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 3.65 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0301434, upper bound: 0.0302427
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0301818, upper bound: 0.0301975
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0301771, upper bound: 0.0302332
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0301762, upper bound: 0.0302332
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0280682, upper bound: 0.0279623
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0280719, upper bound: 0.0279623
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0295590, upper bound: 0.0295153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0296748, upper bound: 0.0294084
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0304372, upper bound: 0.0302608
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0304307, upper bound: 0.0302710
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0285626, upper bound: 0.0284157
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0285626, upper bound: 0.0284157
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0248338, upper bound: 0.0247405
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0248338, upper bound: 0.0247405
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0283358, upper bound: 0.0281414
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0283362, upper bound: 0.0281404
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0286648, upper bound: 0.0283301
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0286728, upper bound: 0.0283301
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0305583, upper bound: 0.0301180
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0305679, upper bound: 0.0301151
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0292262, upper bound: 0.0294734
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0293034, upper bound: 0.0293646
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0261623, upper bound: 0.0262120
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.65
Output dim: 8, lower bound: -0.0261623, upper bound: 0.0262120

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105665, 0.0105647
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271067, upper bound: 0.0269653
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271067, upper bound: 0.0269653
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105683, 0.0105627
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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302582, upper bound: 0.0301400
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0302999, upper bound: 0.0300968
time: 2.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104244, 0.0104048
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0213323, 0.0211080
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304167, upper bound: 0.0299638
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304167, upper bound: 0.0299638
time: 1.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104268, 0.0104071
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0213592, 0.0211347
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300294, upper bound: 0.0296334
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300751, upper bound: 0.0296082
time: 1.59 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 4.86 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 4.86
Output dim: 8, lower bound: -0.0271067, upper bound: 0.0269653
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 4.86
Output dim: 8, lower bound: -0.0271067, upper bound: 0.0269653
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 4.86
Output dim: 8, lower bound: -0.0302582, upper bound: 0.0301400
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.86
Output dim: 8, lower bound: -0.0302999, upper bound: 0.0300968
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 4.86
Output dim: 8, lower bound: -0.0304167, upper bound: 0.0299638
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.86
Output dim: 8, lower bound: -0.0304167, upper bound: 0.0299638
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 4.86
Output dim: 8, lower bound: -0.0300294, upper bound: 0.0296334
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 4.86
Output dim: 8, lower bound: -0.0300751, upper bound: 0.0296082

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104552, 0.0104436
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0216775, 0.0215445
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294312, upper bound: 0.0293476
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295481, upper bound: 0.0292337
time: 2.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104055, 0.0103831
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0211095, 0.0208547
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0280962, upper bound: 0.0277524
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0281001, upper bound: 0.0277524
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104028, 0.0103839
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0210790, 0.0208630
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298039, upper bound: 0.0293994
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298039, upper bound: 0.0293994
time: 1.87 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 4.74 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.74
Output dim: 8, lower bound: -0.0294312, upper bound: 0.0293476
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.74
Output dim: 8, lower bound: -0.0295481, upper bound: 0.0292337
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.74
Output dim: 8, lower bound: -0.0280962, upper bound: 0.0277524
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.74
Output dim: 8, lower bound: -0.0281001, upper bound: 0.0277524
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.74
Output dim: 8, lower bound: -0.0298039, upper bound: 0.0293994
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.74
Output dim: 8, lower bound: -0.0298039, upper bound: 0.0293994

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 4.10 + 594.98 = 599.08 seconds
