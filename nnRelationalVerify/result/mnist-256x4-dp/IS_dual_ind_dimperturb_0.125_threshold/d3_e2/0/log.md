## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00026408


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0011433, -0.0004185, -0.0011433, -0.0004185, -0.0005897, 0.0005897)
1: (-0.0042369, -0.0039800, -0.0042369, -0.0039800, -0.0002243, 0.0002243)
2: (0.0130383, 0.0140342, 0.0130383, 0.0140342, -0.0007970, 0.0007970)
3: (1.0084040, 1.0090557, 1.0084040, 1.0090557, -0.0006517, 0.0006517)
4: (-0.0038804, -0.0037129, -0.0038804, -0.0037129, -0.0001325, 0.0001325)
5: (0.0030682, 0.0036287, 0.0030682, 0.0036287, -0.0004548, 0.0004548)
6: (-0.0024377, -0.0023797, -0.0024377, -0.0023797, -0.0000580, 0.0000580)
7: (-0.0129467, -0.0121339, -0.0129467, -0.0121339, -0.0007963, 0.0007963)
8: (-0.0093802, -0.0075346, -0.0093802, -0.0075346, -0.0014513, 0.0014513)
9: (-0.0006222, 0.0003089, -0.0006222, 0.0003089, -0.0007285, 0.0007285)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.63 + 1.40 = 3.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0003623, upper bound: 0.0003623

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003337, upper bound: 0.0003423
time: 0.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003600, upper bound: 0.0003600
time: 0.52 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.18 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 3, lower bound: -0.0003337, upper bound: 0.0003423
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 3, lower bound: -0.0003600, upper bound: 0.0003600

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0012129, -0.0004513, -0.0011432, -0.0004220, -0.0006347, 0.0005312
1: -0.0042533, -0.0039955, -0.0042369, -0.0039818, -0.0002288, 0.0002030
2: 0.0129518, 0.0139838, 0.0130383, 0.0140288, -0.0008410, 0.0007032
3: 1.0084418, 1.0090969, 1.0084087, 1.0090557, -0.0006139, 0.0006882
4: -0.0038711, -0.0036999, -0.0038794, -0.0037129, -0.0001149, 0.0001374
5: 0.0030151, 0.0036029, 0.0030683, 0.0036259, -0.0004881, 0.0004085
6: -0.0024372, -0.0023763, -0.0024375, -0.0023797, -0.0000575, 0.0000611
7: -0.0129429, -0.0119970, -0.0129463, -0.0121339, -0.0007902, 0.0009314
8: -0.0092709, -0.0074020, -0.0093685, -0.0075346, -0.0012467, 0.0014934
9: -0.0006830, 0.0002504, -0.0006222, 0.0003026, -0.0007449, 0.0006186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003251, upper bound: 0.0003251
time: 0.48 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003251, upper bound: 0.0003423
time: 0.54 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0011432, -0.0004230, -0.0011433, -0.0004185, -0.0005896, 0.0005630
1: -0.0042369, -0.0039822, -0.0042369, -0.0039800, -0.0002243, 0.0002128
2: 0.0130383, 0.0140273, 0.0130383, 0.0140342, -0.0007967, 0.0007528
3: 1.0084090, 1.0090557, 1.0084040, 1.0090557, -0.0006467, 0.0006517
4: -0.0038792, -0.0037129, -0.0038804, -0.0037129, -0.0001242, 0.0001325
5: 0.0030683, 0.0036252, 0.0030682, 0.0036287, -0.0004547, 0.0004334
6: -0.0024374, -0.0023797, -0.0024377, -0.0023797, -0.0000577, 0.0000580
7: -0.0129462, -0.0121339, -0.0129467, -0.0121339, -0.0007948, 0.0007963
8: -0.0093652, -0.0075346, -0.0093802, -0.0075346, -0.0013554, 0.0014511
9: -0.0006222, 0.0003008, -0.0006222, 0.0003089, -0.0007285, 0.0006778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003423, upper bound: 0.0003337
time: 0.47 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003423, upper bound: 0.0003600
time: 0.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.57 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 3, lower bound: -0.0003251, upper bound: 0.0003251
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 3, lower bound: -0.0003251, upper bound: 0.0003423
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 3, lower bound: -0.0003423, upper bound: 0.0003337
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 3, lower bound: -0.0003423, upper bound: 0.0003600

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012129, -0.0004513, -0.0012129, -0.0004513, -0.0005838, 0.0005838
1: -0.0042533, -0.0039955, -0.0042533, -0.0039955, -0.0002099, 0.0002099
2: 0.0129518, 0.0139838, 0.0129518, 0.0139838, -0.0007621, 0.0007621
3: 1.0084418, 1.0090969, 1.0084418, 1.0090969, -0.0006551, 0.0006551
4: -0.0038711, -0.0036999, -0.0038711, -0.0036999, -0.0001218, 0.0001218
5: 0.0030151, 0.0036029, 0.0030151, 0.0036029, -0.0004480, 0.0004480
6: -0.0024372, -0.0023763, -0.0024372, -0.0023763, -0.0000609, 0.0000609
7: -0.0129429, -0.0119970, -0.0129429, -0.0119970, -0.0009262, 0.0009262
8: -0.0092709, -0.0074020, -0.0092709, -0.0074020, -0.0013101, 0.0013101
9: -0.0006830, 0.0002504, -0.0006830, 0.0002504, -0.0006441, 0.0006441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003180, upper bound: 0.0003187
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003161
time: 0.49 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012129, -0.0004513, -0.0011432, -0.0004230, -0.0006376, 0.0005311
1: -0.0042533, -0.0039955, -0.0042369, -0.0039822, -0.0002298, 0.0002030
2: 0.0129518, 0.0139838, 0.0130383, 0.0140273, -0.0008455, 0.0007032
3: 1.0084418, 1.0090969, 1.0084090, 1.0090557, -0.0006139, 0.0006878
4: -0.0038711, -0.0036999, -0.0038792, -0.0037129, -0.0001149, 0.0001382
5: 0.0030151, 0.0036029, 0.0030683, 0.0036252, -0.0004904, 0.0004085
6: -0.0024372, -0.0023763, -0.0024374, -0.0023797, -0.0000575, 0.0000611
7: -0.0129429, -0.0119970, -0.0129462, -0.0121339, -0.0007902, 0.0009318
8: -0.0092709, -0.0074020, -0.0093652, -0.0075346, -0.0012466, 0.0015032
9: -0.0006830, 0.0002504, -0.0006222, 0.0003008, -0.0007502, 0.0006186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003180, upper bound: 0.0003383
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003369
time: 0.52 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011432, -0.0004230, -0.0012129, -0.0004513, -0.0005311, 0.0006376
1: -0.0042369, -0.0039822, -0.0042533, -0.0039955, -0.0002030, 0.0002298
2: 0.0130383, 0.0140273, 0.0129518, 0.0139838, -0.0007032, 0.0008455
3: 1.0084090, 1.0090557, 1.0084418, 1.0090969, -0.0006878, 0.0006139
4: -0.0038792, -0.0037129, -0.0038711, -0.0036999, -0.0001382, 0.0001149
5: 0.0030683, 0.0036252, 0.0030151, 0.0036029, -0.0004085, 0.0004904
6: -0.0024374, -0.0023797, -0.0024372, -0.0023763, -0.0000611, 0.0000575
7: -0.0129462, -0.0121339, -0.0129429, -0.0119970, -0.0009318, 0.0007902
8: -0.0093652, -0.0075346, -0.0092709, -0.0074020, -0.0015032, 0.0012466
9: -0.0006222, 0.0003008, -0.0006830, 0.0002504, -0.0006186, 0.0007502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003372, upper bound: 0.0003292
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003369, upper bound: 0.0003287
time: 0.50 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011432, -0.0004230, -0.0011432, -0.0004230, -0.0005629, 0.0005629
1: -0.0042369, -0.0039822, -0.0042369, -0.0039822, -0.0002128, 0.0002128
2: 0.0130383, 0.0140273, 0.0130383, 0.0140273, -0.0007527, 0.0007527
3: 1.0084090, 1.0090557, 1.0084090, 1.0090557, -0.0006467, 0.0006467
4: -0.0038792, -0.0037129, -0.0038792, -0.0037129, -0.0001242, 0.0001242
5: 0.0030683, 0.0036252, 0.0030683, 0.0036252, -0.0004334, 0.0004334
6: -0.0024374, -0.0023797, -0.0024374, -0.0023797, -0.0000577, 0.0000577
7: -0.0129462, -0.0121339, -0.0129462, -0.0121339, -0.0007948, 0.0007948
8: -0.0093652, -0.0075346, -0.0093652, -0.0075346, -0.0013552, 0.0013552
9: -0.0006222, 0.0003008, -0.0006222, 0.0003008, -0.0006777, 0.0006777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003372, upper bound: 0.0003572
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003369, upper bound: 0.0003571
time: 0.53 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.66 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 3, lower bound: -0.0003180, upper bound: 0.0003187
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003161
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 3, lower bound: -0.0003180, upper bound: 0.0003383
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003369
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 3, lower bound: -0.0003372, upper bound: 0.0003292
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 3, lower bound: -0.0003369, upper bound: 0.0003287
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 3, lower bound: -0.0003372, upper bound: 0.0003572
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 3, lower bound: -0.0003369, upper bound: 0.0003571

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011887, -0.0004532, -0.0012129, -0.0004513, -0.0005583, 0.0005826
1: -0.0042495, -0.0039966, -0.0042533, -0.0039955, -0.0002044, 0.0002077
2: 0.0129795, 0.0139810, 0.0129518, 0.0139838, -0.0007324, 0.0007603
3: 1.0084455, 1.0090874, 1.0084418, 1.0090969, -0.0006509, 0.0006456
4: -0.0038705, -0.0037036, -0.0038711, -0.0036999, -0.0001215, 0.0001176
5: 0.0030335, 0.0036015, 0.0030151, 0.0036029, -0.0004288, 0.0004471
6: -0.0024358, -0.0023771, -0.0024372, -0.0023763, -0.0000595, 0.0000601
7: -0.0129427, -0.0120555, -0.0129429, -0.0119970, -0.0009261, 0.0008668
8: -0.0092648, -0.0074362, -0.0092709, -0.0074020, -0.0013061, 0.0012645
9: -0.0006688, 0.0002472, -0.0006830, 0.0002504, -0.0006237, 0.0006420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003161
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003161
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011690, -0.0004461, -0.0012048, -0.0004518, -0.0005459, 0.0005929
1: -0.0042464, -0.0039930, -0.0042521, -0.0039958, -0.0002040, 0.0002115
2: 0.0130022, 0.0139918, 0.0129611, 0.0139830, -0.0007191, 0.0007788
3: 1.0084372, 1.0090797, 1.0084430, 1.0090936, -0.0006481, 0.0006367
4: -0.0038726, -0.0037067, -0.0038709, -0.0037011, -0.0001255, 0.0001165
5: 0.0030483, 0.0036070, 0.0030213, 0.0036025, -0.0004194, 0.0004553
6: -0.0024356, -0.0023777, -0.0024368, -0.0023766, -0.0000590, 0.0000590
7: -0.0129435, -0.0121070, -0.0129428, -0.0120178, -0.0009074, 0.0008163
8: -0.0092884, -0.0074654, -0.0092693, -0.0074137, -0.0013571, 0.0012618
9: -0.0006572, 0.0002598, -0.0006783, 0.0002496, -0.0006242, 0.0006707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003161
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003161
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011887, -0.0004532, -0.0011432, -0.0004230, -0.0006129, 0.0005300
1: -0.0042495, -0.0039966, -0.0042369, -0.0039822, -0.0002243, 0.0002009
2: 0.0129795, 0.0139810, 0.0130383, 0.0140273, -0.0008161, 0.0007014
3: 1.0084455, 1.0090874, 1.0084090, 1.0090557, -0.0006102, 0.0006763
4: -0.0038705, -0.0037036, -0.0038792, -0.0037129, -0.0001146, 0.0001337
5: 0.0030335, 0.0036015, 0.0030683, 0.0036252, -0.0004718, 0.0004075
6: -0.0024358, -0.0023771, -0.0024374, -0.0023797, -0.0000561, 0.0000603
7: -0.0129427, -0.0120555, -0.0129462, -0.0121339, -0.0007900, 0.0008722
8: -0.0092648, -0.0074362, -0.0093652, -0.0075346, -0.0012426, 0.0014572
9: -0.0006688, 0.0002472, -0.0006222, 0.0003008, -0.0007295, 0.0006165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003287, upper bound: 0.0003369
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003287, upper bound: 0.0003369
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011690, -0.0004461, -0.0011353, -0.0004235, -0.0005994, 0.0005401
1: -0.0042464, -0.0039930, -0.0042357, -0.0039825, -0.0002241, 0.0002045
2: 0.0130022, 0.0139918, 0.0130474, 0.0140265, -0.0008046, 0.0007209
3: 1.0084372, 1.0090797, 1.0084100, 1.0090528, -0.0006156, 0.0006697
4: -0.0038726, -0.0037067, -0.0038790, -0.0037141, -0.0001187, 0.0001331
5: 0.0030483, 0.0036070, 0.0030743, 0.0036248, -0.0004617, 0.0004158
6: -0.0024356, -0.0023777, -0.0024371, -0.0023800, -0.0000556, 0.0000593
7: -0.0129435, -0.0121070, -0.0129461, -0.0121543, -0.0007718, 0.0008222
8: -0.0092884, -0.0074654, -0.0093636, -0.0075461, -0.0012932, 0.0014564
9: -0.0006572, 0.0002598, -0.0006176, 0.0003000, -0.0007308, 0.0006447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003287, upper bound: 0.0003369
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003287, upper bound: 0.0003369
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011193, -0.0004250, -0.0012129, -0.0004513, -0.0005058, 0.0006364
1: -0.0042330, -0.0039833, -0.0042533, -0.0039955, -0.0001979, 0.0002277
2: 0.0130661, 0.0140243, 0.0129518, 0.0139838, -0.0006735, 0.0008436
3: 1.0084126, 1.0090461, 1.0084418, 1.0090969, -0.0006748, 0.0006043
4: -0.0038786, -0.0037167, -0.0038711, -0.0036999, -0.0001378, 0.0001103
5: 0.0030864, 0.0036236, 0.0030151, 0.0036029, -0.0003893, 0.0004895
6: -0.0024363, -0.0023805, -0.0024372, -0.0023763, -0.0000600, 0.0000567
7: -0.0129460, -0.0121902, -0.0129429, -0.0119970, -0.0009316, 0.0007335
8: -0.0093587, -0.0075699, -0.0092709, -0.0074020, -0.0014990, 0.0012014
9: -0.0006075, 0.0002973, -0.0006830, 0.0002504, -0.0005994, 0.0007480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003369, upper bound: 0.0003287
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003369, upper bound: 0.0003287
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011025, -0.0004181, -0.0012048, -0.0004518, -0.0004949, 0.0006433
1: -0.0042303, -0.0039799, -0.0042521, -0.0039958, -0.0001972, 0.0002311
2: 0.0130850, 0.0140349, 0.0129611, 0.0139830, -0.0006647, 0.0008577
3: 1.0084045, 1.0090395, 1.0084430, 1.0090936, -0.0006737, 0.0005965
4: -0.0038806, -0.0037193, -0.0038709, -0.0037011, -0.0001411, 0.0001098
5: 0.0030990, 0.0036291, 0.0030213, 0.0036025, -0.0003812, 0.0004952
6: -0.0024365, -0.0023811, -0.0024368, -0.0023766, -0.0000599, 0.0000557
7: -0.0129468, -0.0122333, -0.0129428, -0.0120178, -0.0009126, 0.0006916
8: -0.0093817, -0.0075944, -0.0092693, -0.0074137, -0.0015405, 0.0011987
9: -0.0005975, 0.0003096, -0.0006783, 0.0002496, -0.0005988, 0.0007715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003369, upper bound: 0.0003287
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003369, upper bound: 0.0003287
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011193, -0.0004250, -0.0011432, -0.0004230, -0.0005386, 0.0005616
1: -0.0042330, -0.0039833, -0.0042369, -0.0039822, -0.0002074, 0.0002105
2: 0.0130661, 0.0140243, 0.0130383, 0.0140273, -0.0007225, 0.0007507
3: 1.0084126, 1.0090461, 1.0084090, 1.0090557, -0.0006360, 0.0006368
4: -0.0038786, -0.0037167, -0.0038792, -0.0037129, -0.0001238, 0.0001195
5: 0.0030864, 0.0036236, 0.0030683, 0.0036252, -0.0004149, 0.0004324
6: -0.0024363, -0.0023805, -0.0024374, -0.0023797, -0.0000566, 0.0000569
7: -0.0129460, -0.0121902, -0.0129462, -0.0121339, -0.0007946, 0.0007380
8: -0.0093587, -0.0075699, -0.0093652, -0.0075346, -0.0013509, 0.0013091
9: -0.0006075, 0.0002973, -0.0006222, 0.0003008, -0.0006573, 0.0006754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003577, upper bound: 0.0003570
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003577, upper bound: 0.0003571
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011025, -0.0004181, -0.0011353, -0.0004235, -0.0005263, 0.0005705
1: -0.0042303, -0.0039799, -0.0042357, -0.0039825, -0.0002069, 0.0002144
2: 0.0130850, 0.0140349, 0.0130474, 0.0140265, -0.0007138, 0.0007684
3: 1.0084045, 1.0090395, 1.0084100, 1.0090528, -0.0006359, 0.0006295
4: -0.0038806, -0.0037193, -0.0038790, -0.0037141, -0.0001276, 0.0001191
5: 0.0030990, 0.0036291, 0.0030743, 0.0036248, -0.0004060, 0.0004396
6: -0.0024365, -0.0023811, -0.0024371, -0.0023800, -0.0000565, 0.0000560
7: -0.0129468, -0.0122333, -0.0129461, -0.0121543, -0.0007763, 0.0006959
8: -0.0093817, -0.0075944, -0.0093636, -0.0075461, -0.0013981, 0.0013077
9: -0.0005975, 0.0003096, -0.0006176, 0.0003000, -0.0006579, 0.0007022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003270, upper bound: 0.0003307
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
time: 0.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.69 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003161
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003161
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003161
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003161
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003287, upper bound: 0.0003369
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003287, upper bound: 0.0003369
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003287, upper bound: 0.0003369
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003287, upper bound: 0.0003369
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003369, upper bound: 0.0003287
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003369, upper bound: 0.0003287
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003369, upper bound: 0.0003287
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003369, upper bound: 0.0003287
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003577, upper bound: 0.0003570
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003577, upper bound: 0.0003571
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003270, upper bound: 0.0003307
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011887, -0.0004532, -0.0011887, -0.0004532, -0.0005571, 0.0005571
1: -0.0042495, -0.0039966, -0.0042495, -0.0039966, -0.0002022, 0.0002022
2: 0.0129795, 0.0139810, 0.0129795, 0.0139810, -0.0007306, 0.0007306
3: 1.0084455, 1.0090874, 1.0084455, 1.0090874, -0.0006372, 0.0006372
4: -0.0038705, -0.0037036, -0.0038705, -0.0037036, -0.0001173, 0.0001173
5: 0.0030335, 0.0036015, 0.0030335, 0.0036015, -0.0004278, 0.0004278
6: -0.0024358, -0.0023771, -0.0024358, -0.0023771, -0.0000587, 0.0000587
7: -0.0129427, -0.0120555, -0.0129427, -0.0120555, -0.0008666, 0.0008666
8: -0.0092648, -0.0074362, -0.0092648, -0.0074362, -0.0012606, 0.0012606
9: -0.0006688, 0.0002472, -0.0006688, 0.0002472, -0.0006216, 0.0006216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003123, upper bound: 0.0003161
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003164, upper bound: 0.0003172
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011887, -0.0004532, -0.0011690, -0.0004461, -0.0005746, 0.0005454
1: -0.0042495, -0.0039966, -0.0042464, -0.0039930, -0.0002066, 0.0002045
2: 0.0129795, 0.0139810, 0.0130022, 0.0139918, -0.0007575, 0.0007189
3: 1.0084455, 1.0090874, 1.0084372, 1.0090797, -0.0006342, 0.0006360
4: -0.0038705, -0.0037036, -0.0038726, -0.0037067, -0.0001173, 0.0001223
5: 0.0030335, 0.0036015, 0.0030483, 0.0036070, -0.0004416, 0.0004191
6: -0.0024358, -0.0023771, -0.0024356, -0.0023777, -0.0000581, 0.0000585
7: -0.0129427, -0.0120555, -0.0129435, -0.0121070, -0.0008160, 0.0008687
8: -0.0092648, -0.0074362, -0.0092884, -0.0074654, -0.0012713, 0.0013190
9: -0.0006688, 0.0002472, -0.0006572, 0.0002598, -0.0006528, 0.0006298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003123, upper bound: 0.0003161
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003164, upper bound: 0.0003172
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011690, -0.0004461, -0.0011887, -0.0004532, -0.0005454, 0.0005746
1: -0.0042464, -0.0039930, -0.0042495, -0.0039966, -0.0002045, 0.0002066
2: 0.0130022, 0.0139918, 0.0129795, 0.0139810, -0.0007189, 0.0007575
3: 1.0084372, 1.0090797, 1.0084455, 1.0090874, -0.0006360, 0.0006342
4: -0.0038726, -0.0037067, -0.0038705, -0.0037036, -0.0001223, 0.0001173
5: 0.0030483, 0.0036070, 0.0030335, 0.0036015, -0.0004191, 0.0004416
6: -0.0024356, -0.0023777, -0.0024358, -0.0023771, -0.0000585, 0.0000581
7: -0.0129435, -0.0121070, -0.0129427, -0.0120555, -0.0008687, 0.0008160
8: -0.0092884, -0.0074654, -0.0092648, -0.0074362, -0.0013190, 0.0012713
9: -0.0006572, 0.0002598, -0.0006688, 0.0002472, -0.0006298, 0.0006528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003103, upper bound: 0.0003127
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003145, upper bound: 0.0003145
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011690, -0.0004461, -0.0011690, -0.0004461, -0.0005526, 0.0005526
1: -0.0042464, -0.0039930, -0.0042464, -0.0039930, -0.0002033, 0.0002033
2: 0.0130022, 0.0139918, 0.0130022, 0.0139918, -0.0007295, 0.0007295
3: 1.0084372, 1.0090797, 1.0084372, 1.0090797, -0.0006291, 0.0006291
4: -0.0038726, -0.0037067, -0.0038726, -0.0037067, -0.0001185, 0.0001185
5: 0.0030483, 0.0036070, 0.0030483, 0.0036070, -0.0004247, 0.0004247
6: -0.0024356, -0.0023777, -0.0024356, -0.0023777, -0.0000579, 0.0000579
7: -0.0129435, -0.0121070, -0.0129435, -0.0121070, -0.0008171, 0.0008171
8: -0.0092884, -0.0074654, -0.0092884, -0.0074654, -0.0012843, 0.0012843
9: -0.0006572, 0.0002598, -0.0006572, 0.0002598, -0.0006362, 0.0006362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003103, upper bound: 0.0003127
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003145, upper bound: 0.0003145
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011887, -0.0004532, -0.0011193, -0.0004250, -0.0006117, 0.0005046
1: -0.0042495, -0.0039966, -0.0042330, -0.0039833, -0.0002222, 0.0001958
2: 0.0129795, 0.0139810, 0.0130661, 0.0140243, -0.0008142, 0.0006717
3: 1.0084455, 1.0090874, 1.0084126, 1.0090461, -0.0006006, 0.0006611
4: -0.0038705, -0.0037036, -0.0038786, -0.0037167, -0.0001100, 0.0001333
5: 0.0030335, 0.0036015, 0.0030864, 0.0036236, -0.0004708, 0.0003884
6: -0.0024358, -0.0023771, -0.0024363, -0.0023805, -0.0000553, 0.0000592
7: -0.0129427, -0.0120555, -0.0129460, -0.0121902, -0.0007334, 0.0008721
8: -0.0092648, -0.0074362, -0.0093587, -0.0075699, -0.0011974, 0.0014531
9: -0.0006688, 0.0002472, -0.0006075, 0.0002973, -0.0007272, 0.0005973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003247, upper bound: 0.0003356
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003279, upper bound: 0.0003368
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011887, -0.0004532, -0.0011025, -0.0004181, -0.0006263, 0.0004941
1: -0.0042495, -0.0039966, -0.0042303, -0.0039799, -0.0002262, 0.0001968
2: 0.0129795, 0.0139810, 0.0130850, 0.0140349, -0.0008366, 0.0006659
3: 1.0084455, 1.0090874, 1.0084045, 1.0090395, -0.0005940, 0.0006616
4: -0.0038705, -0.0037036, -0.0038806, -0.0037193, -0.0001101, 0.0001375
5: 0.0030335, 0.0036015, 0.0030990, 0.0036291, -0.0004823, 0.0003808
6: -0.0024358, -0.0023771, -0.0024365, -0.0023811, -0.0000548, 0.0000594
7: -0.0129427, -0.0120555, -0.0129468, -0.0122333, -0.0006914, 0.0008738
8: -0.0092648, -0.0074362, -0.0093817, -0.0075944, -0.0012023, 0.0015016
9: -0.0006688, 0.0002472, -0.0005975, 0.0003096, -0.0007532, 0.0006009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003247, upper bound: 0.0003356
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003279, upper bound: 0.0003368
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011690, -0.0004461, -0.0011193, -0.0004250, -0.0005981, 0.0005221
1: -0.0042464, -0.0039930, -0.0042330, -0.0039833, -0.0002242, 0.0002002
2: 0.0130022, 0.0139918, 0.0130661, 0.0140243, -0.0008063, 0.0006987
3: 1.0084372, 1.0090797, 1.0084126, 1.0090461, -0.0006089, 0.0006662
4: -0.0038726, -0.0037067, -0.0038786, -0.0037167, -0.0001150, 0.0001337
5: 0.0030483, 0.0036070, 0.0030864, 0.0036236, -0.0004608, 0.0004022
6: -0.0024356, -0.0023777, -0.0024363, -0.0023805, -0.0000551, 0.0000586
7: -0.0129435, -0.0121070, -0.0129460, -0.0121902, -0.0007354, 0.0008218
8: -0.0092884, -0.0074654, -0.0093587, -0.0075699, -0.0012559, 0.0014634
9: -0.0006572, 0.0002598, -0.0006075, 0.0002973, -0.0007349, 0.0006285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003239, upper bound: 0.0003336
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003273, upper bound: 0.0003353
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011690, -0.0004461, -0.0011025, -0.0004181, -0.0006055, 0.0005017
1: -0.0042464, -0.0039930, -0.0042303, -0.0039799, -0.0002238, 0.0001965
2: 0.0130022, 0.0139918, 0.0130850, 0.0140349, -0.0008139, 0.0006751
3: 1.0084372, 1.0090797, 1.0084045, 1.0090395, -0.0006024, 0.0006568
4: -0.0038726, -0.0037067, -0.0038806, -0.0037193, -0.0001118, 0.0001349
5: 0.0030483, 0.0036070, 0.0030990, 0.0036291, -0.0004665, 0.0003865
6: -0.0024356, -0.0023777, -0.0024365, -0.0023811, -0.0000545, 0.0000587
7: -0.0129435, -0.0121070, -0.0129468, -0.0122333, -0.0006924, 0.0008229
8: -0.0092884, -0.0074654, -0.0093817, -0.0075944, -0.0012212, 0.0014765
9: -0.0006572, 0.0002598, -0.0005975, 0.0003096, -0.0007416, 0.0006108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003239, upper bound: 0.0003336
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003273, upper bound: 0.0003353
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011193, -0.0004250, -0.0011887, -0.0004532, -0.0005046, 0.0006117
1: -0.0042330, -0.0039833, -0.0042495, -0.0039966, -0.0001958, 0.0002222
2: 0.0130661, 0.0140243, 0.0129795, 0.0139810, -0.0006717, 0.0008142
3: 1.0084126, 1.0090461, 1.0084455, 1.0090874, -0.0006611, 0.0006006
4: -0.0038786, -0.0037167, -0.0038705, -0.0037036, -0.0001333, 0.0001100
5: 0.0030864, 0.0036236, 0.0030335, 0.0036015, -0.0003884, 0.0004708
6: -0.0024363, -0.0023805, -0.0024358, -0.0023771, -0.0000592, 0.0000553
7: -0.0129460, -0.0121902, -0.0129427, -0.0120555, -0.0008721, 0.0007334
8: -0.0093587, -0.0075699, -0.0092648, -0.0074362, -0.0014531, 0.0011974
9: -0.0006075, 0.0002973, -0.0006688, 0.0002472, -0.0005973, 0.0007272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003298, upper bound: 0.0003292
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003350, upper bound: 0.0003288
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011193, -0.0004250, -0.0011690, -0.0004461, -0.0005221, 0.0005981
1: -0.0042330, -0.0039833, -0.0042464, -0.0039930, -0.0002002, 0.0002242
2: 0.0130661, 0.0140243, 0.0130022, 0.0139918, -0.0006987, 0.0008063
3: 1.0084126, 1.0090461, 1.0084372, 1.0090797, -0.0006662, 0.0006089
4: -0.0038786, -0.0037167, -0.0038726, -0.0037067, -0.0001337, 0.0001150
5: 0.0030864, 0.0036236, 0.0030483, 0.0036070, -0.0004022, 0.0004608
6: -0.0024363, -0.0023805, -0.0024356, -0.0023777, -0.0000586, 0.0000551
7: -0.0129460, -0.0121902, -0.0129435, -0.0121070, -0.0008218, 0.0007354
8: -0.0093587, -0.0075699, -0.0092884, -0.0074654, -0.0014634, 0.0012559
9: -0.0006075, 0.0002973, -0.0006572, 0.0002598, -0.0006285, 0.0007349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003298, upper bound: 0.0003292
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003350, upper bound: 0.0003288
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011025, -0.0004181, -0.0011887, -0.0004532, -0.0004941, 0.0006263
1: -0.0042303, -0.0039799, -0.0042495, -0.0039966, -0.0001968, 0.0002262
2: 0.0130850, 0.0140349, 0.0129795, 0.0139810, -0.0006659, 0.0008366
3: 1.0084045, 1.0090395, 1.0084455, 1.0090874, -0.0006616, 0.0005940
4: -0.0038806, -0.0037193, -0.0038705, -0.0037036, -0.0001375, 0.0001101
5: 0.0030990, 0.0036291, 0.0030335, 0.0036015, -0.0003808, 0.0004823
6: -0.0024365, -0.0023811, -0.0024358, -0.0023771, -0.0000594, 0.0000548
7: -0.0129468, -0.0122333, -0.0129427, -0.0120555, -0.0008738, 0.0006914
8: -0.0093817, -0.0075944, -0.0092648, -0.0074362, -0.0015016, 0.0012023
9: -0.0005975, 0.0003096, -0.0006688, 0.0002472, -0.0006009, 0.0007532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003294, upper bound: 0.0003287
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003348, upper bound: 0.0003285
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011025, -0.0004181, -0.0011690, -0.0004461, -0.0005017, 0.0006055
1: -0.0042303, -0.0039799, -0.0042464, -0.0039930, -0.0001965, 0.0002238
2: 0.0130850, 0.0140349, 0.0130022, 0.0139918, -0.0006751, 0.0008139
3: 1.0084045, 1.0090395, 1.0084372, 1.0090797, -0.0006568, 0.0006024
4: -0.0038806, -0.0037193, -0.0038726, -0.0037067, -0.0001349, 0.0001118
5: 0.0030990, 0.0036291, 0.0030483, 0.0036070, -0.0003865, 0.0004665
6: -0.0024365, -0.0023811, -0.0024356, -0.0023777, -0.0000587, 0.0000545
7: -0.0129468, -0.0122333, -0.0129435, -0.0121070, -0.0008229, 0.0006924
8: -0.0093817, -0.0075944, -0.0092884, -0.0074654, -0.0014765, 0.0012212
9: -0.0005975, 0.0003096, -0.0006572, 0.0002598, -0.0006108, 0.0007416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003294, upper bound: 0.0003287
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003348, upper bound: 0.0003285
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011193, -0.0004250, -0.0011193, -0.0004250, -0.0005373, 0.0005373
1: -0.0042330, -0.0039833, -0.0042330, -0.0039833, -0.0002051, 0.0002051
2: 0.0130661, 0.0140243, 0.0130661, 0.0140243, -0.0007205, 0.0007205
3: 1.0084126, 1.0090461, 1.0084126, 1.0090461, -0.0006224, 0.0006224
4: -0.0038786, -0.0037167, -0.0038786, -0.0037167, -0.0001192, 0.0001192
5: 0.0030864, 0.0036236, 0.0030864, 0.0036236, -0.0004139, 0.0004139
6: -0.0024363, -0.0023805, -0.0024363, -0.0023805, -0.0000558, 0.0000558
7: -0.0129460, -0.0121902, -0.0129460, -0.0121902, -0.0007379, 0.0007379
8: -0.0093587, -0.0075699, -0.0093587, -0.0075699, -0.0013048, 0.0013048
9: -0.0006075, 0.0002973, -0.0006075, 0.0002973, -0.0006550, 0.0006550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003322, upper bound: 0.0003246
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003144, upper bound: 0.0003132
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011193, -0.0004250, -0.0011025, -0.0004181, -0.0005537, 0.0005266
1: -0.0042330, -0.0039833, -0.0042303, -0.0039799, -0.0002096, 0.0002071
2: 0.0130661, 0.0140243, 0.0130850, 0.0140349, -0.0007457, 0.0007159
3: 1.0084126, 1.0090461, 1.0084045, 1.0090395, -0.0006269, 0.0006239
4: -0.0038786, -0.0037167, -0.0038806, -0.0037193, -0.0001196, 0.0001239
5: 0.0030864, 0.0036236, 0.0030990, 0.0036291, -0.0004268, 0.0004064
6: -0.0024363, -0.0023805, -0.0024365, -0.0023811, -0.0000553, 0.0000559
7: -0.0129460, -0.0121902, -0.0129468, -0.0122333, -0.0006957, 0.0007398
8: -0.0093587, -0.0075699, -0.0093817, -0.0075944, -0.0013153, 0.0013595
9: -0.0006075, 0.0002973, -0.0005975, 0.0003096, -0.0006842, 0.0006625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003322, upper bound: 0.0003246
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003144, upper bound: 0.0003132
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011025, -0.0004181, -0.0011352, -0.0004272, -0.0005227, 0.0005704
1: -0.0042303, -0.0039799, -0.0042357, -0.0039843, -0.0002052, 0.0002144
2: 0.0130850, 0.0140349, 0.0130475, 0.0140209, -0.0007082, 0.0007683
3: 1.0084045, 1.0090395, 1.0084145, 1.0090528, -0.0006359, 0.0006250
4: -0.0038806, -0.0037193, -0.0038780, -0.0037142, -0.0001276, 0.0001181
5: 0.0030990, 0.0036291, 0.0030743, 0.0036219, -0.0004032, 0.0004396
6: -0.0024365, -0.0023811, -0.0024368, -0.0023800, -0.0000565, 0.0000558
7: -0.0129468, -0.0122333, -0.0129457, -0.0121543, -0.0007763, 0.0006955
8: -0.0093817, -0.0075944, -0.0093513, -0.0075461, -0.0013979, 0.0012956
9: -0.0005975, 0.0003096, -0.0006176, 0.0002934, -0.0006515, 0.0007021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011025, -0.0004219, -0.0012077, -0.0004558, -0.0005216, 0.0006474
1: -0.0042303, -0.0039816, -0.0042511, -0.0039970, -0.0002112, 0.0002307
2: 0.0130851, 0.0140291, 0.0129600, 0.0139770, -0.0007065, 0.0008606
3: 1.0084085, 1.0090395, 1.0084425, 1.0090911, -0.0006763, 0.0005970
4: -0.0038795, -0.0037193, -0.0038698, -0.0037015, -0.0001407, 0.0001178
5: 0.0030990, 0.0036261, 0.0030193, 0.0035994, -0.0004023, 0.0004981
6: -0.0024362, -0.0023811, -0.0024371, -0.0023768, -0.0000594, 0.0000560
7: -0.0129463, -0.0122333, -0.0129424, -0.0119991, -0.0009313, 0.0006954
8: -0.0093691, -0.0075945, -0.0092562, -0.0074191, -0.0015303, 0.0012922
9: -0.0005975, 0.0003029, -0.0006747, 0.0002425, -0.0006497, 0.0007617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
time: 0.51 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.67 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003123, upper bound: 0.0003161
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003164, upper bound: 0.0003172
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003123, upper bound: 0.0003161
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003164, upper bound: 0.0003172
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003103, upper bound: 0.0003127
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003145, upper bound: 0.0003145
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003103, upper bound: 0.0003127
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003145, upper bound: 0.0003145
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003247, upper bound: 0.0003356
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003279, upper bound: 0.0003368
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003247, upper bound: 0.0003356
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003279, upper bound: 0.0003368
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003239, upper bound: 0.0003336
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003273, upper bound: 0.0003353
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003239, upper bound: 0.0003336
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003273, upper bound: 0.0003353
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003298, upper bound: 0.0003292
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003350, upper bound: 0.0003288
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003298, upper bound: 0.0003292
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003350, upper bound: 0.0003288
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003294, upper bound: 0.0003287
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003348, upper bound: 0.0003285
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003294, upper bound: 0.0003287
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003348, upper bound: 0.0003285
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003322, upper bound: 0.0003246
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003144, upper bound: 0.0003132
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003322, upper bound: 0.0003246
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003144, upper bound: 0.0003132
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011776, -0.0004535, -0.0011867, -0.0004540, -0.0005461, 0.0005540
1: -0.0042482, -0.0039970, -0.0042493, -0.0039970, -0.0001991, 0.0002001
2: 0.0129917, 0.0139805, 0.0129816, 0.0139797, -0.0007166, 0.0007263
3: 1.0084479, 1.0090839, 1.0084463, 1.0090868, -0.0006279, 0.0006285
4: -0.0038704, -0.0037051, -0.0038703, -0.0037038, -0.0001166, 0.0001152
5: 0.0030417, 0.0036012, 0.0030349, 0.0036008, -0.0004194, 0.0004255
6: -0.0024351, -0.0023774, -0.0024357, -0.0023772, -0.0000579, 0.0000583
7: -0.0129426, -0.0120867, -0.0129426, -0.0120610, -0.0008609, 0.0008353
8: -0.0092638, -0.0074502, -0.0092621, -0.0074386, -0.0012527, 0.0012382
9: -0.0006637, 0.0002466, -0.0006679, 0.0002457, -0.0006109, 0.0006176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003185, upper bound: 0.0003185
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003185, upper bound: 0.0003219
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011818, -0.0004543, -0.0011887, -0.0004532, -0.0005486, 0.0005563
1: -0.0042486, -0.0039972, -0.0042495, -0.0039966, -0.0001999, 0.0002009
2: 0.0129871, 0.0139792, 0.0129795, 0.0139810, -0.0007191, 0.0007294
3: 1.0084468, 1.0090851, 1.0084455, 1.0090874, -0.0006309, 0.0006328
4: -0.0038702, -0.0037046, -0.0038705, -0.0037036, -0.0001171, 0.0001155
5: 0.0030386, 0.0036005, 0.0030335, 0.0036015, -0.0004213, 0.0004272
6: -0.0024354, -0.0023773, -0.0024358, -0.0023771, -0.0000583, 0.0000586
7: -0.0129425, -0.0120730, -0.0129427, -0.0120555, -0.0008666, 0.0008489
8: -0.0092609, -0.0074452, -0.0092648, -0.0074362, -0.0012581, 0.0012407
9: -0.0006654, 0.0002451, -0.0006688, 0.0002472, -0.0006120, 0.0006202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003219, upper bound: 0.0003185
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003219, upper bound: 0.0003229
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011776, -0.0004535, -0.0011671, -0.0004469, -0.0005636, 0.0005424
1: -0.0042482, -0.0039970, -0.0042462, -0.0039934, -0.0002035, 0.0002024
2: 0.0129917, 0.0139805, 0.0130043, 0.0139906, -0.0007434, 0.0007148
3: 1.0084479, 1.0090839, 1.0084382, 1.0090791, -0.0006312, 0.0006274
4: -0.0038704, -0.0037051, -0.0038723, -0.0037070, -0.0001166, 0.0001202
5: 0.0030417, 0.0036012, 0.0030497, 0.0036064, -0.0004332, 0.0004167
6: -0.0024351, -0.0023774, -0.0024354, -0.0023778, -0.0000573, 0.0000581
7: -0.0129426, -0.0120867, -0.0129434, -0.0121123, -0.0008106, 0.0008374
8: -0.0092638, -0.0074502, -0.0092857, -0.0074678, -0.0012639, 0.0012964
9: -0.0006637, 0.0002466, -0.0006563, 0.0002583, -0.0006420, 0.0006260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003123, upper bound: 0.0003131
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003123, upper bound: 0.0003161
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011818, -0.0004543, -0.0011690, -0.0004461, -0.0005679, 0.0005447
1: -0.0042486, -0.0039972, -0.0042464, -0.0039930, -0.0002045, 0.0002032
2: 0.0129871, 0.0139792, 0.0130022, 0.0139918, -0.0007487, 0.0007178
3: 1.0084468, 1.0090851, 1.0084372, 1.0090797, -0.0006329, 0.0006307
4: -0.0038702, -0.0037046, -0.0038726, -0.0037067, -0.0001171, 0.0001210
5: 0.0030386, 0.0036005, 0.0030483, 0.0036070, -0.0004364, 0.0004185
6: -0.0024354, -0.0023773, -0.0024356, -0.0023777, -0.0000577, 0.0000583
7: -0.0129425, -0.0120730, -0.0129435, -0.0121070, -0.0008159, 0.0008511
8: -0.0092609, -0.0074452, -0.0092884, -0.0074654, -0.0012688, 0.0013050
9: -0.0006654, 0.0002451, -0.0006572, 0.0002598, -0.0006464, 0.0006284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003152, upper bound: 0.0003131
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003152, upper bound: 0.0003172
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011584, -0.0004461, -0.0011867, -0.0004540, -0.0005344, 0.0005727
1: -0.0042451, -0.0039932, -0.0042493, -0.0039970, -0.0002014, 0.0002047
2: 0.0130140, 0.0139919, 0.0129816, 0.0139797, -0.0007058, 0.0007551
3: 1.0084381, 1.0090765, 1.0084463, 1.0090868, -0.0006265, 0.0006301
4: -0.0038726, -0.0037082, -0.0038703, -0.0037038, -0.0001220, 0.0001153
5: 0.0030563, 0.0036070, 0.0030349, 0.0036008, -0.0004106, 0.0004402
6: -0.0024350, -0.0023780, -0.0024357, -0.0023772, -0.0000579, 0.0000577
7: -0.0129435, -0.0121365, -0.0129426, -0.0120610, -0.0008631, 0.0007873
8: -0.0092884, -0.0074789, -0.0092621, -0.0074386, -0.0013150, 0.0012496
9: -0.0006521, 0.0002598, -0.0006679, 0.0002457, -0.0006192, 0.0006509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003131, upper bound: 0.0003123
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003131, upper bound: 0.0003152
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011620, -0.0004473, -0.0011887, -0.0004532, -0.0005366, 0.0005737
1: -0.0042455, -0.0039937, -0.0042495, -0.0039966, -0.0002023, 0.0002054
2: 0.0130100, 0.0139899, 0.0129795, 0.0139810, -0.0007084, 0.0007562
3: 1.0084387, 1.0090773, 1.0084455, 1.0090874, -0.0006296, 0.0006318
4: -0.0038722, -0.0037077, -0.0038705, -0.0037036, -0.0001221, 0.0001156
5: 0.0030536, 0.0036060, 0.0030335, 0.0036015, -0.0004123, 0.0004409
6: -0.0024352, -0.0023779, -0.0024358, -0.0023771, -0.0000581, 0.0000579
7: -0.0129433, -0.0121250, -0.0129427, -0.0120555, -0.0008686, 0.0007975
8: -0.0092842, -0.0074746, -0.0092648, -0.0074362, -0.0013160, 0.0012529
9: -0.0006536, 0.0002575, -0.0006688, 0.0002472, -0.0006206, 0.0006512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003123
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003164
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011584, -0.0004461, -0.0011671, -0.0004469, -0.0005416, 0.0005491
1: -0.0042451, -0.0039932, -0.0042462, -0.0039934, -0.0002002, 0.0002014
2: 0.0130140, 0.0139919, 0.0130043, 0.0139906, -0.0007157, 0.0007246
3: 1.0084381, 1.0090765, 1.0084382, 1.0090791, -0.0006204, 0.0006207
4: -0.0038726, -0.0037082, -0.0038723, -0.0037070, -0.0001177, 0.0001164
5: 0.0030563, 0.0036070, 0.0030497, 0.0036064, -0.0004164, 0.0004220
6: -0.0024350, -0.0023780, -0.0024354, -0.0023778, -0.0000572, 0.0000574
7: -0.0129435, -0.0121365, -0.0129434, -0.0121123, -0.0008116, 0.0007883
8: -0.0092884, -0.0074789, -0.0092857, -0.0074678, -0.0012752, 0.0012624
9: -0.0006521, 0.0002598, -0.0006563, 0.0002583, -0.0006256, 0.0006315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003102, upper bound: 0.0003102
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003102, upper bound: 0.0003127
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011620, -0.0004473, -0.0011690, -0.0004461, -0.0005430, 0.0005518
1: -0.0042455, -0.0039937, -0.0042464, -0.0039930, -0.0002007, 0.0002020
2: 0.0130100, 0.0139899, 0.0130022, 0.0139918, -0.0007168, 0.0007282
3: 1.0084387, 1.0090773, 1.0084372, 1.0090797, -0.0006228, 0.0006240
4: -0.0038722, -0.0037077, -0.0038726, -0.0037067, -0.0001182, 0.0001165
5: 0.0030536, 0.0036060, 0.0030483, 0.0036070, -0.0004174, 0.0004241
6: -0.0024352, -0.0023779, -0.0024356, -0.0023777, -0.0000575, 0.0000577
7: -0.0129433, -0.0121250, -0.0129435, -0.0121070, -0.0008170, 0.0007987
8: -0.0092842, -0.0074746, -0.0092884, -0.0074654, -0.0012816, 0.0012624
9: -0.0006536, 0.0002575, -0.0006572, 0.0002598, -0.0006255, 0.0006348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003127, upper bound: 0.0003103
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003127, upper bound: 0.0003145
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011776, -0.0004535, -0.0011173, -0.0004257, -0.0005988, 0.0005016
1: -0.0042482, -0.0039970, -0.0042327, -0.0039836, -0.0002182, 0.0001937
2: 0.0129917, 0.0139805, 0.0130683, 0.0140233, -0.0007974, 0.0006676
3: 1.0084479, 1.0090839, 1.0084136, 1.0090455, -0.0005976, 0.0006501
4: -0.0038704, -0.0037051, -0.0038784, -0.0037169, -0.0001093, 0.0001307
5: 0.0030417, 0.0036012, 0.0030878, 0.0036231, -0.0004609, 0.0003861
6: -0.0024351, -0.0023774, -0.0024362, -0.0023806, -0.0000545, 0.0000588
7: -0.0129426, -0.0120867, -0.0129459, -0.0121958, -0.0007277, 0.0008406
8: -0.0092638, -0.0074502, -0.0093564, -0.0075724, -0.0011902, 0.0014246
9: -0.0006637, 0.0002466, -0.0006066, 0.0002961, -0.0007132, 0.0005935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003282, upper bound: 0.0003304
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003280, upper bound: 0.0003347
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011818, -0.0004543, -0.0011193, -0.0004250, -0.0006060, 0.0005038
1: -0.0042486, -0.0039972, -0.0042330, -0.0039833, -0.0002201, 0.0001945
2: 0.0129871, 0.0139792, 0.0130661, 0.0140243, -0.0008068, 0.0006705
3: 1.0084468, 1.0090851, 1.0084126, 1.0090461, -0.0005993, 0.0006564
4: -0.0038702, -0.0037046, -0.0038786, -0.0037167, -0.0001098, 0.0001324
5: 0.0030386, 0.0036005, 0.0030864, 0.0036236, -0.0004664, 0.0003878
6: -0.0024354, -0.0023773, -0.0024363, -0.0023805, -0.0000549, 0.0000591
7: -0.0129425, -0.0120730, -0.0129460, -0.0121902, -0.0007333, 0.0008547
8: -0.0092609, -0.0074452, -0.0093587, -0.0075699, -0.0011949, 0.0014424
9: -0.0006654, 0.0002451, -0.0006075, 0.0002973, -0.0007243, 0.0005960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003314, upper bound: 0.0003313
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003307, upper bound: 0.0003355
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011776, -0.0004535, -0.0011006, -0.0004188, -0.0006133, 0.0004911
1: -0.0042482, -0.0039970, -0.0042301, -0.0039802, -0.0002221, 0.0001947
2: 0.0129917, 0.0139805, 0.0130871, 0.0140338, -0.0008196, 0.0006619
3: 1.0084479, 1.0090839, 1.0084053, 1.0090389, -0.0005910, 0.0006508
4: -0.0038704, -0.0037051, -0.0038804, -0.0037195, -0.0001094, 0.0001348
5: 0.0030417, 0.0036012, 0.0031004, 0.0036285, -0.0004723, 0.0003785
6: -0.0024351, -0.0023774, -0.0024363, -0.0023811, -0.0000540, 0.0000590
7: -0.0129426, -0.0120867, -0.0129467, -0.0122385, -0.0006860, 0.0008423
8: -0.0092638, -0.0074502, -0.0093793, -0.0075968, -0.0011950, 0.0014728
9: -0.0006637, 0.0002466, -0.0005966, 0.0003083, -0.0007390, 0.0005972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003243, upper bound: 0.0003275
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003241, upper bound: 0.0003327
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011818, -0.0004543, -0.0011025, -0.0004181, -0.0006215, 0.0004933
1: -0.0042486, -0.0039972, -0.0042303, -0.0039799, -0.0002249, 0.0001955
2: 0.0129871, 0.0139792, 0.0130850, 0.0140349, -0.0008306, 0.0006648
3: 1.0084468, 1.0090851, 1.0084045, 1.0090395, -0.0005927, 0.0006560
4: -0.0038702, -0.0037046, -0.0038806, -0.0037193, -0.0001099, 0.0001368
5: 0.0030386, 0.0036005, 0.0030990, 0.0036291, -0.0004786, 0.0003802
6: -0.0024354, -0.0023773, -0.0024365, -0.0023811, -0.0000544, 0.0000592
7: -0.0129425, -0.0120730, -0.0129468, -0.0122333, -0.0006913, 0.0008566
8: -0.0092609, -0.0074452, -0.0093817, -0.0075944, -0.0011998, 0.0014941
9: -0.0006654, 0.0002451, -0.0005975, 0.0003096, -0.0007519, 0.0005996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003276, upper bound: 0.0003285
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003272, upper bound: 0.0003337
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011584, -0.0004461, -0.0011173, -0.0004257, -0.0005856, 0.0005203
1: -0.0042451, -0.0039932, -0.0042327, -0.0039836, -0.0002202, 0.0001984
2: 0.0130140, 0.0139919, 0.0130683, 0.0140233, -0.0007898, 0.0006964
3: 1.0084381, 1.0090765, 1.0084136, 1.0090455, -0.0006074, 0.0006551
4: -0.0038726, -0.0037082, -0.0038784, -0.0037169, -0.0001147, 0.0001311
5: 0.0030563, 0.0036070, 0.0030878, 0.0036231, -0.0004514, 0.0004008
6: -0.0024350, -0.0023780, -0.0024362, -0.0023806, -0.0000545, 0.0000582
7: -0.0129435, -0.0121365, -0.0129459, -0.0121958, -0.0007299, 0.0007931
8: -0.0092884, -0.0074789, -0.0093564, -0.0075724, -0.0012525, 0.0014352
9: -0.0006521, 0.0002598, -0.0006066, 0.0002961, -0.0007208, 0.0006268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003250, upper bound: 0.0003259
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003247, upper bound: 0.0003305
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011620, -0.0004473, -0.0011193, -0.0004250, -0.0005925, 0.0005212
1: -0.0042455, -0.0039937, -0.0042330, -0.0039833, -0.0002221, 0.0001989
2: 0.0130100, 0.0139899, 0.0130661, 0.0140243, -0.0007990, 0.0006973
3: 1.0084387, 1.0090773, 1.0084126, 1.0090461, -0.0006074, 0.0006615
4: -0.0038722, -0.0037077, -0.0038786, -0.0037167, -0.0001148, 0.0001326
5: 0.0030536, 0.0036060, 0.0030864, 0.0036236, -0.0004568, 0.0004015
6: -0.0024352, -0.0023779, -0.0024363, -0.0023805, -0.0000547, 0.0000584
7: -0.0129433, -0.0121250, -0.0129460, -0.0121902, -0.0007353, 0.0008038
8: -0.0092842, -0.0074746, -0.0093587, -0.0075699, -0.0012529, 0.0014540
9: -0.0006536, 0.0002575, -0.0006075, 0.0002973, -0.0007319, 0.0006269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003278, upper bound: 0.0003273
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003273, upper bound: 0.0003325
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011584, -0.0004461, -0.0011006, -0.0004188, -0.0005926, 0.0004983
1: -0.0042451, -0.0039932, -0.0042301, -0.0039802, -0.0002199, 0.0001947
2: 0.0130140, 0.0139919, 0.0130871, 0.0140338, -0.0007975, 0.0006704
3: 1.0084381, 1.0090765, 1.0084053, 1.0090389, -0.0006008, 0.0006460
4: -0.0038726, -0.0037082, -0.0038804, -0.0037195, -0.0001110, 0.0001322
5: 0.0030563, 0.0036070, 0.0031004, 0.0036285, -0.0004566, 0.0003839
6: -0.0024350, -0.0023780, -0.0024363, -0.0023811, -0.0000539, 0.0000583
7: -0.0129435, -0.0121365, -0.0129467, -0.0122385, -0.0006870, 0.0007940
8: -0.0092884, -0.0074789, -0.0093793, -0.0075968, -0.0012126, 0.0014483
9: -0.0006521, 0.0002598, -0.0005966, 0.0003083, -0.0007276, 0.0006064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003232, upper bound: 0.0003253
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003232, upper bound: 0.0003302
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011620, -0.0004473, -0.0011025, -0.0004181, -0.0005991, 0.0005009
1: -0.0042455, -0.0039937, -0.0042303, -0.0039799, -0.0002214, 0.0001952
2: 0.0130100, 0.0139899, 0.0130850, 0.0140349, -0.0008069, 0.0006739
3: 1.0084387, 1.0090773, 1.0084045, 1.0090395, -0.0006008, 0.0006507
4: -0.0038722, -0.0037077, -0.0038806, -0.0037193, -0.0001115, 0.0001337
5: 0.0030536, 0.0036060, 0.0030990, 0.0036291, -0.0004616, 0.0003859
6: -0.0024352, -0.0023779, -0.0024365, -0.0023811, -0.0000542, 0.0000585
7: -0.0129433, -0.0121250, -0.0129468, -0.0122333, -0.0006923, 0.0008049
8: -0.0092842, -0.0074746, -0.0093817, -0.0075944, -0.0012186, 0.0014661
9: -0.0006536, 0.0002575, -0.0005975, 0.0003096, -0.0007372, 0.0006094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003272, upper bound: 0.0003271
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003267, upper bound: 0.0003324
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010700, -0.0004208, -0.0011829, -0.0004534, -0.0004506, 0.0005766
1: -0.0042265, -0.0039823, -0.0042488, -0.0039967, -0.0001855, 0.0002005
2: 0.0131211, 0.0140308, 0.0129859, 0.0139807, -0.0006086, 0.0007631
3: 1.0084143, 1.0090301, 1.0084459, 1.0090855, -0.0005855, 0.0005842
4: -0.0038798, -0.0037237, -0.0038705, -0.0037044, -0.0001238, 0.0001012
5: 0.0031234, 0.0036269, 0.0030378, 0.0036013, -0.0003475, 0.0004433
6: -0.0024341, -0.0023818, -0.0024354, -0.0023773, -0.0000568, 0.0000536
7: -0.0129465, -0.0123208, -0.0129426, -0.0120706, -0.0008548, 0.0006016
8: -0.0093728, -0.0076331, -0.0092642, -0.0074438, -0.0013376, 0.0011100
9: -0.0005833, 0.0003049, -0.0006659, 0.0002468, -0.0005608, 0.0006644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002056, upper bound: 0.0002207
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0001989, upper bound: 0.0002129
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003304, upper bound: 0.0003283
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003313, upper bound: 0.0003314
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010639, -0.0004268, -0.0011850, -0.0004532, -0.0004338, 0.0006080
1: -0.0042245, -0.0039845, -0.0042491, -0.0039966, -0.0001804, 0.0002171
2: 0.0131291, 0.0140215, 0.0129835, 0.0139809, -0.0005826, 0.0008099
3: 1.0084174, 1.0090250, 1.0084456, 1.0090863, -0.0006226, 0.0005794
4: -0.0038781, -0.0037250, -0.0038705, -0.0037041, -0.0001327, 0.0000962
5: 0.0031281, 0.0036222, 0.0030362, 0.0036014, -0.0003343, 0.0004680
6: -0.0024344, -0.0023822, -0.0024356, -0.0023772, -0.0000572, 0.0000534
7: -0.0129458, -0.0123239, -0.0129427, -0.0120652, -0.0008622, 0.0005991
8: -0.0093526, -0.0076482, -0.0092646, -0.0074409, -0.0014477, 0.0010529
9: -0.0005758, 0.0002941, -0.0006670, 0.0002470, -0.0005296, 0.0007250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002301, upper bound: 0.0002311
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002060, upper bound: 0.0002126
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003347, upper bound: 0.0003280
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003355, upper bound: 0.0003307
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010700, -0.0004208, -0.0011627, -0.0004463, -0.0004682, 0.0005646
1: -0.0042265, -0.0039823, -0.0042456, -0.0039932, -0.0001899, 0.0002027
2: 0.0131211, 0.0140308, 0.0130091, 0.0139915, -0.0006356, 0.0007508
3: 1.0084143, 1.0090301, 1.0084376, 1.0090778, -0.0005908, 0.0005914
4: -0.0038798, -0.0037237, -0.0038725, -0.0037076, -0.0001237, 0.0001062
5: 0.0031234, 0.0036269, 0.0030530, 0.0036068, -0.0003613, 0.0004344
6: -0.0024341, -0.0023818, -0.0024352, -0.0023779, -0.0000562, 0.0000533
7: -0.0129465, -0.0123208, -0.0129435, -0.0121246, -0.0008015, 0.0006037
8: -0.0093728, -0.0076331, -0.0092876, -0.0074734, -0.0013481, 0.0011685
9: -0.0005833, 0.0003049, -0.0006541, 0.0002594, -0.0005920, 0.0006722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003259, upper bound: 0.0003250
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003273, upper bound: 0.0003278
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010639, -0.0004268, -0.0011656, -0.0004462, -0.0004505, 0.0005945
1: -0.0042245, -0.0039845, -0.0042460, -0.0039931, -0.0001835, 0.0002192
2: 0.0131291, 0.0140215, 0.0130060, 0.0139917, -0.0006083, 0.0008025
3: 1.0084174, 1.0090250, 1.0084372, 1.0090785, -0.0006278, 0.0005849
4: -0.0038781, -0.0037250, -0.0038725, -0.0037072, -0.0001332, 0.0001010
5: 0.0031281, 0.0036222, 0.0030509, 0.0036070, -0.0003475, 0.0004582
6: -0.0024344, -0.0023822, -0.0024354, -0.0023778, -0.0000565, 0.0000532
7: -0.0129458, -0.0123239, -0.0129435, -0.0121156, -0.0008133, 0.0006011
8: -0.0093526, -0.0076482, -0.0092881, -0.0074699, -0.0014587, 0.0011085
9: -0.0005758, 0.0002941, -0.0006554, 0.0002596, -0.0005593, 0.0007327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003305, upper bound: 0.0003247
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003325, upper bound: 0.0003273
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010504, -0.0004145, -0.0011829, -0.0004534, -0.0004369, 0.0005889
1: -0.0042239, -0.0039792, -0.0042488, -0.0039967, -0.0001858, 0.0002043
2: 0.0131426, 0.0140404, 0.0129859, 0.0139807, -0.0005955, 0.0007820
3: 1.0084080, 1.0090235, 1.0084459, 1.0090855, -0.0005822, 0.0005777
4: -0.0038816, -0.0037265, -0.0038705, -0.0037044, -0.0001273, 0.0001004
5: 0.0031382, 0.0036319, 0.0030378, 0.0036013, -0.0003374, 0.0004530
6: -0.0024342, -0.0023824, -0.0024354, -0.0023773, -0.0000569, 0.0000530
7: -0.0129472, -0.0123795, -0.0129426, -0.0120706, -0.0008562, 0.0005445
8: -0.0093935, -0.0076588, -0.0092642, -0.0074438, -0.0013785, 0.0011087
9: -0.0005734, 0.0003159, -0.0006659, 0.0002468, -0.0005619, 0.0006862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003275, upper bound: 0.0003243
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003285, upper bound: 0.0003276
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010488, -0.0004200, -0.0011850, -0.0004532, -0.0004254, 0.0006226
1: -0.0042219, -0.0039812, -0.0042491, -0.0039966, -0.0001808, 0.0002217
2: 0.0131465, 0.0140319, 0.0129835, 0.0139809, -0.0005781, 0.0008323
3: 1.0084093, 1.0090184, 1.0084456, 1.0090863, -0.0006257, 0.0005728
4: -0.0038800, -0.0037275, -0.0038705, -0.0037041, -0.0001369, 0.0000962
5: 0.0031395, 0.0036275, 0.0030362, 0.0036014, -0.0003284, 0.0004795
6: -0.0024347, -0.0023828, -0.0024356, -0.0023772, -0.0000575, 0.0000529
7: -0.0129465, -0.0123577, -0.0129427, -0.0120652, -0.0008639, 0.0005661
8: -0.0093751, -0.0076715, -0.0092646, -0.0074409, -0.0014963, 0.0010564
9: -0.0005659, 0.0003061, -0.0006670, 0.0002470, -0.0005313, 0.0007510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003327, upper bound: 0.0003241
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003337, upper bound: 0.0003272
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010504, -0.0004145, -0.0011627, -0.0004463, -0.0004442, 0.0005701
1: -0.0042239, -0.0039792, -0.0042456, -0.0039932, -0.0001859, 0.0002012
2: 0.0131426, 0.0140404, 0.0130091, 0.0139915, -0.0006069, 0.0007591
3: 1.0084080, 1.0090235, 1.0084376, 1.0090778, -0.0005779, 0.0005827
4: -0.0038816, -0.0037265, -0.0038725, -0.0037076, -0.0001243, 0.0001021
5: 0.0031382, 0.0036319, 0.0030530, 0.0036068, -0.0003431, 0.0004387
6: -0.0024342, -0.0023824, -0.0024352, -0.0023779, -0.0000563, 0.0000528
7: -0.0129472, -0.0123795, -0.0129435, -0.0121246, -0.0008025, 0.0005458
8: -0.0093935, -0.0076588, -0.0092876, -0.0074734, -0.0013550, 0.0011288
9: -0.0005734, 0.0003159, -0.0006541, 0.0002594, -0.0005729, 0.0006755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003252, upper bound: 0.0003232
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003271, upper bound: 0.0003272
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010488, -0.0004200, -0.0011656, -0.0004462, -0.0004327, 0.0006020
1: -0.0042219, -0.0039812, -0.0042460, -0.0039931, -0.0001807, 0.0002193
2: 0.0131465, 0.0140319, 0.0130060, 0.0139917, -0.0005877, 0.0008100
3: 1.0084093, 1.0090184, 1.0084372, 1.0090785, -0.0006209, 0.0005798
4: -0.0038800, -0.0037275, -0.0038725, -0.0037072, -0.0001343, 0.0000980
5: 0.0031395, 0.0036275, 0.0030509, 0.0036070, -0.0003340, 0.0004638
6: -0.0024347, -0.0023828, -0.0024354, -0.0023778, -0.0000569, 0.0000527
7: -0.0129465, -0.0123577, -0.0129435, -0.0121156, -0.0008144, 0.0005671
8: -0.0093751, -0.0076715, -0.0092881, -0.0074699, -0.0014714, 0.0010761
9: -0.0005659, 0.0003061, -0.0006554, 0.0002596, -0.0005426, 0.0007394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003302, upper bound: 0.0003232
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003324, upper bound: 0.0003268
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011192, -0.0004287, -0.0011193, -0.0004250, -0.0005372, 0.0005337
1: -0.0042330, -0.0039851, -0.0042330, -0.0039833, -0.0002051, 0.0002033
2: 0.0130662, 0.0140186, 0.0130661, 0.0140243, -0.0007204, 0.0007149
3: 1.0084174, 1.0090461, 1.0084126, 1.0090461, -0.0006176, 0.0006224
4: -0.0038775, -0.0037167, -0.0038786, -0.0037167, -0.0001181, 0.0001191
5: 0.0030864, 0.0036207, 0.0030864, 0.0036236, -0.0004139, 0.0004111
6: -0.0024361, -0.0023805, -0.0024363, -0.0023805, -0.0000556, 0.0000558
7: -0.0129455, -0.0121902, -0.0129460, -0.0121902, -0.0007375, 0.0007379
8: -0.0093464, -0.0075700, -0.0093587, -0.0075699, -0.0012927, 0.0013046
9: -0.0006075, 0.0002908, -0.0006075, 0.0002973, -0.0006549, 0.0006485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003180, upper bound: 0.0003179
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003180, upper bound: 0.0003179
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011897, -0.0004571, -0.0011192, -0.0004286, -0.0006113, 0.0005325
1: -0.0042486, -0.0039979, -0.0042330, -0.0039850, -0.0002231, 0.0002093
2: 0.0129804, 0.0139749, 0.0130662, 0.0140188, -0.0008118, 0.0007132
3: 1.0084454, 1.0090853, 1.0084163, 1.0090461, -0.0006007, 0.0006674
4: -0.0038694, -0.0037039, -0.0038776, -0.0037167, -0.0001178, 0.0001326
5: 0.0030328, 0.0035983, 0.0030864, 0.0036208, -0.0004702, 0.0004102
6: -0.0024360, -0.0023773, -0.0024361, -0.0023805, -0.0000555, 0.0000588
7: -0.0129422, -0.0120447, -0.0129455, -0.0121902, -0.0007373, 0.0008838
8: -0.0092516, -0.0074415, -0.0093467, -0.0075700, -0.0012890, 0.0014413
9: -0.0006655, 0.0002401, -0.0006075, 0.0002910, -0.0007209, 0.0006466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003180, upper bound: 0.0003179
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003180, upper bound: 0.0003179
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011192, -0.0004287, -0.0011025, -0.0004181, -0.0005536, 0.0005229
1: -0.0042330, -0.0039851, -0.0042303, -0.0039799, -0.0002096, 0.0002053
2: 0.0130662, 0.0140186, 0.0130850, 0.0140349, -0.0007456, 0.0007103
3: 1.0084174, 1.0090461, 1.0084045, 1.0090395, -0.0006222, 0.0006239
4: -0.0038775, -0.0037167, -0.0038806, -0.0037193, -0.0001186, 0.0001238
5: 0.0030864, 0.0036207, 0.0030990, 0.0036291, -0.0004268, 0.0004035
6: -0.0024361, -0.0023805, -0.0024365, -0.0023811, -0.0000550, 0.0000559
7: -0.0129455, -0.0121902, -0.0129468, -0.0122333, -0.0006952, 0.0007398
8: -0.0093464, -0.0075700, -0.0093817, -0.0075944, -0.0013032, 0.0013593
9: -0.0006075, 0.0002908, -0.0005975, 0.0003096, -0.0006842, 0.0006560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003144, upper bound: 0.0003132
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003144, upper bound: 0.0003132
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011897, -0.0004571, -0.0011025, -0.0004219, -0.0006279, 0.0005217
1: -0.0042486, -0.0039979, -0.0042303, -0.0039816, -0.0002277, 0.0002113
2: 0.0129804, 0.0139749, 0.0130851, 0.0140291, -0.0008374, 0.0007085
3: 1.0084454, 1.0090853, 1.0084085, 1.0090395, -0.0005941, 0.0006689
4: -0.0038694, -0.0037039, -0.0038795, -0.0037193, -0.0001182, 0.0001374
5: 0.0030328, 0.0035983, 0.0030990, 0.0036261, -0.0004833, 0.0004026
6: -0.0024360, -0.0023773, -0.0024362, -0.0023811, -0.0000550, 0.0000589
7: -0.0129422, -0.0120447, -0.0129463, -0.0122333, -0.0006951, 0.0008857
8: -0.0092516, -0.0074415, -0.0093691, -0.0075945, -0.0012995, 0.0014968
9: -0.0006655, 0.0002401, -0.0005975, 0.0003029, -0.0007506, 0.0006542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003144, upper bound: 0.0003132
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003144, upper bound: 0.0003132
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011025, -0.0004216, -0.0011352, -0.0004272, -0.0005225, 0.0005668
1: -0.0042303, -0.0039817, -0.0042357, -0.0039843, -0.0002051, 0.0002124
2: 0.0130851, 0.0140295, 0.0130475, 0.0140209, -0.0007080, 0.0007626
3: 1.0084090, 1.0090395, 1.0084145, 1.0090528, -0.0006311, 0.0006250
4: -0.0038796, -0.0037193, -0.0038780, -0.0037142, -0.0001265, 0.0001180
5: 0.0030990, 0.0036263, 0.0030743, 0.0036219, -0.0004031, 0.0004367
6: -0.0024362, -0.0023811, -0.0024368, -0.0023800, -0.0000562, 0.0000558
7: -0.0129464, -0.0122333, -0.0129457, -0.0121543, -0.0007759, 0.0006955
8: -0.0093700, -0.0075945, -0.0093513, -0.0075461, -0.0013857, 0.0012954
9: -0.0005975, 0.0003034, -0.0006176, 0.0002934, -0.0006514, 0.0006956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003270, upper bound: 0.0003307
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003270, upper bound: 0.0003307
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011751, -0.0004504, -0.0011352, -0.0004272, -0.0006025, 0.0005458
1: -0.0042455, -0.0039945, -0.0042357, -0.0039843, -0.0002233, 0.0002076
2: 0.0129975, 0.0139852, 0.0130475, 0.0140209, -0.0008039, 0.0007304
3: 1.0084372, 1.0090773, 1.0084145, 1.0090528, -0.0006156, 0.0006628
4: -0.0038713, -0.0037066, -0.0038780, -0.0037142, -0.0001205, 0.0001320
5: 0.0030439, 0.0036036, 0.0030743, 0.0036219, -0.0004638, 0.0004202
6: -0.0024359, -0.0023779, -0.0024368, -0.0023800, -0.0000560, 0.0000589
7: -0.0129430, -0.0120811, -0.0129457, -0.0121543, -0.0007734, 0.0008482
8: -0.0092740, -0.0074684, -0.0093513, -0.0075461, -0.0013158, 0.0014388
9: -0.0006537, 0.0002521, -0.0006176, 0.0002934, -0.0007182, 0.0006583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003270, upper bound: 0.0003307
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003270, upper bound: 0.0003307
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011025, -0.0004216, -0.0012077, -0.0004558, -0.0005029, 0.0006469
1: -0.0042303, -0.0039817, -0.0042511, -0.0039970, -0.0001997, 0.0002302
2: 0.0130851, 0.0140295, 0.0129600, 0.0139770, -0.0006778, 0.0008599
3: 1.0084090, 1.0090395, 1.0084425, 1.0090911, -0.0006753, 0.0005970
4: -0.0038796, -0.0037193, -0.0038698, -0.0037015, -0.0001406, 0.0001124
5: 0.0030990, 0.0036263, 0.0030193, 0.0035994, -0.0003876, 0.0004977
6: -0.0024362, -0.0023811, -0.0024371, -0.0023768, -0.0000594, 0.0000560
7: -0.0129464, -0.0122333, -0.0129424, -0.0119991, -0.0009313, 0.0006932
8: -0.0093700, -0.0075945, -0.0092562, -0.0074191, -0.0015288, 0.0012300
9: -0.0005975, 0.0003034, -0.0006747, 0.0002425, -0.0006165, 0.0007609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011751, -0.0004504, -0.0012077, -0.0004558, -0.0005647, 0.0006085
1: -0.0042455, -0.0039945, -0.0042511, -0.0039970, -0.0002126, 0.0002194
2: 0.0129975, 0.0139852, 0.0129600, 0.0139770, -0.0007465, 0.0008018
3: 1.0084372, 1.0090773, 1.0084425, 1.0090911, -0.0006540, 0.0006348
4: -0.0038713, -0.0037066, -0.0038698, -0.0037015, -0.0001299, 0.0001216
5: 0.0030439, 0.0036036, 0.0030193, 0.0035994, -0.0004340, 0.0004675
6: -0.0024359, -0.0023779, -0.0024371, -0.0023768, -0.0000592, 0.0000592
7: -0.0129430, -0.0120811, -0.0129424, -0.0119991, -0.0009271, 0.0008444
8: -0.0092740, -0.0074684, -0.0092562, -0.0074191, -0.0014083, 0.0013203
9: -0.0006537, 0.0002521, -0.0006747, 0.0002425, -0.0006556, 0.0006986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
time: 0.54 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.73 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003185, upper bound: 0.0003185
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003185, upper bound: 0.0003219
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003219, upper bound: 0.0003185
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003219, upper bound: 0.0003229
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003123, upper bound: 0.0003131
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003123, upper bound: 0.0003161
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003152, upper bound: 0.0003131
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003152, upper bound: 0.0003172
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003131, upper bound: 0.0003123
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003131, upper bound: 0.0003152
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003123
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003164
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003102, upper bound: 0.0003102
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003102, upper bound: 0.0003127
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003127, upper bound: 0.0003103
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003127, upper bound: 0.0003145
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003282, upper bound: 0.0003304
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003280, upper bound: 0.0003347
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003314, upper bound: 0.0003313
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003307, upper bound: 0.0003355
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003243, upper bound: 0.0003275
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003241, upper bound: 0.0003327
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003276, upper bound: 0.0003285
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003272, upper bound: 0.0003337
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003250, upper bound: 0.0003259
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003247, upper bound: 0.0003305
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003278, upper bound: 0.0003273
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003273, upper bound: 0.0003325
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003232, upper bound: 0.0003253
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003232, upper bound: 0.0003302
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003272, upper bound: 0.0003271
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003267, upper bound: 0.0003324
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003304, upper bound: 0.0003283
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003313, upper bound: 0.0003314
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003347, upper bound: 0.0003280
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003355, upper bound: 0.0003307
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003259, upper bound: 0.0003250
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003273, upper bound: 0.0003278
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003305, upper bound: 0.0003247
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003325, upper bound: 0.0003273
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003275, upper bound: 0.0003243
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003285, upper bound: 0.0003276
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003327, upper bound: 0.0003241
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003337, upper bound: 0.0003272
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003252, upper bound: 0.0003232
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003271, upper bound: 0.0003272
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003302, upper bound: 0.0003232
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003324, upper bound: 0.0003268
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003180, upper bound: 0.0003179
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003180, upper bound: 0.0003179
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003180, upper bound: 0.0003179
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003180, upper bound: 0.0003179
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003144, upper bound: 0.0003132
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003144, upper bound: 0.0003132
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003144, upper bound: 0.0003132
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003144, upper bound: 0.0003132
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003270, upper bound: 0.0003307
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003270, upper bound: 0.0003307
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003270, upper bound: 0.0003307
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003270, upper bound: 0.0003307
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -0.0003133, upper bound: 0.0003119

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011776, -0.0004535, -0.0011776, -0.0004535, -0.0005454, 0.0005454
1: -0.0042482, -0.0039970, -0.0042482, -0.0039970, -0.0001979, 0.0001979
2: 0.0129917, 0.0139805, 0.0129917, 0.0139805, -0.0007155, 0.0007155
3: 1.0084479, 1.0090839, 1.0084479, 1.0090839, -0.0006224, 0.0006224
4: -0.0038704, -0.0037051, -0.0038704, -0.0037051, -0.0001150, 0.0001150
5: 0.0030417, 0.0036012, 0.0030417, 0.0036012, -0.0004188, 0.0004188
6: -0.0024351, -0.0023774, -0.0024351, -0.0023774, -0.0000577, 0.0000577
7: -0.0129426, -0.0120867, -0.0129426, -0.0120867, -0.0008352, 0.0008352
8: -0.0092638, -0.0074502, -0.0092638, -0.0074502, -0.0012358, 0.0012358
9: -0.0006637, 0.0002466, -0.0006637, 0.0002466, -0.0006096, 0.0006096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003149, upper bound: 0.0003188
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003153, upper bound: 0.0003153
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011776, -0.0004535, -0.0011818, -0.0004543, -0.0005458, 0.0005481
1: -0.0042482, -0.0039970, -0.0042486, -0.0039972, -0.0001982, 0.0001982
2: 0.0129917, 0.0139805, 0.0129871, 0.0139792, -0.0007161, 0.0007183
3: 1.0084479, 1.0090839, 1.0084468, 1.0090851, -0.0006230, 0.0006242
4: -0.0038704, -0.0037051, -0.0038702, -0.0037046, -0.0001152, 0.0001151
5: 0.0030417, 0.0036012, 0.0030386, 0.0036005, -0.0004192, 0.0004209
6: -0.0024351, -0.0023774, -0.0024354, -0.0023773, -0.0000578, 0.0000580
7: -0.0129426, -0.0120867, -0.0129425, -0.0120730, -0.0008491, 0.0008353
8: -0.0092638, -0.0074502, -0.0092609, -0.0074452, -0.0012387, 0.0012372
9: -0.0006637, 0.0002466, -0.0006654, 0.0002451, -0.0006103, 0.0006106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003149, upper bound: 0.0003215
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003153, upper bound: 0.0003175
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011818, -0.0004543, -0.0011776, -0.0004535, -0.0005481, 0.0005458
1: -0.0042486, -0.0039972, -0.0042482, -0.0039970, -0.0001982, 0.0001982
2: 0.0129871, 0.0139792, 0.0129917, 0.0139805, -0.0007183, 0.0007161
3: 1.0084468, 1.0090851, 1.0084479, 1.0090839, -0.0006242, 0.0006230
4: -0.0038702, -0.0037046, -0.0038704, -0.0037051, -0.0001151, 0.0001152
5: 0.0030386, 0.0036005, 0.0030417, 0.0036012, -0.0004209, 0.0004192
6: -0.0024354, -0.0023773, -0.0024351, -0.0023774, -0.0000580, 0.0000578
7: -0.0129425, -0.0120730, -0.0129426, -0.0120867, -0.0008353, 0.0008491
8: -0.0092609, -0.0074452, -0.0092638, -0.0074502, -0.0012372, 0.0012387
9: -0.0006654, 0.0002451, -0.0006637, 0.0002466, -0.0006106, 0.0006103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003172, upper bound: 0.0003184
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003175, upper bound: 0.0003153
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011818, -0.0004543, -0.0011818, -0.0004543, -0.0005478, 0.0005478
1: -0.0042486, -0.0039972, -0.0042486, -0.0039972, -0.0001987, 0.0001987
2: 0.0129871, 0.0139792, 0.0129871, 0.0139792, -0.0007179, 0.0007179
3: 1.0084468, 1.0090851, 1.0084468, 1.0090851, -0.0006267, 0.0006267
4: -0.0038702, -0.0037046, -0.0038702, -0.0037046, -0.0001152, 0.0001152
5: 0.0030386, 0.0036005, 0.0030386, 0.0036005, -0.0004206, 0.0004206
6: -0.0024354, -0.0023773, -0.0024354, -0.0023773, -0.0000581, 0.0000581
7: -0.0129425, -0.0120730, -0.0129425, -0.0120730, -0.0008488, 0.0008488
8: -0.0092609, -0.0074452, -0.0092609, -0.0074452, -0.0012381, 0.0012381
9: -0.0006654, 0.0002451, -0.0006654, 0.0002451, -0.0006106, 0.0006106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003172, upper bound: 0.0003184
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003175, upper bound: 0.0003153
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011776, -0.0004535, -0.0011584, -0.0004461, -0.0005641, 0.0005337
1: -0.0042482, -0.0039970, -0.0042451, -0.0039932, -0.0002026, 0.0002002
2: 0.0129917, 0.0139805, 0.0130140, 0.0139919, -0.0007442, 0.0007046
3: 1.0084479, 1.0090839, 1.0084381, 1.0090765, -0.0006282, 0.0006211
4: -0.0038704, -0.0037051, -0.0038726, -0.0037082, -0.0001151, 0.0001203
5: 0.0030417, 0.0036012, 0.0030563, 0.0036070, -0.0004336, 0.0004101
6: -0.0024351, -0.0023774, -0.0024350, -0.0023780, -0.0000571, 0.0000576
7: -0.0129426, -0.0120867, -0.0129435, -0.0121365, -0.0007872, 0.0008374
8: -0.0092638, -0.0074502, -0.0092884, -0.0074789, -0.0012472, 0.0012981
9: -0.0006637, 0.0002466, -0.0006521, 0.0002598, -0.0006429, 0.0006179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003078, upper bound: 0.0003129
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0003105
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011776, -0.0004535, -0.0011620, -0.0004473, -0.0005632, 0.0005363
1: -0.0042482, -0.0039970, -0.0042455, -0.0039937, -0.0002027, 0.0002003
2: 0.0129917, 0.0139805, 0.0130100, 0.0139899, -0.0007428, 0.0007069
3: 1.0084479, 1.0090839, 1.0084387, 1.0090773, -0.0006283, 0.0006229
4: -0.0038704, -0.0037051, -0.0038722, -0.0037077, -0.0001152, 0.0001201
5: 0.0030417, 0.0036012, 0.0030536, 0.0036060, -0.0004329, 0.0004120
6: -0.0024351, -0.0023774, -0.0024352, -0.0023779, -0.0000571, 0.0000578
7: -0.0129426, -0.0120867, -0.0129433, -0.0121250, -0.0007977, 0.0008373
8: -0.0092638, -0.0074502, -0.0092842, -0.0074746, -0.0012479, 0.0012951
9: -0.0006637, 0.0002466, -0.0006536, 0.0002575, -0.0006413, 0.0006181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003078, upper bound: 0.0003160
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0003125
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011818, -0.0004543, -0.0011584, -0.0004461, -0.0005668, 0.0005341
1: -0.0042486, -0.0039972, -0.0042451, -0.0039932, -0.0002028, 0.0002005
2: 0.0129871, 0.0139792, 0.0130140, 0.0139919, -0.0007471, 0.0007053
3: 1.0084468, 1.0090851, 1.0084381, 1.0090765, -0.0006297, 0.0006217
4: -0.0038702, -0.0037046, -0.0038726, -0.0037082, -0.0001152, 0.0001206
5: 0.0030386, 0.0036005, 0.0030563, 0.0036070, -0.0004356, 0.0004104
6: -0.0024354, -0.0023773, -0.0024350, -0.0023780, -0.0000574, 0.0000577
7: -0.0129425, -0.0120730, -0.0129435, -0.0121365, -0.0007873, 0.0008513
8: -0.0092609, -0.0074452, -0.0092884, -0.0074789, -0.0012486, 0.0013010
9: -0.0006654, 0.0002451, -0.0006521, 0.0002598, -0.0006438, 0.0006187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003115, upper bound: 0.0003128
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003121, upper bound: 0.0003104
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011818, -0.0004543, -0.0011620, -0.0004473, -0.0005669, 0.0005358
1: -0.0042486, -0.0039972, -0.0042455, -0.0039937, -0.0002032, 0.0002010
2: 0.0129871, 0.0139792, 0.0130100, 0.0139899, -0.0007473, 0.0007072
3: 1.0084468, 1.0090851, 1.0084387, 1.0090773, -0.0006305, 0.0006246
4: -0.0038702, -0.0037046, -0.0038722, -0.0037077, -0.0001153, 0.0001207
5: 0.0030386, 0.0036005, 0.0030536, 0.0036060, -0.0004357, 0.0004116
6: -0.0024354, -0.0023773, -0.0024352, -0.0023779, -0.0000575, 0.0000579
7: -0.0129425, -0.0120730, -0.0129433, -0.0121250, -0.0007974, 0.0008510
8: -0.0092609, -0.0074452, -0.0092842, -0.0074746, -0.0012503, 0.0013018
9: -0.0006654, 0.0002451, -0.0006536, 0.0002575, -0.0006447, 0.0006192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003115, upper bound: 0.0003154
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003121, upper bound: 0.0003133
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011584, -0.0004461, -0.0011776, -0.0004535, -0.0005337, 0.0005641
1: -0.0042451, -0.0039932, -0.0042482, -0.0039970, -0.0002002, 0.0002026
2: 0.0130140, 0.0139919, 0.0129917, 0.0139805, -0.0007046, 0.0007442
3: 1.0084381, 1.0090765, 1.0084479, 1.0090839, -0.0006211, 0.0006282
4: -0.0038726, -0.0037082, -0.0038704, -0.0037051, -0.0001203, 0.0001151
5: 0.0030563, 0.0036070, 0.0030417, 0.0036012, -0.0004101, 0.0004336
6: -0.0024350, -0.0023780, -0.0024351, -0.0023774, -0.0000576, 0.0000571
7: -0.0129435, -0.0121365, -0.0129426, -0.0120867, -0.0008374, 0.0007872
8: -0.0092884, -0.0074789, -0.0092638, -0.0074502, -0.0012981, 0.0012472
9: -0.0006521, 0.0002598, -0.0006637, 0.0002466, -0.0006179, 0.0006429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003087, upper bound: 0.0003106
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003104, upper bound: 0.0003088
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011584, -0.0004461, -0.0011818, -0.0004543, -0.0005341, 0.0005668
1: -0.0042451, -0.0039932, -0.0042486, -0.0039972, -0.0002005, 0.0002028
2: 0.0130140, 0.0139919, 0.0129871, 0.0139792, -0.0007053, 0.0007471
3: 1.0084381, 1.0090765, 1.0084468, 1.0090851, -0.0006217, 0.0006297
4: -0.0038726, -0.0037082, -0.0038702, -0.0037046, -0.0001206, 0.0001152
5: 0.0030563, 0.0036070, 0.0030386, 0.0036005, -0.0004104, 0.0004356
6: -0.0024350, -0.0023780, -0.0024354, -0.0023773, -0.0000577, 0.0000574
7: -0.0129435, -0.0121365, -0.0129425, -0.0120730, -0.0008513, 0.0007873
8: -0.0092884, -0.0074789, -0.0092609, -0.0074452, -0.0013010, 0.0012486
9: -0.0006521, 0.0002598, -0.0006654, 0.0002451, -0.0006187, 0.0006438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003087, upper bound: 0.0003143
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003104, upper bound: 0.0003121
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011620, -0.0004473, -0.0011776, -0.0004535, -0.0005363, 0.0005632
1: -0.0042455, -0.0039937, -0.0042482, -0.0039970, -0.0002003, 0.0002027
2: 0.0130100, 0.0139899, 0.0129917, 0.0139805, -0.0007069, 0.0007428
3: 1.0084387, 1.0090773, 1.0084479, 1.0090839, -0.0006229, 0.0006283
4: -0.0038722, -0.0037077, -0.0038704, -0.0037051, -0.0001201, 0.0001152
5: 0.0030536, 0.0036060, 0.0030417, 0.0036012, -0.0004120, 0.0004329
6: -0.0024352, -0.0023779, -0.0024351, -0.0023774, -0.0000578, 0.0000571
7: -0.0129433, -0.0121250, -0.0129426, -0.0120867, -0.0008373, 0.0007977
8: -0.0092842, -0.0074746, -0.0092638, -0.0074502, -0.0012952, 0.0012479
9: -0.0006536, 0.0002575, -0.0006637, 0.0002466, -0.0006181, 0.0006413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003106, upper bound: 0.0003108
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003125, upper bound: 0.0003090
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011620, -0.0004473, -0.0011818, -0.0004543, -0.0005358, 0.0005669
1: -0.0042455, -0.0039937, -0.0042486, -0.0039972, -0.0002010, 0.0002032
2: 0.0130100, 0.0139899, 0.0129871, 0.0139792, -0.0007072, 0.0007473
3: 1.0084387, 1.0090773, 1.0084468, 1.0090851, -0.0006246, 0.0006305
4: -0.0038722, -0.0037077, -0.0038702, -0.0037046, -0.0001207, 0.0001153
5: 0.0030536, 0.0036060, 0.0030386, 0.0036005, -0.0004116, 0.0004357
6: -0.0024352, -0.0023779, -0.0024354, -0.0023773, -0.0000579, 0.0000575
7: -0.0129433, -0.0121250, -0.0129425, -0.0120730, -0.0008510, 0.0007974
8: -0.0092842, -0.0074746, -0.0092609, -0.0074452, -0.0013018, 0.0012503
9: -0.0006536, 0.0002575, -0.0006654, 0.0002451, -0.0006192, 0.0006447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003106, upper bound: 0.0003141
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003125, upper bound: 0.0003122
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011584, -0.0004461, -0.0011584, -0.0004461, -0.0005404, 0.0005404
1: -0.0042451, -0.0039932, -0.0042451, -0.0039932, -0.0001992, 0.0001992
2: 0.0130140, 0.0139919, 0.0130140, 0.0139919, -0.0007139, 0.0007139
3: 1.0084381, 1.0090765, 1.0084381, 1.0090765, -0.0006149, 0.0006149
4: -0.0038726, -0.0037082, -0.0038726, -0.0037082, -0.0001161, 0.0001161
5: 0.0030563, 0.0036070, 0.0030563, 0.0036070, -0.0004155, 0.0004155
6: -0.0024350, -0.0023780, -0.0024350, -0.0023780, -0.0000570, 0.0000570
7: -0.0129435, -0.0121365, -0.0129435, -0.0121365, -0.0007882, 0.0007882
8: -0.0092884, -0.0074789, -0.0092884, -0.0074789, -0.0012584, 0.0012584
9: -0.0006521, 0.0002598, -0.0006521, 0.0002598, -0.0006235, 0.0006235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003052, upper bound: 0.0003088
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003070, upper bound: 0.0003070
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011584, -0.0004461, -0.0011620, -0.0004473, -0.0005412, 0.0005432
1: -0.0042451, -0.0039932, -0.0042455, -0.0039937, -0.0001993, 0.0001994
2: 0.0130140, 0.0139919, 0.0130100, 0.0139899, -0.0007151, 0.0007163
3: 1.0084381, 1.0090765, 1.0084387, 1.0090773, -0.0006153, 0.0006161
4: -0.0038726, -0.0037082, -0.0038722, -0.0037077, -0.0001163, 0.0001163
5: 0.0030563, 0.0036070, 0.0030536, 0.0036060, -0.0004161, 0.0004175
6: -0.0024350, -0.0023780, -0.0024352, -0.0023779, -0.0000571, 0.0000572
7: -0.0129435, -0.0121365, -0.0129433, -0.0121250, -0.0007990, 0.0007883
8: -0.0092884, -0.0074789, -0.0092842, -0.0074746, -0.0012598, 0.0012611
9: -0.0006521, 0.0002598, -0.0006536, 0.0002575, -0.0006250, 0.0006241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003052, upper bound: 0.0003120
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003070, upper bound: 0.0003097
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011620, -0.0004473, -0.0011584, -0.0004461, -0.0005432, 0.0005412
1: -0.0042455, -0.0039937, -0.0042451, -0.0039932, -0.0001994, 0.0001993
2: 0.0130100, 0.0139899, 0.0130140, 0.0139919, -0.0007163, 0.0007151
3: 1.0084387, 1.0090773, 1.0084381, 1.0090765, -0.0006161, 0.0006153
4: -0.0038722, -0.0037077, -0.0038726, -0.0037082, -0.0001163, 0.0001163
5: 0.0030536, 0.0036060, 0.0030563, 0.0036070, -0.0004175, 0.0004161
6: -0.0024352, -0.0023779, -0.0024350, -0.0023780, -0.0000572, 0.0000571
7: -0.0129433, -0.0121250, -0.0129435, -0.0121365, -0.0007883, 0.0007990
8: -0.0092842, -0.0074746, -0.0092884, -0.0074789, -0.0012611, 0.0012598
9: -0.0006536, 0.0002575, -0.0006521, 0.0002598, -0.0006241, 0.0006250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003083, upper bound: 0.0003089
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003097, upper bound: 0.0003072
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011620, -0.0004473, -0.0011620, -0.0004473, -0.0005421, 0.0005421
1: -0.0042455, -0.0039937, -0.0042455, -0.0039937, -0.0001995, 0.0001995
2: 0.0130100, 0.0139899, 0.0130100, 0.0139899, -0.0007155, 0.0007155
3: 1.0084387, 1.0090773, 1.0084387, 1.0090773, -0.0006178, 0.0006178
4: -0.0038722, -0.0037077, -0.0038722, -0.0037077, -0.0001162, 0.0001162
5: 0.0030536, 0.0036060, 0.0030536, 0.0036060, -0.0004167, 0.0004167
6: -0.0024352, -0.0023779, -0.0024352, -0.0023779, -0.0000573, 0.0000573
7: -0.0129433, -0.0121250, -0.0129433, -0.0121250, -0.0007986, 0.0007986
8: -0.0092842, -0.0074746, -0.0092842, -0.0074746, -0.0012594, 0.0012594
9: -0.0006536, 0.0002575, -0.0006536, 0.0002575, -0.0006239, 0.0006239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003083, upper bound: 0.0003130
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003097, upper bound: 0.0003116
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011718, -0.0004537, -0.0010680, -0.0004214, -0.0005654, 0.0004476
1: -0.0042474, -0.0039971, -0.0042263, -0.0039827, -0.0001974, 0.0001835
2: 0.0129982, 0.0139802, 0.0131232, 0.0140297, -0.0007488, 0.0006045
3: 1.0084481, 1.0090820, 1.0084152, 1.0090296, -0.0005815, 0.0005766
4: -0.0038704, -0.0037060, -0.0038796, -0.0037239, -0.0001006, 0.0001216
5: 0.0030461, 0.0036011, 0.0031249, 0.0036264, -0.0004348, 0.0003452
6: -0.0024346, -0.0023775, -0.0024340, -0.0023819, -0.0000528, 0.0000565
7: -0.0129426, -0.0121019, -0.0129464, -0.0123263, -0.0005959, 0.0008231
8: -0.0092631, -0.0074579, -0.0093705, -0.0076356, -0.0011035, 0.0013146
9: -0.0006607, 0.0002463, -0.0005824, 0.0003036, -0.0006533, 0.0005572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003278, upper bound: 0.0003304
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003278, upper bound: 0.0003304
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011740, -0.0004535, -0.0010621, -0.0004275, -0.0005952, 0.0004284
1: -0.0042477, -0.0039971, -0.0042243, -0.0039848, -0.0002131, 0.0001777
2: 0.0129958, 0.0139804, 0.0131312, 0.0140204, -0.0007931, 0.0005749
3: 1.0084481, 1.0090828, 1.0084182, 1.0090244, -0.0005763, 0.0006118
4: -0.0038704, -0.0037057, -0.0038779, -0.0037253, -0.0000949, 0.0001301
5: 0.0030445, 0.0036012, 0.0031295, 0.0036216, -0.0004581, 0.0003301
6: -0.0024349, -0.0023775, -0.0024342, -0.0023823, -0.0000526, 0.0000568
7: -0.0129426, -0.0120964, -0.0129457, -0.0123290, -0.0005936, 0.0008307
8: -0.0092636, -0.0074549, -0.0093503, -0.0076505, -0.0010380, 0.0014193
9: -0.0006618, 0.0002465, -0.0005749, 0.0002929, -0.0007110, 0.0005217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002147, upper bound: 0.0001979
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0001971, upper bound: 0.0001771
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003278, upper bound: 0.0003347
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003278, upper bound: 0.0003347
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011760, -0.0004545, -0.0010700, -0.0004208, -0.0005679, 0.0004499
1: -0.0042478, -0.0039973, -0.0042265, -0.0039823, -0.0001986, 0.0001842
2: 0.0129936, 0.0139789, 0.0131211, 0.0140308, -0.0007511, 0.0006075
3: 1.0084474, 1.0090832, 1.0084143, 1.0090301, -0.0005827, 0.0005818
4: -0.0038701, -0.0037054, -0.0038798, -0.0037237, -0.0001010, 0.0001219
5: 0.0030430, 0.0036004, 0.0031234, 0.0036269, -0.0004366, 0.0003469
6: -0.0024350, -0.0023775, -0.0024341, -0.0023818, -0.0000532, 0.0000566
7: -0.0129425, -0.0120882, -0.0129465, -0.0123208, -0.0006015, 0.0008369
8: -0.0092602, -0.0074529, -0.0093728, -0.0076331, -0.0011075, 0.0013173
9: -0.0006624, 0.0002447, -0.0005833, 0.0003049, -0.0006539, 0.0005595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002164, upper bound: 0.0001993
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002084, upper bound: 0.0001923
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002393, upper bound: 0.0002947
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003306, upper bound: 0.0003313
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003306, upper bound: 0.0003313
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011782, -0.0004544, -0.0010639, -0.0004268, -0.0006023, 0.0004331
1: -0.0042481, -0.0039972, -0.0042245, -0.0039845, -0.0002150, 0.0001791
2: 0.0129912, 0.0139791, 0.0131291, 0.0140215, -0.0008025, 0.0005815
3: 1.0084471, 1.0090839, 1.0084174, 1.0090250, -0.0005779, 0.0006182
4: -0.0038702, -0.0037051, -0.0038781, -0.0037250, -0.0000960, 0.0001318
5: 0.0030413, 0.0036005, 0.0031281, 0.0036222, -0.0004636, 0.0003337
6: -0.0024352, -0.0023774, -0.0024344, -0.0023822, -0.0000530, 0.0000570
7: -0.0129425, -0.0120827, -0.0129458, -0.0123239, -0.0005990, 0.0008450
8: -0.0092607, -0.0074500, -0.0093526, -0.0076482, -0.0010504, 0.0014374
9: -0.0006635, 0.0002450, -0.0005758, 0.0002941, -0.0007222, 0.0005283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002304, upper bound: 0.0002252
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002079, upper bound: 0.0001996
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002327, upper bound: 0.0002956
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003306, upper bound: 0.0003355
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003306, upper bound: 0.0003355
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011718, -0.0004537, -0.0010484, -0.0004153, -0.0005776, 0.0004338
1: -0.0042474, -0.0039971, -0.0042237, -0.0039796, -0.0002012, 0.0001839
2: 0.0129982, 0.0139802, 0.0131448, 0.0140392, -0.0007676, 0.0005915
3: 1.0084481, 1.0090820, 1.0084088, 1.0090230, -0.0005748, 0.0005733
4: -0.0038704, -0.0037060, -0.0038814, -0.0037268, -0.0000997, 0.0001251
5: 0.0030461, 0.0036011, 0.0031396, 0.0036313, -0.0004444, 0.0003350
6: -0.0024346, -0.0023775, -0.0024341, -0.0023824, -0.0000522, 0.0000566
7: -0.0129426, -0.0121019, -0.0129471, -0.0123850, -0.0005389, 0.0008246
8: -0.0092631, -0.0074579, -0.0093910, -0.0076613, -0.0011014, 0.0013555
9: -0.0006607, 0.0002463, -0.0005725, 0.0003146, -0.0006751, 0.0005584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003239, upper bound: 0.0003275
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003239, upper bound: 0.0003275
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011740, -0.0004535, -0.0010471, -0.0004208, -0.0006097, 0.0004201
1: -0.0042477, -0.0039971, -0.0042216, -0.0039816, -0.0002176, 0.0001781
2: 0.0129958, 0.0139804, 0.0131484, 0.0140307, -0.0008154, 0.0005704
3: 1.0084481, 1.0090828, 1.0084103, 1.0090179, -0.0005698, 0.0006151
4: -0.0038704, -0.0037057, -0.0038798, -0.0037277, -0.0000949, 0.0001343
5: 0.0030445, 0.0036012, 0.0031408, 0.0036269, -0.0004696, 0.0003243
6: -0.0024349, -0.0023775, -0.0024346, -0.0023828, -0.0000521, 0.0000571
7: -0.0129426, -0.0120964, -0.0129465, -0.0123622, -0.0005612, 0.0008324
8: -0.0092636, -0.0074549, -0.0093727, -0.0076739, -0.0010411, 0.0014677
9: -0.0006618, 0.0002465, -0.0005650, 0.0003048, -0.0007368, 0.0005232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003239, upper bound: 0.0003327
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003239, upper bound: 0.0003327
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011760, -0.0004545, -0.0010504, -0.0004145, -0.0005827, 0.0004361
1: -0.0042478, -0.0039973, -0.0042239, -0.0039792, -0.0002023, 0.0001845
2: 0.0129936, 0.0139789, 0.0131426, 0.0140404, -0.0007739, 0.0005944
3: 1.0084474, 1.0090832, 1.0084080, 1.0090235, -0.0005761, 0.0005790
4: -0.0038701, -0.0037054, -0.0038816, -0.0037265, -0.0001002, 0.0001261
5: 0.0030430, 0.0036004, 0.0031382, 0.0036319, -0.0004483, 0.0003368
6: -0.0024350, -0.0023775, -0.0024342, -0.0023824, -0.0000526, 0.0000567
7: -0.0129425, -0.0120882, -0.0129472, -0.0123795, -0.0005445, 0.0008387
8: -0.0092602, -0.0074529, -0.0093935, -0.0076588, -0.0011062, 0.0013667
9: -0.0006624, 0.0002447, -0.0005734, 0.0003159, -0.0006803, 0.0005606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002214, upper bound: 0.0002808
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003270, upper bound: 0.0003285
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003270, upper bound: 0.0003285
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011782, -0.0004544, -0.0010488, -0.0004200, -0.0006177, 0.0004246
1: -0.0042481, -0.0039972, -0.0042219, -0.0039812, -0.0002204, 0.0001795
2: 0.0129912, 0.0139791, 0.0131465, 0.0140319, -0.0008262, 0.0005769
3: 1.0084471, 1.0090839, 1.0084093, 1.0090184, -0.0005714, 0.0006202
4: -0.0038702, -0.0037051, -0.0038800, -0.0037275, -0.0000960, 0.0001362
5: 0.0030413, 0.0036005, 0.0031395, 0.0036275, -0.0004757, 0.0003278
6: -0.0024352, -0.0023774, -0.0024347, -0.0023828, -0.0000524, 0.0000573
7: -0.0129425, -0.0120827, -0.0129465, -0.0123577, -0.0005660, 0.0008468
8: -0.0092607, -0.0074500, -0.0093751, -0.0076715, -0.0010539, 0.0014887
9: -0.0006635, 0.0002450, -0.0005659, 0.0003061, -0.0007497, 0.0005300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002198, upper bound: 0.0002819
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003270, upper bound: 0.0003337
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003270, upper bound: 0.0003337
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011521, -0.0004463, -0.0010680, -0.0004214, -0.0005533, 0.0004662
1: -0.0042443, -0.0039934, -0.0042263, -0.0039827, -0.0001996, 0.0001881
2: 0.0130210, 0.0139915, 0.0131232, 0.0140297, -0.0007376, 0.0006332
3: 1.0084389, 1.0090744, 1.0084152, 1.0090296, -0.0005821, 0.0005821
4: -0.0038725, -0.0037091, -0.0038796, -0.0037239, -0.0001059, 0.0001216
5: 0.0030610, 0.0036068, 0.0031249, 0.0036264, -0.0004258, 0.0003599
6: -0.0024346, -0.0023782, -0.0024340, -0.0023819, -0.0000527, 0.0000558
7: -0.0129435, -0.0121545, -0.0129464, -0.0123263, -0.0005981, 0.0007723
8: -0.0092877, -0.0074870, -0.0093705, -0.0076356, -0.0011657, 0.0013256
9: -0.0006491, 0.0002594, -0.0005824, 0.0003036, -0.0006613, 0.0005904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003241, upper bound: 0.0003259
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003241, upper bound: 0.0003259
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011550, -0.0004461, -0.0010621, -0.0004275, -0.0005821, 0.0004464
1: -0.0042446, -0.0039933, -0.0042243, -0.0039848, -0.0002152, 0.0001814
2: 0.0130178, 0.0139917, 0.0131312, 0.0140204, -0.0007861, 0.0006026
3: 1.0084386, 1.0090753, 1.0084182, 1.0090244, -0.0005760, 0.0006169
4: -0.0038725, -0.0037087, -0.0038779, -0.0037253, -0.0001000, 0.0001305
5: 0.0030588, 0.0036070, 0.0031295, 0.0036216, -0.0004489, 0.0003443
6: -0.0024349, -0.0023781, -0.0024342, -0.0023823, -0.0000526, 0.0000561
7: -0.0129435, -0.0121449, -0.0129457, -0.0123290, -0.0005957, 0.0007848
8: -0.0092882, -0.0074834, -0.0093503, -0.0076505, -0.0010980, 0.0014304
9: -0.0006504, 0.0002597, -0.0005749, 0.0002929, -0.0007187, 0.0005538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003241, upper bound: 0.0003305
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003241, upper bound: 0.0003305
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011557, -0.0004475, -0.0010700, -0.0004208, -0.0005555, 0.0004673
1: -0.0042447, -0.0039938, -0.0042265, -0.0039823, -0.0002010, 0.0001886
2: 0.0130170, 0.0139896, 0.0131211, 0.0140308, -0.0007402, 0.0006342
3: 1.0084394, 1.0090754, 1.0084143, 1.0090301, -0.0005850, 0.0005876
4: -0.0038721, -0.0037086, -0.0038798, -0.0037237, -0.0001060, 0.0001219
5: 0.0030583, 0.0036059, 0.0031234, 0.0036269, -0.0004274, 0.0003606
6: -0.0024348, -0.0023781, -0.0024341, -0.0023818, -0.0000530, 0.0000560
7: -0.0129433, -0.0121427, -0.0129465, -0.0123208, -0.0006036, 0.0007829
8: -0.0092835, -0.0074827, -0.0093728, -0.0076331, -0.0011655, 0.0013285
9: -0.0006506, 0.0002572, -0.0005833, 0.0003049, -0.0006625, 0.0005904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003269, upper bound: 0.0003273
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003269, upper bound: 0.0003273
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011586, -0.0004474, -0.0010639, -0.0004268, -0.0005892, 0.0004498
1: -0.0042450, -0.0039937, -0.0042245, -0.0039845, -0.0002171, 0.0001822
2: 0.0130139, 0.0139898, 0.0131291, 0.0140215, -0.0007951, 0.0006072
3: 1.0084389, 1.0090762, 1.0084174, 1.0090250, -0.0005783, 0.0006232
4: -0.0038722, -0.0037082, -0.0038781, -0.0037250, -0.0001007, 0.0001321
5: 0.0030561, 0.0036060, 0.0031281, 0.0036222, -0.0004543, 0.0003469
6: -0.0024351, -0.0023780, -0.0024344, -0.0023822, -0.0000528, 0.0000563
7: -0.0129433, -0.0121336, -0.0129458, -0.0123239, -0.0006010, 0.0007954
8: -0.0092840, -0.0074792, -0.0093526, -0.0076482, -0.0011060, 0.0014493
9: -0.0006519, 0.0002574, -0.0005758, 0.0002941, -0.0007298, 0.0005580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003269, upper bound: 0.0003325
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003269, upper bound: 0.0003325
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011521, -0.0004463, -0.0010484, -0.0004153, -0.0005587, 0.0004407
1: -0.0042443, -0.0039934, -0.0042237, -0.0039796, -0.0001981, 0.0001840
2: 0.0130210, 0.0139915, 0.0131448, 0.0140392, -0.0007449, 0.0006022
3: 1.0084389, 1.0090744, 1.0084088, 1.0090230, -0.0005740, 0.0005692
4: -0.0038725, -0.0037091, -0.0038814, -0.0037268, -0.0001013, 0.0001222
5: 0.0030610, 0.0036068, 0.0031396, 0.0036313, -0.0004301, 0.0003404
6: -0.0024346, -0.0023782, -0.0024341, -0.0023824, -0.0000522, 0.0000559
7: -0.0129435, -0.0121545, -0.0129471, -0.0123850, -0.0005401, 0.0007732
8: -0.0092877, -0.0074870, -0.0093910, -0.0076613, -0.0011202, 0.0013325
9: -0.0006491, 0.0002594, -0.0005725, 0.0003146, -0.0006645, 0.0005685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003223, upper bound: 0.0003253
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003223, upper bound: 0.0003253
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011550, -0.0004461, -0.0010471, -0.0004208, -0.0005892, 0.0004278
1: -0.0042446, -0.0039933, -0.0042216, -0.0039816, -0.0002153, 0.0001782
2: 0.0130178, 0.0139917, 0.0131484, 0.0140307, -0.0007937, 0.0005806
3: 1.0084386, 1.0090753, 1.0084103, 1.0090179, -0.0005698, 0.0006103
4: -0.0038725, -0.0037087, -0.0038798, -0.0037277, -0.0000968, 0.0001317
5: 0.0030588, 0.0036070, 0.0031408, 0.0036269, -0.0004540, 0.0003302
6: -0.0024349, -0.0023781, -0.0024346, -0.0023828, -0.0000520, 0.0000565
7: -0.0129435, -0.0121449, -0.0129465, -0.0123622, -0.0005622, 0.0007858
8: -0.0092882, -0.0074834, -0.0093727, -0.0076739, -0.0010621, 0.0014432
9: -0.0006504, 0.0002597, -0.0005650, 0.0003048, -0.0007254, 0.0005354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003223, upper bound: 0.0003302
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003223, upper bound: 0.0003302
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011557, -0.0004475, -0.0010504, -0.0004145, -0.0005608, 0.0004434
1: -0.0042447, -0.0039938, -0.0042239, -0.0039792, -0.0001990, 0.0001845
2: 0.0130170, 0.0139896, 0.0131426, 0.0140404, -0.0007471, 0.0006057
3: 1.0084394, 1.0090754, 1.0084080, 1.0090235, -0.0005763, 0.0005731
4: -0.0038721, -0.0037086, -0.0038816, -0.0037265, -0.0001018, 0.0001225
5: 0.0030583, 0.0036059, 0.0031382, 0.0036319, -0.0004316, 0.0003425
6: -0.0024348, -0.0023781, -0.0024342, -0.0023824, -0.0000524, 0.0000561
7: -0.0129433, -0.0121427, -0.0129472, -0.0123795, -0.0005457, 0.0007840
8: -0.0092835, -0.0074827, -0.0093935, -0.0076588, -0.0011261, 0.0013346
9: -0.0006506, 0.0002572, -0.0005734, 0.0003159, -0.0006655, 0.0005715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003263, upper bound: 0.0003271
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003263, upper bound: 0.0003271
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011586, -0.0004474, -0.0010488, -0.0004200, -0.0005957, 0.0004320
1: -0.0042450, -0.0039937, -0.0042219, -0.0039812, -0.0002169, 0.0001794
2: 0.0130139, 0.0139898, 0.0131465, 0.0140319, -0.0008030, 0.0005867
3: 1.0084389, 1.0090762, 1.0084093, 1.0090184, -0.0005731, 0.0006152
4: -0.0038722, -0.0037082, -0.0038800, -0.0037275, -0.0000978, 0.0001332
5: 0.0030561, 0.0036060, 0.0031395, 0.0036275, -0.0004590, 0.0003334
6: -0.0024351, -0.0023780, -0.0024347, -0.0023828, -0.0000523, 0.0000567
7: -0.0129433, -0.0121336, -0.0129465, -0.0123577, -0.0005670, 0.0007964
8: -0.0092840, -0.0074792, -0.0093751, -0.0076715, -0.0010738, 0.0014613
9: -0.0006519, 0.0002574, -0.0005659, 0.0003061, -0.0007350, 0.0005414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003263, upper bound: 0.0003324
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003263, upper bound: 0.0003324
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010680, -0.0004214, -0.0011718, -0.0004537, -0.0004476, 0.0005654
1: -0.0042263, -0.0039827, -0.0042474, -0.0039971, -0.0001835, 0.0001974
2: 0.0131232, 0.0140297, 0.0129982, 0.0139802, -0.0006045, 0.0007488
3: 1.0084152, 1.0090296, 1.0084481, 1.0090820, -0.0005766, 0.0005815
4: -0.0038796, -0.0037239, -0.0038704, -0.0037060, -0.0001216, 0.0001006
5: 0.0031249, 0.0036264, 0.0030461, 0.0036011, -0.0003452, 0.0004348
6: -0.0024340, -0.0023819, -0.0024346, -0.0023775, -0.0000565, 0.0000528
7: -0.0129464, -0.0123263, -0.0129426, -0.0121019, -0.0008231, 0.0005959
8: -0.0093705, -0.0076356, -0.0092631, -0.0074579, -0.0013146, 0.0011035
9: -0.0005824, 0.0003036, -0.0006607, 0.0002463, -0.0005572, 0.0006533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003251, upper bound: 0.0003283
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003251, upper bound: 0.0003283
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010700, -0.0004208, -0.0011760, -0.0004545, -0.0004499, 0.0005679
1: -0.0042265, -0.0039823, -0.0042478, -0.0039973, -0.0001842, 0.0001986
2: 0.0131211, 0.0140308, 0.0129936, 0.0139789, -0.0006075, 0.0007511
3: 1.0084143, 1.0090301, 1.0084474, 1.0090832, -0.0005818, 0.0005827
4: -0.0038798, -0.0037237, -0.0038701, -0.0037054, -0.0001219, 0.0001010
5: 0.0031234, 0.0036269, 0.0030430, 0.0036004, -0.0003469, 0.0004366
6: -0.0024341, -0.0023818, -0.0024350, -0.0023775, -0.0000566, 0.0000532
7: -0.0129465, -0.0123208, -0.0129425, -0.0120882, -0.0008369, 0.0006015
8: -0.0093728, -0.0076331, -0.0092602, -0.0074529, -0.0013173, 0.0011075
9: -0.0005833, 0.0003049, -0.0006624, 0.0002447, -0.0005595, 0.0006539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003251, upper bound: 0.0003302
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003251, upper bound: 0.0003314
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010621, -0.0004275, -0.0011740, -0.0004535, -0.0004284, 0.0005952
1: -0.0042243, -0.0039848, -0.0042477, -0.0039971, -0.0001777, 0.0002131
2: 0.0131312, 0.0140204, 0.0129958, 0.0139804, -0.0005749, 0.0007931
3: 1.0084182, 1.0090244, 1.0084481, 1.0090828, -0.0006118, 0.0005763
4: -0.0038779, -0.0037253, -0.0038704, -0.0037057, -0.0001301, 0.0000949
5: 0.0031295, 0.0036216, 0.0030445, 0.0036012, -0.0003301, 0.0004581
6: -0.0024342, -0.0023823, -0.0024349, -0.0023775, -0.0000568, 0.0000526
7: -0.0129457, -0.0123290, -0.0129426, -0.0120964, -0.0008307, 0.0005936
8: -0.0093503, -0.0076505, -0.0092636, -0.0074549, -0.0014193, 0.0010380
9: -0.0005749, 0.0002929, -0.0006618, 0.0002465, -0.0005217, 0.0007110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003291, upper bound: 0.0003280
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003291, upper bound: 0.0003280
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010639, -0.0004268, -0.0011782, -0.0004544, -0.0004331, 0.0006023
1: -0.0042245, -0.0039845, -0.0042481, -0.0039972, -0.0001791, 0.0002150
2: 0.0131291, 0.0140215, 0.0129912, 0.0139791, -0.0005815, 0.0008025
3: 1.0084174, 1.0090250, 1.0084471, 1.0090839, -0.0006182, 0.0005779
4: -0.0038781, -0.0037250, -0.0038702, -0.0037051, -0.0001318, 0.0000960
5: 0.0031281, 0.0036222, 0.0030413, 0.0036005, -0.0003337, 0.0004636
6: -0.0024344, -0.0023822, -0.0024352, -0.0023774, -0.0000570, 0.0000530
7: -0.0129458, -0.0123239, -0.0129425, -0.0120827, -0.0008450, 0.0005990
8: -0.0093526, -0.0076482, -0.0092607, -0.0074500, -0.0014374, 0.0010504
9: -0.0005758, 0.0002941, -0.0006635, 0.0002450, -0.0005283, 0.0007222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003291, upper bound: 0.0003297
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003291, upper bound: 0.0003307
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010680, -0.0004214, -0.0011521, -0.0004463, -0.0004662, 0.0005533
1: -0.0042263, -0.0039827, -0.0042443, -0.0039934, -0.0001881, 0.0001996
2: 0.0131232, 0.0140297, 0.0130210, 0.0139915, -0.0006332, 0.0007376
3: 1.0084152, 1.0090296, 1.0084389, 1.0090744, -0.0005821, 0.0005821
4: -0.0038796, -0.0037239, -0.0038725, -0.0037091, -0.0001216, 0.0001059
5: 0.0031249, 0.0036264, 0.0030610, 0.0036068, -0.0003599, 0.0004258
6: -0.0024340, -0.0023819, -0.0024346, -0.0023782, -0.0000558, 0.0000527
7: -0.0129464, -0.0123263, -0.0129435, -0.0121545, -0.0007723, 0.0005981
8: -0.0093705, -0.0076356, -0.0092877, -0.0074870, -0.0013256, 0.0011657
9: -0.0005824, 0.0003036, -0.0006491, 0.0002594, -0.0005904, 0.0006613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003188, upper bound: 0.0003250
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003188, upper bound: 0.0003250
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010700, -0.0004208, -0.0011557, -0.0004475, -0.0004673, 0.0005555
1: -0.0042265, -0.0039823, -0.0042447, -0.0039938, -0.0001886, 0.0002010
2: 0.0131211, 0.0140308, 0.0130170, 0.0139896, -0.0006342, 0.0007402
3: 1.0084143, 1.0090301, 1.0084394, 1.0090754, -0.0005876, 0.0005850
4: -0.0038798, -0.0037237, -0.0038721, -0.0037086, -0.0001219, 0.0001060
5: 0.0031234, 0.0036269, 0.0030583, 0.0036059, -0.0003606, 0.0004274
6: -0.0024341, -0.0023818, -0.0024348, -0.0023781, -0.0000560, 0.0000530
7: -0.0129465, -0.0123208, -0.0129433, -0.0121427, -0.0007829, 0.0006036
8: -0.0093728, -0.0076331, -0.0092835, -0.0074827, -0.0013285, 0.0011655
9: -0.0005833, 0.0003049, -0.0006506, 0.0002572, -0.0005904, 0.0006625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003188, upper bound: 0.0003264
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003188, upper bound: 0.0003278
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010621, -0.0004275, -0.0011550, -0.0004461, -0.0004464, 0.0005821
1: -0.0042243, -0.0039848, -0.0042446, -0.0039933, -0.0001814, 0.0002152
2: 0.0131312, 0.0140204, 0.0130178, 0.0139917, -0.0006026, 0.0007861
3: 1.0084182, 1.0090244, 1.0084386, 1.0090753, -0.0006169, 0.0005760
4: -0.0038779, -0.0037253, -0.0038725, -0.0037087, -0.0001305, 0.0001000
5: 0.0031295, 0.0036216, 0.0030588, 0.0036070, -0.0003443, 0.0004489
6: -0.0024342, -0.0023823, -0.0024349, -0.0023781, -0.0000561, 0.0000526
7: -0.0129457, -0.0123290, -0.0129435, -0.0121449, -0.0007848, 0.0005957
8: -0.0093503, -0.0076505, -0.0092882, -0.0074834, -0.0014304, 0.0010980
9: -0.0005749, 0.0002929, -0.0006504, 0.0002597, -0.0005538, 0.0007187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003239, upper bound: 0.0003247
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003239, upper bound: 0.0003247
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010639, -0.0004268, -0.0011586, -0.0004474, -0.0004498, 0.0005892
1: -0.0042245, -0.0039845, -0.0042450, -0.0039937, -0.0001822, 0.0002171
2: 0.0131291, 0.0140215, 0.0130139, 0.0139898, -0.0006072, 0.0007951
3: 1.0084174, 1.0090250, 1.0084389, 1.0090762, -0.0006232, 0.0005783
4: -0.0038781, -0.0037250, -0.0038722, -0.0037082, -0.0001321, 0.0001007
5: 0.0031281, 0.0036222, 0.0030561, 0.0036060, -0.0003469, 0.0004543
6: -0.0024344, -0.0023822, -0.0024351, -0.0023780, -0.0000563, 0.0000528
7: -0.0129458, -0.0123239, -0.0129433, -0.0121336, -0.0007954, 0.0006010
8: -0.0093526, -0.0076482, -0.0092840, -0.0074792, -0.0014493, 0.0011060
9: -0.0005758, 0.0002941, -0.0006519, 0.0002574, -0.0005580, 0.0007298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003239, upper bound: 0.0003260
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003239, upper bound: 0.0003273
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010484, -0.0004153, -0.0011718, -0.0004537, -0.0004338, 0.0005776
1: -0.0042237, -0.0039796, -0.0042474, -0.0039971, -0.0001839, 0.0002012
2: 0.0131448, 0.0140392, 0.0129982, 0.0139802, -0.0005915, 0.0007676
3: 1.0084088, 1.0090230, 1.0084481, 1.0090820, -0.0005733, 0.0005748
4: -0.0038814, -0.0037268, -0.0038704, -0.0037060, -0.0001251, 0.0000997
5: 0.0031396, 0.0036313, 0.0030461, 0.0036011, -0.0003350, 0.0004444
6: -0.0024341, -0.0023824, -0.0024346, -0.0023775, -0.0000566, 0.0000522
7: -0.0129471, -0.0123850, -0.0129426, -0.0121019, -0.0008246, 0.0005389
8: -0.0093910, -0.0076613, -0.0092631, -0.0074579, -0.0013555, 0.0011014
9: -0.0005725, 0.0003146, -0.0006607, 0.0002463, -0.0005584, 0.0006751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.03 + 597.82 = 600.85 seconds
