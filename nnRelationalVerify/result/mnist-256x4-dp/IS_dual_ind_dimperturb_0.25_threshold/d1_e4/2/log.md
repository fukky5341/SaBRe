## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 6.43e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0001288, 0.0009471, -0.0001288, 0.0009471, -0.0007579, 0.0007579)
1: (-0.0035063, -0.0033086, -0.0035063, -0.0033086, -0.0001383, 0.0001383)
2: (0.0147743, 0.0161257, 0.0147743, 0.0161257, -0.0008606, 0.0008606)
3: (1.0066677, 1.0069965, 1.0066677, 1.0069965, -0.0003288, 0.0003288)
4: (-0.0042691, -0.0040629, -0.0042691, -0.0040629, -0.0001149, 0.0001149)
5: (0.0038815, 0.0047028, 0.0038815, 0.0047028, -0.0005708, 0.0005708)
6: (-0.0026098, -0.0025608, -0.0026098, -0.0025608, -0.0000490, 0.0000490)
7: (-0.0131697, -0.0111883, -0.0131697, -0.0111883, -0.0019462, 0.0019462)
8: (-0.0138999, -0.0117600, -0.0138999, -0.0117600, -0.0010809, 0.0010809)
9: (0.0017152, 0.0027260, 0.0017152, 0.0027260, -0.0004456, 0.0004456)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.57 + 1.36 = 2.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0001101, upper bound: 0.0001101

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 45

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 189

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000926, upper bound: 0.0000957
time: 0.53 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000973, upper bound: 0.0000974
time: 0.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.32 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.32
Output dim: 3, lower bound: -0.0000926, upper bound: 0.0000957
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.32
Output dim: 3, lower bound: -0.0000973, upper bound: 0.0000974

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007769, -0.0001286, 0.0008734, -0.0004701, 0.0004134
1: -0.0035067, -0.0033148, -0.0035063, -0.0033129, -0.0000694, 0.0000675
2: 0.0147868, 0.0159336, 0.0147745, 0.0160424, -0.0005650, 0.0005011
3: 1.0066483, 1.0069599, 1.0066805, 1.0069799, -0.0002019, 0.0001737
4: -0.0042439, -0.0040637, -0.0042581, -0.0040630, -0.0000733, 0.0000818
5: 0.0038926, 0.0045746, 0.0038816, 0.0046473, -0.0003567, 0.0003140
6: -0.0026078, -0.0025764, -0.0026096, -0.0025681, -0.0000348, 0.0000286
7: -0.0127334, -0.0112484, -0.0129760, -0.0111883, -0.0008804, 0.0010318
8: -0.0136690, -0.0117601, -0.0137995, -0.0117604, -0.0007398, 0.0008207
9: 0.0017133, 0.0026348, 0.0017152, 0.0026867, -0.0003722, 0.0003388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000801, upper bound: 0.0000906
time: 0.62 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000880, upper bound: 0.0000906
time: 0.61 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0001287, 0.0009294, -0.0001287, 0.0009432, -0.0007549, 0.0006801
1: -0.0035063, -0.0033109, -0.0035063, -0.0033092, -0.0001312, 0.0001550
2: 0.0147744, 0.0161034, 0.0147743, 0.0161203, -0.0008570, 0.0007452
3: 1.0066744, 1.0069788, 1.0066694, 1.0069926, -0.0003182, 0.0003093
4: -0.0042658, -0.0040630, -0.0042684, -0.0040629, -0.0000937, 0.0001146
5: 0.0038815, 0.0046893, 0.0038815, 0.0046997, -0.0005685, 0.0005100
6: -0.0026097, -0.0025616, -0.0026098, -0.0025610, -0.0000487, 0.0000482
7: -0.0131336, -0.0111883, -0.0131606, -0.0111883, -0.0019036, 0.0019375
8: -0.0138683, -0.0117602, -0.0138927, -0.0117600, -0.0008397, 0.0010777
9: 0.0017152, 0.0027124, 0.0017152, 0.0027229, -0.0004444, 0.0003316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 45

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 189

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000928, upper bound: 0.0000926
time: 0.55 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000928, upper bound: 0.0000973
time: 0.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.41 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 3, lower bound: -0.0000801, upper bound: 0.0000906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 3, lower bound: -0.0000880, upper bound: 0.0000906
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 3, lower bound: -0.0000928, upper bound: 0.0000926
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 3, lower bound: -0.0000928, upper bound: 0.0000973

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007674, -0.0001396, 0.0008506, -0.0004366, 0.0003972
1: -0.0035044, -0.0033148, -0.0035010, -0.0033122, -0.0000674, 0.0000629
2: 0.0147868, 0.0159242, 0.0147639, 0.0160204, -0.0005332, 0.0004861
3: 1.0066483, 1.0069450, 1.0066861, 1.0069450, -0.0001591, 0.0001444
4: -0.0042430, -0.0040637, -0.0042559, -0.0040618, -0.0000719, 0.0000788
5: 0.0038926, 0.0045675, 0.0038735, 0.0046305, -0.0003320, 0.0003021
6: -0.0026078, -0.0025785, -0.0026103, -0.0025730, -0.0000278, 0.0000243
7: -0.0126992, -0.0112484, -0.0128909, -0.0111541, -0.0008099, 0.0009023
8: -0.0136635, -0.0117601, -0.0137865, -0.0117515, -0.0007323, 0.0008020
9: 0.0017133, 0.0026344, 0.0017145, 0.0026857, -0.0003708, 0.0003384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 3

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000847
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000906
time: 0.61 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007728, -0.0001286, 0.0008606, -0.0004279, 0.0004115
1: -0.0035062, -0.0033148, -0.0035046, -0.0033129, -0.0000689, 0.0000637
2: 0.0147868, 0.0159298, 0.0147745, 0.0160296, -0.0005256, 0.0004990
3: 1.0066483, 1.0069578, 1.0066805, 1.0069735, -0.0001540, 0.0001729
4: -0.0042435, -0.0040637, -0.0042568, -0.0040630, -0.0000730, 0.0000783
5: 0.0038926, 0.0045716, 0.0038816, 0.0046378, -0.0003256, 0.0003126
6: -0.0026078, -0.0025769, -0.0026096, -0.0025695, -0.0000259, 0.0000285
7: -0.0127195, -0.0112484, -0.0129358, -0.0111883, -0.0008752, 0.0008616
8: -0.0136665, -0.0117601, -0.0137914, -0.0117604, -0.0007382, 0.0008000
9: 0.0017133, 0.0026346, 0.0017152, 0.0026861, -0.0003709, 0.0003387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 3

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000880, upper bound: 0.0000847
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000880, upper bound: 0.0000906
time: 0.59 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001287, 0.0009294, -0.0001136, 0.0007769, -0.0005792, 0.0005521
1: -0.0035063, -0.0033109, -0.0035067, -0.0033148, -0.0000675, 0.0001265
2: 0.0147744, 0.0161034, 0.0147868, 0.0159336, -0.0006553, 0.0006580
3: 1.0066744, 1.0069788, 1.0066483, 1.0069599, -0.0002855, 0.0002299
4: -0.0042658, -0.0040630, -0.0042439, -0.0040637, -0.0000940, 0.0000871
5: 0.0038815, 0.0046893, 0.0038926, 0.0045746, -0.0004359, 0.0004184
6: -0.0026097, -0.0025616, -0.0026078, -0.0025764, -0.0000333, 0.0000421
7: -0.0131336, -0.0111883, -0.0127334, -0.0112484, -0.0012369, 0.0015090
8: -0.0138683, -0.0117602, -0.0136690, -0.0117601, -0.0009319, 0.0008137
9: 0.0017152, 0.0027124, 0.0017133, 0.0026348, -0.0003405, 0.0004153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 3

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000880, upper bound: 0.0000802
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000880, upper bound: 0.0000880
time: 0.56 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001287, 0.0009294, -0.0001287, 0.0009294, -0.0006800, 0.0006800
1: -0.0035063, -0.0033109, -0.0035063, -0.0033109, -0.0001539, 0.0001539
2: 0.0147744, 0.0161034, 0.0147744, 0.0161034, -0.0007451, 0.0007451
3: 1.0066744, 1.0069788, 1.0066744, 1.0069788, -0.0003043, 0.0003043
4: -0.0042658, -0.0040630, -0.0042658, -0.0040630, -0.0000937, 0.0000937
5: 0.0038815, 0.0046893, 0.0038815, 0.0046893, -0.0005100, 0.0005100
6: -0.0026097, -0.0025616, -0.0026097, -0.0025616, -0.0000481, 0.0000481
7: -0.0131336, -0.0111883, -0.0131336, -0.0111883, -0.0019036, 0.0019036
8: -0.0138683, -0.0117602, -0.0138683, -0.0117602, -0.0008395, 0.0008395
9: 0.0017152, 0.0027124, 0.0017152, 0.0027124, -0.0003315, 0.0003315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 3

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000880, upper bound: 0.0000843
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000880, upper bound: 0.0000927
time: 0.55 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.45 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000847
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000906
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 3, lower bound: -0.0000880, upper bound: 0.0000847
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 3, lower bound: -0.0000880, upper bound: 0.0000906
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 3, lower bound: -0.0000880, upper bound: 0.0000802
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 3, lower bound: -0.0000880, upper bound: 0.0000880
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 3, lower bound: -0.0000880, upper bound: 0.0000843
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 3, lower bound: -0.0000880, upper bound: 0.0000927

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001293, 0.0007551, -0.0001396, 0.0008506, -0.0004303, 0.0003787
1: -0.0035013, -0.0033142, -0.0035010, -0.0033122, -0.0000651, 0.0000629
2: 0.0147715, 0.0159117, 0.0147639, 0.0160204, -0.0005283, 0.0004686
3: 1.0066478, 1.0069275, 1.0066861, 1.0069450, -0.0001482, 0.0001237
4: -0.0042419, -0.0040621, -0.0042559, -0.0040618, -0.0000703, 0.0000786
5: 0.0038810, 0.0045585, 0.0038735, 0.0046305, -0.0003274, 0.0002885
6: -0.0026089, -0.0025810, -0.0026103, -0.0025730, -0.0000257, 0.0000207
7: -0.0126541, -0.0111921, -0.0128909, -0.0111541, -0.0007410, 0.0008683
8: -0.0136562, -0.0117497, -0.0137865, -0.0117515, -0.0007223, 0.0008021
9: 0.0017124, 0.0026338, 0.0017145, 0.0026857, -0.0003710, 0.0003377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 189

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000851
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000850
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007632, -0.0001396, 0.0008506, -0.0004366, 0.0004061
1: -0.0035050, -0.0033148, -0.0035010, -0.0033122, -0.0000681, 0.0000629
2: 0.0147868, 0.0159207, 0.0147639, 0.0160204, -0.0005332, 0.0004939
3: 1.0066483, 1.0069526, 1.0066861, 1.0069450, -0.0001591, 0.0001616
4: -0.0042426, -0.0040637, -0.0042559, -0.0040618, -0.0000725, 0.0000788
5: 0.0038926, 0.0045646, 0.0038735, 0.0046305, -0.0003320, 0.0003086
6: -0.0026078, -0.0025780, -0.0026103, -0.0025730, -0.0000278, 0.0000271
7: -0.0126866, -0.0112484, -0.0128909, -0.0111541, -0.0008524, 0.0009023
8: -0.0136606, -0.0117601, -0.0137865, -0.0117515, -0.0007351, 0.0008020
9: 0.0017133, 0.0026341, 0.0017145, 0.0026857, -0.0003708, 0.0003384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 189

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000906
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000906
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001293, 0.0007551, -0.0001286, 0.0008606, -0.0004584, 0.0003798
1: -0.0035013, -0.0033142, -0.0035046, -0.0033129, -0.0000650, 0.0000660
2: 0.0147715, 0.0159117, 0.0147745, 0.0160296, -0.0005537, 0.0004692
3: 1.0066478, 1.0069275, 1.0066805, 1.0069735, -0.0001892, 0.0001334
4: -0.0042419, -0.0040621, -0.0042568, -0.0040630, -0.0000703, 0.0000808
5: 0.0038810, 0.0045585, 0.0038816, 0.0046378, -0.0003481, 0.0002893
6: -0.0026089, -0.0025810, -0.0026096, -0.0025695, -0.0000322, 0.0000218
7: -0.0126541, -0.0111921, -0.0129358, -0.0111883, -0.0007529, 0.0009835
8: -0.0136562, -0.0117497, -0.0137914, -0.0117604, -0.0007218, 0.0008157
9: 0.0017124, 0.0026338, 0.0017152, 0.0026861, -0.0003720, 0.0003376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 189

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000847
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000847
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007632, -0.0001286, 0.0008606, -0.0004279, 0.0003763
1: -0.0035050, -0.0033148, -0.0035046, -0.0033129, -0.0000659, 0.0000637
2: 0.0147868, 0.0159207, 0.0147745, 0.0160296, -0.0005256, 0.0004658
3: 1.0066483, 1.0069526, 1.0066805, 1.0069735, -0.0001540, 0.0001294
4: -0.0042426, -0.0040637, -0.0042568, -0.0040630, -0.0000700, 0.0000783
5: 0.0038926, 0.0045646, 0.0038816, 0.0046378, -0.0003256, 0.0002867
6: -0.0026078, -0.0025780, -0.0026096, -0.0025695, -0.0000259, 0.0000209
7: -0.0126866, -0.0112484, -0.0129358, -0.0111883, -0.0007342, 0.0008616
8: -0.0136606, -0.0117601, -0.0137914, -0.0117604, -0.0007202, 0.0008000
9: 0.0017133, 0.0026341, 0.0017152, 0.0026861, -0.0003709, 0.0003375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 189

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000906
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000906
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001396, 0.0009033, -0.0001136, 0.0007674, -0.0003972, 0.0005204
1: -0.0035010, -0.0033114, -0.0035044, -0.0033148, -0.0000629, 0.0000684
2: 0.0147639, 0.0160779, 0.0147868, 0.0159242, -0.0004861, 0.0006273
3: 1.0066861, 1.0069412, 1.0066483, 1.0069450, -0.0001444, 0.0001907
4: -0.0042635, -0.0040618, -0.0042430, -0.0040637, -0.0000910, 0.0000719
5: 0.0038735, 0.0046699, 0.0038926, 0.0045675, -0.0003021, 0.0003950
6: -0.0026103, -0.0025667, -0.0026078, -0.0025785, -0.0000243, 0.0000360
7: -0.0130413, -0.0111541, -0.0126992, -0.0112484, -0.0011194, 0.0008099
8: -0.0138544, -0.0117515, -0.0136635, -0.0117601, -0.0009135, 0.0007323
9: 0.0017145, 0.0027113, 0.0017133, 0.0026344, -0.0003384, 0.0004139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000847, upper bound: 0.0000802
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000847, upper bound: 0.0000802
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001286, 0.0009178, -0.0001136, 0.0007728, -0.0004115, 0.0005117
1: -0.0035046, -0.0033123, -0.0035062, -0.0033148, -0.0000637, 0.0000697
2: 0.0147745, 0.0160921, 0.0147868, 0.0159298, -0.0004990, 0.0006196
3: 1.0066805, 1.0069728, 1.0066483, 1.0069578, -0.0001729, 0.0001856
4: -0.0042646, -0.0040630, -0.0042435, -0.0040637, -0.0000905, 0.0000730
5: 0.0038816, 0.0046807, 0.0038926, 0.0045716, -0.0003126, 0.0003887
6: -0.0026096, -0.0025631, -0.0026078, -0.0025769, -0.0000285, 0.0000341
7: -0.0130946, -0.0111883, -0.0127195, -0.0112484, -0.0010786, 0.0008753
8: -0.0138606, -0.0117604, -0.0136665, -0.0117601, -0.0009114, 0.0007382
9: 0.0017152, 0.0027118, 0.0017133, 0.0026346, -0.0003387, 0.0004140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000847, upper bound: 0.0000880
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000847, upper bound: 0.0000880
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001396, 0.0009033, -0.0001286, 0.0009182, -0.0003816, 0.0003695
1: -0.0035010, -0.0033114, -0.0035040, -0.0033123, -0.0000635, 0.0000659
2: 0.0147639, 0.0160779, 0.0147745, 0.0160924, -0.0004668, 0.0004543
3: 1.0066861, 1.0069412, 1.0066805, 1.0069624, -0.0001348, 0.0001250
4: -0.0042635, -0.0040618, -0.0042648, -0.0040630, -0.0000678, 0.0000693
5: 0.0038735, 0.0046699, 0.0038816, 0.0046810, -0.0002902, 0.0002812
6: -0.0026103, -0.0025667, -0.0026096, -0.0025639, -0.0000237, 0.0000222
7: -0.0130413, -0.0111541, -0.0130930, -0.0111883, -0.0007457, 0.0007806
8: -0.0138544, -0.0117515, -0.0138623, -0.0117604, -0.0006975, 0.0007076
9: 0.0017145, 0.0027113, 0.0017152, 0.0027119, -0.0003293, 0.0003284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000845, upper bound: 0.0000843
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000845, upper bound: 0.0000843
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0001286, 0.0009178, -0.0001287, 0.0009260, -0.0004012, 0.0006687
1: -0.0035046, -0.0033123, -0.0035058, -0.0033114, -0.0001517, 0.0000675
2: 0.0147745, 0.0160921, 0.0147744, 0.0161001, -0.0004841, 0.0007340
3: 1.0066805, 1.0069728, 1.0066766, 1.0069770, -0.0001645, 0.0002962
4: -0.0042646, -0.0040630, -0.0042654, -0.0040630, -0.0000926, 0.0000706
5: 0.0038816, 0.0046807, 0.0038816, 0.0046868, -0.0003045, 0.0005016
6: -0.0026096, -0.0025631, -0.0026097, -0.0025621, -0.0000289, 0.0000465
7: -0.0130946, -0.0111883, -0.0131220, -0.0111883, -0.0018647, 0.0008681
8: -0.0138606, -0.0117604, -0.0138660, -0.0117603, -0.0008322, 0.0007139
9: 0.0017152, 0.0027118, 0.0017152, 0.0027122, -0.0003295, 0.0003306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000845, upper bound: 0.0000924
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000845, upper bound: 0.0000926
time: 0.60 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.50 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000851
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000850
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000906
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000906
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000847
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000847
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000906
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000802, upper bound: 0.0000906
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000847, upper bound: 0.0000802
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000847, upper bound: 0.0000802
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000847, upper bound: 0.0000880
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000847, upper bound: 0.0000880
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000845, upper bound: 0.0000843
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000845, upper bound: 0.0000843
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000845, upper bound: 0.0000924
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 3, lower bound: -0.0000845, upper bound: 0.0000926

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001293, 0.0007551, -0.0001293, 0.0007551, -0.0003315, 0.0003315
1: -0.0035013, -0.0033142, -0.0035013, -0.0033142, -0.0000631, 0.0000631
2: 0.0147715, 0.0159117, 0.0147715, 0.0159117, -0.0004150, 0.0004150
3: 1.0066478, 1.0069275, 1.0066478, 1.0069275, -0.0001172, 0.0001172
4: -0.0042419, -0.0040621, -0.0042419, -0.0040621, -0.0000634, 0.0000634
5: 0.0038810, 0.0045585, 0.0038810, 0.0045585, -0.0002529, 0.0002529
6: -0.0026089, -0.0025810, -0.0026089, -0.0025810, -0.0000169, 0.0000169
7: -0.0126541, -0.0111921, -0.0126541, -0.0111921, -0.0006225, 0.0006225
8: -0.0136562, -0.0117497, -0.0136562, -0.0117497, -0.0006601, 0.0006601
9: 0.0017124, 0.0026338, 0.0017124, 0.0026338, -0.0003145, 0.0003145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000696, upper bound: 0.0000673
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000672
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001293, 0.0007551, -0.0001396, 0.0009033, -0.0005141, 0.0003787
1: -0.0035013, -0.0033142, -0.0035010, -0.0033114, -0.0000661, 0.0000629
2: 0.0147715, 0.0159117, 0.0147639, 0.0160779, -0.0006224, 0.0004686
3: 1.0066478, 1.0069275, 1.0066861, 1.0069412, -0.0001798, 0.0001237
4: -0.0042419, -0.0040621, -0.0042635, -0.0040618, -0.0000703, 0.0000908
5: 0.0038810, 0.0045585, 0.0038735, 0.0046699, -0.0003904, 0.0002885
6: -0.0026089, -0.0025810, -0.0026103, -0.0025667, -0.0000339, 0.0000207
7: -0.0126541, -0.0111921, -0.0130413, -0.0111541, -0.0007410, 0.0010853
8: -0.0136562, -0.0117497, -0.0138544, -0.0117515, -0.0007223, 0.0009135
9: 0.0017124, 0.0026338, 0.0017145, 0.0027113, -0.0004142, 0.0003377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000696, upper bound: 0.0000673
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000672
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007632, -0.0001293, 0.0007551, -0.0003378, 0.0003589
1: -0.0035050, -0.0033148, -0.0035013, -0.0033142, -0.0000661, 0.0000631
2: 0.0147868, 0.0159207, 0.0147715, 0.0159117, -0.0004199, 0.0004404
3: 1.0066483, 1.0069526, 1.0066478, 1.0069275, -0.0001281, 0.0001551
4: -0.0042426, -0.0040637, -0.0042419, -0.0040621, -0.0000657, 0.0000636
5: 0.0038926, 0.0045646, 0.0038810, 0.0045585, -0.0002575, 0.0002730
6: -0.0026078, -0.0025780, -0.0026089, -0.0025810, -0.0000189, 0.0000232
7: -0.0126866, -0.0112484, -0.0126541, -0.0111921, -0.0007339, 0.0006565
8: -0.0136606, -0.0117601, -0.0136562, -0.0117497, -0.0006729, 0.0006601
9: 0.0017133, 0.0026341, 0.0017124, 0.0026338, -0.0003143, 0.0003153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000467
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000763, upper bound: 0.0000878
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007632, -0.0001396, 0.0009033, -0.0005204, 0.0004061
1: -0.0035050, -0.0033148, -0.0035010, -0.0033114, -0.0000691, 0.0000629
2: 0.0147868, 0.0159207, 0.0147639, 0.0160779, -0.0006273, 0.0004939
3: 1.0066483, 1.0069526, 1.0066861, 1.0069412, -0.0001907, 0.0001616
4: -0.0042426, -0.0040637, -0.0042635, -0.0040618, -0.0000725, 0.0000910
5: 0.0038926, 0.0045646, 0.0038735, 0.0046699, -0.0003950, 0.0003086
6: -0.0026078, -0.0025780, -0.0026103, -0.0025667, -0.0000360, 0.0000271
7: -0.0126866, -0.0112484, -0.0130413, -0.0111541, -0.0008524, 0.0011194
8: -0.0136606, -0.0117601, -0.0138544, -0.0117515, -0.0007351, 0.0009135
9: 0.0017133, 0.0026341, 0.0017145, 0.0027113, -0.0004139, 0.0003384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000495
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000763, upper bound: 0.0000878
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001293, 0.0007551, -0.0001136, 0.0007632, -0.0003589, 0.0003378
1: -0.0035013, -0.0033142, -0.0035050, -0.0033148, -0.0000631, 0.0000661
2: 0.0147715, 0.0159117, 0.0147868, 0.0159207, -0.0004404, 0.0004199
3: 1.0066478, 1.0069275, 1.0066483, 1.0069526, -0.0001551, 0.0001281
4: -0.0042419, -0.0040621, -0.0042426, -0.0040637, -0.0000636, 0.0000657
5: 0.0038810, 0.0045585, 0.0038926, 0.0045646, -0.0002730, 0.0002575
6: -0.0026089, -0.0025810, -0.0026078, -0.0025780, -0.0000232, 0.0000189
7: -0.0126541, -0.0111921, -0.0126866, -0.0112484, -0.0006565, 0.0007339
8: -0.0136562, -0.0117497, -0.0136606, -0.0117601, -0.0006601, 0.0006729
9: 0.0017124, 0.0026338, 0.0017133, 0.0026341, -0.0003153, 0.0003143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000754, upper bound: 0.0000659
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000735, upper bound: 0.0000659
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001293, 0.0007551, -0.0001286, 0.0009178, -0.0005381, 0.0003798
1: -0.0035013, -0.0033142, -0.0035046, -0.0033123, -0.0000657, 0.0000660
2: 0.0147715, 0.0159117, 0.0147745, 0.0160921, -0.0006451, 0.0004692
3: 1.0066478, 1.0069275, 1.0066805, 1.0069728, -0.0002153, 0.0001334
4: -0.0042419, -0.0040621, -0.0042646, -0.0040630, -0.0000703, 0.0000930
5: 0.0038810, 0.0045585, 0.0038816, 0.0046807, -0.0004082, 0.0002893
6: -0.0026089, -0.0025810, -0.0026096, -0.0025631, -0.0000392, 0.0000218
7: -0.0126541, -0.0111921, -0.0130946, -0.0111883, -0.0007529, 0.0011816
8: -0.0136562, -0.0117497, -0.0138606, -0.0117604, -0.0007218, 0.0009268
9: 0.0017124, 0.0026338, 0.0017152, 0.0027118, -0.0004151, 0.0003376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000754, upper bound: 0.0000659
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000735, upper bound: 0.0000659
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007632, -0.0001136, 0.0007632, -0.0003291, 0.0003291
1: -0.0035050, -0.0033148, -0.0035050, -0.0033148, -0.0000640, 0.0000640
2: 0.0147868, 0.0159207, 0.0147868, 0.0159207, -0.0004123, 0.0004123
3: 1.0066483, 1.0069526, 1.0066483, 1.0069526, -0.0001230, 0.0001230
4: -0.0042426, -0.0040637, -0.0042426, -0.0040637, -0.0000631, 0.0000631
5: 0.0038926, 0.0045646, 0.0038926, 0.0045646, -0.0002511, 0.0002511
6: -0.0026078, -0.0025780, -0.0026078, -0.0025780, -0.0000170, 0.0000170
7: -0.0126866, -0.0112484, -0.0126866, -0.0112484, -0.0006157, 0.0006157
8: -0.0136606, -0.0117601, -0.0136606, -0.0117601, -0.0006580, 0.0006580
9: 0.0017133, 0.0026341, 0.0017133, 0.0026341, -0.0003144, 0.0003144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000665, upper bound: 0.0000508
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000765, upper bound: 0.0000878
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007632, -0.0001286, 0.0009178, -0.0005117, 0.0003763
1: -0.0035050, -0.0033148, -0.0035046, -0.0033123, -0.0000669, 0.0000637
2: 0.0147868, 0.0159207, 0.0147745, 0.0160921, -0.0006196, 0.0004658
3: 1.0066483, 1.0069526, 1.0066805, 1.0069728, -0.0001856, 0.0001294
4: -0.0042426, -0.0040637, -0.0042646, -0.0040630, -0.0000700, 0.0000905
5: 0.0038926, 0.0045646, 0.0038816, 0.0046807, -0.0003887, 0.0002867
6: -0.0026078, -0.0025780, -0.0026096, -0.0025631, -0.0000341, 0.0000209
7: -0.0126866, -0.0112484, -0.0130946, -0.0111883, -0.0007342, 0.0010786
8: -0.0136606, -0.0117601, -0.0138606, -0.0117604, -0.0007202, 0.0009114
9: 0.0017133, 0.0026341, 0.0017152, 0.0027118, -0.0004140, 0.0003375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000665, upper bound: 0.0000539
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000765, upper bound: 0.0000878
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001396, 0.0009033, -0.0001293, 0.0007551, -0.0003787, 0.0005141
1: -0.0035010, -0.0033114, -0.0035013, -0.0033142, -0.0000629, 0.0000661
2: 0.0147639, 0.0160779, 0.0147715, 0.0159117, -0.0004686, 0.0006224
3: 1.0066861, 1.0069412, 1.0066478, 1.0069275, -0.0001237, 0.0001798
4: -0.0042635, -0.0040618, -0.0042419, -0.0040621, -0.0000908, 0.0000703
5: 0.0038735, 0.0046699, 0.0038810, 0.0045585, -0.0002885, 0.0003904
6: -0.0026103, -0.0025667, -0.0026089, -0.0025810, -0.0000207, 0.0000339
7: -0.0130413, -0.0111541, -0.0126541, -0.0111921, -0.0010853, 0.0007410
8: -0.0138544, -0.0117515, -0.0136562, -0.0117497, -0.0009135, 0.0007223
9: 0.0017145, 0.0027113, 0.0017124, 0.0026338, -0.0003377, 0.0004142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000704, upper bound: 0.0000514
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000827, upper bound: 0.0000763
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001396, 0.0009033, -0.0001136, 0.0007632, -0.0004061, 0.0005204
1: -0.0035010, -0.0033114, -0.0035050, -0.0033148, -0.0000629, 0.0000691
2: 0.0147639, 0.0160779, 0.0147868, 0.0159207, -0.0004939, 0.0006273
3: 1.0066861, 1.0069412, 1.0066483, 1.0069526, -0.0001616, 0.0001907
4: -0.0042635, -0.0040618, -0.0042426, -0.0040637, -0.0000910, 0.0000725
5: 0.0038735, 0.0046699, 0.0038926, 0.0045646, -0.0003086, 0.0003950
6: -0.0026103, -0.0025667, -0.0026078, -0.0025780, -0.0000271, 0.0000360
7: -0.0130413, -0.0111541, -0.0126866, -0.0112484, -0.0011194, 0.0008524
8: -0.0138544, -0.0117515, -0.0136606, -0.0117601, -0.0009135, 0.0007351
9: 0.0017145, 0.0027113, 0.0017133, 0.0026341, -0.0003384, 0.0004139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000704, upper bound: 0.0000514
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000827, upper bound: 0.0000763
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001286, 0.0009178, -0.0001293, 0.0007551, -0.0003798, 0.0005381
1: -0.0035046, -0.0033123, -0.0035013, -0.0033142, -0.0000660, 0.0000657
2: 0.0147745, 0.0160921, 0.0147715, 0.0159117, -0.0004692, 0.0006451
3: 1.0066805, 1.0069728, 1.0066478, 1.0069275, -0.0001334, 0.0002153
4: -0.0042646, -0.0040630, -0.0042419, -0.0040621, -0.0000930, 0.0000703
5: 0.0038816, 0.0046807, 0.0038810, 0.0045585, -0.0002893, 0.0004082
6: -0.0026096, -0.0025631, -0.0026089, -0.0025810, -0.0000218, 0.0000392
7: -0.0130946, -0.0111883, -0.0126541, -0.0111921, -0.0011816, 0.0007529
8: -0.0138606, -0.0117604, -0.0136562, -0.0117497, -0.0009268, 0.0007218
9: 0.0017152, 0.0027118, 0.0017124, 0.0026338, -0.0003376, 0.0004151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000697, upper bound: 0.0000646
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000820, upper bound: 0.0000845
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001286, 0.0009178, -0.0001136, 0.0007632, -0.0003763, 0.0005117
1: -0.0035046, -0.0033123, -0.0035050, -0.0033148, -0.0000637, 0.0000669
2: 0.0147745, 0.0160921, 0.0147868, 0.0159207, -0.0004658, 0.0006196
3: 1.0066805, 1.0069728, 1.0066483, 1.0069526, -0.0001294, 0.0001856
4: -0.0042646, -0.0040630, -0.0042426, -0.0040637, -0.0000905, 0.0000700
5: 0.0038816, 0.0046807, 0.0038926, 0.0045646, -0.0002867, 0.0003887
6: -0.0026096, -0.0025631, -0.0026078, -0.0025780, -0.0000209, 0.0000341
7: -0.0130946, -0.0111883, -0.0126866, -0.0112484, -0.0010786, 0.0007342
8: -0.0138606, -0.0117604, -0.0136606, -0.0117601, -0.0009114, 0.0007202
9: 0.0017152, 0.0027118, 0.0017133, 0.0026341, -0.0003375, 0.0004140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000697, upper bound: 0.0000667
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000820, upper bound: 0.0000845
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001396, 0.0009033, -0.0001396, 0.0009033, -0.0003632, 0.0003632
1: -0.0035010, -0.0033114, -0.0035010, -0.0033114, -0.0000636, 0.0000636
2: 0.0147639, 0.0160779, 0.0147639, 0.0160779, -0.0004493, 0.0004493
3: 1.0066861, 1.0069412, 1.0066861, 1.0069412, -0.0001141, 0.0001141
4: -0.0042635, -0.0040618, -0.0042635, -0.0040618, -0.0000676, 0.0000676
5: 0.0038735, 0.0046699, 0.0038735, 0.0046699, -0.0002766, 0.0002766
6: -0.0026103, -0.0025667, -0.0026103, -0.0025667, -0.0000201, 0.0000201
7: -0.0130413, -0.0111541, -0.0130413, -0.0111541, -0.0007117, 0.0007117
8: -0.0138544, -0.0117515, -0.0138544, -0.0117515, -0.0006976, 0.0006976
9: 0.0017145, 0.0027113, 0.0017145, 0.0027113, -0.0003286, 0.0003286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000813, upper bound: 0.0000720
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000821, upper bound: 0.0000805
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001396, 0.0009033, -0.0001286, 0.0009178, -0.0003905, 0.0003695
1: -0.0035010, -0.0033114, -0.0035046, -0.0033123, -0.0000635, 0.0000666
2: 0.0147639, 0.0160779, 0.0147745, 0.0160921, -0.0004747, 0.0004543
3: 1.0066861, 1.0069412, 1.0066805, 1.0069728, -0.0001520, 0.0001250
4: -0.0042635, -0.0040618, -0.0042646, -0.0040630, -0.0000678, 0.0000699
5: 0.0038735, 0.0046699, 0.0038816, 0.0046807, -0.0002967, 0.0002812
6: -0.0026103, -0.0025667, -0.0026096, -0.0025631, -0.0000265, 0.0000222
7: -0.0130413, -0.0111541, -0.0130946, -0.0111883, -0.0007457, 0.0008231
8: -0.0138544, -0.0117515, -0.0138606, -0.0117604, -0.0006975, 0.0007104
9: 0.0017145, 0.0027113, 0.0017152, 0.0027118, -0.0003293, 0.0003284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000813, upper bound: 0.0000720
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000821, upper bound: 0.0000805
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001286, 0.0009178, -0.0001396, 0.0009033, -0.0003695, 0.0003905
1: -0.0035046, -0.0033123, -0.0035010, -0.0033114, -0.0000666, 0.0000635
2: 0.0147745, 0.0160921, 0.0147639, 0.0160779, -0.0004543, 0.0004747
3: 1.0066805, 1.0069728, 1.0066861, 1.0069412, -0.0001250, 0.0001520
4: -0.0042646, -0.0040630, -0.0042635, -0.0040618, -0.0000699, 0.0000678
5: 0.0038816, 0.0046807, 0.0038735, 0.0046699, -0.0002812, 0.0002967
6: -0.0026096, -0.0025631, -0.0026103, -0.0025667, -0.0000222, 0.0000265
7: -0.0130946, -0.0111883, -0.0130413, -0.0111541, -0.0008231, 0.0007457
8: -0.0138606, -0.0117604, -0.0138544, -0.0117515, -0.0007104, 0.0006975
9: 0.0017152, 0.0027118, 0.0017145, 0.0027113, -0.0003284, 0.0003293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000801, upper bound: 0.0000814
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000807, upper bound: 0.0000889
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001286, 0.0009178, -0.0001286, 0.0009178, -0.0003625, 0.0003625
1: -0.0035046, -0.0033123, -0.0035046, -0.0033123, -0.0000644, 0.0000644
2: 0.0147745, 0.0160921, 0.0147745, 0.0160921, -0.0004485, 0.0004485
3: 1.0066805, 1.0069728, 1.0066805, 1.0069728, -0.0001212, 0.0001212
4: -0.0042646, -0.0040630, -0.0042646, -0.0040630, -0.0000675, 0.0000675
5: 0.0038816, 0.0046807, 0.0038816, 0.0046807, -0.0002761, 0.0002761
6: -0.0026096, -0.0025631, -0.0026096, -0.0025631, -0.0000205, 0.0000205
7: -0.0130946, -0.0111883, -0.0130946, -0.0111883, -0.0007125, 0.0007125
8: -0.0138606, -0.0117604, -0.0138606, -0.0117604, -0.0006963, 0.0006963
9: 0.0017152, 0.0027118, 0.0017152, 0.0027118, -0.0003285, 0.0003285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000801, upper bound: 0.0000819
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000807, upper bound: 0.0000892
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.71 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000696, upper bound: 0.0000673
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000672
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000696, upper bound: 0.0000673
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000672
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000467
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000763, upper bound: 0.0000878
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000495
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000763, upper bound: 0.0000878
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000754, upper bound: 0.0000659
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000735, upper bound: 0.0000659
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000754, upper bound: 0.0000659
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000735, upper bound: 0.0000659
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000665, upper bound: 0.0000508
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000765, upper bound: 0.0000878
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000665, upper bound: 0.0000539
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000765, upper bound: 0.0000878
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000704, upper bound: 0.0000514
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000827, upper bound: 0.0000763
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000704, upper bound: 0.0000514
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000827, upper bound: 0.0000763
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000697, upper bound: 0.0000646
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000820, upper bound: 0.0000845
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000697, upper bound: 0.0000667
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000820, upper bound: 0.0000845
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000813, upper bound: 0.0000720
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000821, upper bound: 0.0000805
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000813, upper bound: 0.0000720
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000821, upper bound: 0.0000805
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000801, upper bound: 0.0000814
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000807, upper bound: 0.0000889
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000801, upper bound: 0.0000819
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 3, lower bound: -0.0000807, upper bound: 0.0000892

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001274, 0.0007551, -0.0001293, 0.0007551, -0.0003288, 0.0003315
1: -0.0035013, -0.0033151, -0.0035013, -0.0033142, -0.0000631, 0.0000628
2: 0.0147726, 0.0159117, 0.0147715, 0.0159117, -0.0004135, 0.0004150
3: 1.0066508, 1.0069275, 1.0066478, 1.0069275, -0.0001133, 0.0001172
4: -0.0042418, -0.0040621, -0.0042419, -0.0040621, -0.0000634, 0.0000634
5: 0.0038823, 0.0045585, 0.0038810, 0.0045585, -0.0002510, 0.0002529
6: -0.0026084, -0.0025810, -0.0026089, -0.0025810, -0.0000161, 0.0000169
7: -0.0126541, -0.0112027, -0.0126541, -0.0111921, -0.0006225, 0.0006077
8: -0.0136545, -0.0117497, -0.0136562, -0.0117497, -0.0006595, 0.0006601
9: 0.0017124, 0.0026322, 0.0017124, 0.0026338, -0.0003145, 0.0003140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000672, upper bound: 0.0000672
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000672, upper bound: 0.0000672
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001144, 0.0007745, -0.0001258, 0.0007551, -0.0003290, 0.0003623
1: -0.0035104, -0.0033273, -0.0035013, -0.0033174, -0.0000687, 0.0000626
2: 0.0147801, 0.0159233, 0.0147736, 0.0159117, -0.0004136, 0.0004323
3: 1.0066979, 1.0069686, 1.0066618, 1.0069275, -0.0001132, 0.0001841
4: -0.0042410, -0.0040616, -0.0042417, -0.0040621, -0.0000634, 0.0000632
5: 0.0038914, 0.0045722, 0.0038835, 0.0045585, -0.0002512, 0.0002746
6: -0.0026026, -0.0025743, -0.0026072, -0.0025810, -0.0000159, 0.0000280
7: -0.0127699, -0.0112848, -0.0126541, -0.0112149, -0.0008075, 0.0006095
8: -0.0136279, -0.0117318, -0.0136495, -0.0117497, -0.0006594, 0.0006561
9: 0.0016961, 0.0026093, 0.0017124, 0.0026280, -0.0003128, 0.0003139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000627, upper bound: 0.0000608
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000609, upper bound: 0.0000609
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001274, 0.0007551, -0.0001396, 0.0009033, -0.0005113, 0.0003787
1: -0.0035013, -0.0033151, -0.0035010, -0.0033114, -0.0000661, 0.0000625
2: 0.0147726, 0.0159117, 0.0147639, 0.0160779, -0.0006208, 0.0004686
3: 1.0066508, 1.0069275, 1.0066861, 1.0069412, -0.0001759, 0.0001237
4: -0.0042418, -0.0040621, -0.0042635, -0.0040618, -0.0000703, 0.0000908
5: 0.0038823, 0.0045585, 0.0038735, 0.0046699, -0.0003885, 0.0002885
6: -0.0026084, -0.0025810, -0.0026103, -0.0025667, -0.0000332, 0.0000207
7: -0.0126541, -0.0112027, -0.0130413, -0.0111541, -0.0007410, 0.0010705
8: -0.0136545, -0.0117497, -0.0138544, -0.0117515, -0.0007217, 0.0009135
9: 0.0017124, 0.0026322, 0.0017145, 0.0027113, -0.0004142, 0.0003371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000660, upper bound: 0.0000672
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000660, upper bound: 0.0000672
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0001144, 0.0007745, -0.0001361, 0.0009033, -0.0005116, 0.0004085
1: -0.0035104, -0.0033273, -0.0035010, -0.0033148, -0.0000717, 0.0000624
2: 0.0147801, 0.0159233, 0.0147659, 0.0160779, -0.0006210, 0.0004853
3: 1.0066979, 1.0069686, 1.0066977, 1.0069412, -0.0001758, 0.0001920
4: -0.0042410, -0.0040616, -0.0042633, -0.0040618, -0.0000703, 0.0000907
5: 0.0038914, 0.0045722, 0.0038760, 0.0046699, -0.0003887, 0.0003095
6: -0.0026026, -0.0025743, -0.0026090, -0.0025667, -0.0000329, 0.0000317
7: -0.0127699, -0.0112848, -0.0130413, -0.0111755, -0.0009208, 0.0010723
8: -0.0136279, -0.0117318, -0.0138474, -0.0117515, -0.0007216, 0.0009106
9: 0.0016961, 0.0026093, 0.0017145, 0.0027050, -0.0004130, 0.0003370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000617, upper bound: 0.0000608
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000594, upper bound: 0.0000609
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007570, -0.0001293, 0.0007551, -0.0003378, 0.0003624
1: -0.0035046, -0.0033148, -0.0035013, -0.0033142, -0.0000600, 0.0000631
2: 0.0147868, 0.0159151, 0.0147715, 0.0159117, -0.0004199, 0.0004443
3: 1.0066483, 1.0069501, 1.0066478, 1.0069275, -0.0001281, 0.0001452
4: -0.0042422, -0.0040637, -0.0042419, -0.0040621, -0.0000660, 0.0000636
5: 0.0038926, 0.0045600, 0.0038810, 0.0045585, -0.0002575, 0.0002757
6: -0.0026078, -0.0025788, -0.0026089, -0.0025810, -0.0000189, 0.0000229
7: -0.0126628, -0.0112484, -0.0126541, -0.0111921, -0.0007387, 0.0006565
8: -0.0136596, -0.0117601, -0.0136562, -0.0117497, -0.0006740, 0.0006601
9: 0.0017136, 0.0026341, 0.0017124, 0.0026338, -0.0003101, 0.0003153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000631, upper bound: 0.0000762
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000631, upper bound: 0.0000724
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007570, -0.0001396, 0.0009033, -0.0005204, 0.0004131
1: -0.0035046, -0.0033148, -0.0035010, -0.0033114, -0.0000627, 0.0000629
2: 0.0147868, 0.0159151, 0.0147639, 0.0160779, -0.0006273, 0.0005004
3: 1.0066483, 1.0069501, 1.0066861, 1.0069412, -0.0001907, 0.0001698
4: -0.0042422, -0.0040637, -0.0042635, -0.0040618, -0.0000731, 0.0000910
5: 0.0038926, 0.0045600, 0.0038735, 0.0046699, -0.0003950, 0.0003138
6: -0.0026078, -0.0025788, -0.0026103, -0.0025667, -0.0000360, 0.0000283
7: -0.0126628, -0.0112484, -0.0130413, -0.0111541, -0.0008759, 0.0011194
8: -0.0136596, -0.0117601, -0.0138544, -0.0117515, -0.0007367, 0.0009135
9: 0.0017136, 0.0026341, 0.0017145, 0.0027113, -0.0004095, 0.0003384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000771
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000878
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001274, 0.0007551, -0.0001136, 0.0007632, -0.0003561, 0.0003378
1: -0.0035013, -0.0033151, -0.0035050, -0.0033148, -0.0000631, 0.0000658
2: 0.0147726, 0.0159117, 0.0147868, 0.0159207, -0.0004389, 0.0004199
3: 1.0066508, 1.0069275, 1.0066483, 1.0069526, -0.0001512, 0.0001281
4: -0.0042418, -0.0040621, -0.0042426, -0.0040637, -0.0000636, 0.0000657
5: 0.0038823, 0.0045585, 0.0038926, 0.0045646, -0.0002711, 0.0002575
6: -0.0026084, -0.0025810, -0.0026078, -0.0025780, -0.0000225, 0.0000189
7: -0.0126541, -0.0112027, -0.0126866, -0.0112484, -0.0006565, 0.0007191
8: -0.0136545, -0.0117497, -0.0136606, -0.0117601, -0.0006594, 0.0006729
9: 0.0017124, 0.0026322, 0.0017133, 0.0026341, -0.0003153, 0.0003137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000659
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000659
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001144, 0.0007745, -0.0001101, 0.0007632, -0.0003564, 0.0003687
1: -0.0035104, -0.0033273, -0.0035050, -0.0033179, -0.0000693, 0.0000656
2: 0.0147801, 0.0159233, 0.0147888, 0.0159207, -0.0004390, 0.0004373
3: 1.0066979, 1.0069686, 1.0066603, 1.0069526, -0.0001511, 0.0001971
4: -0.0042410, -0.0040616, -0.0042424, -0.0040637, -0.0000635, 0.0000655
5: 0.0038914, 0.0045722, 0.0038950, 0.0045646, -0.0002713, 0.0002793
6: -0.0026026, -0.0025743, -0.0026062, -0.0025780, -0.0000223, 0.0000301
7: -0.0127699, -0.0112848, -0.0126866, -0.0112709, -0.0008404, 0.0007209
8: -0.0136279, -0.0117318, -0.0136540, -0.0117601, -0.0006593, 0.0006684
9: 0.0016961, 0.0026093, 0.0017133, 0.0026283, -0.0003134, 0.0003136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000703, upper bound: 0.0000596
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000638, upper bound: 0.0000596
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001274, 0.0007551, -0.0001286, 0.0009178, -0.0005354, 0.0003798
1: -0.0035013, -0.0033151, -0.0035046, -0.0033123, -0.0000657, 0.0000656
2: 0.0147726, 0.0159117, 0.0147745, 0.0160921, -0.0006435, 0.0004692
3: 1.0066508, 1.0069275, 1.0066805, 1.0069728, -0.0002114, 0.0001334
4: -0.0042418, -0.0040621, -0.0042646, -0.0040630, -0.0000703, 0.0000930
5: 0.0038823, 0.0045585, 0.0038816, 0.0046807, -0.0004062, 0.0002893
6: -0.0026084, -0.0025810, -0.0026096, -0.0025631, -0.0000385, 0.0000218
7: -0.0126541, -0.0112027, -0.0130946, -0.0111883, -0.0007529, 0.0011668
8: -0.0136545, -0.0117497, -0.0138606, -0.0117604, -0.0007212, 0.0009268
9: 0.0017124, 0.0026322, 0.0017152, 0.0027118, -0.0004151, 0.0003370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000738, upper bound: 0.0000659
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000738, upper bound: 0.0000659
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0001144, 0.0007745, -0.0001253, 0.0009178, -0.0005357, 0.0004097
1: -0.0035104, -0.0033273, -0.0035046, -0.0033153, -0.0000716, 0.0000654
2: 0.0147801, 0.0159233, 0.0147764, 0.0160921, -0.0006437, 0.0004859
3: 1.0066979, 1.0069686, 1.0066906, 1.0069728, -0.0002112, 0.0002025
4: -0.0042410, -0.0040616, -0.0042644, -0.0040630, -0.0000703, 0.0000928
5: 0.0038914, 0.0045722, 0.0038839, 0.0046807, -0.0004064, 0.0003103
6: -0.0026026, -0.0025743, -0.0026083, -0.0025631, -0.0000383, 0.0000328
7: -0.0127699, -0.0112848, -0.0130946, -0.0112094, -0.0009324, 0.0011686
8: -0.0136279, -0.0117318, -0.0138536, -0.0117604, -0.0007211, 0.0009238
9: 0.0016961, 0.0026093, 0.0017152, 0.0027055, -0.0004139, 0.0003369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000596
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000619, upper bound: 0.0000596
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001397, 0.0007565, -0.0001136, 0.0007608, -0.0003440, 0.0003176
1: -0.0034971, -0.0033147, -0.0035022, -0.0033148, -0.0000558, 0.0000607
2: 0.0147638, 0.0159147, 0.0147868, 0.0159185, -0.0004262, 0.0004024
3: 1.0066440, 1.0069162, 1.0066483, 1.0069393, -0.0001152, 0.0000906
4: -0.0042422, -0.0040620, -0.0042425, -0.0040637, -0.0000624, 0.0000641
5: 0.0038735, 0.0045597, 0.0038926, 0.0045628, -0.0002621, 0.0002427
6: -0.0026117, -0.0025807, -0.0026078, -0.0025791, -0.0000186, 0.0000139
7: -0.0126565, -0.0111468, -0.0126754, -0.0112484, -0.0005648, 0.0006666
8: -0.0136596, -0.0117559, -0.0136603, -0.0117601, -0.0006564, 0.0006608
9: 0.0017196, 0.0026342, 0.0017155, 0.0026341, -0.0003085, 0.0003128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000429, upper bound: 0.0000467
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000429, upper bound: 0.0000508
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007570, -0.0001136, 0.0007632, -0.0003291, 0.0003324
1: -0.0035046, -0.0033148, -0.0035050, -0.0033148, -0.0000570, 0.0000640
2: 0.0147868, 0.0159151, 0.0147868, 0.0159207, -0.0004123, 0.0004161
3: 1.0066483, 1.0069501, 1.0066483, 1.0069526, -0.0001230, 0.0001050
4: -0.0042422, -0.0040637, -0.0042426, -0.0040637, -0.0000634, 0.0000631
5: 0.0038926, 0.0045600, 0.0038926, 0.0045646, -0.0002511, 0.0002536
6: -0.0026078, -0.0025788, -0.0026078, -0.0025780, -0.0000170, 0.0000168
7: -0.0126628, -0.0112484, -0.0126866, -0.0112484, -0.0006212, 0.0006157
8: -0.0136596, -0.0117601, -0.0136606, -0.0117601, -0.0006590, 0.0006580
9: 0.0017136, 0.0026341, 0.0017133, 0.0026341, -0.0003102, 0.0003144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000451, upper bound: 0.0000721
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000451, upper bound: 0.0000878
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001397, 0.0007565, -0.0001286, 0.0009154, -0.0005280, 0.0003648
1: -0.0034971, -0.0033147, -0.0035017, -0.0033123, -0.0000587, 0.0000605
2: 0.0147638, 0.0159147, 0.0147745, 0.0160900, -0.0006347, 0.0004560
3: 1.0066440, 1.0069162, 1.0066805, 1.0069603, -0.0001863, 0.0000970
4: -0.0042422, -0.0040620, -0.0042645, -0.0040630, -0.0000693, 0.0000916
5: 0.0038735, 0.0045597, 0.0038816, 0.0046789, -0.0004007, 0.0002783
6: -0.0026117, -0.0025807, -0.0026096, -0.0025640, -0.0000362, 0.0000178
7: -0.0126565, -0.0111468, -0.0130835, -0.0111883, -0.0006833, 0.0011366
8: -0.0136596, -0.0117559, -0.0138602, -0.0117604, -0.0007186, 0.0009143
9: 0.0017196, 0.0026342, 0.0017175, 0.0027118, -0.0004081, 0.0003360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000539
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000539
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007570, -0.0001286, 0.0009178, -0.0005117, 0.0003831
1: -0.0035046, -0.0033148, -0.0035046, -0.0033123, -0.0000598, 0.0000637
2: 0.0147868, 0.0159151, 0.0147745, 0.0160921, -0.0006196, 0.0004722
3: 1.0066483, 1.0069501, 1.0066805, 1.0069728, -0.0001856, 0.0001296
4: -0.0042422, -0.0040637, -0.0042646, -0.0040630, -0.0000705, 0.0000905
5: 0.0038926, 0.0045600, 0.0038816, 0.0046807, -0.0003887, 0.0002917
6: -0.0026078, -0.0025788, -0.0026096, -0.0025631, -0.0000341, 0.0000222
7: -0.0126628, -0.0112484, -0.0130946, -0.0111883, -0.0007584, 0.0010786
8: -0.0136596, -0.0117601, -0.0138606, -0.0117604, -0.0007216, 0.0009114
9: 0.0017136, 0.0026341, 0.0017152, 0.0027118, -0.0004096, 0.0003375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000564, upper bound: 0.0000780
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000564, upper bound: 0.0000878
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001657, 0.0008965, -0.0001293, 0.0007524, -0.0003971, 0.0005055
1: -0.0034929, -0.0033116, -0.0034984, -0.0033142, -0.0000548, 0.0000626
2: 0.0147405, 0.0160721, 0.0147715, 0.0159096, -0.0004851, 0.0006150
3: 1.0066777, 1.0069085, 1.0066478, 1.0069132, -0.0001341, 0.0001627
4: -0.0042631, -0.0040601, -0.0042417, -0.0040621, -0.0000903, 0.0000715
5: 0.0038544, 0.0046649, 0.0038810, 0.0045565, -0.0003020, 0.0003842
6: -0.0026150, -0.0025689, -0.0026089, -0.0025820, -0.0000238, 0.0000317
7: -0.0130115, -0.0110511, -0.0126429, -0.0111921, -0.0010486, 0.0008106
8: -0.0138534, -0.0117473, -0.0136559, -0.0117497, -0.0009122, 0.0007255
9: 0.0017210, 0.0027112, 0.0017147, 0.0026338, -0.0003316, 0.0004124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000412, upper bound: 0.0000148
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 253

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000246
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000694, upper bound: 0.0000517
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001396, 0.0008971, -0.0001293, 0.0007551, -0.0003787, 0.0005203
1: -0.0035007, -0.0033114, -0.0035013, -0.0033142, -0.0000561, 0.0000661
2: 0.0147639, 0.0160724, 0.0147715, 0.0159117, -0.0004686, 0.0006286
3: 1.0066861, 1.0069381, 1.0066478, 1.0069275, -0.0001237, 0.0001779
4: -0.0042631, -0.0040618, -0.0042419, -0.0040621, -0.0000913, 0.0000703
5: 0.0038735, 0.0046654, 0.0038810, 0.0045585, -0.0002885, 0.0003951
6: -0.0026103, -0.0025679, -0.0026089, -0.0025810, -0.0000207, 0.0000346
7: -0.0130167, -0.0111541, -0.0126541, -0.0111921, -0.0011056, 0.0007410
8: -0.0138534, -0.0117515, -0.0136562, -0.0117497, -0.0009148, 0.0007223
9: 0.0017148, 0.0027113, 0.0017124, 0.0026338, -0.0003333, 0.0004142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000649, upper bound: 0.0000666
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000647, upper bound: 0.0000631
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001657, 0.0008965, -0.0001136, 0.0007608, -0.0004246, 0.0005118
1: -0.0034929, -0.0033116, -0.0035022, -0.0033148, -0.0000547, 0.0000658
2: 0.0147405, 0.0160721, 0.0147868, 0.0159185, -0.0005106, 0.0006199
3: 1.0066777, 1.0069085, 1.0066483, 1.0069393, -0.0001772, 0.0001736
4: -0.0042631, -0.0040601, -0.0042425, -0.0040637, -0.0000905, 0.0000738
5: 0.0038544, 0.0046649, 0.0038926, 0.0045628, -0.0003222, 0.0003887
6: -0.0026150, -0.0025689, -0.0026078, -0.0025791, -0.0000303, 0.0000338
7: -0.0130115, -0.0110511, -0.0126754, -0.0112484, -0.0010826, 0.0009217
8: -0.0138534, -0.0117473, -0.0136603, -0.0117601, -0.0009121, 0.0007384
9: 0.0017210, 0.0027112, 0.0017155, 0.0026341, -0.0003323, 0.0004122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000491, upper bound: 0.0000395
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000491, upper bound: 0.0000514
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0001396, 0.0008971, -0.0001136, 0.0007632, -0.0004061, 0.0005260
1: -0.0035007, -0.0033114, -0.0035050, -0.0033148, -0.0000570, 0.0000691
2: 0.0147639, 0.0160724, 0.0147868, 0.0159207, -0.0004939, 0.0006330
3: 1.0066861, 1.0069381, 1.0066483, 1.0069526, -0.0001616, 0.0001979
4: -0.0042631, -0.0040618, -0.0042426, -0.0040637, -0.0000915, 0.0000725
5: 0.0038735, 0.0046654, 0.0038926, 0.0045646, -0.0003086, 0.0003992
6: -0.0026103, -0.0025679, -0.0026078, -0.0025780, -0.0000271, 0.0000368
7: -0.0130167, -0.0111541, -0.0126866, -0.0112484, -0.0011369, 0.0008524
8: -0.0138534, -0.0117515, -0.0136606, -0.0117601, -0.0009150, 0.0007351
9: 0.0017148, 0.0027113, 0.0017133, 0.0026341, -0.0003340, 0.0004139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000572
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000763
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001545, 0.0009111, -0.0001293, 0.0007524, -0.0003981, 0.0005294
1: -0.0034966, -0.0033122, -0.0034984, -0.0033142, -0.0000580, 0.0000627
2: 0.0147516, 0.0160862, 0.0147715, 0.0159096, -0.0004853, 0.0006376
3: 1.0066711, 1.0069400, 1.0066478, 1.0069132, -0.0001419, 0.0001960
4: -0.0042642, -0.0040613, -0.0042417, -0.0040621, -0.0000924, 0.0000715
5: 0.0038627, 0.0046758, 0.0038810, 0.0045565, -0.0003026, 0.0004018
6: -0.0026142, -0.0025654, -0.0026089, -0.0025820, -0.0000245, 0.0000367
7: -0.0130647, -0.0110871, -0.0126429, -0.0111921, -0.0011410, 0.0008189
8: -0.0138596, -0.0117561, -0.0136559, -0.0117497, -0.0009254, 0.0007251
9: 0.0017217, 0.0027117, 0.0017147, 0.0026338, -0.0003315, 0.0004133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 253

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000377
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000639
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001286, 0.0009116, -0.0001293, 0.0007551, -0.0003798, 0.0005443
1: -0.0035042, -0.0033123, -0.0035013, -0.0033142, -0.0000592, 0.0000657
2: 0.0147745, 0.0160867, 0.0147715, 0.0159117, -0.0004692, 0.0006513
3: 1.0066805, 1.0069697, 1.0066478, 1.0069275, -0.0001334, 0.0002112
4: -0.0042642, -0.0040630, -0.0042419, -0.0040621, -0.0000934, 0.0000703
5: 0.0038816, 0.0046762, 0.0038810, 0.0045585, -0.0002893, 0.0004127
6: -0.0026096, -0.0025642, -0.0026089, -0.0025810, -0.0000218, 0.0000396
7: -0.0130700, -0.0111883, -0.0126541, -0.0111921, -0.0011981, 0.0007529
8: -0.0138596, -0.0117604, -0.0136562, -0.0117497, -0.0009280, 0.0007218
9: 0.0017155, 0.0027118, 0.0017124, 0.0026338, -0.0003332, 0.0004151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000631, upper bound: 0.0000724
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000631, upper bound: 0.0000711
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001545, 0.0009111, -0.0001136, 0.0007608, -0.0003947, 0.0005031
1: -0.0034966, -0.0033122, -0.0035022, -0.0033148, -0.0000556, 0.0000634
2: 0.0147516, 0.0160862, 0.0147868, 0.0159185, -0.0004823, 0.0006122
3: 1.0066711, 1.0069400, 1.0066483, 1.0069393, -0.0001398, 0.0001684
4: -0.0042642, -0.0040613, -0.0042425, -0.0040637, -0.0000900, 0.0000712
5: 0.0038627, 0.0046758, 0.0038926, 0.0045628, -0.0003002, 0.0003824
6: -0.0026142, -0.0025654, -0.0026078, -0.0025791, -0.0000240, 0.0000319
7: -0.0130647, -0.0110871, -0.0126754, -0.0112484, -0.0010419, 0.0008039
8: -0.0138596, -0.0117561, -0.0136603, -0.0117601, -0.0009101, 0.0007234
9: 0.0017217, 0.0027117, 0.0017155, 0.0026341, -0.0003314, 0.0004122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000492, upper bound: 0.0000627
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000492, upper bound: 0.0000666
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0001286, 0.0009116, -0.0001136, 0.0007632, -0.0003763, 0.0005179
1: -0.0035042, -0.0033123, -0.0035050, -0.0033148, -0.0000569, 0.0000669
2: 0.0147745, 0.0160867, 0.0147868, 0.0159207, -0.0004658, 0.0006259
3: 1.0066805, 1.0069697, 1.0066483, 1.0069526, -0.0001294, 0.0001829
4: -0.0042642, -0.0040630, -0.0042426, -0.0040637, -0.0000910, 0.0000700
5: 0.0038816, 0.0046762, 0.0038926, 0.0045646, -0.0002867, 0.0003933
6: -0.0026096, -0.0025642, -0.0026078, -0.0025780, -0.0000209, 0.0000348
7: -0.0130700, -0.0111883, -0.0126866, -0.0112484, -0.0010982, 0.0007342
8: -0.0138596, -0.0117604, -0.0136606, -0.0117601, -0.0009127, 0.0007202
9: 0.0017155, 0.0027118, 0.0017133, 0.0026341, -0.0003331, 0.0004140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000492, upper bound: 0.0000736
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000492, upper bound: 0.0000845
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001657, 0.0008965, -0.0001396, 0.0009008, -0.0003780, 0.0003516
1: -0.0034929, -0.0033116, -0.0034981, -0.0033114, -0.0000554, 0.0000603
2: 0.0147405, 0.0160721, 0.0147639, 0.0160758, -0.0004633, 0.0004395
3: 1.0066777, 1.0069085, 1.0066861, 1.0069288, -0.0001063, 0.0000817
4: -0.0042631, -0.0040601, -0.0042634, -0.0040618, -0.0000670, 0.0000687
5: 0.0038544, 0.0046649, 0.0038735, 0.0046681, -0.0002876, 0.0002682
6: -0.0026150, -0.0025689, -0.0026103, -0.0025675, -0.0000217, 0.0000170
7: -0.0130115, -0.0110511, -0.0130304, -0.0111541, -0.0006607, 0.0007625
8: -0.0138534, -0.0117473, -0.0138540, -0.0117515, -0.0006960, 0.0007004
9: 0.0017210, 0.0027112, 0.0017168, 0.0027113, -0.0003227, 0.0003271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000708, upper bound: 0.0000519
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000661, upper bound: 0.0000491
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001396, 0.0008971, -0.0001396, 0.0009033, -0.0003632, 0.0003663
1: -0.0035007, -0.0033114, -0.0035010, -0.0033114, -0.0000567, 0.0000636
2: 0.0147639, 0.0160724, 0.0147639, 0.0160779, -0.0004493, 0.0004530
3: 1.0066861, 1.0069381, 1.0066861, 1.0069412, -0.0001141, 0.0000951
4: -0.0042631, -0.0040618, -0.0042635, -0.0040618, -0.0000680, 0.0000676
5: 0.0038735, 0.0046654, 0.0038735, 0.0046699, -0.0002766, 0.0002790
6: -0.0026103, -0.0025679, -0.0026103, -0.0025667, -0.0000201, 0.0000195
7: -0.0130167, -0.0111541, -0.0130413, -0.0111541, -0.0007168, 0.0007117
8: -0.0138534, -0.0117515, -0.0138544, -0.0117515, -0.0006985, 0.0006976
9: 0.0017148, 0.0027113, 0.0017145, 0.0027113, -0.0003245, 0.0003286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000717, upper bound: 0.0000810
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000717, upper bound: 0.0000819
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001657, 0.0008965, -0.0001286, 0.0009154, -0.0004055, 0.0003579
1: -0.0034929, -0.0033116, -0.0035017, -0.0033123, -0.0000554, 0.0000635
2: 0.0147405, 0.0160721, 0.0147745, 0.0160900, -0.0004888, 0.0004444
3: 1.0066777, 1.0069085, 1.0066805, 1.0069603, -0.0001495, 0.0000926
4: -0.0042631, -0.0040601, -0.0042645, -0.0040630, -0.0000672, 0.0000710
5: 0.0038544, 0.0046649, 0.0038816, 0.0046789, -0.0003078, 0.0002728
6: -0.0026150, -0.0025689, -0.0026096, -0.0025640, -0.0000282, 0.0000191
7: -0.0130115, -0.0110511, -0.0130835, -0.0111883, -0.0006948, 0.0008736
8: -0.0138534, -0.0117473, -0.0138602, -0.0117604, -0.0006959, 0.0007132
9: 0.0017210, 0.0027112, 0.0017175, 0.0027118, -0.0003235, 0.0003268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000747, upper bound: 0.0000495
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000637, upper bound: 0.0000395
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0001396, 0.0008971, -0.0001286, 0.0009178, -0.0003905, 0.0003719
1: -0.0035007, -0.0033114, -0.0035046, -0.0033123, -0.0000577, 0.0000666
2: 0.0147639, 0.0160724, 0.0147745, 0.0160921, -0.0004747, 0.0004574
3: 1.0066861, 1.0069381, 1.0066805, 1.0069728, -0.0001520, 0.0001151
4: -0.0042631, -0.0040618, -0.0042646, -0.0040630, -0.0000682, 0.0000699
5: 0.0038735, 0.0046654, 0.0038816, 0.0046807, -0.0002967, 0.0002831
6: -0.0026103, -0.0025679, -0.0026096, -0.0025631, -0.0000265, 0.0000217
7: -0.0130167, -0.0111541, -0.0130946, -0.0111883, -0.0007480, 0.0008231
8: -0.0138534, -0.0117515, -0.0138606, -0.0117604, -0.0006987, 0.0007104
9: 0.0017148, 0.0027113, 0.0017152, 0.0027118, -0.0003252, 0.0003284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000799, upper bound: 0.0000799
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000799, upper bound: 0.0000805
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001545, 0.0009111, -0.0001396, 0.0009008, -0.0003837, 0.0003792
1: -0.0034966, -0.0033122, -0.0034981, -0.0033114, -0.0000592, 0.0000613
2: 0.0147516, 0.0160862, 0.0147639, 0.0160758, -0.0004677, 0.0004650
3: 1.0066711, 1.0069400, 1.0066861, 1.0069288, -0.0001263, 0.0001268
4: -0.0042642, -0.0040613, -0.0042634, -0.0040618, -0.0000692, 0.0000689
5: 0.0038627, 0.0046758, 0.0038735, 0.0046681, -0.0002917, 0.0002885
6: -0.0026142, -0.0025654, -0.0026103, -0.0025675, -0.0000239, 0.0000232
7: -0.0130647, -0.0110871, -0.0130304, -0.0111541, -0.0007708, 0.0007937
8: -0.0138596, -0.0117561, -0.0138540, -0.0117515, -0.0007089, 0.0007005
9: 0.0017217, 0.0027117, 0.0017168, 0.0027113, -0.0003225, 0.0003278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000602, upper bound: 0.0000471
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000588, upper bound: 0.0000440
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001286, 0.0009116, -0.0001396, 0.0009033, -0.0003695, 0.0003938
1: -0.0035042, -0.0033123, -0.0035010, -0.0033114, -0.0000604, 0.0000635
2: 0.0147745, 0.0160867, 0.0147639, 0.0160779, -0.0004543, 0.0004785
3: 1.0066805, 1.0069697, 1.0066861, 1.0069412, -0.0001250, 0.0001403
4: -0.0042642, -0.0040630, -0.0042635, -0.0040618, -0.0000702, 0.0000678
5: 0.0038816, 0.0046762, 0.0038735, 0.0046699, -0.0002812, 0.0002993
6: -0.0026096, -0.0025642, -0.0026103, -0.0025667, -0.0000222, 0.0000257
7: -0.0130700, -0.0111883, -0.0130413, -0.0111541, -0.0008269, 0.0007457
8: -0.0138596, -0.0117604, -0.0138544, -0.0117515, -0.0007114, 0.0006975
9: 0.0017155, 0.0027118, 0.0017145, 0.0027113, -0.0003242, 0.0003293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000882
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000889
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001545, 0.0009111, -0.0001286, 0.0009154, -0.0003774, 0.0003510
1: -0.0034966, -0.0033122, -0.0035017, -0.0033123, -0.0000562, 0.0000611
2: 0.0147516, 0.0160862, 0.0147745, 0.0160900, -0.0004624, 0.0004386
3: 1.0066711, 1.0069400, 1.0066805, 1.0069603, -0.0001134, 0.0000888
4: -0.0042642, -0.0040613, -0.0042645, -0.0040630, -0.0000668, 0.0000685
5: 0.0038627, 0.0046758, 0.0038816, 0.0046789, -0.0002871, 0.0002677
6: -0.0026142, -0.0025654, -0.0026096, -0.0025640, -0.0000220, 0.0000173
7: -0.0130647, -0.0110871, -0.0130835, -0.0111883, -0.0006616, 0.0007634
8: -0.0138596, -0.0117561, -0.0138602, -0.0117604, -0.0006947, 0.0006991
9: 0.0017217, 0.0027117, 0.0017175, 0.0027118, -0.0003226, 0.0003269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 133

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000602, upper bound: 0.0000470
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000588, upper bound: 0.0000398
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0001286, 0.0009116, -0.0001286, 0.0009178, -0.0003625, 0.0003657
1: -0.0035042, -0.0033123, -0.0035046, -0.0033123, -0.0000575, 0.0000644
2: 0.0147745, 0.0160867, 0.0147745, 0.0160921, -0.0004485, 0.0004522
3: 1.0066805, 1.0069697, 1.0066805, 1.0069728, -0.0001212, 0.0001019
4: -0.0042642, -0.0040630, -0.0042646, -0.0040630, -0.0000678, 0.0000675
5: 0.0038816, 0.0046762, 0.0038816, 0.0046807, -0.0002761, 0.0002785
6: -0.0026096, -0.0025642, -0.0026096, -0.0025631, -0.0000205, 0.0000199
7: -0.0130700, -0.0111883, -0.0130946, -0.0111883, -0.0007173, 0.0007125
8: -0.0138596, -0.0117604, -0.0138606, -0.0117604, -0.0006972, 0.0006963
9: 0.0017155, 0.0027118, 0.0017152, 0.0027118, -0.0003244, 0.0003285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000886
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000892
time: 0.65 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.76 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000672, upper bound: 0.0000672
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000672, upper bound: 0.0000672
IS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000627, upper bound: 0.0000608
IS_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000609, upper bound: 0.0000609
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000660, upper bound: 0.0000672
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000660, upper bound: 0.0000672
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000617, upper bound: 0.0000608
IS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000594, upper bound: 0.0000609
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000631, upper bound: 0.0000762
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000631, upper bound: 0.0000724
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000771
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000878
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000659
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000750, upper bound: 0.0000659
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000703, upper bound: 0.0000596
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000638, upper bound: 0.0000596
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000738, upper bound: 0.0000659
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000738, upper bound: 0.0000659
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000596
IS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000619, upper bound: 0.0000596
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000429, upper bound: 0.0000467
IS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000429, upper bound: 0.0000508
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000451, upper bound: 0.0000721
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000451, upper bound: 0.0000878
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000539
IS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000539
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000564, upper bound: 0.0000780
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000564, upper bound: 0.0000878
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000246
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000694, upper bound: 0.0000517
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000649, upper bound: 0.0000666
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000647, upper bound: 0.0000631
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000491, upper bound: 0.0000395
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000491, upper bound: 0.0000514
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000572
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000763
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000377
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000639
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000631, upper bound: 0.0000724
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000631, upper bound: 0.0000711
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000492, upper bound: 0.0000627
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000492, upper bound: 0.0000666
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000492, upper bound: 0.0000736
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000492, upper bound: 0.0000845
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000708, upper bound: 0.0000519
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000661, upper bound: 0.0000491
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000717, upper bound: 0.0000810
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000717, upper bound: 0.0000819
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000747, upper bound: 0.0000495
IS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000637, upper bound: 0.0000395
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000799, upper bound: 0.0000799
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000799, upper bound: 0.0000805
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000602, upper bound: 0.0000471
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000588, upper bound: 0.0000440
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000882
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000889
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000602, upper bound: 0.0000470
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000588, upper bound: 0.0000398
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000886
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000892

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001274, 0.0007551, -0.0001274, 0.0007551, -0.0003288, 0.0003288
1: -0.0035013, -0.0033151, -0.0035013, -0.0033151, -0.0000628, 0.0000628
2: 0.0147726, 0.0159117, 0.0147726, 0.0159117, -0.0004135, 0.0004135
3: 1.0066508, 1.0069275, 1.0066508, 1.0069275, -0.0001133, 0.0001133
4: -0.0042418, -0.0040621, -0.0042418, -0.0040621, -0.0000634, 0.0000634
5: 0.0038823, 0.0045585, 0.0038823, 0.0045585, -0.0002510, 0.0002510
6: -0.0026084, -0.0025810, -0.0026084, -0.0025810, -0.0000161, 0.0000161
7: -0.0126541, -0.0112027, -0.0126541, -0.0112027, -0.0006077, 0.0006077
8: -0.0136545, -0.0117497, -0.0136545, -0.0117497, -0.0006595, 0.0006595
9: 0.0017124, 0.0026322, 0.0017124, 0.0026322, -0.0003140, 0.0003140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000671, upper bound: 0.0000630
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000670, upper bound: 0.0000609
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001274, 0.0007551, -0.0001144, 0.0007745, -0.0003610, 0.0003269
1: -0.0035013, -0.0033151, -0.0035104, -0.0033273, -0.0000508, 0.0000725
2: 0.0147726, 0.0159117, 0.0147801, 0.0159233, -0.0004316, 0.0004120
3: 1.0066508, 1.0069275, 1.0066979, 1.0069686, -0.0001900, 0.0000972
4: -0.0042418, -0.0040621, -0.0042410, -0.0040616, -0.0000635, 0.0000621
5: 0.0038823, 0.0045585, 0.0038914, 0.0045722, -0.0002737, 0.0002496
6: -0.0026084, -0.0025810, -0.0026026, -0.0025743, -0.0000281, 0.0000153
7: -0.0126541, -0.0112027, -0.0127699, -0.0112848, -0.0006014, 0.0008022
8: -0.0136545, -0.0117497, -0.0136279, -0.0117318, -0.0006649, 0.0006211
9: 0.0017124, 0.0026322, 0.0016961, 0.0026093, -0.0002815, 0.0003204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000671, upper bound: 0.0000630
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000670, upper bound: 0.0000609
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001274, 0.0007551, -0.0001377, 0.0009033, -0.0005113, 0.0003763
1: -0.0035013, -0.0033151, -0.0035010, -0.0033122, -0.0000657, 0.0000625
2: 0.0147726, 0.0159117, 0.0147650, 0.0160779, -0.0006208, 0.0004672
3: 1.0066508, 1.0069275, 1.0066903, 1.0069412, -0.0001759, 0.0001184
4: -0.0042418, -0.0040621, -0.0042635, -0.0040618, -0.0000703, 0.0000908
5: 0.0038823, 0.0045585, 0.0038748, 0.0046699, -0.0003885, 0.0002868
6: -0.0026084, -0.0025810, -0.0026097, -0.0025667, -0.0000332, 0.0000199
7: -0.0126541, -0.0112027, -0.0130413, -0.0111649, -0.0007276, 0.0010705
8: -0.0136545, -0.0117497, -0.0138527, -0.0117515, -0.0007217, 0.0009124
9: 0.0017124, 0.0026322, 0.0017145, 0.0027099, -0.0004133, 0.0003371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000639, upper bound: 0.0000630
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000639, upper bound: 0.0000609
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001274, 0.0007551, -0.0001250, 0.0009239, -0.0005374, 0.0003689
1: -0.0035013, -0.0033151, -0.0035098, -0.0033247, -0.0000529, 0.0000717
2: 0.0147726, 0.0159117, 0.0147724, 0.0160902, -0.0006357, 0.0004629
3: 1.0066508, 1.0069275, 1.0067281, 1.0069915, -0.0002461, 0.0000996
4: -0.0042418, -0.0040621, -0.0042626, -0.0040613, -0.0000706, 0.0000898
5: 0.0038823, 0.0045585, 0.0038838, 0.0046845, -0.0004069, 0.0002815
6: -0.0026084, -0.0025810, -0.0026052, -0.0025589, -0.0000430, 0.0000173
7: -0.0126541, -0.0112027, -0.0131616, -0.0112428, -0.0006848, 0.0012281
8: -0.0136545, -0.0117497, -0.0138249, -0.0117346, -0.0007324, 0.0008789
9: 0.0017124, 0.0026322, 0.0017003, 0.0026850, -0.0003835, 0.0003467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000639, upper bound: 0.0000630
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000639, upper bound: 0.0000609
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007570, -0.0001274, 0.0007551, -0.0003378, 0.0003595
1: -0.0035046, -0.0033148, -0.0035013, -0.0033151, -0.0000596, 0.0000631
2: 0.0147868, 0.0159151, 0.0147726, 0.0159117, -0.0004199, 0.0004427
3: 1.0066483, 1.0069501, 1.0066508, 1.0069275, -0.0001281, 0.0001373
4: -0.0042422, -0.0040637, -0.0042418, -0.0040621, -0.0000660, 0.0000636
5: 0.0038926, 0.0045600, 0.0038823, 0.0045585, -0.0002575, 0.0002737
6: -0.0026078, -0.0025788, -0.0026084, -0.0025810, -0.0000189, 0.0000218
7: -0.0126628, -0.0112484, -0.0126541, -0.0112027, -0.0007211, 0.0006565
8: -0.0136596, -0.0117601, -0.0136545, -0.0117497, -0.0006740, 0.0006594
9: 0.0017136, 0.0026341, 0.0017124, 0.0026322, -0.0003096, 0.0003153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000631, upper bound: 0.0000724
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000631, upper bound: 0.0000724
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001101, 0.0007570, -0.0001144, 0.0007745, -0.0003687, 0.0003596
1: -0.0035046, -0.0033179, -0.0035104, -0.0033273, -0.0000593, 0.0000693
2: 0.0147888, 0.0159151, 0.0147801, 0.0159233, -0.0004373, 0.0004428
3: 1.0066603, 1.0069501, 1.0066979, 1.0069686, -0.0001971, 0.0001352
4: -0.0042420, -0.0040637, -0.0042410, -0.0040616, -0.0000658, 0.0000635
5: 0.0038950, 0.0045600, 0.0038914, 0.0045722, -0.0002793, 0.0002737
6: -0.0026062, -0.0025788, -0.0026026, -0.0025743, -0.0000301, 0.0000211
7: -0.0126628, -0.0112709, -0.0127699, -0.0112848, -0.0007201, 0.0008404
8: -0.0136529, -0.0117601, -0.0136279, -0.0117318, -0.0006694, 0.0006593
9: 0.0017136, 0.0026283, 0.0016961, 0.0026093, -0.0003095, 0.0003134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000566, upper bound: 0.0000675
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000565, upper bound: 0.0000607
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007570, -0.0001657, 0.0008965, -0.0005118, 0.0004245
1: -0.0035046, -0.0033148, -0.0034929, -0.0033116, -0.0000684, 0.0000547
2: 0.0147868, 0.0159151, 0.0147405, 0.0160721, -0.0006199, 0.0005100
3: 1.0066483, 1.0069501, 1.0066777, 1.0069085, -0.0001736, 0.0001845
4: -0.0042422, -0.0040637, -0.0042631, -0.0040601, -0.0000737, 0.0000905
5: 0.0038926, 0.0045600, 0.0038544, 0.0046649, -0.0003887, 0.0003221
6: -0.0026078, -0.0025788, -0.0026150, -0.0025689, -0.0000338, 0.0000309
7: -0.0126628, -0.0112484, -0.0130115, -0.0110511, -0.0009271, 0.0010826
8: -0.0136596, -0.0117601, -0.0138534, -0.0117473, -0.0007381, 0.0009121
9: 0.0017136, 0.0026341, 0.0017210, 0.0027112, -0.0004140, 0.0003323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 253

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000272, upper bound: 0.0000571
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000506, upper bound: 0.0000761
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007570, -0.0001396, 0.0008971, -0.0005260, 0.0004131
1: -0.0035046, -0.0033148, -0.0035007, -0.0033114, -0.0000627, 0.0000570
2: 0.0147868, 0.0159151, 0.0147639, 0.0160724, -0.0006330, 0.0005004
3: 1.0066483, 1.0069501, 1.0066861, 1.0069381, -0.0001979, 0.0001698
4: -0.0042422, -0.0040637, -0.0042631, -0.0040618, -0.0000731, 0.0000915
5: 0.0038926, 0.0045600, 0.0038735, 0.0046654, -0.0003992, 0.0003138
6: -0.0026078, -0.0025788, -0.0026103, -0.0025679, -0.0000368, 0.0000283
7: -0.0126628, -0.0112484, -0.0130167, -0.0111541, -0.0008759, 0.0011369
8: -0.0136596, -0.0117601, -0.0138534, -0.0117515, -0.0007367, 0.0009150
9: 0.0017136, 0.0026341, 0.0017148, 0.0027113, -0.0004095, 0.0003340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 253

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000272, upper bound: 0.0000775
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000506, upper bound: 0.0000870
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001274, 0.0007551, -0.0001117, 0.0007632, -0.0003561, 0.0003350
1: -0.0035013, -0.0033151, -0.0035050, -0.0033158, -0.0000625, 0.0000658
2: 0.0147726, 0.0159117, 0.0147879, 0.0159207, -0.0004389, 0.0004183
3: 1.0066508, 1.0069275, 1.0066521, 1.0069526, -0.0001512, 0.0001228
4: -0.0042418, -0.0040621, -0.0042426, -0.0040637, -0.0000636, 0.0000656
5: 0.0038823, 0.0045585, 0.0038940, 0.0045646, -0.0002711, 0.0002555
6: -0.0026084, -0.0025810, -0.0026072, -0.0025780, -0.0000225, 0.0000181
7: -0.0126541, -0.0112027, -0.0126866, -0.0112588, -0.0006412, 0.0007191
8: -0.0136545, -0.0117497, -0.0136588, -0.0117601, -0.0006594, 0.0006725
9: 0.0017124, 0.0026322, 0.0017133, 0.0026325, -0.0003148, 0.0003137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000698, upper bound: 0.0000595
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000697, upper bound: 0.0000596
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001274, 0.0007551, -0.0000989, 0.0007832, -0.0003881, 0.0003326
1: -0.0035013, -0.0033151, -0.0035145, -0.0033276, -0.0000527, 0.0000780
2: 0.0147726, 0.0159117, 0.0147954, 0.0159321, -0.0004569, 0.0004170
3: 1.0066508, 1.0069275, 1.0066929, 1.0070002, -0.0002229, 0.0001048
4: -0.0042418, -0.0040621, -0.0042417, -0.0040631, -0.0000637, 0.0000644
5: 0.0038823, 0.0045585, 0.0039030, 0.0045787, -0.0002936, 0.0002538
6: -0.0026084, -0.0025810, -0.0026025, -0.0025702, -0.0000334, 0.0000163
7: -0.0126541, -0.0112027, -0.0128088, -0.0113363, -0.0006236, 0.0009050
8: -0.0136545, -0.0117497, -0.0136322, -0.0117428, -0.0006638, 0.0006318
9: 0.0017124, 0.0026322, 0.0016970, 0.0026096, -0.0002821, 0.0003201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000698, upper bound: 0.0000595
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000697, upper bound: 0.0000596
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001144, 0.0007745, -0.0001034, 0.0007632, -0.0003564, 0.0003577
1: -0.0035104, -0.0033273, -0.0035031, -0.0033179, -0.0000693, 0.0000638
2: 0.0147801, 0.0159233, 0.0147980, 0.0159207, -0.0004390, 0.0004218
3: 1.0066979, 1.0069686, 1.0066603, 1.0069491, -0.0001497, 0.0001971
4: -0.0042410, -0.0040616, -0.0042424, -0.0040653, -0.0000610, 0.0000655
5: 0.0038914, 0.0045722, 0.0039003, 0.0045646, -0.0002713, 0.0002707
6: -0.0026026, -0.0025743, -0.0026062, -0.0025781, -0.0000222, 0.0000301
7: -0.0127699, -0.0112848, -0.0126866, -0.0112764, -0.0008315, 0.0007209
8: -0.0136279, -0.0117318, -0.0136540, -0.0117776, -0.0006319, 0.0006684
9: 0.0016961, 0.0026093, 0.0017220, 0.0026283, -0.0003134, 0.0003011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000635, upper bound: 0.0000595
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000635, upper bound: 0.0000596
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001274, 0.0007551, -0.0001266, 0.0009178, -0.0005354, 0.0003775
1: -0.0035013, -0.0033151, -0.0035046, -0.0033132, -0.0000651, 0.0000656
2: 0.0147726, 0.0159117, 0.0147757, 0.0160921, -0.0006435, 0.0004679
3: 1.0066508, 1.0069275, 1.0066850, 1.0069728, -0.0002114, 0.0001276
4: -0.0042418, -0.0040621, -0.0042646, -0.0040630, -0.0000703, 0.0000929
5: 0.0038823, 0.0045585, 0.0038830, 0.0046807, -0.0004062, 0.0002876
6: -0.0026084, -0.0025810, -0.0026089, -0.0025631, -0.0000385, 0.0000210
7: -0.0126541, -0.0112027, -0.0130946, -0.0111998, -0.0007396, 0.0011668
8: -0.0136545, -0.0117497, -0.0138589, -0.0117604, -0.0007212, 0.0009256
9: 0.0017124, 0.0026322, 0.0017152, 0.0027104, -0.0004142, 0.0003370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000670, upper bound: 0.0000595
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000670, upper bound: 0.0000596
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001274, 0.0007551, -0.0001145, 0.0009386, -0.0005623, 0.0003703
1: -0.0035013, -0.0033151, -0.0035140, -0.0033250, -0.0000540, 0.0000767
2: 0.0147726, 0.0159117, 0.0147825, 0.0161041, -0.0006588, 0.0004633
3: 1.0066508, 1.0069275, 1.0067180, 1.0070282, -0.0002792, 0.0001071
4: -0.0042418, -0.0040621, -0.0042637, -0.0040625, -0.0000706, 0.0000919
5: 0.0038823, 0.0045585, 0.0038916, 0.0046954, -0.0004252, 0.0002825
6: -0.0026084, -0.0025810, -0.0026049, -0.0025549, -0.0000481, 0.0000183
7: -0.0126541, -0.0112027, -0.0132152, -0.0112762, -0.0006982, 0.0013267
8: -0.0136545, -0.0117497, -0.0138309, -0.0117438, -0.0007319, 0.0008914
9: 0.0017124, 0.0026322, 0.0017010, 0.0026854, -0.0003845, 0.0003466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000670, upper bound: 0.0000595
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000670, upper bound: 0.0000596
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001144, 0.0007745, -0.0001180, 0.0009178, -0.0005357, 0.0004000
1: -0.0035104, -0.0033273, -0.0035026, -0.0033153, -0.0000716, 0.0000634
2: 0.0147801, 0.0159233, 0.0147868, 0.0160921, -0.0006437, 0.0004724
3: 1.0066979, 1.0069686, 1.0066906, 1.0069698, -0.0002097, 0.0002025
4: -0.0042410, -0.0040616, -0.0042644, -0.0040648, -0.0000680, 0.0000928
5: 0.0038914, 0.0045722, 0.0038896, 0.0046807, -0.0004064, 0.0003028
6: -0.0026026, -0.0025743, -0.0026083, -0.0025633, -0.0000382, 0.0000328
7: -0.0127699, -0.0112848, -0.0130946, -0.0112156, -0.0009242, 0.0011686
8: -0.0136279, -0.0117318, -0.0138536, -0.0117794, -0.0006957, 0.0009238
9: 0.0016961, 0.0026093, 0.0017248, 0.0027055, -0.0004139, 0.0003238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000614, upper bound: 0.0000595
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000614, upper bound: 0.0000596
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007570, -0.0001397, 0.0007565, -0.0003176, 0.0003438
1: -0.0035046, -0.0033148, -0.0034971, -0.0033147, -0.0000636, 0.0000558
2: 0.0147868, 0.0159151, 0.0147638, 0.0159147, -0.0004024, 0.0004259
3: 1.0066483, 1.0069501, 1.0066440, 1.0069162, -0.0000906, 0.0001307
4: -0.0042422, -0.0040637, -0.0042422, -0.0040620, -0.0000641, 0.0000624
5: 0.0038926, 0.0045600, 0.0038735, 0.0045597, -0.0002427, 0.0002619
6: -0.0026078, -0.0025788, -0.0026117, -0.0025807, -0.0000139, 0.0000194
7: -0.0126628, -0.0112484, -0.0126565, -0.0111468, -0.0006723, 0.0005648
8: -0.0136596, -0.0117601, -0.0136596, -0.0117559, -0.0006605, 0.0006564
9: 0.0017136, 0.0026341, 0.0017196, 0.0026342, -0.0003146, 0.0003085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 253

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000346, upper bound: 0.0000493
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000692
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007570, -0.0001136, 0.0007570, -0.0003324, 0.0003324
1: -0.0035046, -0.0033148, -0.0035046, -0.0033148, -0.0000570, 0.0000570
2: 0.0147868, 0.0159151, 0.0147868, 0.0159151, -0.0004161, 0.0004161
3: 1.0066483, 1.0069501, 1.0066483, 1.0069501, -0.0001050, 0.0001050
4: -0.0042422, -0.0040637, -0.0042422, -0.0040637, -0.0000634, 0.0000634
5: 0.0038926, 0.0045600, 0.0038926, 0.0045600, -0.0002536, 0.0002536
6: -0.0026078, -0.0025788, -0.0026078, -0.0025788, -0.0000168, 0.0000168
7: -0.0126628, -0.0112484, -0.0126628, -0.0112484, -0.0006212, 0.0006212
8: -0.0136596, -0.0117601, -0.0136596, -0.0117601, -0.0006590, 0.0006590
9: 0.0017136, 0.0026341, 0.0017136, 0.0026341, -0.0003102, 0.0003102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 253

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000346, upper bound: 0.0000785
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000850
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007570, -0.0001545, 0.0009111, -0.0005031, 0.0003945
1: -0.0035046, -0.0033148, -0.0034966, -0.0033122, -0.0000663, 0.0000556
2: 0.0147868, 0.0159151, 0.0147516, 0.0160862, -0.0006122, 0.0004820
3: 1.0066483, 1.0069501, 1.0066711, 1.0069400, -0.0001684, 0.0001553
4: -0.0042422, -0.0040637, -0.0042642, -0.0040613, -0.0000711, 0.0000900
5: 0.0038926, 0.0045600, 0.0038627, 0.0046758, -0.0003824, 0.0003000
6: -0.0026078, -0.0025788, -0.0026142, -0.0025654, -0.0000319, 0.0000248
7: -0.0126628, -0.0112484, -0.0130647, -0.0110871, -0.0008095, 0.0010419
8: -0.0136596, -0.0117601, -0.0138596, -0.0117561, -0.0007231, 0.0009101
9: 0.0017136, 0.0026341, 0.0017217, 0.0027117, -0.0004140, 0.0003314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 253

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000492, upper bound: 0.0000672
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000537, upper bound: 0.0000752
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001136, 0.0007570, -0.0001286, 0.0009116, -0.0005179, 0.0003831
1: -0.0035046, -0.0033148, -0.0035042, -0.0033123, -0.0000598, 0.0000569
2: 0.0147868, 0.0159151, 0.0147745, 0.0160867, -0.0006259, 0.0004722
3: 1.0066483, 1.0069501, 1.0066805, 1.0069697, -0.0001829, 0.0001296
4: -0.0042422, -0.0040637, -0.0042642, -0.0040630, -0.0000705, 0.0000910
5: 0.0038926, 0.0045600, 0.0038816, 0.0046762, -0.0003933, 0.0002917
6: -0.0026078, -0.0025788, -0.0026096, -0.0025642, -0.0000348, 0.0000222
7: -0.0126628, -0.0112484, -0.0130700, -0.0111883, -0.0007584, 0.0010982
8: -0.0136596, -0.0117601, -0.0138596, -0.0117604, -0.0007216, 0.0009127
9: 0.0017136, 0.0026341, 0.0017155, 0.0027118, -0.0004096, 0.0003331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 253

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000492, upper bound: 0.0000786
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000537, upper bound: 0.0000850
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001650, 0.0008965, -0.0001211, 0.0007524, -0.0003964, 0.0004451
1: -0.0034929, -0.0033116, -0.0034975, -0.0033142, -0.0000548, 0.0000544
2: 0.0147414, 0.0160721, 0.0147816, 0.0159096, -0.0004843, 0.0005377
3: 1.0066777, 1.0069085, 1.0066495, 1.0069132, -0.0001340, 0.0001607
4: -0.0042631, -0.0040602, -0.0042417, -0.0040636, -0.0000783, 0.0000714
5: 0.0038549, 0.0046649, 0.0038872, 0.0045565, -0.0003015, 0.0003379
6: -0.0026149, -0.0025689, -0.0026083, -0.0025820, -0.0000238, 0.0000299
7: -0.0130115, -0.0110527, -0.0126429, -0.0112103, -0.0009551, 0.0008091
8: -0.0138534, -0.0117484, -0.0136559, -0.0117648, -0.0007889, 0.0007245
9: 0.0017215, 0.0027112, 0.0017218, 0.0026338, -0.0003312, 0.0003564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000487, upper bound: 0.0000329
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 253

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000592, upper bound: 0.0000359
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000592, upper bound: 0.0000517
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001396, 0.0008971, -0.0001274, 0.0007551, -0.0003787, 0.0005175
1: -0.0035007, -0.0033114, -0.0035013, -0.0033151, -0.0000557, 0.0000661
2: 0.0147639, 0.0160724, 0.0147726, 0.0159117, -0.0004686, 0.0006270
3: 1.0066861, 1.0069381, 1.0066508, 1.0069275, -0.0001237, 0.0001701
4: -0.0042631, -0.0040618, -0.0042418, -0.0040621, -0.0000913, 0.0000703
5: 0.0038735, 0.0046654, 0.0038823, 0.0045585, -0.0002885, 0.0003931
6: -0.0026103, -0.0025679, -0.0026084, -0.0025810, -0.0000207, 0.0000335
7: -0.0130167, -0.0111541, -0.0126541, -0.0112027, -0.0010880, 0.0007410
8: -0.0138534, -0.0117515, -0.0136545, -0.0117497, -0.0009148, 0.0007217
9: 0.0017148, 0.0027113, 0.0017124, 0.0026322, -0.0003328, 0.0004142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000647, upper bound: 0.0000631
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000647, upper bound: 0.0000631
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001361, 0.0008971, -0.0001144, 0.0007745, -0.0004085, 0.0005176
1: -0.0035007, -0.0033148, -0.0035104, -0.0033273, -0.0000554, 0.0000717
2: 0.0147659, 0.0160724, 0.0147801, 0.0159233, -0.0004853, 0.0006270
3: 1.0066977, 1.0069381, 1.0066979, 1.0069686, -0.0001920, 0.0001679
4: -0.0042629, -0.0040618, -0.0042410, -0.0040616, -0.0000912, 0.0000703
5: 0.0038760, 0.0046654, 0.0038914, 0.0045722, -0.0003095, 0.0003931
6: -0.0026090, -0.0025679, -0.0026026, -0.0025743, -0.0000317, 0.0000329
7: -0.0130167, -0.0111755, -0.0127699, -0.0112848, -0.0010870, 0.0009208
8: -0.0138464, -0.0117515, -0.0136279, -0.0117318, -0.0009119, 0.0007216
9: 0.0017148, 0.0027050, 0.0016961, 0.0026093, -0.0003327, 0.0004130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000578, upper bound: 0.0000588
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000579, upper bound: 0.0000559
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001396, 0.0008971, -0.0001136, 0.0007570, -0.0004131, 0.0005260
1: -0.0035007, -0.0033114, -0.0035046, -0.0033148, -0.0000570, 0.0000627
2: 0.0147639, 0.0160724, 0.0147868, 0.0159151, -0.0005004, 0.0006330
3: 1.0066861, 1.0069381, 1.0066483, 1.0069501, -0.0001698, 0.0001979
4: -0.0042631, -0.0040618, -0.0042422, -0.0040637, -0.0000915, 0.0000731
5: 0.0038735, 0.0046654, 0.0038926, 0.0045600, -0.0003138, 0.0003992
6: -0.0026103, -0.0025679, -0.0026078, -0.0025788, -0.0000283, 0.0000368
7: -0.0130167, -0.0111541, -0.0126628, -0.0112484, -0.0011369, 0.0008759
8: -0.0138534, -0.0117515, -0.0136596, -0.0117601, -0.0009150, 0.0007367
9: 0.0017148, 0.0027113, 0.0017136, 0.0026341, -0.0003340, 0.0004095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 133

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 253

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 107

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5

No IS candidates found

### IS candidates at layer 7

No IS candidates found

No IS candidates found

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.93 + 355.15 = 358.08 seconds
