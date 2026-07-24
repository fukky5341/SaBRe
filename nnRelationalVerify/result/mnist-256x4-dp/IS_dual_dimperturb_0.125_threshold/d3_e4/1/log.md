## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000418


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0011383, -0.0006029, -0.0011383, -0.0006029, -0.0003135, 0.0003135)
1: (-0.0071992, -0.0058406, -0.0071992, -0.0058406, -0.0007955, 0.0007955)
2: (0.0305636, 0.0314065, 0.0305636, 0.0314065, -0.0004935, 0.0004935)
3: (0.0009186, 0.0024925, 0.0009186, 0.0024925, -0.0009215, 0.0009215)
4: (-0.0062158, -0.0048339, -0.0062158, -0.0048339, -0.0008092, 0.0008092)
5: (0.0113838, 0.0119072, 0.0113838, 0.0119072, -0.0003065, 0.0003065)
6: (0.0015540, 0.0035514, 0.0015540, 0.0035514, -0.0011696, 0.0011696)
7: (0.9791467, 0.9805443, 0.9791467, 0.9805443, -0.0008184, 0.0008184)
8: (-0.0089223, -0.0074237, -0.0089223, -0.0074237, -0.0008775, 0.0008775)
9: (-0.0000958, 0.0008941, -0.0000958, 0.0008941, -0.0005796, 0.0005796)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.83 + 1.39 = 3.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0005080, upper bound: 0.0005080

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004701
time: 0.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004969
time: 0.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.25 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004701
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004969

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0011158, -0.0006069, -0.0011298, -0.0006031, -0.0002877, 0.0002930
1: -0.0071421, -0.0058506, -0.0071776, -0.0058410, -0.0007301, 0.0007436
2: 0.0305990, 0.0314003, 0.0305770, 0.0314062, -0.0004530, 0.0004613
3: 0.0009302, 0.0024264, 0.0009191, 0.0024675, -0.0008614, 0.0008458
4: -0.0061578, -0.0048441, -0.0061938, -0.0048343, -0.0007427, 0.0007564
5: 0.0114058, 0.0119034, 0.0113921, 0.0119071, -0.0002813, 0.0002865
6: 0.0015687, 0.0034675, 0.0015546, 0.0035196, -0.0010933, 0.0010735
7: 0.9791570, 0.9804857, 0.9791471, 0.9805221, -0.0007650, 0.0007512
8: -0.0089113, -0.0074867, -0.0089218, -0.0074476, -0.0008202, 0.0008054
9: -0.0000542, 0.0008868, -0.0000801, 0.0008938, -0.0005320, 0.0005418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004735, upper bound: 0.0004539
time: 0.51 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004815, upper bound: 0.0004539
time: 0.51 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0011271, -0.0006032, -0.0011345, -0.0006030, -0.0002902, 0.0003108
1: -0.0071708, -0.0058412, -0.0071894, -0.0058408, -0.0007365, 0.0007886
2: 0.0305812, 0.0314061, 0.0305697, 0.0314064, -0.0004569, 0.0004893
3: 0.0009193, 0.0024596, 0.0009189, 0.0024812, -0.0009136, 0.0008532
4: -0.0061869, -0.0048345, -0.0062059, -0.0048341, -0.0007492, 0.0008022
5: 0.0113947, 0.0119070, 0.0113876, 0.0119072, -0.0002838, 0.0003038
6: 0.0015549, 0.0035096, 0.0015543, 0.0035371, -0.0011594, 0.0010828
7: 0.9791473, 0.9805151, 0.9791469, 0.9805343, -0.0008113, 0.0007577
8: -0.0089216, -0.0074551, -0.0089221, -0.0074345, -0.0008699, 0.0008124
9: -0.0000751, 0.0008936, -0.0000887, 0.0008939, -0.0005366, 0.0005746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004827, upper bound: 0.0004740
time: 0.52 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004827, upper bound: 0.0004827
time: 0.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.90 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 7, lower bound: -0.0004735, upper bound: 0.0004539
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 7, lower bound: -0.0004815, upper bound: 0.0004539
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 7, lower bound: -0.0004827, upper bound: 0.0004740
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 7, lower bound: -0.0004827, upper bound: 0.0004827

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011137, -0.0006069, -0.0011212, -0.0006033, -0.0002857, 0.0002867
1: -0.0071366, -0.0058507, -0.0071557, -0.0058414, -0.0007249, 0.0007276
2: 0.0306024, 0.0314002, 0.0305906, 0.0314060, -0.0004497, 0.0004514
3: 0.0009303, 0.0024200, 0.0009196, 0.0024422, -0.0008429, 0.0008398
4: -0.0061522, -0.0048442, -0.0061716, -0.0048347, -0.0007374, 0.0007401
5: 0.0114079, 0.0119033, 0.0114005, 0.0119069, -0.0002793, 0.0002803
6: 0.0015689, 0.0034594, 0.0015552, 0.0034876, -0.0010697, 0.0010658
7: 0.9791571, 0.9804800, 0.9791475, 0.9804997, -0.0007485, 0.0007458
8: -0.0089111, -0.0074927, -0.0089214, -0.0074716, -0.0008026, 0.0007996
9: -0.0000502, 0.0008867, -0.0000642, 0.0008935, -0.0005282, 0.0005301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004585, upper bound: 0.0004366
time: 0.52 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004585, upper bound: 0.0004375
time: 0.54 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011098, -0.0006070, -0.0011154, -0.0005930, -0.0002864, 0.0002901
1: -0.0071268, -0.0058509, -0.0071411, -0.0058154, -0.0007269, 0.0007361
2: 0.0306085, 0.0314001, 0.0305997, 0.0314221, -0.0004510, 0.0004567
3: 0.0009306, 0.0024086, 0.0008895, 0.0024252, -0.0008527, 0.0008421
4: -0.0061421, -0.0048444, -0.0061567, -0.0048083, -0.0007394, 0.0007487
5: 0.0114117, 0.0119033, 0.0114062, 0.0119169, -0.0002801, 0.0002836
6: 0.0015691, 0.0034450, 0.0015170, 0.0034660, -0.0010822, 0.0010687
7: 0.9791573, 0.9804698, 0.9791207, 0.9804846, -0.0007573, 0.0007478
8: -0.0089109, -0.0075036, -0.0089500, -0.0074878, -0.0008119, 0.0008018
9: -0.0000430, 0.0008866, -0.0000535, 0.0009124, -0.0005296, 0.0005363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004662, upper bound: 0.0004366
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004662, upper bound: 0.0004375
time: 0.54 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -0.0011185, -0.0006033, -0.0011323, -0.0006031, -0.0002839, 0.0003086
1: -0.0071489, -0.0058416, -0.0071839, -0.0058409, -0.0007205, 0.0007831
2: 0.0305948, 0.0314059, 0.0305731, 0.0314063, -0.0004470, 0.0004858
3: 0.0009198, 0.0024343, 0.0009190, 0.0024748, -0.0009071, 0.0008346
4: -0.0061647, -0.0048349, -0.0062003, -0.0048342, -0.0007329, 0.0007965
5: 0.0114032, 0.0119068, 0.0113897, 0.0119071, -0.0002776, 0.0003017
6: 0.0015555, 0.0034775, 0.0015544, 0.0035290, -0.0011513, 0.0010593
7: 0.9791478, 0.9804927, 0.9791470, 0.9805288, -0.0008056, 0.0007412
8: -0.0089212, -0.0074791, -0.0089219, -0.0074405, -0.0008637, 0.0007947
9: -0.0000592, 0.0008933, -0.0000847, 0.0008939, -0.0005250, 0.0005706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004664, upper bound: 0.0004588
time: 0.54 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004663, upper bound: 0.0004586
time: 0.61 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -0.0011132, -0.0005931, -0.0011284, -0.0006031, -0.0002875, 0.0003096
1: -0.0071354, -0.0058156, -0.0071741, -0.0058411, -0.0007295, 0.0007857
2: 0.0306032, 0.0314220, 0.0305792, 0.0314062, -0.0004526, 0.0004875
3: 0.0008897, 0.0024186, 0.0009192, 0.0024634, -0.0009103, 0.0008451
4: -0.0061510, -0.0048085, -0.0061903, -0.0048344, -0.0007420, 0.0007992
5: 0.0114084, 0.0119168, 0.0113935, 0.0119071, -0.0002810, 0.0003027
6: 0.0015173, 0.0034577, 0.0015547, 0.0035145, -0.0011552, 0.0010725
7: 0.9791210, 0.9804788, 0.9791472, 0.9805186, -0.0008084, 0.0007505
8: -0.0089498, -0.0074940, -0.0089218, -0.0074514, -0.0008667, 0.0008046
9: -0.0000494, 0.0009123, -0.0000775, 0.0008937, -0.0005315, 0.0005725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004664, upper bound: 0.0004659
time: 0.53 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004663, upper bound: 0.0004664
time: 0.54 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.11 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 7, lower bound: -0.0004585, upper bound: 0.0004366
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 7, lower bound: -0.0004585, upper bound: 0.0004375
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 7, lower bound: -0.0004662, upper bound: 0.0004366
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 7, lower bound: -0.0004662, upper bound: 0.0004375
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 7, lower bound: -0.0004664, upper bound: 0.0004588
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 7, lower bound: -0.0004663, upper bound: 0.0004586
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 7, lower bound: -0.0004664, upper bound: 0.0004659
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 7, lower bound: -0.0004663, upper bound: 0.0004664

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011133, -0.0006094, -0.0011199, -0.0006131, -0.0002756, 0.0002838
1: -0.0071358, -0.0058571, -0.0071526, -0.0058664, -0.0006993, 0.0007203
2: 0.0306029, 0.0313962, 0.0305925, 0.0313905, -0.0004339, 0.0004469
3: 0.0009378, 0.0024191, 0.0009485, 0.0024385, -0.0008344, 0.0008101
4: -0.0061514, -0.0048507, -0.0061684, -0.0048601, -0.0007113, 0.0007326
5: 0.0114082, 0.0119009, 0.0114018, 0.0118973, -0.0002694, 0.0002775
6: 0.0015783, 0.0034583, 0.0015919, 0.0034829, -0.0010589, 0.0010282
7: 0.9791636, 0.9804792, 0.9791732, 0.9804964, -0.0007410, 0.0007195
8: -0.0089041, -0.0074936, -0.0088938, -0.0074751, -0.0007945, 0.0007714
9: -0.0000496, 0.0008820, -0.0000619, 0.0008753, -0.0005095, 0.0005248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004523, upper bound: 0.0004303
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011126, -0.0006152, -0.0011297, -0.0006230, -0.0002712, 0.0002954
1: -0.0071339, -0.0058717, -0.0071775, -0.0058916, -0.0006881, 0.0007495
2: 0.0306041, 0.0313872, 0.0305771, 0.0313748, -0.0004269, 0.0004650
3: 0.0009546, 0.0024169, 0.0009778, 0.0024673, -0.0008683, 0.0007972
4: -0.0061494, -0.0048655, -0.0061937, -0.0048858, -0.0007000, 0.0007624
5: 0.0114089, 0.0118953, 0.0113922, 0.0118876, -0.0002651, 0.0002888
6: 0.0015997, 0.0034555, 0.0016290, 0.0035195, -0.0011020, 0.0010117
7: 0.9791786, 0.9804773, 0.9791992, 0.9805220, -0.0007711, 0.0007080
8: -0.0088880, -0.0074957, -0.0088660, -0.0074477, -0.0008267, 0.0007590
9: -0.0000483, 0.0008714, -0.0000800, 0.0008569, -0.0005014, 0.0005461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004310
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004523, upper bound: 0.0004318
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011095, -0.0006095, -0.0011142, -0.0006031, -0.0002744, 0.0002872
1: -0.0071260, -0.0058573, -0.0071380, -0.0058411, -0.0006964, 0.0007288
2: 0.0306090, 0.0313961, 0.0306016, 0.0314062, -0.0004320, 0.0004521
3: 0.0009380, 0.0024077, 0.0009192, 0.0024216, -0.0008443, 0.0008067
4: -0.0061413, -0.0048509, -0.0061536, -0.0048344, -0.0007083, 0.0007413
5: 0.0114120, 0.0119008, 0.0114074, 0.0119070, -0.0002683, 0.0002808
6: 0.0015785, 0.0034438, 0.0015547, 0.0034615, -0.0010715, 0.0010238
7: 0.9791639, 0.9804691, 0.9791472, 0.9804814, -0.0007498, 0.0007164
8: -0.0089039, -0.0075045, -0.0089217, -0.0074912, -0.0008039, 0.0007681
9: -0.0000425, 0.0008819, -0.0000512, 0.0008937, -0.0005074, 0.0005310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004377, upper bound: 0.0004294
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004602, upper bound: 0.0004302
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011087, -0.0006152, -0.0011252, -0.0006125, -0.0002729, 0.0002984
1: -0.0071240, -0.0058719, -0.0071658, -0.0058649, -0.0006925, 0.0007574
2: 0.0306102, 0.0313871, 0.0305843, 0.0313914, -0.0004296, 0.0004699
3: 0.0009548, 0.0024054, 0.0009468, 0.0024539, -0.0008774, 0.0008022
4: -0.0061394, -0.0048657, -0.0061819, -0.0048586, -0.0007044, 0.0007704
5: 0.0114127, 0.0118952, 0.0113966, 0.0118979, -0.0002668, 0.0002918
6: 0.0015999, 0.0034409, 0.0015897, 0.0035024, -0.0011135, 0.0010182
7: 0.9791788, 0.9804671, 0.9791717, 0.9805101, -0.0007792, 0.0007125
8: -0.0088878, -0.0075066, -0.0088955, -0.0074605, -0.0008354, 0.0007639
9: -0.0000411, 0.0008713, -0.0000715, 0.0008764, -0.0005046, 0.0005518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004377, upper bound: 0.0004310
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004602, upper bound: 0.0004318
time: 0.55 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011182, -0.0006058, -0.0011311, -0.0006129, -0.0002741, 0.0003056
1: -0.0071481, -0.0058479, -0.0071808, -0.0058659, -0.0006957, 0.0007755
2: 0.0305953, 0.0314020, 0.0305750, 0.0313908, -0.0004316, 0.0004811
3: 0.0009271, 0.0024333, 0.0009479, 0.0024712, -0.0008983, 0.0008059
4: -0.0061639, -0.0048413, -0.0061971, -0.0048596, -0.0007076, 0.0007888
5: 0.0114035, 0.0119044, 0.0113909, 0.0118975, -0.0002680, 0.0002988
6: 0.0015647, 0.0034763, 0.0015911, 0.0035244, -0.0011401, 0.0010228
7: 0.9791542, 0.9804918, 0.9791727, 0.9805254, -0.0007978, 0.0007157
8: -0.0089143, -0.0074800, -0.0088944, -0.0074440, -0.0008553, 0.0007674
9: -0.0000586, 0.0008888, -0.0000824, 0.0008757, -0.0005069, 0.0005650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004377, upper bound: 0.0004500
time: 0.54 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004607, upper bound: 0.0004524
time: 0.54 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011176, -0.0006116, -0.0011415, -0.0006228, -0.0002694, 0.0003190
1: -0.0071465, -0.0058627, -0.0072073, -0.0058911, -0.0006836, 0.0008095
2: 0.0305963, 0.0313928, 0.0305586, 0.0313751, -0.0004241, 0.0005022
3: 0.0009442, 0.0024315, 0.0009772, 0.0025018, -0.0009377, 0.0007919
4: -0.0061622, -0.0048564, -0.0062240, -0.0048853, -0.0006954, 0.0008234
5: 0.0114041, 0.0118987, 0.0113807, 0.0118878, -0.0002634, 0.0003119
6: 0.0015865, 0.0034740, 0.0016283, 0.0035633, -0.0011901, 0.0010051
7: 0.9791694, 0.9804902, 0.9791986, 0.9805527, -0.0008328, 0.0007033
8: -0.0088979, -0.0074818, -0.0088665, -0.0074148, -0.0008929, 0.0007540
9: -0.0000574, 0.0008780, -0.0001017, 0.0008573, -0.0004981, 0.0005898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004377, upper bound: 0.0004504
time: 0.53 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004607, upper bound: 0.0004527
time: 0.56 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011128, -0.0005957, -0.0011272, -0.0006130, -0.0002777, 0.0003061
1: -0.0071346, -0.0058221, -0.0071710, -0.0058660, -0.0007047, 0.0007768
2: 0.0306037, 0.0314179, 0.0305811, 0.0313907, -0.0004372, 0.0004820
3: 0.0008973, 0.0024176, 0.0009481, 0.0024598, -0.0008999, 0.0008164
4: -0.0061501, -0.0048151, -0.0061871, -0.0048598, -0.0007168, 0.0007902
5: 0.0114087, 0.0119143, 0.0113947, 0.0118974, -0.0002715, 0.0002993
6: 0.0015269, 0.0034564, 0.0015914, 0.0035100, -0.0011421, 0.0010361
7: 0.9791277, 0.9804779, 0.9791729, 0.9805154, -0.0007992, 0.0007250
8: -0.0089426, -0.0074950, -0.0088942, -0.0074548, -0.0008569, 0.0007773
9: -0.0000487, 0.0009075, -0.0000753, 0.0008755, -0.0005135, 0.0005660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004589, upper bound: 0.0004352
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004607, upper bound: 0.0004599
time: 0.52 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011121, -0.0006011, -0.0011379, -0.0006229, -0.0002731, 0.0003244
1: -0.0071327, -0.0058361, -0.0071981, -0.0058913, -0.0006930, 0.0008232
2: 0.0306048, 0.0314093, 0.0305643, 0.0313750, -0.0004299, 0.0005107
3: 0.0009134, 0.0024155, 0.0009774, 0.0024913, -0.0009536, 0.0008028
4: -0.0061482, -0.0048293, -0.0062147, -0.0048855, -0.0007049, 0.0008373
5: 0.0114094, 0.0119090, 0.0113842, 0.0118877, -0.0002670, 0.0003171
6: 0.0015473, 0.0034537, 0.0016285, 0.0035499, -0.0012102, 0.0010189
7: 0.9791420, 0.9804760, 0.9791988, 0.9805433, -0.0008469, 0.0007130
8: -0.0089273, -0.0074970, -0.0088664, -0.0074249, -0.0009080, 0.0007644
9: -0.0000474, 0.0008974, -0.0000950, 0.0008571, -0.0005049, 0.0005998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004377, upper bound: 0.0004589
time: 0.53 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004607, upper bound: 0.0004607
time: 0.54 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.88 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004523, upper bound: 0.0004303
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004310
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004523, upper bound: 0.0004318
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004377, upper bound: 0.0004294
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004602, upper bound: 0.0004302
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004377, upper bound: 0.0004310
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004602, upper bound: 0.0004318
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004377, upper bound: 0.0004500
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004607, upper bound: 0.0004524
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004377, upper bound: 0.0004504
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004607, upper bound: 0.0004527
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004589, upper bound: 0.0004352
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004607, upper bound: 0.0004599
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004377, upper bound: 0.0004589
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0004607, upper bound: 0.0004607

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011051, -0.0006099, -0.0011010, -0.0006155, -0.0002640, 0.0002638
1: -0.0071149, -0.0058583, -0.0071044, -0.0058726, -0.0006700, 0.0006695
2: 0.0306159, 0.0313955, 0.0306224, 0.0313867, -0.0004157, 0.0004154
3: 0.0009391, 0.0023949, 0.0009557, 0.0023827, -0.0007756, 0.0007762
4: -0.0061301, -0.0048519, -0.0061194, -0.0048664, -0.0006815, 0.0006810
5: 0.0114163, 0.0119004, 0.0114203, 0.0118949, -0.0002581, 0.0002579
6: 0.0015800, 0.0034276, 0.0016010, 0.0034121, -0.0009843, 0.0009851
7: 0.9791649, 0.9804577, 0.9791796, 0.9804469, -0.0006888, 0.0006893
8: -0.0089028, -0.0075167, -0.0088870, -0.0075283, -0.0007385, 0.0007390
9: -0.0000344, 0.0008812, -0.0000268, 0.0008708, -0.0004882, 0.0004878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004294
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004294
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011124, -0.0006095, -0.0011162, -0.0006134, -0.0002746, 0.0002643
1: -0.0071334, -0.0058573, -0.0071430, -0.0058671, -0.0006968, 0.0006707
2: 0.0306044, 0.0313961, 0.0305985, 0.0313900, -0.0004323, 0.0004161
3: 0.0009380, 0.0024163, 0.0009494, 0.0024274, -0.0007769, 0.0008072
4: -0.0061489, -0.0048509, -0.0061586, -0.0048609, -0.0007087, 0.0006822
5: 0.0114091, 0.0119008, 0.0114055, 0.0118970, -0.0002684, 0.0002584
6: 0.0015786, 0.0034548, 0.0015930, 0.0034688, -0.0009860, 0.0010244
7: 0.9791639, 0.9804767, 0.9791740, 0.9804866, -0.0006900, 0.0007168
8: -0.0089038, -0.0074962, -0.0088930, -0.0074857, -0.0007398, 0.0007685
9: -0.0000479, 0.0008819, -0.0000549, 0.0008747, -0.0005077, 0.0004887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004303
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004303
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011044, -0.0006156, -0.0011108, -0.0006263, -0.0002612, 0.0002778
1: -0.0071131, -0.0058729, -0.0071294, -0.0058998, -0.0006627, 0.0007050
2: 0.0306171, 0.0313865, 0.0306069, 0.0313698, -0.0004112, 0.0004374
3: 0.0009560, 0.0023927, 0.0009872, 0.0024117, -0.0008167, 0.0007678
4: -0.0061282, -0.0048667, -0.0061449, -0.0048941, -0.0006741, 0.0007171
5: 0.0114170, 0.0118948, 0.0114107, 0.0118844, -0.0002553, 0.0002716
6: 0.0016014, 0.0034248, 0.0016410, 0.0034489, -0.0010366, 0.0009744
7: 0.9791799, 0.9804558, 0.9792075, 0.9804726, -0.0007253, 0.0006818
8: -0.0088867, -0.0075187, -0.0088570, -0.0075007, -0.0007777, 0.0007310
9: -0.0000331, 0.0008706, -0.0000450, 0.0008510, -0.0004829, 0.0005137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004310
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004310
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011116, -0.0006153, -0.0011262, -0.0006233, -0.0002702, 0.0002803
1: -0.0071315, -0.0058719, -0.0071686, -0.0058924, -0.0006856, 0.0007113
2: 0.0306056, 0.0313871, 0.0305826, 0.0313743, -0.0004253, 0.0004413
3: 0.0009549, 0.0024141, 0.0009787, 0.0024570, -0.0008240, 0.0007942
4: -0.0061470, -0.0048657, -0.0061847, -0.0048866, -0.0006973, 0.0007235
5: 0.0114099, 0.0118952, 0.0113956, 0.0118873, -0.0002641, 0.0002740
6: 0.0016000, 0.0034520, 0.0016302, 0.0035064, -0.0010458, 0.0010079
7: 0.9791789, 0.9804748, 0.9792000, 0.9805129, -0.0007318, 0.0007053
8: -0.0088878, -0.0074983, -0.0088651, -0.0074575, -0.0007846, 0.0007562
9: -0.0000465, 0.0008713, -0.0000735, 0.0008563, -0.0004995, 0.0005183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004318
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004318
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011012, -0.0006100, -0.0010956, -0.0006069, -0.0002675, 0.0002672
1: -0.0071051, -0.0058585, -0.0070908, -0.0058506, -0.0006788, 0.0006782
2: 0.0306220, 0.0313954, 0.0306309, 0.0314003, -0.0004211, 0.0004207
3: 0.0009394, 0.0023835, 0.0009303, 0.0023669, -0.0007856, 0.0007863
4: -0.0061201, -0.0048521, -0.0061055, -0.0048441, -0.0006904, 0.0006898
5: 0.0114201, 0.0119003, 0.0114256, 0.0119034, -0.0002615, 0.0002613
6: 0.0015803, 0.0034131, 0.0015688, 0.0033920, -0.0009971, 0.0009980
7: 0.9791651, 0.9804476, 0.9791570, 0.9804328, -0.0006977, 0.0006983
8: -0.0089025, -0.0075275, -0.0089112, -0.0075433, -0.0007480, 0.0007487
9: -0.0000273, 0.0008810, -0.0000168, 0.0008868, -0.0004946, 0.0004941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004294
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004294
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011085, -0.0006096, -0.0011104, -0.0006034, -0.0002733, 0.0002675
1: -0.0071236, -0.0058575, -0.0071285, -0.0058418, -0.0006935, 0.0006788
2: 0.0306105, 0.0313960, 0.0306075, 0.0314057, -0.0004302, 0.0004211
3: 0.0009382, 0.0024049, 0.0009201, 0.0024105, -0.0007863, 0.0008034
4: -0.0061389, -0.0048511, -0.0061439, -0.0048352, -0.0007054, 0.0006904
5: 0.0114129, 0.0119007, 0.0114111, 0.0119067, -0.0002672, 0.0002615
6: 0.0015788, 0.0034403, 0.0015558, 0.0034474, -0.0009980, 0.0010196
7: 0.9791641, 0.9804665, 0.9791480, 0.9804716, -0.0006983, 0.0007135
8: -0.0089036, -0.0075071, -0.0089209, -0.0075017, -0.0007487, 0.0007649
9: -0.0000407, 0.0008818, -0.0000443, 0.0008932, -0.0005053, 0.0004946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004481, upper bound: 0.0004303
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004481, upper bound: 0.0004302
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011005, -0.0006157, -0.0011061, -0.0006139, -0.0002651, 0.0002810
1: -0.0071032, -0.0058730, -0.0071175, -0.0058685, -0.0006728, 0.0007130
2: 0.0306232, 0.0313864, 0.0306143, 0.0313892, -0.0004174, 0.0004423
3: 0.0009562, 0.0023813, 0.0009509, 0.0023979, -0.0008259, 0.0007794
4: -0.0061182, -0.0048669, -0.0061327, -0.0048622, -0.0006844, 0.0007252
5: 0.0114208, 0.0118947, 0.0114153, 0.0118965, -0.0002592, 0.0002747
6: 0.0016017, 0.0034103, 0.0015949, 0.0034313, -0.0010482, 0.0009892
7: 0.9791800, 0.9804456, 0.9791753, 0.9804603, -0.0007335, 0.0006922
8: -0.0088865, -0.0075296, -0.0088916, -0.0075138, -0.0007864, 0.0007421
9: -0.0000259, 0.0008704, -0.0000363, 0.0008738, -0.0004902, 0.0005195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004310
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004310
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011078, -0.0006153, -0.0011216, -0.0006129, -0.0002717, 0.0002831
1: -0.0071217, -0.0058721, -0.0071569, -0.0058658, -0.0006896, 0.0007185
2: 0.0306117, 0.0313870, 0.0305898, 0.0313908, -0.0004278, 0.0004458
3: 0.0009551, 0.0024027, 0.0009478, 0.0024435, -0.0008323, 0.0007988
4: -0.0061370, -0.0048659, -0.0061728, -0.0048595, -0.0007014, 0.0007308
5: 0.0114137, 0.0118951, 0.0114001, 0.0118975, -0.0002657, 0.0002768
6: 0.0016003, 0.0034374, 0.0015911, 0.0034893, -0.0010563, 0.0010138
7: 0.9791790, 0.9804646, 0.9791725, 0.9805009, -0.0007392, 0.0007094
8: -0.0088876, -0.0075092, -0.0088945, -0.0074704, -0.0007925, 0.0007606
9: -0.0000393, 0.0008711, -0.0000650, 0.0008757, -0.0005024, 0.0005235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004481, upper bound: 0.0004318
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004481, upper bound: 0.0004318
time: 0.57 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011099, -0.0006062, -0.0011123, -0.0006153, -0.0002608, 0.0002859
1: -0.0071272, -0.0058489, -0.0071331, -0.0058719, -0.0006619, 0.0007255
2: 0.0306083, 0.0314013, 0.0306046, 0.0313870, -0.0004106, 0.0004501
3: 0.0009283, 0.0024091, 0.0009549, 0.0024160, -0.0008404, 0.0007668
4: -0.0061426, -0.0048424, -0.0061486, -0.0048658, -0.0006733, 0.0007379
5: 0.0114115, 0.0119040, 0.0114092, 0.0118952, -0.0002550, 0.0002795
6: 0.0015662, 0.0034455, 0.0016001, 0.0034543, -0.0010666, 0.0009731
7: 0.9791552, 0.9804703, 0.9791789, 0.9804764, -0.0007463, 0.0006810
8: -0.0089131, -0.0075032, -0.0088877, -0.0074966, -0.0008002, 0.0007301
9: -0.0000433, 0.0008880, -0.0000477, 0.0008712, -0.0004823, 0.0005286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004499
time: 0.58 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004499
time: 0.54 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011172, -0.0006059, -0.0011273, -0.0006132, -0.0002732, 0.0002883
1: -0.0071457, -0.0058481, -0.0071712, -0.0058666, -0.0006932, 0.0007316
2: 0.0305968, 0.0314019, 0.0305810, 0.0313904, -0.0004301, 0.0004539
3: 0.0009273, 0.0024305, 0.0009488, 0.0024601, -0.0008475, 0.0008030
4: -0.0061614, -0.0048415, -0.0061873, -0.0048603, -0.0007051, 0.0007442
5: 0.0114044, 0.0119044, 0.0113946, 0.0118972, -0.0002671, 0.0002819
6: 0.0015650, 0.0034728, 0.0015922, 0.0035102, -0.0010756, 0.0010192
7: 0.9791544, 0.9804894, 0.9791735, 0.9805155, -0.0007527, 0.0007132
8: -0.0089140, -0.0074827, -0.0088936, -0.0074546, -0.0008070, 0.0007646
9: -0.0000568, 0.0008886, -0.0000754, 0.0008751, -0.0005051, 0.0005331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004520
time: 0.55 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004524
time: 0.56 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011093, -0.0006120, -0.0011225, -0.0006260, -0.0002576, 0.0003017
1: -0.0071256, -0.0058637, -0.0071590, -0.0058991, -0.0006536, 0.0007656
2: 0.0306093, 0.0313921, 0.0305885, 0.0313702, -0.0004055, 0.0004750
3: 0.0009454, 0.0024072, 0.0009864, 0.0024460, -0.0008869, 0.0007572
4: -0.0061409, -0.0048574, -0.0061750, -0.0048934, -0.0006648, 0.0007788
5: 0.0114122, 0.0118983, 0.0113993, 0.0118847, -0.0002518, 0.0002950
6: 0.0015880, 0.0034432, 0.0016399, 0.0034924, -0.0011256, 0.0009610
7: 0.9791705, 0.9804686, 0.9792069, 0.9805031, -0.0007877, 0.0006724
8: -0.0088968, -0.0075049, -0.0088578, -0.0074680, -0.0008445, 0.0007210
9: -0.0000422, 0.0008772, -0.0000666, 0.0008515, -0.0004762, 0.0005578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004504
time: 0.56 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004504
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011166, -0.0006117, -0.0011377, -0.0006231, -0.0002684, 0.0003057
1: -0.0071441, -0.0058629, -0.0071976, -0.0058919, -0.0006811, 0.0007758
2: 0.0305978, 0.0313927, 0.0305646, 0.0313747, -0.0004225, 0.0004813
3: 0.0009444, 0.0024287, 0.0009781, 0.0024907, -0.0008987, 0.0007890
4: -0.0061598, -0.0048565, -0.0062142, -0.0048861, -0.0006928, 0.0007891
5: 0.0114050, 0.0118987, 0.0113844, 0.0118875, -0.0002624, 0.0002989
6: 0.0015867, 0.0034704, 0.0016294, 0.0035491, -0.0011406, 0.0010013
7: 0.9791696, 0.9804877, 0.9791994, 0.9805428, -0.0007981, 0.0007007
8: -0.0088977, -0.0074845, -0.0088657, -0.0074255, -0.0008557, 0.0007512
9: -0.0000557, 0.0008779, -0.0000947, 0.0008567, -0.0004962, 0.0005653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004523
time: 0.66 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004527
time: 0.60 seconds

## BFS IS instance: IS_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010943, -0.0005993, -0.0011190, -0.0006134, -0.0002577, 0.0002985
1: -0.0070876, -0.0058314, -0.0071503, -0.0058671, -0.0006539, 0.0007575
2: 0.0306329, 0.0314122, 0.0305939, 0.0313901, -0.0004057, 0.0004700
3: 0.0009080, 0.0023632, 0.0009493, 0.0024359, -0.0008775, 0.0007575
4: -0.0061023, -0.0048245, -0.0061661, -0.0048608, -0.0006651, 0.0007705
5: 0.0114268, 0.0119108, 0.0114026, 0.0118970, -0.0002519, 0.0002918
6: 0.0015405, 0.0033873, 0.0015929, 0.0034795, -0.0011137, 0.0009614
7: 0.9791372, 0.9804295, 0.9791740, 0.9804941, -0.0007793, 0.0006727
8: -0.0089324, -0.0075468, -0.0088931, -0.0074776, -0.0008355, 0.0007213
9: -0.0000145, 0.0009008, -0.0000602, 0.0008748, -0.0004764, 0.0005519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004310, upper bound: 0.0004352
time: 0.57 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004310, upper bound: 0.0004352
time: 0.60 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011089, -0.0005960, -0.0011262, -0.0006130, -0.0002572, 0.0003049
1: -0.0071246, -0.0058230, -0.0071686, -0.0058662, -0.0006527, 0.0007738
2: 0.0306099, 0.0314174, 0.0305826, 0.0313906, -0.0004050, 0.0004801
3: 0.0008982, 0.0024061, 0.0009483, 0.0024570, -0.0008964, 0.0007562
4: -0.0061399, -0.0048160, -0.0061847, -0.0048600, -0.0006640, 0.0007871
5: 0.0114125, 0.0119140, 0.0113956, 0.0118974, -0.0002515, 0.0002981
6: 0.0015281, 0.0034418, 0.0015917, 0.0035064, -0.0011377, 0.0009597
7: 0.9791286, 0.9804677, 0.9791731, 0.9805129, -0.0007961, 0.0006715
8: -0.0089417, -0.0075060, -0.0088940, -0.0074575, -0.0008535, 0.0007200
9: -0.0000415, 0.0009069, -0.0000735, 0.0008754, -0.0004756, 0.0005638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004594
time: 0.62 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004599
time: 0.60 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011041, -0.0006016, -0.0011189, -0.0006261, -0.0002613, 0.0003073
1: -0.0071123, -0.0058372, -0.0071499, -0.0058993, -0.0006630, 0.0007799
2: 0.0306175, 0.0314086, 0.0305942, 0.0313701, -0.0004113, 0.0004839
3: 0.0009147, 0.0023918, 0.0009866, 0.0024354, -0.0009035, 0.0007681
4: -0.0061274, -0.0048304, -0.0061656, -0.0048936, -0.0006744, 0.0007933
5: 0.0114173, 0.0119085, 0.0114028, 0.0118846, -0.0002554, 0.0003005
6: 0.0015490, 0.0034237, 0.0016403, 0.0034789, -0.0011466, 0.0009748
7: 0.9791431, 0.9804550, 0.9792070, 0.9804936, -0.0008024, 0.0006821
8: -0.0089261, -0.0075196, -0.0088576, -0.0074781, -0.0008603, 0.0007313
9: -0.0000325, 0.0008966, -0.0000599, 0.0008513, -0.0004831, 0.0005683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004589
time: 0.56 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004589
time: 0.55 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011111, -0.0006012, -0.0011341, -0.0006232, -0.0002721, 0.0003127
1: -0.0071302, -0.0058363, -0.0071885, -0.0058921, -0.0006904, 0.0007934
2: 0.0306064, 0.0314092, 0.0305702, 0.0313745, -0.0004283, 0.0004922
3: 0.0009136, 0.0024126, 0.0009783, 0.0024801, -0.0009191, 0.0007998
4: -0.0061457, -0.0048295, -0.0062050, -0.0048863, -0.0007023, 0.0008070
5: 0.0114104, 0.0119089, 0.0113879, 0.0118874, -0.0002660, 0.0003057
6: 0.0015476, 0.0034500, 0.0016297, 0.0035357, -0.0011665, 0.0010151
7: 0.9791422, 0.9804734, 0.9791996, 0.9805334, -0.0008162, 0.0007103
8: -0.0089270, -0.0074998, -0.0088655, -0.0074355, -0.0008751, 0.0007616
9: -0.0000456, 0.0008972, -0.0000880, 0.0008566, -0.0005030, 0.0005781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004602
time: 0.61 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004607
time: 0.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.39 seconds
IS_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004294
IS_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004294
IS_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004303
IS_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004303
IS_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004310
IS_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004310
IS_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004318
IS_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004318
IS_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004294
IS_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004294
IS_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004481, upper bound: 0.0004303
IS_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004481, upper bound: 0.0004302
IS_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004310
IS_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004310
IS_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004481, upper bound: 0.0004318
IS_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004481, upper bound: 0.0004318
IS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004499
IS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004499
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004520
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004524
IS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004504
IS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004504
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004523
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004527
IS_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004310, upper bound: 0.0004352
IS_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004310, upper bound: 0.0004352
IS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004594
IS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004599
IS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004589
IS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004589
IS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004602
IS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004607

## BFS IS instance: IS_A1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011051, -0.0006099, -0.0010872, -0.0006198, -0.0002537, 0.0002495
1: -0.0071149, -0.0058583, -0.0070694, -0.0058834, -0.0006438, 0.0006331
2: 0.0306159, 0.0313955, 0.0306441, 0.0313800, -0.0003994, 0.0003928
3: 0.0009391, 0.0023949, 0.0009682, 0.0023421, -0.0007335, 0.0007459
4: -0.0061301, -0.0048519, -0.0060838, -0.0048774, -0.0006549, 0.0006440
5: 0.0114163, 0.0119004, 0.0114338, 0.0118908, -0.0002481, 0.0002439
6: 0.0015800, 0.0034276, 0.0016169, 0.0033606, -0.0009309, 0.0009466
7: 0.9791649, 0.9804577, 0.9791906, 0.9804108, -0.0006514, 0.0006624
8: -0.0089028, -0.0075167, -0.0088751, -0.0075669, -0.0006984, 0.0007102
9: -0.0000344, 0.0008812, -0.0000012, 0.0008629, -0.0004691, 0.0004613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004270
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004294
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011051, -0.0006099, -0.0010986, -0.0006156, -0.0002640, 0.0002670
1: -0.0071149, -0.0058583, -0.0070983, -0.0058728, -0.0006698, 0.0006775
2: 0.0306159, 0.0313955, 0.0306262, 0.0313865, -0.0004156, 0.0004203
3: 0.0009391, 0.0023949, 0.0009560, 0.0023756, -0.0007848, 0.0007760
4: -0.0061301, -0.0048519, -0.0061132, -0.0048667, -0.0006813, 0.0006891
5: 0.0114163, 0.0119004, 0.0114227, 0.0118948, -0.0002581, 0.0002610
6: 0.0015800, 0.0034276, 0.0016014, 0.0034031, -0.0009961, 0.0009848
7: 0.9791649, 0.9804577, 0.9791798, 0.9804407, -0.0006970, 0.0006891
8: -0.0089028, -0.0075167, -0.0088867, -0.0075350, -0.0007473, 0.0007389
9: -0.0000344, 0.0008812, -0.0000223, 0.0008706, -0.0004881, 0.0004936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004270
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004294
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011124, -0.0006095, -0.0011023, -0.0006172, -0.0002648, 0.0002503
1: -0.0071334, -0.0058573, -0.0071078, -0.0058769, -0.0006720, 0.0006353
2: 0.0306044, 0.0313961, 0.0306203, 0.0313840, -0.0004169, 0.0003941
3: 0.0009380, 0.0024163, 0.0009606, 0.0023866, -0.0007360, 0.0007785
4: -0.0061489, -0.0048509, -0.0061229, -0.0048708, -0.0006836, 0.0006462
5: 0.0114091, 0.0119008, 0.0114190, 0.0118933, -0.0002589, 0.0002448
6: 0.0015786, 0.0034548, 0.0016073, 0.0034171, -0.0009340, 0.0009880
7: 0.9791639, 0.9804767, 0.9791840, 0.9804503, -0.0006536, 0.0006914
8: -0.0089038, -0.0074962, -0.0088823, -0.0075245, -0.0007008, 0.0007413
9: -0.0000479, 0.0008819, -0.0000292, 0.0008677, -0.0004896, 0.0004629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004285
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004303
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011124, -0.0006095, -0.0011135, -0.0006135, -0.0002745, 0.0002679
1: -0.0071334, -0.0058573, -0.0071362, -0.0058673, -0.0006966, 0.0006798
2: 0.0306044, 0.0313961, 0.0306027, 0.0313899, -0.0004322, 0.0004218
3: 0.0009380, 0.0024163, 0.0009496, 0.0024195, -0.0007875, 0.0008070
4: -0.0061489, -0.0048509, -0.0061517, -0.0048611, -0.0007086, 0.0006915
5: 0.0114091, 0.0119008, 0.0114081, 0.0118969, -0.0002684, 0.0002619
6: 0.0015786, 0.0034548, 0.0015933, 0.0034587, -0.0009995, 0.0010242
7: 0.9791639, 0.9804767, 0.9791742, 0.9804795, -0.0006994, 0.0007167
8: -0.0089038, -0.0074962, -0.0088928, -0.0074933, -0.0007498, 0.0007684
9: -0.0000479, 0.0008819, -0.0000499, 0.0008746, -0.0005076, 0.0004953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004285
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004303
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011044, -0.0006156, -0.0010970, -0.0006306, -0.0002502, 0.0002637
1: -0.0071131, -0.0058729, -0.0070944, -0.0059108, -0.0006350, 0.0006693
2: 0.0306171, 0.0313865, 0.0306286, 0.0313630, -0.0003939, 0.0004152
3: 0.0009560, 0.0023927, 0.0009999, 0.0023711, -0.0007753, 0.0007356
4: -0.0061282, -0.0048667, -0.0061092, -0.0049053, -0.0006459, 0.0006807
5: 0.0114170, 0.0118948, 0.0114242, 0.0118802, -0.0002446, 0.0002578
6: 0.0016014, 0.0034248, 0.0016571, 0.0033973, -0.0009840, 0.0009335
7: 0.9791799, 0.9804558, 0.9792188, 0.9804365, -0.0006885, 0.0006532
8: -0.0088867, -0.0075187, -0.0088449, -0.0075393, -0.0007382, 0.0007004
9: -0.0000331, 0.0008706, -0.0000194, 0.0008430, -0.0004626, 0.0004876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004291
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004310
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011044, -0.0006156, -0.0011091, -0.0006264, -0.0002611, 0.0002810
1: -0.0071131, -0.0058729, -0.0071250, -0.0059001, -0.0006625, 0.0007130
2: 0.0306171, 0.0313865, 0.0306096, 0.0313696, -0.0004110, 0.0004423
3: 0.0009560, 0.0023927, 0.0009876, 0.0024066, -0.0008260, 0.0007675
4: -0.0061282, -0.0048667, -0.0061404, -0.0048944, -0.0006739, 0.0007252
5: 0.0114170, 0.0118948, 0.0114124, 0.0118843, -0.0002553, 0.0002747
6: 0.0016014, 0.0034248, 0.0016415, 0.0034424, -0.0010483, 0.0009741
7: 0.9791799, 0.9804558, 0.9792079, 0.9804681, -0.0007335, 0.0006816
8: -0.0088867, -0.0075187, -0.0088567, -0.0075055, -0.0007864, 0.0007308
9: -0.0000331, 0.0008706, -0.0000418, 0.0008507, -0.0004827, 0.0005195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004291
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004310
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011116, -0.0006153, -0.0011124, -0.0006272, -0.0002603, 0.0002663
1: -0.0071315, -0.0058719, -0.0071334, -0.0059021, -0.0006606, 0.0006757
2: 0.0306056, 0.0313871, 0.0306044, 0.0313683, -0.0004098, 0.0004192
3: 0.0009549, 0.0024141, 0.0009899, 0.0024163, -0.0007828, 0.0007653
4: -0.0061470, -0.0048657, -0.0061489, -0.0048964, -0.0006720, 0.0006873
5: 0.0114099, 0.0118952, 0.0114091, 0.0118835, -0.0002545, 0.0002603
6: 0.0016000, 0.0034520, 0.0016444, 0.0034547, -0.0009935, 0.0009712
7: 0.9791789, 0.9804748, 0.9792099, 0.9804767, -0.0006952, 0.0006796
8: -0.0088878, -0.0074983, -0.0088545, -0.0074963, -0.0007453, 0.0007287
9: -0.0000465, 0.0008713, -0.0000479, 0.0008493, -0.0004813, 0.0004923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004302
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004318
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011116, -0.0006153, -0.0011243, -0.0006234, -0.0002701, 0.0002836
1: -0.0071315, -0.0058719, -0.0071635, -0.0058926, -0.0006854, 0.0007196
2: 0.0306056, 0.0313871, 0.0305857, 0.0313742, -0.0004252, 0.0004464
3: 0.0009549, 0.0024141, 0.0009789, 0.0024512, -0.0008336, 0.0007940
4: -0.0061470, -0.0048657, -0.0061795, -0.0048868, -0.0006972, 0.0007319
5: 0.0114099, 0.0118952, 0.0113975, 0.0118872, -0.0002641, 0.0002772
6: 0.0016000, 0.0034520, 0.0016304, 0.0034990, -0.0010580, 0.0010077
7: 0.9791789, 0.9804748, 0.9792001, 0.9805077, -0.0007403, 0.0007052
8: -0.0088878, -0.0074983, -0.0088649, -0.0074630, -0.0007937, 0.0007560
9: -0.0000465, 0.0008713, -0.0000698, 0.0008562, -0.0004994, 0.0005243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004302
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004318
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011012, -0.0006100, -0.0010815, -0.0006111, -0.0002550, 0.0002530
1: -0.0071051, -0.0058585, -0.0070551, -0.0058614, -0.0006471, 0.0006419
2: 0.0306220, 0.0313954, 0.0306530, 0.0313936, -0.0004014, 0.0003983
3: 0.0009394, 0.0023835, 0.0009428, 0.0023256, -0.0007436, 0.0007496
4: -0.0061201, -0.0048521, -0.0060693, -0.0048551, -0.0006582, 0.0006529
5: 0.0114201, 0.0119003, 0.0114393, 0.0118992, -0.0002493, 0.0002473
6: 0.0015803, 0.0034131, 0.0015846, 0.0033396, -0.0009438, 0.0009513
7: 0.9791651, 0.9804476, 0.9791682, 0.9803962, -0.0006604, 0.0006657
8: -0.0089025, -0.0075275, -0.0088993, -0.0075826, -0.0007081, 0.0007137
9: -0.0000273, 0.0008810, 0.0000091, 0.0008789, -0.0004715, 0.0004677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004294
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004294
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011012, -0.0006100, -0.0010934, -0.0006070, -0.0002674, 0.0002702
1: -0.0071051, -0.0058585, -0.0070852, -0.0058509, -0.0006786, 0.0006858
2: 0.0306220, 0.0313954, 0.0306343, 0.0314001, -0.0004210, 0.0004255
3: 0.0009394, 0.0023835, 0.0009306, 0.0023604, -0.0007945, 0.0007861
4: -0.0061201, -0.0048521, -0.0060998, -0.0048444, -0.0006903, 0.0006976
5: 0.0114201, 0.0119003, 0.0114277, 0.0119033, -0.0002615, 0.0002642
6: 0.0015803, 0.0034131, 0.0015692, 0.0033838, -0.0010083, 0.0009977
7: 0.9791651, 0.9804476, 0.9791573, 0.9804271, -0.0007055, 0.0006982
8: -0.0089025, -0.0075275, -0.0089109, -0.0075495, -0.0007564, 0.0007485
9: -0.0000273, 0.0008810, -0.0000127, 0.0008866, -0.0004945, 0.0004997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004294
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004294
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011085, -0.0006096, -0.0010966, -0.0006080, -0.0002619, 0.0002536
1: -0.0071236, -0.0058575, -0.0070932, -0.0058534, -0.0006647, 0.0006436
2: 0.0306105, 0.0313960, 0.0306294, 0.0313985, -0.0004124, 0.0003993
3: 0.0009382, 0.0024049, 0.0009335, 0.0023698, -0.0007456, 0.0007700
4: -0.0061389, -0.0048511, -0.0061080, -0.0048469, -0.0006761, 0.0006547
5: 0.0114129, 0.0119007, 0.0114246, 0.0119023, -0.0002561, 0.0002480
6: 0.0015788, 0.0034403, 0.0015728, 0.0033956, -0.0009463, 0.0009772
7: 0.9791641, 0.9804665, 0.9791599, 0.9804354, -0.0006621, 0.0006838
8: -0.0089036, -0.0075071, -0.0089082, -0.0075406, -0.0007099, 0.0007332
9: -0.0000407, 0.0008818, -0.0000186, 0.0008847, -0.0004843, 0.0004689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004474, upper bound: 0.0004303
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004474, upper bound: 0.0004302
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011085, -0.0006096, -0.0011080, -0.0006035, -0.0002732, 0.0002709
1: -0.0071236, -0.0058575, -0.0071223, -0.0058420, -0.0006934, 0.0006874
2: 0.0306105, 0.0313960, 0.0306113, 0.0314056, -0.0004302, 0.0004265
3: 0.0009382, 0.0024049, 0.0009203, 0.0024035, -0.0007964, 0.0008033
4: -0.0061389, -0.0048511, -0.0061376, -0.0048353, -0.0007053, 0.0006992
5: 0.0114129, 0.0119007, 0.0114134, 0.0119067, -0.0002671, 0.0002648
6: 0.0015788, 0.0034403, 0.0015561, 0.0034385, -0.0010107, 0.0010194
7: 0.9791641, 0.9804665, 0.9791481, 0.9804654, -0.0007072, 0.0007134
8: -0.0089036, -0.0075071, -0.0089207, -0.0075085, -0.0007583, 0.0007648
9: -0.0000407, 0.0008818, -0.0000398, 0.0008930, -0.0005052, 0.0005009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004474, upper bound: 0.0004303
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004474, upper bound: 0.0004303
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011005, -0.0006157, -0.0010918, -0.0006198, -0.0002530, 0.0002668
1: -0.0071032, -0.0058730, -0.0070813, -0.0058833, -0.0006421, 0.0006771
2: 0.0306232, 0.0313864, 0.0306368, 0.0313800, -0.0003984, 0.0004200
3: 0.0009562, 0.0023813, 0.0009681, 0.0023559, -0.0007843, 0.0007439
4: -0.0061182, -0.0048669, -0.0060959, -0.0048774, -0.0006532, 0.0006887
5: 0.0114208, 0.0118947, 0.0114292, 0.0118908, -0.0002474, 0.0002609
6: 0.0016017, 0.0034103, 0.0016168, 0.0033781, -0.0009954, 0.0009441
7: 0.9791800, 0.9804456, 0.9791906, 0.9804231, -0.0006965, 0.0006606
8: -0.0088865, -0.0075296, -0.0088751, -0.0075538, -0.0007468, 0.0007083
9: -0.0000259, 0.0008704, -0.0000099, 0.0008629, -0.0004679, 0.0004933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004276
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004310
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011005, -0.0006157, -0.0011042, -0.0006140, -0.0002651, 0.0002840
1: -0.0071032, -0.0058730, -0.0071128, -0.0058687, -0.0006726, 0.0007206
2: 0.0306232, 0.0313864, 0.0306172, 0.0313890, -0.0004173, 0.0004471
3: 0.0009562, 0.0023813, 0.0009512, 0.0023924, -0.0008348, 0.0007792
4: -0.0061182, -0.0048669, -0.0061279, -0.0048625, -0.0006842, 0.0007330
5: 0.0114208, 0.0118947, 0.0114171, 0.0118964, -0.0002591, 0.0002776
6: 0.0016017, 0.0034103, 0.0015953, 0.0034244, -0.0010595, 0.0009889
7: 0.9791800, 0.9804456, 0.9791756, 0.9804555, -0.0007414, 0.0006920
8: -0.0088865, -0.0075296, -0.0088913, -0.0075190, -0.0007949, 0.0007419
9: -0.0000259, 0.0008704, -0.0000328, 0.0008736, -0.0004901, 0.0005251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004276
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004310
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011078, -0.0006153, -0.0011072, -0.0006180, -0.0002598, 0.0002691
1: -0.0071217, -0.0058721, -0.0071203, -0.0058789, -0.0006593, 0.0006829
2: 0.0306117, 0.0313870, 0.0306126, 0.0313827, -0.0004090, 0.0004237
3: 0.0009551, 0.0024027, 0.0009630, 0.0024011, -0.0007911, 0.0007637
4: -0.0061370, -0.0048659, -0.0061356, -0.0048729, -0.0006706, 0.0006946
5: 0.0114137, 0.0118951, 0.0114142, 0.0118925, -0.0002540, 0.0002631
6: 0.0016003, 0.0034374, 0.0016103, 0.0034355, -0.0010040, 0.0009693
7: 0.9791790, 0.9804646, 0.9791861, 0.9804633, -0.0007026, 0.0006782
8: -0.0088876, -0.0075092, -0.0088800, -0.0075107, -0.0007533, 0.0007272
9: -0.0000393, 0.0008711, -0.0000383, 0.0008662, -0.0004803, 0.0004976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004102
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004318
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011078, -0.0006153, -0.0011196, -0.0006130, -0.0002717, 0.0002863
1: -0.0071217, -0.0058721, -0.0071518, -0.0058660, -0.0006894, 0.0007266
2: 0.0306117, 0.0313870, 0.0305930, 0.0313907, -0.0004277, 0.0004508
3: 0.0009551, 0.0024027, 0.0009481, 0.0024376, -0.0008418, 0.0007987
4: -0.0061370, -0.0048659, -0.0061676, -0.0048598, -0.0007013, 0.0007391
5: 0.0114137, 0.0118951, 0.0114021, 0.0118974, -0.0002656, 0.0002800
6: 0.0016003, 0.0034374, 0.0015914, 0.0034818, -0.0010683, 0.0010136
7: 0.9791790, 0.9804646, 0.9791729, 0.9804957, -0.0007476, 0.0007093
8: -0.0088876, -0.0075092, -0.0088942, -0.0074760, -0.0008015, 0.0007605
9: -0.0000393, 0.0008711, -0.0000613, 0.0008755, -0.0005023, 0.0005294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004103
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004318
time: 0.58 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011099, -0.0006062, -0.0010935, -0.0006196, -0.0002671, 0.0002634
1: -0.0071272, -0.0058489, -0.0070856, -0.0058829, -0.0006778, 0.0006684
2: 0.0306083, 0.0314013, 0.0306341, 0.0313802, -0.0004205, 0.0004147
3: 0.0009283, 0.0024091, 0.0009677, 0.0023609, -0.0007743, 0.0007852
4: -0.0061426, -0.0048424, -0.0061003, -0.0048770, -0.0006895, 0.0006799
5: 0.0114115, 0.0119040, 0.0114276, 0.0118909, -0.0002611, 0.0002575
6: 0.0015662, 0.0034455, 0.0016162, 0.0033844, -0.0009827, 0.0009965
7: 0.9791552, 0.9804703, 0.9791902, 0.9804275, -0.0006876, 0.0006973
8: -0.0089131, -0.0075032, -0.0088756, -0.0075490, -0.0007372, 0.0007477
9: -0.0000433, 0.0008880, -0.0000130, 0.0008632, -0.0004939, 0.0004870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004048, upper bound: 0.0004500
time: 0.60 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004048, upper bound: 0.0004499
time: 0.58 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011099, -0.0006062, -0.0011051, -0.0006155, -0.0002607, 0.0002655
1: -0.0071272, -0.0058489, -0.0071150, -0.0058725, -0.0006615, 0.0006737
2: 0.0306083, 0.0314013, 0.0306159, 0.0313867, -0.0004104, 0.0004179
3: 0.0009283, 0.0024091, 0.0009555, 0.0023949, -0.0007804, 0.0007664
4: -0.0061426, -0.0048424, -0.0061301, -0.0048663, -0.0006729, 0.0006852
5: 0.0114115, 0.0119040, 0.0114162, 0.0118950, -0.0002549, 0.0002595
6: 0.0015662, 0.0034455, 0.0016008, 0.0034276, -0.0009905, 0.0009726
7: 0.9791552, 0.9804703, 0.9791794, 0.9804578, -0.0006931, 0.0006806
8: -0.0089131, -0.0075032, -0.0088871, -0.0075166, -0.0007431, 0.0007297
9: -0.0000433, 0.0008880, -0.0000344, 0.0008709, -0.0004820, 0.0004908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004499
time: 0.56 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004499
time: 0.58 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011172, -0.0006059, -0.0011087, -0.0006171, -0.0002785, 0.0002657
1: -0.0071457, -0.0058481, -0.0071242, -0.0058765, -0.0007067, 0.0006743
2: 0.0305968, 0.0314019, 0.0306102, 0.0313842, -0.0004385, 0.0004183
3: 0.0009273, 0.0024305, 0.0009603, 0.0024056, -0.0007811, 0.0008187
4: -0.0061614, -0.0048415, -0.0061395, -0.0048705, -0.0007189, 0.0006859
5: 0.0114044, 0.0119044, 0.0114127, 0.0118934, -0.0002723, 0.0002598
6: 0.0015650, 0.0034728, 0.0016068, 0.0034411, -0.0009913, 0.0010390
7: 0.9791544, 0.9804894, 0.9791837, 0.9804671, -0.0006937, 0.0007271
8: -0.0089140, -0.0074827, -0.0088826, -0.0075065, -0.0007437, 0.0007795
9: -0.0000568, 0.0008886, -0.0000411, 0.0008679, -0.0005149, 0.0004913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004520
time: 0.57 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004520
time: 0.57 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011172, -0.0006059, -0.0011199, -0.0006133, -0.0002731, 0.0002663
1: -0.0071457, -0.0058481, -0.0071526, -0.0058670, -0.0006929, 0.0006759
2: 0.0305968, 0.0314019, 0.0305925, 0.0313901, -0.0004299, 0.0004193
3: 0.0009273, 0.0024305, 0.0009492, 0.0024385, -0.0007830, 0.0008027
4: -0.0061614, -0.0048415, -0.0061684, -0.0048608, -0.0007048, 0.0006875
5: 0.0114044, 0.0119044, 0.0114018, 0.0118970, -0.0002670, 0.0002604
6: 0.0015650, 0.0034728, 0.0015928, 0.0034829, -0.0009937, 0.0010188
7: 0.9791544, 0.9804894, 0.9791739, 0.9804964, -0.0006953, 0.0007129
8: -0.0089140, -0.0074827, -0.0088931, -0.0074751, -0.0007455, 0.0007643
9: -0.0000568, 0.0008886, -0.0000618, 0.0008748, -0.0005049, 0.0004925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004524
time: 0.66 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004524
time: 0.72 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011093, -0.0006120, -0.0011035, -0.0006304, -0.0002635, 0.0002792
1: -0.0071256, -0.0058637, -0.0071108, -0.0059103, -0.0006688, 0.0007086
2: 0.0306093, 0.0313921, 0.0306184, 0.0313632, -0.0004149, 0.0004396
3: 0.0009454, 0.0024072, 0.0009994, 0.0023901, -0.0008208, 0.0007747
4: -0.0061409, -0.0048574, -0.0061259, -0.0049048, -0.0006803, 0.0007207
5: 0.0114122, 0.0118983, 0.0114178, 0.0118804, -0.0002577, 0.0002730
6: 0.0015880, 0.0034432, 0.0016565, 0.0034215, -0.0010418, 0.0009832
7: 0.9791705, 0.9804686, 0.9792184, 0.9804534, -0.0007290, 0.0006880
8: -0.0088968, -0.0075049, -0.0088454, -0.0075212, -0.0007816, 0.0007377
9: -0.0000422, 0.0008772, -0.0000314, 0.0008433, -0.0004873, 0.0005163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B2_B1_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004048, upper bound: 0.0004504
time: 0.60 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004048, upper bound: 0.0004504
time: 0.61 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011093, -0.0006120, -0.0011157, -0.0006262, -0.0002574, 0.0002814
1: -0.0071256, -0.0058637, -0.0071418, -0.0058996, -0.0006533, 0.0007140
2: 0.0306093, 0.0313921, 0.0305992, 0.0313699, -0.0004053, 0.0004430
3: 0.0009454, 0.0024072, 0.0009870, 0.0024261, -0.0008271, 0.0007568
4: -0.0061409, -0.0048574, -0.0061575, -0.0048940, -0.0006645, 0.0007263
5: 0.0114122, 0.0118983, 0.0114059, 0.0118845, -0.0002517, 0.0002751
6: 0.0015880, 0.0034432, 0.0016408, 0.0034671, -0.0010498, 0.0009605
7: 0.9791705, 0.9804686, 0.9792074, 0.9804854, -0.0007346, 0.0006721
8: -0.0088968, -0.0075049, -0.0088572, -0.0074870, -0.0007876, 0.0007206
9: -0.0000422, 0.0008772, -0.0000540, 0.0008511, -0.0004760, 0.0005202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B2_B1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004048, upper bound: 0.0004504
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004048, upper bound: 0.0004504
time: 0.61 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011166, -0.0006117, -0.0011188, -0.0006270, -0.0002740, 0.0002835
1: -0.0071441, -0.0058629, -0.0071497, -0.0059018, -0.0006952, 0.0007193
2: 0.0305978, 0.0313927, 0.0305943, 0.0313685, -0.0004313, 0.0004463
3: 0.0009444, 0.0024287, 0.0009895, 0.0024352, -0.0008333, 0.0008054
4: -0.0061598, -0.0048565, -0.0061655, -0.0048961, -0.0007072, 0.0007317
5: 0.0114050, 0.0118987, 0.0114029, 0.0118837, -0.0002679, 0.0002771
6: 0.0015867, 0.0034704, 0.0016439, 0.0034787, -0.0010576, 0.0010221
7: 0.9791696, 0.9804877, 0.9792096, 0.9804935, -0.0007400, 0.0007152
8: -0.0088977, -0.0074845, -0.0088548, -0.0074783, -0.0007934, 0.0007669
9: -0.0000557, 0.0008779, -0.0000598, 0.0008495, -0.0005066, 0.0005241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B2_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004523
time: 0.61 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004523
time: 0.60 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011166, -0.0006117, -0.0011309, -0.0006233, -0.0002683, 0.0002842
1: -0.0071441, -0.0058629, -0.0071804, -0.0058923, -0.0006807, 0.0007212
2: 0.0305978, 0.0313927, 0.0305753, 0.0313744, -0.0004223, 0.0004474
3: 0.0009444, 0.0024287, 0.0009785, 0.0024707, -0.0008354, 0.0007886
4: -0.0061598, -0.0048565, -0.0061967, -0.0048865, -0.0006924, 0.0007336
5: 0.0114050, 0.0118987, 0.0113910, 0.0118873, -0.0002623, 0.0002778
6: 0.0015867, 0.0034704, 0.0016300, 0.0035238, -0.0010603, 0.0010009
7: 0.9791696, 0.9804877, 0.9791998, 0.9805251, -0.0007419, 0.0007003
8: -0.0088977, -0.0074845, -0.0088652, -0.0074445, -0.0007955, 0.0007509
9: -0.0000557, 0.0008779, -0.0000821, 0.0008564, -0.0004960, 0.0005255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B2_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004527
time: 0.60 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004527
time: 0.61 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010943, -0.0005993, -0.0011004, -0.0006173, -0.0002630, 0.0002753
1: -0.0070876, -0.0058314, -0.0071029, -0.0058771, -0.0006674, 0.0006987
2: 0.0306329, 0.0314122, 0.0306233, 0.0313839, -0.0004141, 0.0004335
3: 0.0009080, 0.0023632, 0.0009609, 0.0023810, -0.0008094, 0.0007732
4: -0.0061023, -0.0048245, -0.0061179, -0.0048710, -0.0006789, 0.0007107
5: 0.0114268, 0.0119108, 0.0114209, 0.0118932, -0.0002571, 0.0002692
6: 0.0015405, 0.0033873, 0.0016076, 0.0034099, -0.0010272, 0.0009813
7: 0.9791372, 0.9804295, 0.9791842, 0.9804453, -0.0007188, 0.0006867
8: -0.0089324, -0.0075468, -0.0088821, -0.0075299, -0.0007707, 0.0007362
9: -0.0000145, 0.0009008, -0.0000257, 0.0008675, -0.0004863, 0.0005091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004299, upper bound: 0.0004352
time: 0.61 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004299, upper bound: 0.0004352
time: 0.61 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010943, -0.0005993, -0.0011118, -0.0006135, -0.0002576, 0.0002756
1: -0.0070876, -0.0058314, -0.0071320, -0.0058675, -0.0006536, 0.0006993
2: 0.0306329, 0.0314122, 0.0306053, 0.0313898, -0.0004055, 0.0004339
3: 0.0009080, 0.0023632, 0.0009498, 0.0024146, -0.0008101, 0.0007572
4: -0.0061023, -0.0048245, -0.0061475, -0.0048613, -0.0006648, 0.0007113
5: 0.0114268, 0.0119108, 0.0114097, 0.0118968, -0.0002518, 0.0002694
6: 0.0015405, 0.0033873, 0.0015936, 0.0034526, -0.0010282, 0.0009610
7: 0.9791372, 0.9804295, 0.9791744, 0.9804752, -0.0007195, 0.0006724
8: -0.0089324, -0.0075468, -0.0088926, -0.0074978, -0.0007714, 0.0007210
9: -0.0000145, 0.0009008, -0.0000468, 0.0008744, -0.0004762, 0.0005095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004299, upper bound: 0.0004352
time: 0.61 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004299, upper bound: 0.0004352
time: 0.60 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011089, -0.0005960, -0.0011076, -0.0006169, -0.0002638, 0.0002818
1: -0.0071246, -0.0058230, -0.0071214, -0.0058762, -0.0006695, 0.0007150
2: 0.0306099, 0.0314174, 0.0306119, 0.0313844, -0.0004153, 0.0004436
3: 0.0008982, 0.0024061, 0.0009598, 0.0024024, -0.0008283, 0.0007755
4: -0.0061399, -0.0048160, -0.0061367, -0.0048701, -0.0006810, 0.0007273
5: 0.0114125, 0.0119140, 0.0114138, 0.0118935, -0.0002579, 0.0002755
6: 0.0015281, 0.0034418, 0.0016063, 0.0034370, -0.0010513, 0.0009843
7: 0.9791286, 0.9804677, 0.9791833, 0.9804643, -0.0007356, 0.0006887
8: -0.0089417, -0.0075060, -0.0088831, -0.0075095, -0.0007887, 0.0007384
9: -0.0000415, 0.0009069, -0.0000391, 0.0008682, -0.0004878, 0.0005210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004308, upper bound: 0.0004594
time: 0.57 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004308, upper bound: 0.0004594
time: 0.56 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011089, -0.0005960, -0.0011189, -0.0006132, -0.0002571, 0.0002828
1: -0.0071246, -0.0058230, -0.0071499, -0.0058666, -0.0006525, 0.0007176
2: 0.0306099, 0.0314174, 0.0305942, 0.0313903, -0.0004048, 0.0004452
3: 0.0008982, 0.0024061, 0.0009488, 0.0024354, -0.0008313, 0.0007559
4: -0.0061399, -0.0048160, -0.0061657, -0.0048604, -0.0006637, 0.0007299
5: 0.0114125, 0.0119140, 0.0114028, 0.0118972, -0.0002514, 0.0002765
6: 0.0015281, 0.0034418, 0.0015923, 0.0034790, -0.0010550, 0.0009593
7: 0.9791286, 0.9804677, 0.9791735, 0.9804937, -0.0007382, 0.0006713
8: -0.0089417, -0.0075060, -0.0088936, -0.0074781, -0.0007915, 0.0007197
9: -0.0000415, 0.0009069, -0.0000599, 0.0008751, -0.0004754, 0.0005228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004308, upper bound: 0.0004599
time: 0.63 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004308, upper bound: 0.0004599
time: 0.65 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011041, -0.0006016, -0.0010997, -0.0006305, -0.0002669, 0.0002845
1: -0.0071123, -0.0058372, -0.0071012, -0.0059106, -0.0006774, 0.0007220
2: 0.0306175, 0.0314086, 0.0306244, 0.0313631, -0.0004203, 0.0004479
3: 0.0009147, 0.0023918, 0.0009997, 0.0023790, -0.0008364, 0.0007848
4: -0.0061274, -0.0048304, -0.0061162, -0.0049051, -0.0006890, 0.0007343
5: 0.0114173, 0.0119085, 0.0114215, 0.0118803, -0.0002610, 0.0002782
6: 0.0015490, 0.0034237, 0.0016569, 0.0034074, -0.0010614, 0.0009960
7: 0.9791431, 0.9804550, 0.9792187, 0.9804436, -0.0007427, 0.0006969
8: -0.0089261, -0.0075196, -0.0088451, -0.0075317, -0.0007963, 0.0007472
9: -0.0000325, 0.0008966, -0.0000245, 0.0008431, -0.0004936, 0.0005260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004589
time: 0.58 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004589
time: 0.61 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011041, -0.0006016, -0.0011121, -0.0006263, -0.0002612, 0.0002860
1: -0.0071123, -0.0058372, -0.0071327, -0.0058999, -0.0006627, 0.0007259
2: 0.0306175, 0.0314086, 0.0306049, 0.0313697, -0.0004112, 0.0004503
3: 0.0009147, 0.0023918, 0.0009873, 0.0024155, -0.0008409, 0.0007677
4: -0.0061274, -0.0048304, -0.0061482, -0.0048942, -0.0006741, 0.0007383
5: 0.0114173, 0.0119085, 0.0114094, 0.0118844, -0.0002553, 0.0002797
6: 0.0015490, 0.0034237, 0.0016411, 0.0034537, -0.0010672, 0.0009744
7: 0.9791431, 0.9804550, 0.9792077, 0.9804760, -0.0007468, 0.0006818
8: -0.0089261, -0.0075196, -0.0088569, -0.0074970, -0.0008007, 0.0007310
9: -0.0000325, 0.0008966, -0.0000474, 0.0008509, -0.0004829, 0.0005289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004589
time: 0.60 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004589
time: 0.60 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011111, -0.0006012, -0.0011150, -0.0006271, -0.0002774, 0.0002900
1: -0.0071302, -0.0058363, -0.0071401, -0.0059020, -0.0007039, 0.0007359
2: 0.0306064, 0.0314092, 0.0306003, 0.0313684, -0.0004367, 0.0004566
3: 0.0009136, 0.0024126, 0.0009897, 0.0024240, -0.0008525, 0.0008155
4: -0.0061457, -0.0048295, -0.0061557, -0.0048963, -0.0007160, 0.0007485
5: 0.0114104, 0.0119089, 0.0114066, 0.0118836, -0.0002712, 0.0002835
6: 0.0015476, 0.0034500, 0.0016442, 0.0034646, -0.0010819, 0.0010349
7: 0.9791422, 0.9804734, 0.9792098, 0.9804836, -0.0007571, 0.0007242
8: -0.0089270, -0.0074998, -0.0088546, -0.0074889, -0.0008117, 0.0007764
9: -0.0000456, 0.0008972, -0.0000528, 0.0008494, -0.0005129, 0.0005362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004377
time: 0.60 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004602
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011111, -0.0006012, -0.0011273, -0.0006234, -0.0002719, 0.0002910
1: -0.0071302, -0.0058363, -0.0071713, -0.0058925, -0.0006901, 0.0007385
2: 0.0306064, 0.0314092, 0.0305809, 0.0313743, -0.0004281, 0.0004582
3: 0.0009136, 0.0024126, 0.0009788, 0.0024602, -0.0008556, 0.0007995
4: -0.0061457, -0.0048295, -0.0061875, -0.0048867, -0.0007020, 0.0007512
5: 0.0114104, 0.0119089, 0.0113945, 0.0118872, -0.0002659, 0.0002845
6: 0.0015476, 0.0034500, 0.0016303, 0.0035105, -0.0010858, 0.0010146
7: 0.9791422, 0.9804734, 0.9792001, 0.9805158, -0.0007598, 0.0007100
8: -0.0089270, -0.0074998, -0.0088650, -0.0074544, -0.0008146, 0.0007612
9: -0.0000456, 0.0008972, -0.0000755, 0.0008562, -0.0005028, 0.0005381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004377
time: 0.56 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004607
time: 0.57 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.13 seconds
IS_A1_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004270
IS_A1_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004294
IS_A1_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004270
IS_A1_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004294
IS_A1_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004285
IS_A1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004303
IS_A1_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004285
IS_A1_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004303
IS_A1_B1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004291
IS_A1_B1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004310
IS_A1_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004291
IS_A1_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004310
IS_A1_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004302
IS_A1_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004318
IS_A1_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004302
IS_A1_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004410, upper bound: 0.0004318
IS_A1_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004294
IS_A1_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004294
IS_A1_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004294
IS_A1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004294
IS_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004474, upper bound: 0.0004303
IS_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004474, upper bound: 0.0004302
IS_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004474, upper bound: 0.0004303
IS_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004474, upper bound: 0.0004303
IS_A1_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004276
IS_A1_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004310
IS_A1_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004276
IS_A1_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004310
IS_A1_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004102
IS_A1_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004318
IS_A1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004103
IS_A1_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004250, upper bound: 0.0004318
IS_A2_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004048, upper bound: 0.0004500
IS_A2_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004048, upper bound: 0.0004499
IS_A2_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004499
IS_A2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004499
IS_A2_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004520
IS_A2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004520
IS_A2_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004524
IS_A2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004524
IS_A2_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004048, upper bound: 0.0004504
IS_A2_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004048, upper bound: 0.0004504
IS_A2_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004048, upper bound: 0.0004504
IS_A2_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004048, upper bound: 0.0004504
IS_A2_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004523
IS_A2_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004523
IS_A2_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004527
IS_A2_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004302, upper bound: 0.0004527
IS_A2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004299, upper bound: 0.0004352
IS_A2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004299, upper bound: 0.0004352
IS_A2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004299, upper bound: 0.0004352
IS_A2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004299, upper bound: 0.0004352
IS_A2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004308, upper bound: 0.0004594
IS_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004308, upper bound: 0.0004594
IS_A2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004308, upper bound: 0.0004599
IS_A2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004308, upper bound: 0.0004599
IS_A2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004589
IS_A2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004589
IS_A2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004589
IS_A2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004589
IS_A2_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004377
IS_A2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004602
IS_A2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004377
IS_A2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 7, lower bound: -0.0004103, upper bound: 0.0004607

## BFS IS instance: IS_A1_B1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010987, -0.0006100, -0.0010872, -0.0006198, -0.0002489, 0.0002492
1: -0.0070986, -0.0058586, -0.0070694, -0.0058834, -0.0006317, 0.0006323
2: 0.0306260, 0.0313953, 0.0306441, 0.0313800, -0.0003919, 0.0003923
3: 0.0009395, 0.0023760, 0.0009682, 0.0023421, -0.0007324, 0.0007318
4: -0.0061135, -0.0048522, -0.0060838, -0.0048774, -0.0006425, 0.0006431
5: 0.0114225, 0.0119003, 0.0114338, 0.0118908, -0.0002434, 0.0002436
6: 0.0015805, 0.0034036, 0.0016169, 0.0033606, -0.0009296, 0.0009287
7: 0.9791652, 0.9804409, 0.9791906, 0.9804108, -0.0006505, 0.0006499
8: -0.0089024, -0.0075346, -0.0088751, -0.0075669, -0.0006974, 0.0006968
9: -0.0000226, 0.0008809, -0.0000012, 0.0008629, -0.0004602, 0.0004607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004449
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004449
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010929, -0.0006008, -0.0010872, -0.0006198, -0.0002415, 0.0002532
1: -0.0070840, -0.0058353, -0.0070694, -0.0058834, -0.0006128, 0.0006425
2: 0.0306351, 0.0314098, 0.0306441, 0.0313800, -0.0003802, 0.0003986
3: 0.0009125, 0.0023591, 0.0009682, 0.0023421, -0.0007444, 0.0007099
4: -0.0060987, -0.0048285, -0.0060838, -0.0048774, -0.0006233, 0.0006536
5: 0.0114282, 0.0119093, 0.0114338, 0.0118908, -0.0002361, 0.0002476
6: 0.0015462, 0.0033821, 0.0016169, 0.0033606, -0.0009447, 0.0009009
7: 0.9791412, 0.9804258, 0.9791906, 0.9804108, -0.0006610, 0.0006304
8: -0.0089281, -0.0075508, -0.0088751, -0.0075669, -0.0007087, 0.0006759
9: -0.0000119, 0.0008979, -0.0000012, 0.0008629, -0.0004465, 0.0004682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004470
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004470
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010987, -0.0006100, -0.0010986, -0.0006156, -0.0002592, 0.0002666
1: -0.0070986, -0.0058586, -0.0070983, -0.0058728, -0.0006577, 0.0006766
2: 0.0306260, 0.0313953, 0.0306262, 0.0313865, -0.0004080, 0.0004198
3: 0.0009395, 0.0023760, 0.0009560, 0.0023756, -0.0007838, 0.0007619
4: -0.0061135, -0.0048522, -0.0061132, -0.0048667, -0.0006690, 0.0006882
5: 0.0114225, 0.0119003, 0.0114227, 0.0118948, -0.0002534, 0.0002607
6: 0.0015805, 0.0034036, 0.0016014, 0.0034031, -0.0009948, 0.0009669
7: 0.9791652, 0.9804409, 0.9791798, 0.9804407, -0.0006961, 0.0006766
8: -0.0089024, -0.0075346, -0.0088867, -0.0075350, -0.0007463, 0.0007254
9: -0.0000226, 0.0008809, -0.0000223, 0.0008706, -0.0004792, 0.0004930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004270
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004270
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010929, -0.0006008, -0.0010986, -0.0006156, -0.0002517, 0.0002707
1: -0.0070840, -0.0058353, -0.0070983, -0.0058728, -0.0006388, 0.0006869
2: 0.0306351, 0.0314098, 0.0306262, 0.0313865, -0.0003963, 0.0004261
3: 0.0009125, 0.0023591, 0.0009560, 0.0023756, -0.0007957, 0.0007400
4: -0.0060987, -0.0048285, -0.0061132, -0.0048667, -0.0006497, 0.0006987
5: 0.0114282, 0.0119093, 0.0114227, 0.0118948, -0.0002461, 0.0002646
6: 0.0015462, 0.0033821, 0.0016014, 0.0034031, -0.0010099, 0.0009391
7: 0.9791412, 0.9804258, 0.9791798, 0.9804407, -0.0007067, 0.0006572
8: -0.0089281, -0.0075508, -0.0088867, -0.0075350, -0.0007577, 0.0007046
9: -0.0000119, 0.0008979, -0.0000223, 0.0008706, -0.0004654, 0.0005005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011060, -0.0006097, -0.0011023, -0.0006172, -0.0002603, 0.0002500
1: -0.0071171, -0.0058577, -0.0071078, -0.0058769, -0.0006606, 0.0006345
2: 0.0306145, 0.0313959, 0.0306203, 0.0313840, -0.0004098, 0.0003937
3: 0.0009384, 0.0023974, 0.0009606, 0.0023866, -0.0007350, 0.0007652
4: -0.0061323, -0.0048512, -0.0061229, -0.0048708, -0.0006719, 0.0006454
5: 0.0114154, 0.0119007, 0.0114190, 0.0118933, -0.0002545, 0.0002445
6: 0.0015791, 0.0034308, 0.0016073, 0.0034171, -0.0009329, 0.0009712
7: 0.9791642, 0.9804599, 0.9791840, 0.9804503, -0.0006528, 0.0006796
8: -0.0089035, -0.0075142, -0.0088823, -0.0075245, -0.0006999, 0.0007286
9: -0.0000360, 0.0008816, -0.0000292, 0.0008677, -0.0004813, 0.0004623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004407, upper bound: 0.0004458
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004407, upper bound: 0.0004458
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011002, -0.0006005, -0.0011023, -0.0006172, -0.0002503, 0.0002572
1: -0.0071024, -0.0058343, -0.0071078, -0.0058769, -0.0006353, 0.0006528
2: 0.0306236, 0.0314104, 0.0306203, 0.0313840, -0.0003941, 0.0004050
3: 0.0009114, 0.0023804, 0.0009606, 0.0023866, -0.0007562, 0.0007360
4: -0.0061174, -0.0048275, -0.0061229, -0.0048708, -0.0006462, 0.0006640
5: 0.0114211, 0.0119096, 0.0114190, 0.0118933, -0.0002448, 0.0002515
6: 0.0015448, 0.0034092, 0.0016073, 0.0034171, -0.0009597, 0.0009340
7: 0.9791402, 0.9804448, 0.9791840, 0.9804503, -0.0006716, 0.0006536
8: -0.0089292, -0.0075304, -0.0088823, -0.0075245, -0.0007200, 0.0007007
9: -0.0000253, 0.0008986, -0.0000292, 0.0008677, -0.0004629, 0.0004756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004407, upper bound: 0.0004476
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004407, upper bound: 0.0004476
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011060, -0.0006097, -0.0011135, -0.0006135, -0.0002700, 0.0002676
1: -0.0071171, -0.0058577, -0.0071362, -0.0058673, -0.0006852, 0.0006790
2: 0.0306145, 0.0313959, 0.0306027, 0.0313899, -0.0004251, 0.0004213
3: 0.0009384, 0.0023974, 0.0009496, 0.0024195, -0.0007866, 0.0007937
4: -0.0061323, -0.0048512, -0.0061517, -0.0048611, -0.0006969, 0.0006907
5: 0.0114154, 0.0119007, 0.0114081, 0.0118969, -0.0002640, 0.0002616
6: 0.0015791, 0.0034308, 0.0015933, 0.0034587, -0.0009983, 0.0010073
7: 0.9791642, 0.9804599, 0.9791742, 0.9804795, -0.0006986, 0.0007049
8: -0.0089035, -0.0075142, -0.0088928, -0.0074933, -0.0007490, 0.0007557
9: -0.0000360, 0.0008816, -0.0000499, 0.0008746, -0.0004992, 0.0004947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004519, upper bound: 0.0004285
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004519, upper bound: 0.0004285
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011002, -0.0006005, -0.0011135, -0.0006135, -0.0002600, 0.0002748
1: -0.0071024, -0.0058343, -0.0071362, -0.0058673, -0.0006599, 0.0006973
2: 0.0306236, 0.0314104, 0.0306027, 0.0313899, -0.0004094, 0.0004326
3: 0.0009114, 0.0023804, 0.0009496, 0.0024195, -0.0008078, 0.0007644
4: -0.0061174, -0.0048275, -0.0061517, -0.0048611, -0.0006712, 0.0007093
5: 0.0114211, 0.0119096, 0.0114081, 0.0118969, -0.0002542, 0.0002686
6: 0.0015448, 0.0034092, 0.0015933, 0.0034587, -0.0010252, 0.0009702
7: 0.9791402, 0.9804448, 0.9791742, 0.9804795, -0.0007174, 0.0006789
8: -0.0089292, -0.0075304, -0.0088928, -0.0074933, -0.0007691, 0.0007279
9: -0.0000253, 0.0008986, -0.0000499, 0.0008746, -0.0004808, 0.0005080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004519, upper bound: 0.0004303
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004519, upper bound: 0.0004303
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010980, -0.0006158, -0.0010970, -0.0006306, -0.0002454, 0.0002634
1: -0.0070968, -0.0058732, -0.0070944, -0.0059108, -0.0006228, 0.0006684
2: 0.0306271, 0.0313863, 0.0306286, 0.0313630, -0.0003864, 0.0004147
3: 0.0009564, 0.0023739, 0.0009999, 0.0023711, -0.0007743, 0.0007215
4: -0.0061117, -0.0048671, -0.0061092, -0.0049053, -0.0006335, 0.0006799
5: 0.0114232, 0.0118947, 0.0114242, 0.0118802, -0.0002400, 0.0002575
6: 0.0016019, 0.0034009, 0.0016571, 0.0033973, -0.0009827, 0.0009157
7: 0.9791802, 0.9804390, 0.9792188, 0.9804365, -0.0006876, 0.0006408
8: -0.0088863, -0.0075366, -0.0088449, -0.0075393, -0.0007372, 0.0006870
9: -0.0000212, 0.0008703, -0.0000194, 0.0008430, -0.0004538, 0.0004870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004457
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004447
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010921, -0.0006065, -0.0010970, -0.0006306, -0.0002377, 0.0002695
1: -0.0070820, -0.0058497, -0.0070944, -0.0059108, -0.0006033, 0.0006840
2: 0.0306363, 0.0314009, 0.0306286, 0.0313630, -0.0003743, 0.0004244
3: 0.0009291, 0.0023567, 0.0009999, 0.0023711, -0.0007924, 0.0006989
4: -0.0060966, -0.0048431, -0.0061092, -0.0049053, -0.0006136, 0.0006958
5: 0.0114290, 0.0119037, 0.0114242, 0.0118802, -0.0002324, 0.0002635
6: 0.0015673, 0.0033791, 0.0016571, 0.0033973, -0.0010056, 0.0008870
7: 0.9791560, 0.9804238, 0.9792188, 0.9804365, -0.0007037, 0.0006206
8: -0.0089123, -0.0075530, -0.0088449, -0.0075393, -0.0007545, 0.0006654
9: -0.0000104, 0.0008875, -0.0000194, 0.0008430, -0.0004396, 0.0004984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004477
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004469
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010980, -0.0006158, -0.0011091, -0.0006264, -0.0002563, 0.0002806
1: -0.0070968, -0.0058732, -0.0071250, -0.0059001, -0.0006504, 0.0007121
2: 0.0306271, 0.0313863, 0.0306096, 0.0313696, -0.0004035, 0.0004418
3: 0.0009564, 0.0023739, 0.0009876, 0.0024066, -0.0008250, 0.0007535
4: -0.0061117, -0.0048671, -0.0061404, -0.0048944, -0.0006616, 0.0007243
5: 0.0114232, 0.0118947, 0.0114124, 0.0118843, -0.0002506, 0.0002744
6: 0.0016019, 0.0034009, 0.0016415, 0.0034424, -0.0010470, 0.0009562
7: 0.9791802, 0.9804390, 0.9792079, 0.9804681, -0.0007326, 0.0006691
8: -0.0088863, -0.0075366, -0.0088567, -0.0075055, -0.0007855, 0.0007174
9: -0.0000212, 0.0008703, -0.0000418, 0.0008507, -0.0004739, 0.0005189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004291
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004270
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010921, -0.0006065, -0.0011091, -0.0006264, -0.0002486, 0.0002868
1: -0.0070820, -0.0058497, -0.0071250, -0.0059001, -0.0006309, 0.0007277
2: 0.0306363, 0.0314009, 0.0306096, 0.0313696, -0.0003914, 0.0004515
3: 0.0009291, 0.0023567, 0.0009876, 0.0024066, -0.0008431, 0.0007308
4: -0.0060966, -0.0048431, -0.0061404, -0.0048944, -0.0006417, 0.0007402
5: 0.0114290, 0.0119037, 0.0114124, 0.0118843, -0.0002431, 0.0002804
6: 0.0015673, 0.0033791, 0.0016415, 0.0034424, -0.0010700, 0.0009275
7: 0.9791560, 0.9804238, 0.9792079, 0.9804681, -0.0007487, 0.0006490
8: -0.0089123, -0.0075530, -0.0088567, -0.0075055, -0.0008027, 0.0006959
9: -0.0000104, 0.0008875, -0.0000418, 0.0008507, -0.0004596, 0.0005302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004310
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011052, -0.0006154, -0.0011124, -0.0006272, -0.0002558, 0.0002660
1: -0.0071152, -0.0058722, -0.0071334, -0.0059021, -0.0006492, 0.0006750
2: 0.0306157, 0.0313869, 0.0306044, 0.0313683, -0.0004028, 0.0004188
3: 0.0009553, 0.0023952, 0.0009899, 0.0024163, -0.0007819, 0.0007521
4: -0.0061304, -0.0048661, -0.0061489, -0.0048964, -0.0006604, 0.0006866
5: 0.0114161, 0.0118950, 0.0114091, 0.0118835, -0.0002501, 0.0002600
6: 0.0016005, 0.0034280, 0.0016444, 0.0034547, -0.0009924, 0.0009545
7: 0.9791792, 0.9804580, 0.9792099, 0.9804767, -0.0006944, 0.0006679
8: -0.0088874, -0.0075163, -0.0088545, -0.0074963, -0.0007445, 0.0007161
9: -0.0000346, 0.0008710, -0.0000479, 0.0008493, -0.0004730, 0.0004918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004188
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004463
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010994, -0.0006061, -0.0011124, -0.0006272, -0.0002457, 0.0002748
1: -0.0071005, -0.0058487, -0.0071334, -0.0059021, -0.0006234, 0.0006973
2: 0.0306249, 0.0314015, 0.0306044, 0.0313683, -0.0003867, 0.0004326
3: 0.0009280, 0.0023781, 0.0009899, 0.0024163, -0.0008078, 0.0007222
4: -0.0061154, -0.0048421, -0.0061489, -0.0048964, -0.0006341, 0.0007093
5: 0.0114218, 0.0119041, 0.0114091, 0.0118835, -0.0002402, 0.0002687
6: 0.0015659, 0.0034063, 0.0016444, 0.0034547, -0.0010252, 0.0009165
7: 0.9791550, 0.9804428, 0.9792099, 0.9804767, -0.0007174, 0.0006413
8: -0.0089133, -0.0075326, -0.0088545, -0.0074963, -0.0007691, 0.0006876
9: -0.0000239, 0.0008882, -0.0000479, 0.0008493, -0.0004542, 0.0005081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004250
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004481
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011052, -0.0006154, -0.0011243, -0.0006234, -0.0002656, 0.0002833
1: -0.0071152, -0.0058722, -0.0071635, -0.0058926, -0.0006740, 0.0007188
2: 0.0306157, 0.0313869, 0.0305857, 0.0313742, -0.0004182, 0.0004460
3: 0.0009553, 0.0023952, 0.0009789, 0.0024512, -0.0008327, 0.0007808
4: -0.0061304, -0.0048661, -0.0061795, -0.0048868, -0.0006856, 0.0007312
5: 0.0114161, 0.0118950, 0.0113975, 0.0118872, -0.0002597, 0.0002769
6: 0.0016005, 0.0034280, 0.0016304, 0.0034990, -0.0010568, 0.0009910
7: 0.9791792, 0.9804580, 0.9792001, 0.9805077, -0.0007395, 0.0006934
8: -0.0088874, -0.0075163, -0.0088649, -0.0074630, -0.0007929, 0.0007435
9: -0.0000346, 0.0008710, -0.0000698, 0.0008562, -0.0004911, 0.0005237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004048
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004302
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010994, -0.0006061, -0.0011243, -0.0006234, -0.0002554, 0.0002921
1: -0.0071005, -0.0058487, -0.0071635, -0.0058926, -0.0006482, 0.0007412
2: 0.0306249, 0.0314015, 0.0305857, 0.0313742, -0.0004021, 0.0004598
3: 0.0009280, 0.0023781, 0.0009789, 0.0024512, -0.0008586, 0.0007509
4: -0.0061154, -0.0048421, -0.0061795, -0.0048868, -0.0006593, 0.0007539
5: 0.0114218, 0.0119041, 0.0113975, 0.0118872, -0.0002497, 0.0002855
6: 0.0015659, 0.0034063, 0.0016304, 0.0034990, -0.0010897, 0.0009530
7: 0.9791550, 0.9804428, 0.9792001, 0.9805077, -0.0007625, 0.0006669
8: -0.0089133, -0.0075326, -0.0088649, -0.0074630, -0.0008175, 0.0007150
9: -0.0000239, 0.0008882, -0.0000698, 0.0008562, -0.0004723, 0.0005400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004103
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004318
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011004, -0.0006173, -0.0010815, -0.0006111, -0.0002547, 0.0002455
1: -0.0071029, -0.0058771, -0.0070551, -0.0058614, -0.0006464, 0.0006229
2: 0.0306233, 0.0313839, 0.0306530, 0.0313936, -0.0004010, 0.0003865
3: 0.0009609, 0.0023810, 0.0009428, 0.0023256, -0.0007216, 0.0007488
4: -0.0061179, -0.0048710, -0.0060693, -0.0048551, -0.0006575, 0.0006336
5: 0.0114209, 0.0118932, 0.0114393, 0.0118992, -0.0002490, 0.0002400
6: 0.0016076, 0.0034099, 0.0015846, 0.0033396, -0.0009158, 0.0009504
7: 0.9791842, 0.9804453, 0.9791682, 0.9803962, -0.0006409, 0.0006650
8: -0.0088821, -0.0075299, -0.0088993, -0.0075826, -0.0006871, 0.0007130
9: -0.0000257, 0.0008675, 0.0000091, 0.0008789, -0.0004710, 0.0004539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004405
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004470
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011105, -0.0006272, -0.0010815, -0.0006111, -0.0002762, 0.0002388
1: -0.0071285, -0.0059022, -0.0070551, -0.0058614, -0.0007008, 0.0006061
2: 0.0306074, 0.0313683, 0.0306530, 0.0313936, -0.0004348, 0.0003760
3: 0.0009900, 0.0024107, 0.0009428, 0.0023256, -0.0007021, 0.0008119
4: -0.0061440, -0.0048965, -0.0060693, -0.0048551, -0.0007129, 0.0006165
5: 0.0114110, 0.0118835, 0.0114393, 0.0118992, -0.0002700, 0.0002335
6: 0.0016445, 0.0034476, 0.0015846, 0.0033396, -0.0008911, 0.0010304
7: 0.9792100, 0.9804717, 0.9791682, 0.9803962, -0.0006235, 0.0007210
8: -0.0088544, -0.0075016, -0.0088993, -0.0075826, -0.0006685, 0.0007731
9: -0.0000443, 0.0008492, 0.0000091, 0.0008789, -0.0005106, 0.0004416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004405
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004470
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011004, -0.0006173, -0.0010934, -0.0006070, -0.0002672, 0.0002628
1: -0.0071029, -0.0058771, -0.0070852, -0.0058509, -0.0006780, 0.0006668
2: 0.0306233, 0.0313839, 0.0306343, 0.0314001, -0.0004206, 0.0004137
3: 0.0009609, 0.0023810, 0.0009306, 0.0023604, -0.0007725, 0.0007854
4: -0.0061179, -0.0048710, -0.0060998, -0.0048444, -0.0006896, 0.0006782
5: 0.0114209, 0.0118932, 0.0114277, 0.0119033, -0.0002612, 0.0002569
6: 0.0016076, 0.0034099, 0.0015692, 0.0033838, -0.0009803, 0.0009968
7: 0.9791842, 0.9804453, 0.9791573, 0.9804271, -0.0006860, 0.0006975
8: -0.0088821, -0.0075299, -0.0089109, -0.0075495, -0.0007355, 0.0007478
9: -0.0000257, 0.0008675, -0.0000127, 0.0008866, -0.0004940, 0.0004858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004257
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011105, -0.0006272, -0.0010934, -0.0006070, -0.0002886, 0.0002561
1: -0.0071285, -0.0059022, -0.0070852, -0.0058509, -0.0007324, 0.0006500
2: 0.0306074, 0.0313683, 0.0306343, 0.0314001, -0.0004544, 0.0004032
3: 0.0009900, 0.0024107, 0.0009306, 0.0023604, -0.0007529, 0.0008485
4: -0.0061440, -0.0048965, -0.0060998, -0.0048444, -0.0007450, 0.0006611
5: 0.0114110, 0.0118835, 0.0114277, 0.0119033, -0.0002822, 0.0002504
6: 0.0016445, 0.0034476, 0.0015692, 0.0033838, -0.0009556, 0.0010768
7: 0.9792100, 0.9804717, 0.9791573, 0.9804271, -0.0006687, 0.0007535
8: -0.0088544, -0.0075016, -0.0089109, -0.0075495, -0.0007169, 0.0008079
9: -0.0000443, 0.0008492, -0.0000127, 0.0008866, -0.0005336, 0.0004736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004257
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011076, -0.0006169, -0.0010966, -0.0006080, -0.0002617, 0.0002462
1: -0.0071214, -0.0058762, -0.0070932, -0.0058534, -0.0006640, 0.0006249
2: 0.0306119, 0.0313844, 0.0306294, 0.0313985, -0.0004119, 0.0003877
3: 0.0009598, 0.0024024, 0.0009335, 0.0023698, -0.0007239, 0.0007692
4: -0.0061367, -0.0048701, -0.0061080, -0.0048469, -0.0006754, 0.0006356
5: 0.0114138, 0.0118935, 0.0114246, 0.0119023, -0.0002558, 0.0002407
6: 0.0016063, 0.0034370, 0.0015728, 0.0033956, -0.0009187, 0.0009762
7: 0.9791833, 0.9804643, 0.9791599, 0.9804354, -0.0006429, 0.0006831
8: -0.0088831, -0.0075095, -0.0089082, -0.0075406, -0.0006893, 0.0007324
9: -0.0000391, 0.0008682, -0.0000186, 0.0008847, -0.0004838, 0.0004553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004238
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004238
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011176, -0.0006269, -0.0010966, -0.0006080, -0.0002815, 0.0002427
1: -0.0071467, -0.0059013, -0.0070932, -0.0058534, -0.0007143, 0.0006158
2: 0.0305962, 0.0313688, 0.0306294, 0.0313985, -0.0004431, 0.0003820
3: 0.0009890, 0.0024317, 0.0009335, 0.0023698, -0.0007133, 0.0008274
4: -0.0061624, -0.0048957, -0.0061080, -0.0048469, -0.0007265, 0.0006263
5: 0.0114040, 0.0118838, 0.0114246, 0.0119023, -0.0002752, 0.0002372
6: 0.0016433, 0.0034743, 0.0015728, 0.0033956, -0.0009053, 0.0010501
7: 0.9792091, 0.9804904, 0.9791599, 0.9804354, -0.0006335, 0.0007348
8: -0.0088553, -0.0074816, -0.0089082, -0.0075406, -0.0006792, 0.0007879
9: -0.0000576, 0.0008498, -0.0000186, 0.0008847, -0.0005204, 0.0004487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004238
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004476
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011076, -0.0006169, -0.0011080, -0.0006035, -0.0002730, 0.0002635
1: -0.0071214, -0.0058762, -0.0071223, -0.0058420, -0.0006927, 0.0006687
2: 0.0306119, 0.0313844, 0.0306113, 0.0314056, -0.0004298, 0.0004149
3: 0.0009598, 0.0024024, 0.0009203, 0.0024035, -0.0007746, 0.0008025
4: -0.0061367, -0.0048701, -0.0061376, -0.0048353, -0.0007046, 0.0006802
5: 0.0114138, 0.0118935, 0.0114134, 0.0119067, -0.0002669, 0.0002576
6: 0.0016063, 0.0034370, 0.0015561, 0.0034385, -0.0009831, 0.0010184
7: 0.9791833, 0.9804643, 0.9791481, 0.9804654, -0.0006879, 0.0007127
8: -0.0088831, -0.0075095, -0.0089207, -0.0075085, -0.0007376, 0.0007641
9: -0.0000391, 0.0008682, -0.0000398, 0.0008930, -0.0005047, 0.0004872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004360, upper bound: 0.0004103
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004360, upper bound: 0.0004303
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011176, -0.0006269, -0.0011080, -0.0006035, -0.0002928, 0.0002599
1: -0.0071467, -0.0059013, -0.0071223, -0.0058420, -0.0007430, 0.0006596
2: 0.0305962, 0.0313688, 0.0306113, 0.0314056, -0.0004609, 0.0004092
3: 0.0009890, 0.0024317, 0.0009203, 0.0024035, -0.0007641, 0.0008607
4: -0.0061624, -0.0048957, -0.0061376, -0.0048353, -0.0007557, 0.0006709
5: 0.0114040, 0.0118838, 0.0114134, 0.0119067, -0.0002863, 0.0002541
6: 0.0016433, 0.0034743, 0.0015561, 0.0034385, -0.0009697, 0.0010923
7: 0.9792091, 0.9804904, 0.9791481, 0.9804654, -0.0006786, 0.0007644
8: -0.0088553, -0.0074816, -0.0089207, -0.0075085, -0.0007275, 0.0008195
9: -0.0000576, 0.0008498, -0.0000398, 0.0008930, -0.0005413, 0.0004806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004360, upper bound: 0.0004103
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004360, upper bound: 0.0004303
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010980, -0.0006158, -0.0010918, -0.0006198, -0.0002536, 0.0002605
1: -0.0070968, -0.0058732, -0.0070813, -0.0058833, -0.0006435, 0.0006610
2: 0.0306271, 0.0313863, 0.0306368, 0.0313800, -0.0003992, 0.0004101
3: 0.0009564, 0.0023739, 0.0009681, 0.0023559, -0.0007658, 0.0007455
4: -0.0061117, -0.0048671, -0.0060959, -0.0048774, -0.0006545, 0.0006724
5: 0.0114232, 0.0118947, 0.0114292, 0.0118908, -0.0002479, 0.0002547
6: 0.0016019, 0.0034009, 0.0016168, 0.0033781, -0.0009718, 0.0009461
7: 0.9791802, 0.9804390, 0.9791906, 0.9804231, -0.0006800, 0.0006620
8: -0.0088863, -0.0075366, -0.0088751, -0.0075538, -0.0007291, 0.0007098
9: -0.0000212, 0.0008703, -0.0000099, 0.0008629, -0.0004689, 0.0004816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004404
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004400
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010921, -0.0006065, -0.0010918, -0.0006198, -0.0002497, 0.0002671
1: -0.0070820, -0.0058497, -0.0070813, -0.0058833, -0.0006337, 0.0006779
2: 0.0306363, 0.0314009, 0.0306368, 0.0313800, -0.0003931, 0.0004206
3: 0.0009291, 0.0023567, 0.0009681, 0.0023559, -0.0007853, 0.0007341
4: -0.0060966, -0.0048431, -0.0060959, -0.0048774, -0.0006445, 0.0006896
5: 0.0114290, 0.0119037, 0.0114292, 0.0118908, -0.0002441, 0.0002612
6: 0.0015673, 0.0033791, 0.0016168, 0.0033781, -0.0009967, 0.0009316
7: 0.9791560, 0.9804238, 0.9791906, 0.9804231, -0.0006974, 0.0006519
8: -0.0089123, -0.0075530, -0.0088751, -0.0075538, -0.0007478, 0.0006989
9: -0.0000104, 0.0008875, -0.0000099, 0.0008629, -0.0004617, 0.0004939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004469
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004469
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010980, -0.0006158, -0.0011042, -0.0006140, -0.0002656, 0.0002794
1: -0.0070968, -0.0058732, -0.0071128, -0.0058687, -0.0006740, 0.0007090
2: 0.0306271, 0.0313863, 0.0306172, 0.0313890, -0.0004181, 0.0004399
3: 0.0009564, 0.0023739, 0.0009512, 0.0023924, -0.0008213, 0.0007808
4: -0.0061117, -0.0048671, -0.0061279, -0.0048625, -0.0006855, 0.0007212
5: 0.0114232, 0.0118947, 0.0114171, 0.0118964, -0.0002597, 0.0002732
6: 0.0016019, 0.0034009, 0.0015953, 0.0034244, -0.0010424, 0.0009909
7: 0.9791802, 0.9804390, 0.9791756, 0.9804555, -0.0007294, 0.0006934
8: -0.0088863, -0.0075366, -0.0088913, -0.0075190, -0.0007820, 0.0007434
9: -0.0000212, 0.0008703, -0.0000328, 0.0008736, -0.0004911, 0.0005166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004276
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004257
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010921, -0.0006065, -0.0011042, -0.0006140, -0.0002606, 0.0002843
1: -0.0070820, -0.0058497, -0.0071128, -0.0058687, -0.0006614, 0.0007215
2: 0.0306363, 0.0314009, 0.0306172, 0.0313890, -0.0004103, 0.0004476
3: 0.0009291, 0.0023567, 0.0009512, 0.0023924, -0.0008358, 0.0007661
4: -0.0060966, -0.0048431, -0.0061279, -0.0048625, -0.0006727, 0.0007339
5: 0.0114290, 0.0119037, 0.0114171, 0.0118964, -0.0002548, 0.0002780
6: 0.0015673, 0.0033791, 0.0015953, 0.0034244, -0.0010608, 0.0009723
7: 0.9791560, 0.9804238, 0.9791756, 0.9804555, -0.0007423, 0.0006804
8: -0.0089123, -0.0075530, -0.0088913, -0.0075190, -0.0007959, 0.0007295
9: -0.0000104, 0.0008875, -0.0000328, 0.0008736, -0.0004819, 0.0005257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004310
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010898, -0.0006187, -0.0011072, -0.0006180, -0.0002432, 0.0002789
1: -0.0070762, -0.0058806, -0.0071203, -0.0058789, -0.0006171, 0.0007077
2: 0.0306399, 0.0313817, 0.0306126, 0.0313827, -0.0003829, 0.0004391
3: 0.0009649, 0.0023500, 0.0009630, 0.0024011, -0.0008199, 0.0007149
4: -0.0060907, -0.0048746, -0.0061356, -0.0048729, -0.0006277, 0.0007199
5: 0.0114312, 0.0118918, 0.0114142, 0.0118925, -0.0002378, 0.0002727
6: 0.0016128, 0.0033706, 0.0016103, 0.0034355, -0.0010405, 0.0009073
7: 0.9791878, 0.9804179, 0.9791861, 0.9804633, -0.0007281, 0.0006349
8: -0.0088782, -0.0075594, -0.0088800, -0.0075107, -0.0007806, 0.0006807
9: -0.0000062, 0.0008649, -0.0000383, 0.0008662, -0.0004497, 0.0005157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004078
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004250
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011050, -0.0006156, -0.0011072, -0.0006180, -0.0002477, 0.0002690
1: -0.0071147, -0.0058727, -0.0071203, -0.0058789, -0.0006287, 0.0006825
2: 0.0306161, 0.0313866, 0.0306126, 0.0313827, -0.0003900, 0.0004234
3: 0.0009558, 0.0023946, 0.0009630, 0.0024011, -0.0007907, 0.0007283
4: -0.0061298, -0.0048666, -0.0061356, -0.0048729, -0.0006395, 0.0006942
5: 0.0114164, 0.0118949, 0.0114142, 0.0118925, -0.0002422, 0.0002630
6: 0.0016012, 0.0034271, 0.0016103, 0.0034355, -0.0010035, 0.0009243
7: 0.9791797, 0.9804574, 0.9791861, 0.9804633, -0.0007022, 0.0006468
8: -0.0088868, -0.0075170, -0.0088800, -0.0075107, -0.0007528, 0.0006934
9: -0.0000342, 0.0008707, -0.0000383, 0.0008662, -0.0004581, 0.0004973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004410
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004481
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010898, -0.0006187, -0.0011196, -0.0006130, -0.0002551, 0.0002957
1: -0.0070762, -0.0058806, -0.0071518, -0.0058660, -0.0006473, 0.0007504
2: 0.0306399, 0.0313817, 0.0305930, 0.0313907, -0.0004016, 0.0004655
3: 0.0009649, 0.0023500, 0.0009481, 0.0024376, -0.0008693, 0.0007499
4: -0.0060907, -0.0048746, -0.0061676, -0.0048598, -0.0006584, 0.0007633
5: 0.0114312, 0.0118918, 0.0114021, 0.0118974, -0.0002494, 0.0002891
6: 0.0016128, 0.0033706, 0.0015914, 0.0034818, -0.0011032, 0.0009517
7: 0.9791878, 0.9804179, 0.9791729, 0.9804957, -0.0007720, 0.0006660
8: -0.0088782, -0.0075594, -0.0088942, -0.0074760, -0.0008277, 0.0007140
9: -0.0000062, 0.0008649, -0.0000613, 0.0008755, -0.0004716, 0.0005467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0003942
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004103
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011050, -0.0006156, -0.0011196, -0.0006130, -0.0002598, 0.0002862
1: -0.0071147, -0.0058727, -0.0071518, -0.0058660, -0.0006593, 0.0007263
2: 0.0306161, 0.0313866, 0.0305930, 0.0313907, -0.0004090, 0.0004506
3: 0.0009558, 0.0023946, 0.0009481, 0.0024376, -0.0008413, 0.0007638
4: -0.0061298, -0.0048666, -0.0061676, -0.0048598, -0.0006706, 0.0007387
5: 0.0114164, 0.0118949, 0.0114021, 0.0118974, -0.0002540, 0.0002798
6: 0.0016012, 0.0034271, 0.0015914, 0.0034818, -0.0010678, 0.0009693
7: 0.9791797, 0.9804574, 0.9791729, 0.9804957, -0.0007472, 0.0006783
8: -0.0088868, -0.0075170, -0.0088942, -0.0074760, -0.0008011, 0.0007272
9: -0.0000342, 0.0008707, -0.0000613, 0.0008755, -0.0004804, 0.0005292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004283
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004318
time: 0.58 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011099, -0.0006062, -0.0010872, -0.0006198, -0.0002668, 0.0002587
1: -0.0071272, -0.0058489, -0.0070694, -0.0058834, -0.0006771, 0.0006565
2: 0.0306083, 0.0314013, 0.0306441, 0.0313800, -0.0004201, 0.0004073
3: 0.0009283, 0.0024091, 0.0009682, 0.0023421, -0.0007606, 0.0007844
4: -0.0061426, -0.0048424, -0.0060838, -0.0048774, -0.0006887, 0.0006678
5: 0.0114115, 0.0119040, 0.0114338, 0.0118908, -0.0002609, 0.0002530
6: 0.0015662, 0.0034455, 0.0016169, 0.0033606, -0.0009653, 0.0009955
7: 0.9791552, 0.9804703, 0.9791906, 0.9804108, -0.0006755, 0.0006966
8: -0.0089131, -0.0075032, -0.0088751, -0.0075669, -0.0007242, 0.0007468
9: -0.0000433, 0.0008880, -0.0000012, 0.0008629, -0.0004933, 0.0004784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B1_B1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004500
time: 0.57 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004499
time: 0.56 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011099, -0.0006062, -0.0010815, -0.0006111, -0.0002734, 0.0002524
1: -0.0071272, -0.0058489, -0.0070551, -0.0058614, -0.0006937, 0.0006405
2: 0.0306083, 0.0314013, 0.0306530, 0.0313936, -0.0004304, 0.0003974
3: 0.0009283, 0.0024091, 0.0009428, 0.0023256, -0.0007420, 0.0008036
4: -0.0061426, -0.0048424, -0.0060693, -0.0048551, -0.0007056, 0.0006515
5: 0.0114115, 0.0119040, 0.0114393, 0.0118992, -0.0002673, 0.0002468
6: 0.0015662, 0.0034455, 0.0015846, 0.0033396, -0.0009417, 0.0010199
7: 0.9791552, 0.9804703, 0.9791682, 0.9803962, -0.0006589, 0.0007137
8: -0.0089131, -0.0075032, -0.0088993, -0.0075826, -0.0007065, 0.0007652
9: -0.0000433, 0.0008880, 0.0000091, 0.0008789, -0.0005054, 0.0004667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B1_B1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004499
time: 0.56 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004500
time: 0.57 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011099, -0.0006062, -0.0010986, -0.0006156, -0.0002604, 0.0002605
1: -0.0071272, -0.0058489, -0.0070983, -0.0058728, -0.0006608, 0.0006611
2: 0.0306083, 0.0314013, 0.0306262, 0.0313865, -0.0004100, 0.0004101
3: 0.0009283, 0.0024091, 0.0009560, 0.0023756, -0.0007658, 0.0007655
4: -0.0061426, -0.0048424, -0.0061132, -0.0048667, -0.0006721, 0.0006724
5: 0.0114115, 0.0119040, 0.0114227, 0.0118948, -0.0002546, 0.0002547
6: 0.0015662, 0.0034455, 0.0016014, 0.0034031, -0.0009719, 0.0009715
7: 0.9791552, 0.9804703, 0.9791798, 0.9804407, -0.0006801, 0.0006798
8: -0.0089131, -0.0075032, -0.0088867, -0.0075350, -0.0007292, 0.0007289
9: -0.0000433, 0.0008880, -0.0000223, 0.0008706, -0.0004815, 0.0004817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004499
time: 0.61 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004499
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011099, -0.0006062, -0.0010934, -0.0006070, -0.0002679, 0.0002546
1: -0.0071272, -0.0058489, -0.0070852, -0.0058509, -0.0006799, 0.0006461
2: 0.0306083, 0.0314013, 0.0306343, 0.0314001, -0.0004218, 0.0004009
3: 0.0009283, 0.0024091, 0.0009306, 0.0023604, -0.0007485, 0.0007876
4: -0.0061426, -0.0048424, -0.0060998, -0.0048444, -0.0006916, 0.0006572
5: 0.0114115, 0.0119040, 0.0114277, 0.0119033, -0.0002619, 0.0002489
6: 0.0015662, 0.0034455, 0.0015692, 0.0033838, -0.0009499, 0.0009996
7: 0.9791552, 0.9804703, 0.9791573, 0.9804271, -0.0006647, 0.0006995
8: -0.0089131, -0.0075032, -0.0089109, -0.0075495, -0.0007127, 0.0007499
9: -0.0000433, 0.0008880, -0.0000127, 0.0008866, -0.0004954, 0.0004708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004499
time: 0.61 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004499
time: 0.61 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011172, -0.0006059, -0.0011023, -0.0006172, -0.0002782, 0.0002607
1: -0.0071457, -0.0058481, -0.0071078, -0.0058769, -0.0007059, 0.0006617
2: 0.0305968, 0.0314019, 0.0306203, 0.0313840, -0.0004379, 0.0004105
3: 0.0009273, 0.0024305, 0.0009606, 0.0023866, -0.0007665, 0.0008177
4: -0.0061614, -0.0048415, -0.0061229, -0.0048708, -0.0007180, 0.0006730
5: 0.0114044, 0.0119044, 0.0114190, 0.0118933, -0.0002720, 0.0002549
6: 0.0015650, 0.0034728, 0.0016073, 0.0034171, -0.0009728, 0.0010378
7: 0.9791544, 0.9804894, 0.9791840, 0.9804503, -0.0006807, 0.0007262
8: -0.0089140, -0.0074827, -0.0088823, -0.0075245, -0.0007298, 0.0007786
9: -0.0000568, 0.0008886, -0.0000292, 0.0008677, -0.0005143, 0.0004821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B1_B2_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004290, upper bound: 0.0004520
time: 0.57 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004290, upper bound: 0.0004520
time: 0.66 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011172, -0.0006059, -0.0010966, -0.0006080, -0.0002811, 0.0002551
1: -0.0071457, -0.0058481, -0.0070932, -0.0058534, -0.0007134, 0.0006474
2: 0.0305968, 0.0314019, 0.0306294, 0.0313985, -0.0004426, 0.0004017
3: 0.0009273, 0.0024305, 0.0009335, 0.0023698, -0.0007500, 0.0008264
4: -0.0061614, -0.0048415, -0.0061080, -0.0048469, -0.0007256, 0.0006585
5: 0.0114044, 0.0119044, 0.0114246, 0.0119023, -0.0002748, 0.0002494
6: 0.0015650, 0.0034728, 0.0015728, 0.0033956, -0.0009519, 0.0010488
7: 0.9791544, 0.9804894, 0.9791599, 0.9804354, -0.0006661, 0.0007339
8: -0.0089140, -0.0074827, -0.0089082, -0.0075406, -0.0007141, 0.0007869
9: -0.0000568, 0.0008886, -0.0000186, 0.0008847, -0.0005198, 0.0004717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B1_B2_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004290, upper bound: 0.0004520
time: 0.59 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004290, upper bound: 0.0004520
time: 0.62 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011172, -0.0006059, -0.0011135, -0.0006135, -0.0002727, 0.0002607
1: -0.0071457, -0.0058481, -0.0071362, -0.0058673, -0.0006921, 0.0006616
2: 0.0305968, 0.0314019, 0.0306027, 0.0313899, -0.0004294, 0.0004104
3: 0.0009273, 0.0024305, 0.0009496, 0.0024195, -0.0007664, 0.0008018
4: -0.0061614, -0.0048415, -0.0061517, -0.0048611, -0.0007040, 0.0006729
5: 0.0114044, 0.0119044, 0.0114081, 0.0118969, -0.0002667, 0.0002549
6: 0.0015650, 0.0034728, 0.0015933, 0.0034587, -0.0009727, 0.0010175
7: 0.9791544, 0.9804894, 0.9791742, 0.9804795, -0.0006806, 0.0007120
8: -0.0089140, -0.0074827, -0.0088928, -0.0074933, -0.0007297, 0.0007634
9: -0.0000568, 0.0008886, -0.0000499, 0.0008746, -0.0005043, 0.0004820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004524
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004524
time: 0.62 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011172, -0.0006059, -0.0011080, -0.0006035, -0.0002753, 0.0002560
1: -0.0071457, -0.0058481, -0.0071223, -0.0058420, -0.0006987, 0.0006496
2: 0.0305968, 0.0314019, 0.0306113, 0.0314056, -0.0004335, 0.0004030
3: 0.0009273, 0.0024305, 0.0009203, 0.0024035, -0.0007525, 0.0008094
4: -0.0061614, -0.0048415, -0.0061376, -0.0048353, -0.0007107, 0.0006607
5: 0.0114044, 0.0119044, 0.0114134, 0.0119067, -0.0002692, 0.0002503
6: 0.0015650, 0.0034728, 0.0015561, 0.0034385, -0.0009550, 0.0010272
7: 0.9791544, 0.9804894, 0.9791481, 0.9804654, -0.0006683, 0.0007188
8: -0.0089140, -0.0074827, -0.0089207, -0.0075085, -0.0007165, 0.0007707
9: -0.0000568, 0.0008886, -0.0000398, 0.0008930, -0.0005091, 0.0004733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004524
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004524
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011093, -0.0006120, -0.0010970, -0.0006306, -0.0002633, 0.0002733
1: -0.0071256, -0.0058637, -0.0070944, -0.0059108, -0.0006682, 0.0006935
2: 0.0306093, 0.0313921, 0.0306286, 0.0313630, -0.0004146, 0.0004302
3: 0.0009454, 0.0024072, 0.0009999, 0.0023711, -0.0008033, 0.0007741
4: -0.0061409, -0.0048574, -0.0061092, -0.0049053, -0.0006797, 0.0007054
5: 0.0114122, 0.0118983, 0.0114242, 0.0118802, -0.0002574, 0.0002672
6: 0.0015880, 0.0034432, 0.0016571, 0.0033973, -0.0010195, 0.0009824
7: 0.9791705, 0.9804686, 0.9792188, 0.9804365, -0.0007134, 0.0006874
8: -0.0088968, -0.0075049, -0.0088449, -0.0075393, -0.0007649, 0.0007370
9: -0.0000422, 0.0008772, -0.0000194, 0.0008430, -0.0004869, 0.0005053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004504
time: 0.58 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004498
time: 0.61 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011093, -0.0006120, -0.0010918, -0.0006198, -0.0002715, 0.0002704
1: -0.0071256, -0.0058637, -0.0070813, -0.0058833, -0.0006889, 0.0006861
2: 0.0306093, 0.0313921, 0.0306368, 0.0313800, -0.0004274, 0.0004257
3: 0.0009454, 0.0024072, 0.0009681, 0.0023559, -0.0007948, 0.0007980
4: -0.0061409, -0.0048574, -0.0060959, -0.0048774, -0.0007007, 0.0006979
5: 0.0114122, 0.0118983, 0.0114292, 0.0118908, -0.0002654, 0.0002643
6: 0.0015880, 0.0034432, 0.0016168, 0.0033781, -0.0010087, 0.0010128
7: 0.9791705, 0.9804686, 0.9791906, 0.9804231, -0.0007058, 0.0007087
8: -0.0088968, -0.0075049, -0.0088751, -0.0075538, -0.0007568, 0.0007598
9: -0.0000422, 0.0008772, -0.0000099, 0.0008629, -0.0005019, 0.0004999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004504
time: 0.55 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004498
time: 0.55 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011093, -0.0006120, -0.0011091, -0.0006264, -0.0002572, 0.0002746
1: -0.0071256, -0.0058637, -0.0071250, -0.0059001, -0.0006527, 0.0006969
2: 0.0306093, 0.0313921, 0.0306096, 0.0313696, -0.0004049, 0.0004324
3: 0.0009454, 0.0024072, 0.0009876, 0.0024066, -0.0008073, 0.0007561
4: -0.0061409, -0.0048574, -0.0061404, -0.0048944, -0.0006639, 0.0007089
5: 0.0114122, 0.0118983, 0.0114124, 0.0118843, -0.0002515, 0.0002685
6: 0.0015880, 0.0034432, 0.0016415, 0.0034424, -0.0010246, 0.0009596
7: 0.9791705, 0.9804686, 0.9792079, 0.9804681, -0.0007170, 0.0006715
8: -0.0088968, -0.0075049, -0.0088567, -0.0075055, -0.0007687, 0.0007200
9: -0.0000422, 0.0008772, -0.0000418, 0.0008507, -0.0004756, 0.0005078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004504
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004498
time: 0.60 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011093, -0.0006120, -0.0011042, -0.0006140, -0.0002659, 0.0002738
1: -0.0071256, -0.0058637, -0.0071128, -0.0058687, -0.0006748, 0.0006947
2: 0.0306093, 0.0313921, 0.0306172, 0.0313890, -0.0004186, 0.0004310
3: 0.0009454, 0.0024072, 0.0009512, 0.0023924, -0.0008048, 0.0007817
4: -0.0061409, -0.0048574, -0.0061279, -0.0048625, -0.0006864, 0.0007067
5: 0.0114122, 0.0118983, 0.0114171, 0.0118964, -0.0002600, 0.0002677
6: 0.0015880, 0.0034432, 0.0015953, 0.0034244, -0.0010214, 0.0009921
7: 0.9791705, 0.9804686, 0.9791756, 0.9804555, -0.0007147, 0.0006942
8: -0.0088968, -0.0075049, -0.0088913, -0.0075190, -0.0007663, 0.0007443
9: -0.0000422, 0.0008772, -0.0000328, 0.0008736, -0.0004917, 0.0005062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004504
time: 0.59 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004498
time: 0.62 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011166, -0.0006117, -0.0011124, -0.0006272, -0.0002737, 0.0002773
1: -0.0071441, -0.0058629, -0.0071334, -0.0059021, -0.0006945, 0.0007038
2: 0.0305978, 0.0313927, 0.0306044, 0.0313683, -0.0004309, 0.0004366
3: 0.0009444, 0.0024287, 0.0009899, 0.0024163, -0.0008153, 0.0008045
4: -0.0061598, -0.0048565, -0.0061489, -0.0048964, -0.0007064, 0.0007159
5: 0.0114050, 0.0118987, 0.0114091, 0.0118835, -0.0002676, 0.0002712
6: 0.0015867, 0.0034704, 0.0016444, 0.0034547, -0.0010347, 0.0010211
7: 0.9791696, 0.9804877, 0.9792099, 0.9804767, -0.0007241, 0.0007145
8: -0.0088977, -0.0074845, -0.0088545, -0.0074963, -0.0007763, 0.0007660
9: -0.0000557, 0.0008779, -0.0000479, 0.0008493, -0.0005060, 0.0005128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B2_B1_B1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004186
time: 0.60 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_B1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004523
time: 0.57 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011166, -0.0006117, -0.0011072, -0.0006180, -0.0002790, 0.0002749
1: -0.0071441, -0.0058629, -0.0071203, -0.0058789, -0.0007081, 0.0006975
2: 0.0305978, 0.0313927, 0.0306126, 0.0313827, -0.0004393, 0.0004327
3: 0.0009444, 0.0024287, 0.0009630, 0.0024011, -0.0008080, 0.0008203
4: -0.0061598, -0.0048565, -0.0061356, -0.0048729, -0.0007202, 0.0007095
5: 0.0114050, 0.0118987, 0.0114142, 0.0118925, -0.0002728, 0.0002687
6: 0.0015867, 0.0034704, 0.0016103, 0.0034355, -0.0010254, 0.0010410
7: 0.9791696, 0.9804877, 0.9791861, 0.9804633, -0.0007176, 0.0007285
8: -0.0088977, -0.0074845, -0.0088800, -0.0075107, -0.0007693, 0.0007810
9: -0.0000557, 0.0008779, -0.0000383, 0.0008662, -0.0005159, 0.0005082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B2_B1_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004186
time: 0.57 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004523
time: 0.58 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011166, -0.0006117, -0.0011243, -0.0006234, -0.0002680, 0.0002774
1: -0.0071441, -0.0058629, -0.0071635, -0.0058926, -0.0006800, 0.0007039
2: 0.0305978, 0.0313927, 0.0305857, 0.0313742, -0.0004219, 0.0004367
3: 0.0009444, 0.0024287, 0.0009789, 0.0024512, -0.0008154, 0.0007878
4: -0.0061598, -0.0048565, -0.0061795, -0.0048868, -0.0006917, 0.0007160
5: 0.0114050, 0.0118987, 0.0113975, 0.0118872, -0.0002620, 0.0002712
6: 0.0015867, 0.0034704, 0.0016304, 0.0034990, -0.0010349, 0.0009998
7: 0.9791696, 0.9804877, 0.9792001, 0.9805077, -0.0007242, 0.0006996
8: -0.0088977, -0.0074845, -0.0088649, -0.0074630, -0.0007764, 0.0007501
9: -0.0000557, 0.0008779, -0.0000698, 0.0008562, -0.0004955, 0.0005129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004186
time: 0.61 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004527
time: 0.60 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011166, -0.0006117, -0.0011196, -0.0006130, -0.0002737, 0.0002768
1: -0.0071441, -0.0058629, -0.0071518, -0.0058660, -0.0006946, 0.0007024
2: 0.0305978, 0.0313927, 0.0305930, 0.0313907, -0.0004310, 0.0004358
3: 0.0009444, 0.0024287, 0.0009481, 0.0024376, -0.0008137, 0.0008047
4: -0.0061598, -0.0048565, -0.0061676, -0.0048598, -0.0007066, 0.0007145
5: 0.0114050, 0.0118987, 0.0114021, 0.0118974, -0.0002676, 0.0002706
6: 0.0015867, 0.0034704, 0.0015914, 0.0034818, -0.0010327, 0.0010213
7: 0.9791696, 0.9804877, 0.9791729, 0.9804957, -0.0007226, 0.0007146
8: -0.0088977, -0.0074845, -0.0088942, -0.0074760, -0.0007748, 0.0007662
9: -0.0000557, 0.0008779, -0.0000613, 0.0008755, -0.0005061, 0.0005118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004186
time: 0.61 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004527
time: 0.61 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010934, -0.0006070, -0.0011004, -0.0006173, -0.0002628, 0.0002672
1: -0.0070852, -0.0058509, -0.0071029, -0.0058771, -0.0006668, 0.0006780
2: 0.0306343, 0.0314001, 0.0306233, 0.0313839, -0.0004137, 0.0004206
3: 0.0009306, 0.0023604, 0.0009609, 0.0023810, -0.0007854, 0.0007725
4: -0.0060998, -0.0048444, -0.0061179, -0.0048710, -0.0006782, 0.0006896
5: 0.0114277, 0.0119033, 0.0114209, 0.0118932, -0.0002569, 0.0002612
6: 0.0015692, 0.0033838, 0.0016076, 0.0034099, -0.0009968, 0.0009803
7: 0.9791573, 0.9804271, 0.9791842, 0.9804453, -0.0006975, 0.0006860
8: -0.0089109, -0.0075495, -0.0088821, -0.0075299, -0.0007478, 0.0007355
9: -0.0000127, 0.0008866, -0.0000257, 0.0008675, -0.0004858, 0.0004940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004260, upper bound: 0.0004352
time: 0.65 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004260, upper bound: 0.0004352
time: 0.61 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011042, -0.0006140, -0.0011004, -0.0006173, -0.0002815, 0.0002664
1: -0.0071128, -0.0058687, -0.0071029, -0.0058771, -0.0007143, 0.0006761
2: 0.0306172, 0.0313890, 0.0306233, 0.0313839, -0.0004431, 0.0004195
3: 0.0009512, 0.0023924, 0.0009609, 0.0023810, -0.0007832, 0.0008274
4: -0.0061279, -0.0048625, -0.0061179, -0.0048710, -0.0007265, 0.0006877
5: 0.0114171, 0.0118964, 0.0114209, 0.0118932, -0.0002752, 0.0002605
6: 0.0015953, 0.0034244, 0.0016076, 0.0034099, -0.0009940, 0.0010501
7: 0.9791756, 0.9804555, 0.9791842, 0.9804453, -0.0006956, 0.0007348
8: -0.0088913, -0.0075190, -0.0088821, -0.0075299, -0.0007458, 0.0007878
9: -0.0000328, 0.0008736, -0.0000257, 0.0008675, -0.0005204, 0.0004926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004260, upper bound: 0.0004352
time: 0.61 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004260, upper bound: 0.0004352
time: 0.60 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010934, -0.0006070, -0.0011118, -0.0006135, -0.0002574, 0.0002674
1: -0.0070852, -0.0058509, -0.0071320, -0.0058675, -0.0006531, 0.0006786
2: 0.0306343, 0.0314001, 0.0306053, 0.0313898, -0.0004052, 0.0004210
3: 0.0009306, 0.0023604, 0.0009498, 0.0024146, -0.0007861, 0.0007566
4: -0.0060998, -0.0048444, -0.0061475, -0.0048613, -0.0006643, 0.0006902
5: 0.0114277, 0.0119033, 0.0114097, 0.0118968, -0.0002516, 0.0002614
6: 0.0015692, 0.0033838, 0.0015936, 0.0034526, -0.0009976, 0.0009602
7: 0.9791573, 0.9804271, 0.9791744, 0.9804752, -0.0006981, 0.0006719
8: -0.0089109, -0.0075495, -0.0088926, -0.0074978, -0.0007485, 0.0007204
9: -0.0000127, 0.0008866, -0.0000468, 0.0008744, -0.0004758, 0.0004944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004265, upper bound: 0.0004352
time: 0.65 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004265, upper bound: 0.0004352
time: 0.63 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011042, -0.0006140, -0.0011118, -0.0006135, -0.0002762, 0.0002675
1: -0.0071128, -0.0058687, -0.0071320, -0.0058675, -0.0007008, 0.0006787
2: 0.0306172, 0.0313890, 0.0306053, 0.0313898, -0.0004348, 0.0004211
3: 0.0009512, 0.0023924, 0.0009498, 0.0024146, -0.0007863, 0.0008118
4: -0.0061279, -0.0048625, -0.0061475, -0.0048613, -0.0007128, 0.0006904
5: 0.0114171, 0.0118964, 0.0114097, 0.0118968, -0.0002700, 0.0002615
6: 0.0015953, 0.0034244, 0.0015936, 0.0034526, -0.0009979, 0.0010303
7: 0.9791756, 0.9804555, 0.9791744, 0.9804752, -0.0006983, 0.0007210
8: -0.0088913, -0.0075190, -0.0088926, -0.0074978, -0.0007487, 0.0007730
9: -0.0000328, 0.0008736, -0.0000468, 0.0008744, -0.0005106, 0.0004945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004265, upper bound: 0.0004352
time: 0.62 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004265, upper bound: 0.0004352
time: 0.63 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011080, -0.0006035, -0.0011076, -0.0006169, -0.0002635, 0.0002730
1: -0.0071223, -0.0058420, -0.0071214, -0.0058762, -0.0006687, 0.0006927
2: 0.0306113, 0.0314056, 0.0306119, 0.0313844, -0.0004149, 0.0004298
3: 0.0009203, 0.0024035, 0.0009598, 0.0024024, -0.0008025, 0.0007746
4: -0.0061376, -0.0048353, -0.0061367, -0.0048701, -0.0006802, 0.0007046
5: 0.0114134, 0.0119067, 0.0114138, 0.0118935, -0.0002576, 0.0002669
6: 0.0015561, 0.0034385, 0.0016063, 0.0034370, -0.0010184, 0.0009831
7: 0.9791481, 0.9804654, 0.9791833, 0.9804643, -0.0007127, 0.0006879
8: -0.0089207, -0.0075085, -0.0088831, -0.0075095, -0.0007641, 0.0007376
9: -0.0000398, 0.0008930, -0.0000391, 0.0008682, -0.0004872, 0.0005047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004579
time: 0.58 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004594
time: 0.68 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011196, -0.0006130, -0.0011076, -0.0006169, -0.0002827, 0.0002719
1: -0.0071518, -0.0058660, -0.0071214, -0.0058762, -0.0007175, 0.0006900
2: 0.0305930, 0.0313907, 0.0306119, 0.0313844, -0.0004451, 0.0004281
3: 0.0009481, 0.0024376, 0.0009598, 0.0024024, -0.0007993, 0.0008312
4: -0.0061676, -0.0048598, -0.0061367, -0.0048701, -0.0007298, 0.0007018
5: 0.0114021, 0.0118974, 0.0114138, 0.0118935, -0.0002764, 0.0002658
6: 0.0015914, 0.0034818, 0.0016063, 0.0034370, -0.0010144, 0.0010549
7: 0.9791729, 0.9804957, 0.9791833, 0.9804643, -0.0007098, 0.0007381
8: -0.0088942, -0.0074760, -0.0088831, -0.0075095, -0.0007611, 0.0007914
9: -0.0000613, 0.0008755, -0.0000391, 0.0008682, -0.0005228, 0.0005027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004578
time: 0.65 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004594
time: 0.64 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011080, -0.0006035, -0.0011189, -0.0006132, -0.0002569, 0.0002740
1: -0.0071223, -0.0058420, -0.0071499, -0.0058666, -0.0006519, 0.0006953
2: 0.0306113, 0.0314056, 0.0305942, 0.0313903, -0.0004045, 0.0004313
3: 0.0009203, 0.0024035, 0.0009488, 0.0024354, -0.0008054, 0.0007552
4: -0.0061376, -0.0048353, -0.0061657, -0.0048604, -0.0006631, 0.0007072
5: 0.0114134, 0.0119067, 0.0114028, 0.0118972, -0.0002512, 0.0002679
6: 0.0015561, 0.0034385, 0.0015923, 0.0034790, -0.0010222, 0.0009585
7: 0.9791481, 0.9804654, 0.9791735, 0.9804937, -0.0007153, 0.0006707
8: -0.0089207, -0.0075085, -0.0088936, -0.0074781, -0.0007669, 0.0007191
9: -0.0000398, 0.0008930, -0.0000599, 0.0008751, -0.0004750, 0.0005066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004580
time: 0.65 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004599
time: 0.60 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011196, -0.0006130, -0.0011189, -0.0006132, -0.0002766, 0.0002735
1: -0.0071518, -0.0058660, -0.0071499, -0.0058666, -0.0007018, 0.0006939
2: 0.0305930, 0.0313907, 0.0305942, 0.0313903, -0.0004354, 0.0004305
3: 0.0009481, 0.0024376, 0.0009488, 0.0024354, -0.0008039, 0.0008130
4: -0.0061676, -0.0048598, -0.0061657, -0.0048604, -0.0007139, 0.0007059
5: 0.0114021, 0.0118974, 0.0114028, 0.0118972, -0.0002704, 0.0002674
6: 0.0015914, 0.0034818, 0.0015923, 0.0034790, -0.0010203, 0.0010318
7: 0.9791729, 0.9804957, 0.9791735, 0.9804937, -0.0007139, 0.0007220
8: -0.0088942, -0.0074760, -0.0088936, -0.0074781, -0.0007654, 0.0007741
9: -0.0000613, 0.0008755, -0.0000599, 0.0008751, -0.0005114, 0.0005056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004580
time: 0.61 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004599
time: 0.59 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0011041, -0.0006016, -0.0010970, -0.0006306, -0.0002575, 0.0002814
1: -0.0071123, -0.0058372, -0.0070944, -0.0059108, -0.0006534, 0.0007141
2: 0.0306175, 0.0314086, 0.0306286, 0.0313630, -0.0004054, 0.0004430
3: 0.0009147, 0.0023918, 0.0009999, 0.0023711, -0.0008272, 0.0007569
4: -0.0061274, -0.0048304, -0.0061092, -0.0049053, -0.0006646, 0.0007263
5: 0.0114173, 0.0119085, 0.0114242, 0.0118802, -0.0002517, 0.0002751
6: 0.0015490, 0.0034237, 0.0016571, 0.0033973, -0.0010499, 0.0009607
7: 0.9791431, 0.9804550, 0.9792188, 0.9804365, -0.0007346, 0.0006722
8: -0.0089261, -0.0075196, -0.0088449, -0.0075393, -0.0007877, 0.0007207
9: -0.0000325, 0.0008966, -0.0000194, 0.0008430, -0.0004761, 0.0005203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_B2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004589
time: 0.67 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004578
time: 0.62 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0011041, -0.0006016, -0.0010918, -0.0006198, -0.0002674, 0.0002770
1: -0.0071123, -0.0058372, -0.0070813, -0.0058833, -0.0006786, 0.0007030
2: 0.0306175, 0.0314086, 0.0306368, 0.0313800, -0.0004210, 0.0004362
3: 0.0009147, 0.0023918, 0.0009681, 0.0023559, -0.0008144, 0.0007861
4: -0.0061274, -0.0048304, -0.0060959, -0.0048774, -0.0006902, 0.0007151
5: 0.0114173, 0.0119085, 0.0114292, 0.0118908, -0.0002614, 0.0002709
6: 0.0015490, 0.0034237, 0.0016168, 0.0033781, -0.0010336, 0.0009977
7: 0.9791431, 0.9804550, 0.9791906, 0.9804231, -0.0007233, 0.0006981
8: -0.0089261, -0.0075196, -0.0088751, -0.0075538, -0.0007755, 0.0007485
9: -0.0000325, 0.0008966, -0.0000099, 0.0008629, -0.0004944, 0.0005122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_B2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004589
time: 0.62 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004578
time: 0.63 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0011041, -0.0006016, -0.0011091, -0.0006264, -0.0002497, 0.0002815
1: -0.0071123, -0.0058372, -0.0071250, -0.0059001, -0.0006337, 0.0007144
2: 0.0306175, 0.0314086, 0.0306096, 0.0313696, -0.0003931, 0.0004432
3: 0.0009147, 0.0023918, 0.0009876, 0.0024066, -0.0008275, 0.0007341
4: -0.0061274, -0.0048304, -0.0061404, -0.0048944, -0.0006445, 0.0007266
5: 0.0114173, 0.0119085, 0.0114124, 0.0118843, -0.0002441, 0.0002752
6: 0.0015490, 0.0034237, 0.0016415, 0.0034424, -0.0010503, 0.0009316
7: 0.9791431, 0.9804550, 0.9792079, 0.9804681, -0.0007349, 0.0006519
8: -0.0089261, -0.0075196, -0.0088567, -0.0075055, -0.0007879, 0.0006990
9: -0.0000325, 0.0008966, -0.0000418, 0.0008507, -0.0004617, 0.0005205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_B2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004589
time: 0.61 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004579
time: 0.59 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0011041, -0.0006016, -0.0011042, -0.0006140, -0.0002617, 0.0002786
1: -0.0071123, -0.0058372, -0.0071128, -0.0058687, -0.0006642, 0.0007071
2: 0.0306175, 0.0314086, 0.0306172, 0.0313890, -0.0004121, 0.0004387
3: 0.0009147, 0.0023918, 0.0009512, 0.0023924, -0.0008191, 0.0007694
4: -0.0061274, -0.0048304, -0.0061279, -0.0048625, -0.0006756, 0.0007192
5: 0.0114173, 0.0119085, 0.0114171, 0.0118964, -0.0002559, 0.0002724
6: 0.0015490, 0.0034237, 0.0015953, 0.0034244, -0.0010396, 0.0009765
7: 0.9791431, 0.9804550, 0.9791756, 0.9804555, -0.0007274, 0.0006833
8: -0.0089261, -0.0075196, -0.0088913, -0.0075190, -0.0007799, 0.0007326
9: -0.0000325, 0.0008966, -0.0000328, 0.0008736, -0.0004839, 0.0005152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_B2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004589
time: 0.60 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004579
time: 0.59 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010936, -0.0006038, -0.0011150, -0.0006271, -0.0002587, 0.0002971
1: -0.0070857, -0.0058429, -0.0071401, -0.0059020, -0.0006566, 0.0007540
2: 0.0306340, 0.0314051, 0.0306003, 0.0313684, -0.0004073, 0.0004678
3: 0.0009213, 0.0023611, 0.0009897, 0.0024240, -0.0008735, 0.0007606
4: -0.0061004, -0.0048362, -0.0061557, -0.0048963, -0.0006678, 0.0007670
5: 0.0114275, 0.0119063, 0.0114066, 0.0118836, -0.0002530, 0.0002905
6: 0.0015574, 0.0033846, 0.0016442, 0.0034646, -0.0011086, 0.0009653
7: 0.9791490, 0.9804277, 0.9792098, 0.9804836, -0.0007757, 0.0006755
8: -0.0089198, -0.0075488, -0.0088546, -0.0074889, -0.0008317, 0.0007242
9: -0.0000132, 0.0008924, -0.0000528, 0.0008494, -0.0004784, 0.0005494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004377
time: 0.62 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004377
time: 0.65 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011082, -0.0006015, -0.0011150, -0.0006271, -0.0002600, 0.0002898
1: -0.0071229, -0.0058369, -0.0071401, -0.0059020, -0.0006599, 0.0007354
2: 0.0306109, 0.0314088, 0.0306003, 0.0313684, -0.0004094, 0.0004563
3: 0.0009144, 0.0024041, 0.0009897, 0.0024240, -0.0008520, 0.0007644
4: -0.0061382, -0.0048301, -0.0061557, -0.0048963, -0.0006712, 0.0007480
5: 0.0114132, 0.0119086, 0.0114066, 0.0118836, -0.0002542, 0.0002833
6: 0.0015486, 0.0034393, 0.0016442, 0.0034646, -0.0010812, 0.0009702
7: 0.9791429, 0.9804658, 0.9792098, 0.9804836, -0.0007566, 0.0006789
8: -0.0089263, -0.0075079, -0.0088546, -0.0074889, -0.0008112, 0.0007279
9: -0.0000402, 0.0008968, -0.0000528, 0.0008494, -0.0004808, 0.0005358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004602
time: 0.61 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004602
time: 0.63 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010936, -0.0006038, -0.0011273, -0.0006234, -0.0002530, 0.0002988
1: -0.0070857, -0.0058429, -0.0071713, -0.0058925, -0.0006420, 0.0007584
2: 0.0306340, 0.0314051, 0.0305809, 0.0313743, -0.0003983, 0.0004705
3: 0.0009213, 0.0023611, 0.0009788, 0.0024602, -0.0008785, 0.0007437
4: -0.0061004, -0.0048362, -0.0061875, -0.0048867, -0.0006530, 0.0007714
5: 0.0114275, 0.0119063, 0.0113945, 0.0118872, -0.0002473, 0.0002922
6: 0.0015574, 0.0033846, 0.0016303, 0.0035105, -0.0011150, 0.0009439
7: 0.9791490, 0.9804277, 0.9792001, 0.9805158, -0.0007802, 0.0006605
8: -0.0089198, -0.0075488, -0.0088650, -0.0074544, -0.0008365, 0.0007081
9: -0.0000132, 0.0008924, -0.0000755, 0.0008562, -0.0004678, 0.0005525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004377
time: 0.70 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004377
time: 0.58 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011082, -0.0006015, -0.0011273, -0.0006234, -0.0002536, 0.0002908
1: -0.0071229, -0.0058369, -0.0071713, -0.0058925, -0.0006436, 0.0007381
2: 0.0306109, 0.0314088, 0.0305809, 0.0313743, -0.0003993, 0.0004579
3: 0.0009144, 0.0024041, 0.0009788, 0.0024602, -0.0008550, 0.0007456
4: -0.0061382, -0.0048301, -0.0061875, -0.0048867, -0.0006546, 0.0007507
5: 0.0114132, 0.0119086, 0.0113945, 0.0118872, -0.0002480, 0.0002844
6: 0.0015486, 0.0034393, 0.0016303, 0.0035105, -0.0010851, 0.0009462
7: 0.9791429, 0.9804658, 0.9792001, 0.9805158, -0.0007593, 0.0006621
8: -0.0089263, -0.0075079, -0.0088650, -0.0074544, -0.0008141, 0.0007099
9: -0.0000402, 0.0008968, -0.0000755, 0.0008562, -0.0004689, 0.0005378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004607
time: 0.59 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004607
time: 0.58 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.24 seconds
IS_A1_B1_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004449
IS_A1_B1_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004449
IS_A1_B1_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004470
IS_A1_B1_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004470
IS_A1_B1_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004270
IS_A1_B1_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004270
IS_A1_B1_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
IS_A1_B1_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
IS_A1_B1_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004407, upper bound: 0.0004458
IS_A1_B1_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004407, upper bound: 0.0004458
IS_A1_B1_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004407, upper bound: 0.0004476
IS_A1_B1_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004407, upper bound: 0.0004476
IS_A1_B1_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004519, upper bound: 0.0004285
IS_A1_B1_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004519, upper bound: 0.0004285
IS_A1_B1_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004519, upper bound: 0.0004303
IS_A1_B1_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004519, upper bound: 0.0004303
IS_A1_B1_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004457
IS_A1_B1_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004447
IS_A1_B1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004477
IS_A1_B1_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004469
IS_A1_B1_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004291
IS_A1_B1_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004270
IS_A1_B1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004310
IS_A1_B1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
IS_A1_B1_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004188
IS_A1_B1_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004463
IS_A1_B1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004250
IS_A1_B1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004481
IS_A1_B1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004048
IS_A1_B1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004302
IS_A1_B1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004103
IS_A1_B1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004318
IS_A1_B2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004405
IS_A1_B2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004470
IS_A1_B2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004405
IS_A1_B2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004470
IS_A1_B2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004257
IS_A1_B2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
IS_A1_B2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004257
IS_A1_B2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
IS_A1_B2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004238
IS_A1_B2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004238
IS_A1_B2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004238
IS_A1_B2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004237, upper bound: 0.0004476
IS_A1_B2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004360, upper bound: 0.0004103
IS_A1_B2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004360, upper bound: 0.0004303
IS_A1_B2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004360, upper bound: 0.0004103
IS_A1_B2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004360, upper bound: 0.0004303
IS_A1_B2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004404
IS_A1_B2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004400
IS_A1_B2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004469
IS_A1_B2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004469
IS_A1_B2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004276
IS_A1_B2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004257
IS_A1_B2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004310
IS_A1_B2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
IS_A1_B2_B2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004078
IS_A1_B2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004250
IS_A1_B2_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004410
IS_A1_B2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004078, upper bound: 0.0004481
IS_A1_B2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0003942
IS_A1_B2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004103
IS_A1_B2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004283
IS_A1_B2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004318
IS_A2_A1_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004500
IS_A2_A1_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004499
IS_A2_A1_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004499
IS_A2_A1_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004500
IS_A2_A1_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004499
IS_A2_A1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004499
IS_A2_A1_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004499
IS_A2_A1_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004499
IS_A2_A1_B1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004290, upper bound: 0.0004520
IS_A2_A1_B1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004290, upper bound: 0.0004520
IS_A2_A1_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004290, upper bound: 0.0004520
IS_A2_A1_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004290, upper bound: 0.0004520
IS_A2_A1_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004524
IS_A2_A1_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004524
IS_A2_A1_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004524
IS_A2_A1_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004524
IS_A2_A1_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004504
IS_A2_A1_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004498
IS_A2_A1_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004504
IS_A2_A1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004498
IS_A2_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004504
IS_A2_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004498
IS_A2_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004504
IS_A2_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004045, upper bound: 0.0004498
IS_A2_A1_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004186
IS_A2_A1_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004523
IS_A2_A1_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004186
IS_A2_A1_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004523
IS_A2_A1_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004186
IS_A2_A1_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004527
IS_A2_A1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004186
IS_A2_A1_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004047, upper bound: 0.0004527
IS_A2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004260, upper bound: 0.0004352
IS_A2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004260, upper bound: 0.0004352
IS_A2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004260, upper bound: 0.0004352
IS_A2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004260, upper bound: 0.0004352
IS_A2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004265, upper bound: 0.0004352
IS_A2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004265, upper bound: 0.0004352
IS_A2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004265, upper bound: 0.0004352
IS_A2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004265, upper bound: 0.0004352
IS_A2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004579
IS_A2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004594
IS_A2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004578
IS_A2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004594
IS_A2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004580
IS_A2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004599
IS_A2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004580
IS_A2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0004100, upper bound: 0.0004599
IS_A2_A2_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004589
IS_A2_A2_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004578
IS_A2_A2_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004589
IS_A2_A2_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004578
IS_A2_A2_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004589
IS_A2_A2_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004579
IS_A2_A2_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004589
IS_A2_A2_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004579
IS_A2_A2_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004377
IS_A2_A2_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004377
IS_A2_A2_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004602
IS_A2_A2_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0004602
IS_A2_A2_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004377
IS_A2_A2_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004377
IS_A2_A2_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004607
IS_A2_A2_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 7, lower bound: -0.0003943, upper bound: 0.0004607

## BFS IS instance: IS_A1_B1_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010978, -0.0006174, -0.0010872, -0.0006198, -0.0002487, 0.0002417
1: -0.0070964, -0.0058772, -0.0070694, -0.0058834, -0.0006312, 0.0006132
2: 0.0306274, 0.0313838, 0.0306441, 0.0313800, -0.0003916, 0.0003805
3: 0.0009611, 0.0023735, 0.0009682, 0.0023421, -0.0007104, 0.0007312
4: -0.0061113, -0.0048711, -0.0060838, -0.0048774, -0.0006420, 0.0006238
5: 0.0114234, 0.0118931, 0.0114338, 0.0118908, -0.0002432, 0.0002363
6: 0.0016078, 0.0034004, 0.0016169, 0.0033606, -0.0009016, 0.0009280
7: 0.9791844, 0.9804387, 0.9791906, 0.9804108, -0.0006309, 0.0006494
8: -0.0088819, -0.0075370, -0.0088751, -0.0075669, -0.0006764, 0.0006962
9: -0.0000210, 0.0008674, -0.0000012, 0.0008629, -0.0004599, 0.0004468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004348
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004466
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0011075, -0.0006273, -0.0010872, -0.0006198, -0.0002670, 0.0002351
1: -0.0071209, -0.0059023, -0.0070694, -0.0058834, -0.0006776, 0.0005967
2: 0.0306122, 0.0313682, 0.0306441, 0.0313800, -0.0004204, 0.0003702
3: 0.0009901, 0.0024018, 0.0009682, 0.0023421, -0.0006912, 0.0007850
4: -0.0061362, -0.0048967, -0.0060838, -0.0048774, -0.0006892, 0.0006069
5: 0.0114140, 0.0118834, 0.0114338, 0.0118908, -0.0002611, 0.0002299
6: 0.0016447, 0.0034363, 0.0016169, 0.0033606, -0.0008773, 0.0009962
7: 0.9792102, 0.9804638, 0.9791906, 0.9804108, -0.0006139, 0.0006971
8: -0.0088542, -0.0075101, -0.0088751, -0.0075669, -0.0006582, 0.0007474
9: -0.0000388, 0.0008491, -0.0000012, 0.0008629, -0.0004937, 0.0004347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004348
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004466
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010920, -0.0006081, -0.0010872, -0.0006198, -0.0002410, 0.0002446
1: -0.0070818, -0.0058538, -0.0070694, -0.0058834, -0.0006116, 0.0006207
2: 0.0306365, 0.0313983, 0.0306441, 0.0313800, -0.0003794, 0.0003851
3: 0.0009339, 0.0023565, 0.0009682, 0.0023421, -0.0007191, 0.0007085
4: -0.0060964, -0.0048473, -0.0060838, -0.0048774, -0.0006221, 0.0006314
5: 0.0114290, 0.0119022, 0.0114338, 0.0118908, -0.0002356, 0.0002392
6: 0.0015733, 0.0033788, 0.0016169, 0.0033606, -0.0009126, 0.0008992
7: 0.9791602, 0.9804236, 0.9791906, 0.9804108, -0.0006386, 0.0006292
8: -0.0089078, -0.0075532, -0.0088751, -0.0075669, -0.0006847, 0.0006746
9: -0.0000103, 0.0008845, -0.0000012, 0.0008629, -0.0004456, 0.0004523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004370
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004470
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0011026, -0.0006181, -0.0010872, -0.0006198, -0.0002634, 0.0002430
1: -0.0071087, -0.0058791, -0.0070694, -0.0058834, -0.0006684, 0.0006167
2: 0.0306198, 0.0313826, 0.0306441, 0.0313800, -0.0004147, 0.0003826
3: 0.0009633, 0.0023876, 0.0009682, 0.0023421, -0.0007144, 0.0007743
4: -0.0061237, -0.0048731, -0.0060838, -0.0048774, -0.0006798, 0.0006273
5: 0.0114187, 0.0118924, 0.0114338, 0.0118908, -0.0002575, 0.0002376
6: 0.0016106, 0.0034183, 0.0016169, 0.0033606, -0.0009067, 0.0009827
7: 0.9791864, 0.9804513, 0.9791906, 0.9804108, -0.0006345, 0.0006876
8: -0.0088798, -0.0075236, -0.0088751, -0.0075669, -0.0006802, 0.0007372
9: -0.0000299, 0.0008660, -0.0000012, 0.0008629, -0.0004870, 0.0004493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004370
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004470
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010978, -0.0006174, -0.0010986, -0.0006156, -0.0002590, 0.0002591
1: -0.0070964, -0.0058772, -0.0070983, -0.0058728, -0.0006572, 0.0006576
2: 0.0306274, 0.0313838, 0.0306262, 0.0313865, -0.0004077, 0.0004080
3: 0.0009611, 0.0023735, 0.0009560, 0.0023756, -0.0007618, 0.0007613
4: -0.0061113, -0.0048711, -0.0061132, -0.0048667, -0.0006685, 0.0006689
5: 0.0114234, 0.0118931, 0.0114227, 0.0118948, -0.0002532, 0.0002534
6: 0.0016078, 0.0034004, 0.0016014, 0.0034031, -0.0009668, 0.0009662
7: 0.9791844, 0.9804387, 0.9791798, 0.9804407, -0.0006765, 0.0006761
8: -0.0088819, -0.0075370, -0.0088867, -0.0075350, -0.0007253, 0.0007249
9: -0.0000210, 0.0008674, -0.0000223, 0.0008706, -0.0004788, 0.0004791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004300, upper bound: 0.0004166
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004300, upper bound: 0.0004285
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0011075, -0.0006273, -0.0010986, -0.0006156, -0.0002773, 0.0002526
1: -0.0071209, -0.0059023, -0.0070983, -0.0058728, -0.0007036, 0.0006410
2: 0.0306122, 0.0313682, 0.0306262, 0.0313865, -0.0004365, 0.0003977
3: 0.0009901, 0.0024018, 0.0009560, 0.0023756, -0.0007426, 0.0008151
4: -0.0061362, -0.0048967, -0.0061132, -0.0048667, -0.0007157, 0.0006520
5: 0.0114140, 0.0118834, 0.0114227, 0.0118948, -0.0002711, 0.0002470
6: 0.0016447, 0.0034363, 0.0016014, 0.0034031, -0.0009424, 0.0010345
7: 0.9792102, 0.9804638, 0.9791798, 0.9804407, -0.0006595, 0.0007239
8: -0.0088542, -0.0075101, -0.0088867, -0.0075350, -0.0007071, 0.0007761
9: -0.0000388, 0.0008491, -0.0000223, 0.0008706, -0.0005127, 0.0004671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004300, upper bound: 0.0004166
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004300, upper bound: 0.0004285
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010920, -0.0006081, -0.0010986, -0.0006156, -0.0002513, 0.0002621
1: -0.0070818, -0.0058538, -0.0070983, -0.0058728, -0.0006376, 0.0006651
2: 0.0306365, 0.0313983, 0.0306262, 0.0313865, -0.0003956, 0.0004126
3: 0.0009339, 0.0023565, 0.0009560, 0.0023756, -0.0007705, 0.0007386
4: -0.0060964, -0.0048473, -0.0061132, -0.0048667, -0.0006485, 0.0006765
5: 0.0114290, 0.0119022, 0.0114227, 0.0118948, -0.0002456, 0.0002562
6: 0.0015733, 0.0033788, 0.0016014, 0.0034031, -0.0009778, 0.0009374
7: 0.9791602, 0.9804236, 0.9791798, 0.9804407, -0.0006842, 0.0006559
8: -0.0089078, -0.0075532, -0.0088867, -0.0075350, -0.0007336, 0.0007033
9: -0.0000103, 0.0008845, -0.0000223, 0.0008706, -0.0004646, 0.0004846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004186
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0011026, -0.0006181, -0.0010986, -0.0006156, -0.0002736, 0.0002605
1: -0.0071087, -0.0058791, -0.0070983, -0.0058728, -0.0006944, 0.0006610
2: 0.0306198, 0.0313826, 0.0306262, 0.0313865, -0.0004308, 0.0004101
3: 0.0009633, 0.0023876, 0.0009560, 0.0023756, -0.0007658, 0.0008044
4: -0.0061237, -0.0048731, -0.0061132, -0.0048667, -0.0007063, 0.0006724
5: 0.0114187, 0.0118924, 0.0114227, 0.0118948, -0.0002675, 0.0002547
6: 0.0016106, 0.0034183, 0.0016014, 0.0034031, -0.0009719, 0.0010209
7: 0.9791864, 0.9804513, 0.9791798, 0.9804407, -0.0006801, 0.0007144
8: -0.0088798, -0.0075236, -0.0088867, -0.0075350, -0.0007292, 0.0007659
9: -0.0000299, 0.0008660, -0.0000223, 0.0008706, -0.0005059, 0.0004816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004186
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0011051, -0.0006170, -0.0011023, -0.0006172, -0.0002601, 0.0002426
1: -0.0071148, -0.0058763, -0.0071078, -0.0058769, -0.0006600, 0.0006157
2: 0.0306159, 0.0313843, 0.0306203, 0.0313840, -0.0004095, 0.0003820
3: 0.0009600, 0.0023948, 0.0009606, 0.0023866, -0.0007132, 0.0007646
4: -0.0061300, -0.0048702, -0.0061229, -0.0048708, -0.0006714, 0.0006263
5: 0.0114163, 0.0118935, 0.0114190, 0.0118933, -0.0002543, 0.0002372
6: 0.0016065, 0.0034274, 0.0016073, 0.0034171, -0.0009052, 0.0009704
7: 0.9791834, 0.9804576, 0.9791840, 0.9804503, -0.0006334, 0.0006790
8: -0.0088829, -0.0075168, -0.0088823, -0.0075245, -0.0006791, 0.0007280
9: -0.0000344, 0.0008681, -0.0000292, 0.0008677, -0.0004809, 0.0004486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004195
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004502
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0011150, -0.0006269, -0.0011023, -0.0006172, -0.0002767, 0.0002391
1: -0.0071399, -0.0059015, -0.0071078, -0.0058769, -0.0007022, 0.0006067
2: 0.0306004, 0.0313687, 0.0306203, 0.0313840, -0.0004356, 0.0003764
3: 0.0009891, 0.0024238, 0.0009606, 0.0023866, -0.0007029, 0.0008135
4: -0.0061555, -0.0048958, -0.0061229, -0.0048708, -0.0007142, 0.0006171
5: 0.0114066, 0.0118838, 0.0114190, 0.0118933, -0.0002705, 0.0002338
6: 0.0016435, 0.0034643, 0.0016073, 0.0034171, -0.0008920, 0.0010324
7: 0.9792093, 0.9804835, 0.9791840, 0.9804503, -0.0006242, 0.0007224
8: -0.0088551, -0.0074891, -0.0088823, -0.0075245, -0.0006692, 0.0007745
9: -0.0000526, 0.0008497, -0.0000292, 0.0008677, -0.0005116, 0.0004421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004195
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004502
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010993, -0.0006078, -0.0011023, -0.0006172, -0.0002499, 0.0002492
1: -0.0071003, -0.0058529, -0.0071078, -0.0058769, -0.0006342, 0.0006324
2: 0.0306250, 0.0313989, 0.0306203, 0.0313840, -0.0003934, 0.0003923
3: 0.0009328, 0.0023779, 0.0009606, 0.0023866, -0.0007326, 0.0007346
4: -0.0061152, -0.0048464, -0.0061229, -0.0048708, -0.0006450, 0.0006432
5: 0.0114219, 0.0119025, 0.0114190, 0.0118933, -0.0002443, 0.0002436
6: 0.0015720, 0.0034060, 0.0016073, 0.0034171, -0.0009297, 0.0009323
7: 0.9791593, 0.9804426, 0.9791840, 0.9804503, -0.0006506, 0.0006524
8: -0.0089088, -0.0075328, -0.0088823, -0.0075245, -0.0006975, 0.0006995
9: -0.0000237, 0.0008851, -0.0000292, 0.0008677, -0.0004621, 0.0004608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004238
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004476
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0011098, -0.0006177, -0.0011023, -0.0006172, -0.0002713, 0.0002489
1: -0.0071269, -0.0058782, -0.0071078, -0.0058769, -0.0006886, 0.0006315
2: 0.0306084, 0.0313832, 0.0306203, 0.0313840, -0.0004272, 0.0003918
3: 0.0009622, 0.0024088, 0.0009606, 0.0023866, -0.0007316, 0.0007977
4: -0.0061423, -0.0048721, -0.0061229, -0.0048708, -0.0007004, 0.0006424
5: 0.0114116, 0.0118927, 0.0114190, 0.0118933, -0.0002653, 0.0002433
6: 0.0016093, 0.0034452, 0.0016073, 0.0034171, -0.0009285, 0.0010124
7: 0.9791853, 0.9804700, 0.9791840, 0.9804503, -0.0006497, 0.0007084
8: -0.0088808, -0.0075034, -0.0088823, -0.0075245, -0.0006966, 0.0007595
9: -0.0000432, 0.0008667, -0.0000292, 0.0008677, -0.0005017, 0.0004601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004238
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004476
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0011051, -0.0006170, -0.0011135, -0.0006135, -0.0002698, 0.0002602
1: -0.0071148, -0.0058763, -0.0071362, -0.0058673, -0.0006846, 0.0006602
2: 0.0306159, 0.0313843, 0.0306027, 0.0313899, -0.0004248, 0.0004096
3: 0.0009600, 0.0023948, 0.0009496, 0.0024195, -0.0007648, 0.0007931
4: -0.0061300, -0.0048702, -0.0061517, -0.0048611, -0.0006964, 0.0006715
5: 0.0114163, 0.0118935, 0.0114081, 0.0118969, -0.0002638, 0.0002544
6: 0.0016065, 0.0034274, 0.0015933, 0.0034587, -0.0009706, 0.0010066
7: 0.9791834, 0.9804576, 0.9791742, 0.9804795, -0.0006792, 0.0007043
8: -0.0088829, -0.0075168, -0.0088928, -0.0074933, -0.0007282, 0.0007552
9: -0.0000344, 0.0008681, -0.0000499, 0.0008746, -0.0004988, 0.0004810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004300, upper bound: 0.0004056
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004300, upper bound: 0.0004319
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0011150, -0.0006269, -0.0011135, -0.0006135, -0.0002864, 0.0002566
1: -0.0071399, -0.0059015, -0.0071362, -0.0058673, -0.0007268, 0.0006512
2: 0.0306004, 0.0313687, 0.0306027, 0.0313899, -0.0004509, 0.0004040
3: 0.0009891, 0.0024238, 0.0009496, 0.0024195, -0.0007544, 0.0008419
4: -0.0061555, -0.0048958, -0.0061517, -0.0048611, -0.0007393, 0.0006624
5: 0.0114066, 0.0118838, 0.0114081, 0.0118969, -0.0002800, 0.0002509
6: 0.0016435, 0.0034643, 0.0015933, 0.0034587, -0.0009574, 0.0010685
7: 0.9792093, 0.9804835, 0.9791742, 0.9804795, -0.0006700, 0.0007477
8: -0.0088551, -0.0074891, -0.0088928, -0.0074933, -0.0007183, 0.0008017
9: -0.0000526, 0.0008497, -0.0000499, 0.0008746, -0.0005295, 0.0004745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004300, upper bound: 0.0004056
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004300, upper bound: 0.0004319
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010993, -0.0006078, -0.0011135, -0.0006135, -0.0002596, 0.0002667
1: -0.0071003, -0.0058529, -0.0071362, -0.0058673, -0.0006587, 0.0006769
2: 0.0306250, 0.0313989, 0.0306027, 0.0313899, -0.0004087, 0.0004199
3: 0.0009328, 0.0023779, 0.0009496, 0.0024195, -0.0007841, 0.0007631
4: -0.0061152, -0.0048464, -0.0061517, -0.0048611, -0.0006701, 0.0006885
5: 0.0114219, 0.0119025, 0.0114081, 0.0118969, -0.0002538, 0.0002608
6: 0.0015720, 0.0034060, 0.0015933, 0.0034587, -0.0009952, 0.0009685
7: 0.9791593, 0.9804426, 0.9791742, 0.9804795, -0.0006964, 0.0006777
8: -0.0089088, -0.0075328, -0.0088928, -0.0074933, -0.0007466, 0.0007266
9: -0.0000237, 0.0008851, -0.0000499, 0.0008746, -0.0004800, 0.0004932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004103
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004303
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0011098, -0.0006177, -0.0011135, -0.0006135, -0.0002810, 0.0002664
1: -0.0071269, -0.0058782, -0.0071362, -0.0058673, -0.0007132, 0.0006760
2: 0.0306084, 0.0313832, 0.0306027, 0.0313899, -0.0004425, 0.0004194
3: 0.0009622, 0.0024088, 0.0009496, 0.0024195, -0.0007831, 0.0008262
4: -0.0061423, -0.0048721, -0.0061517, -0.0048611, -0.0007254, 0.0006876
5: 0.0114116, 0.0118927, 0.0114081, 0.0118969, -0.0002748, 0.0002605
6: 0.0016093, 0.0034452, 0.0015933, 0.0034587, -0.0009939, 0.0010485
7: 0.9791853, 0.9804700, 0.9791742, 0.9804795, -0.0006955, 0.0007337
8: -0.0088808, -0.0075034, -0.0088928, -0.0074933, -0.0007457, 0.0007867
9: -0.0000432, 0.0008667, -0.0000499, 0.0008746, -0.0005196, 0.0004926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004103
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004303
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010978, -0.0006174, -0.0010970, -0.0006306, -0.0002454, 0.0002609
1: -0.0070964, -0.0058772, -0.0070944, -0.0059108, -0.0006226, 0.0006619
2: 0.0306274, 0.0313838, 0.0306286, 0.0313630, -0.0003863, 0.0004107
3: 0.0009611, 0.0023735, 0.0009999, 0.0023711, -0.0007668, 0.0007213
4: -0.0061113, -0.0048711, -0.0061092, -0.0049053, -0.0006333, 0.0006733
5: 0.0114234, 0.0118931, 0.0114242, 0.0118802, -0.0002399, 0.0002550
6: 0.0016078, 0.0034004, 0.0016571, 0.0033973, -0.0009732, 0.0009154
7: 0.9791844, 0.9804387, 0.9792188, 0.9804365, -0.0006810, 0.0006405
8: -0.0088819, -0.0075370, -0.0088449, -0.0075393, -0.0007301, 0.0006868
9: -0.0000210, 0.0008674, -0.0000194, 0.0008430, -0.0004536, 0.0004823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004347
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004474
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0011075, -0.0006273, -0.0010970, -0.0006306, -0.0002526, 0.0002445
1: -0.0071209, -0.0059023, -0.0070944, -0.0059108, -0.0006410, 0.0006205
2: 0.0306122, 0.0313682, 0.0306286, 0.0313630, -0.0003977, 0.0003849
3: 0.0009901, 0.0024018, 0.0009999, 0.0023711, -0.0007188, 0.0007426
4: -0.0061362, -0.0048967, -0.0061092, -0.0049053, -0.0006520, 0.0006311
5: 0.0114140, 0.0118834, 0.0114242, 0.0118802, -0.0002470, 0.0002390
6: 0.0016447, 0.0034363, 0.0016571, 0.0033973, -0.0009122, 0.0009424
7: 0.9792102, 0.9804638, 0.9792188, 0.9804365, -0.0006383, 0.0006595
8: -0.0088542, -0.0075101, -0.0088449, -0.0075393, -0.0006844, 0.0007070
9: -0.0000388, 0.0008491, -0.0000194, 0.0008430, -0.0004670, 0.0004521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004342
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004464
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010920, -0.0006081, -0.0010970, -0.0006306, -0.0002376, 0.0002638
1: -0.0070818, -0.0058538, -0.0070944, -0.0059108, -0.0006030, 0.0006695
2: 0.0306365, 0.0313983, 0.0306286, 0.0313630, -0.0003741, 0.0004153
3: 0.0009339, 0.0023565, 0.0009999, 0.0023711, -0.0007755, 0.0006985
4: -0.0060964, -0.0048473, -0.0061092, -0.0049053, -0.0006133, 0.0006809
5: 0.0114290, 0.0119022, 0.0114242, 0.0118802, -0.0002323, 0.0002579
6: 0.0015733, 0.0033788, 0.0016571, 0.0033973, -0.0009842, 0.0008865
7: 0.9791602, 0.9804236, 0.9792188, 0.9804365, -0.0006887, 0.0006204
8: -0.0089078, -0.0075532, -0.0088449, -0.0075393, -0.0007384, 0.0006651
9: -0.0000103, 0.0008845, -0.0000194, 0.0008430, -0.0004394, 0.0004878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004369
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004477
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0011026, -0.0006181, -0.0010970, -0.0006306, -0.0002464, 0.0002499
1: -0.0071087, -0.0058791, -0.0070944, -0.0059108, -0.0006252, 0.0006341
2: 0.0306198, 0.0313826, 0.0306286, 0.0313630, -0.0003879, 0.0003934
3: 0.0009633, 0.0023876, 0.0009999, 0.0023711, -0.0007345, 0.0007243
4: -0.0061237, -0.0048731, -0.0061092, -0.0049053, -0.0006360, 0.0006449
5: 0.0114187, 0.0118924, 0.0114242, 0.0118802, -0.0002409, 0.0002443
6: 0.0016106, 0.0034183, 0.0016571, 0.0033973, -0.0009322, 0.0009192
7: 0.9791864, 0.9804513, 0.9792188, 0.9804365, -0.0006523, 0.0006432
8: -0.0088798, -0.0075236, -0.0088449, -0.0075393, -0.0006994, 0.0006897
9: -0.0000299, 0.0008660, -0.0000194, 0.0008430, -0.0004556, 0.0004620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004363
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004469
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010978, -0.0006174, -0.0011091, -0.0006264, -0.0002565, 0.0002781
1: -0.0070964, -0.0058772, -0.0071250, -0.0059001, -0.0006509, 0.0007057
2: 0.0306274, 0.0313838, 0.0306096, 0.0313696, -0.0004038, 0.0004378
3: 0.0009611, 0.0023735, 0.0009876, 0.0024066, -0.0008175, 0.0007541
4: -0.0061113, -0.0048711, -0.0061404, -0.0048944, -0.0006621, 0.0007178
5: 0.0114234, 0.0118931, 0.0114124, 0.0118843, -0.0002508, 0.0002719
6: 0.0016078, 0.0034004, 0.0016415, 0.0034424, -0.0010375, 0.0009570
7: 0.9791844, 0.9804387, 0.9792079, 0.9804681, -0.0007260, 0.0006697
8: -0.0088819, -0.0075370, -0.0088567, -0.0075055, -0.0007784, 0.0007180
9: -0.0000210, 0.0008674, -0.0000418, 0.0008507, -0.0004743, 0.0005142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004172
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004310
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0011075, -0.0006273, -0.0011091, -0.0006264, -0.0002635, 0.0002628
1: -0.0071209, -0.0059023, -0.0071250, -0.0059001, -0.0006686, 0.0006670
2: 0.0306122, 0.0313682, 0.0306096, 0.0313696, -0.0004148, 0.0004138
3: 0.0009901, 0.0024018, 0.0009876, 0.0024066, -0.0007727, 0.0007745
4: -0.0061362, -0.0048967, -0.0061404, -0.0048944, -0.0006801, 0.0006784
5: 0.0114140, 0.0118834, 0.0114124, 0.0118843, -0.0002576, 0.0002570
6: 0.0016447, 0.0034363, 0.0016415, 0.0034424, -0.0009806, 0.0009830
7: 0.9792102, 0.9804638, 0.9792079, 0.9804681, -0.0006862, 0.0006878
8: -0.0088542, -0.0075101, -0.0088567, -0.0075055, -0.0007357, 0.0007375
9: -0.0000388, 0.0008491, -0.0000418, 0.0008507, -0.0004871, 0.0004860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004166
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004285
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010920, -0.0006081, -0.0011091, -0.0006264, -0.0002488, 0.0002810
1: -0.0070818, -0.0058538, -0.0071250, -0.0059001, -0.0006313, 0.0007132
2: 0.0306365, 0.0313983, 0.0306096, 0.0313696, -0.0003917, 0.0004425
3: 0.0009339, 0.0023565, 0.0009876, 0.0024066, -0.0008262, 0.0007313
4: -0.0060964, -0.0048473, -0.0061404, -0.0048944, -0.0006421, 0.0007254
5: 0.0114290, 0.0119022, 0.0114124, 0.0118843, -0.0002432, 0.0002748
6: 0.0015733, 0.0033788, 0.0016415, 0.0034424, -0.0010485, 0.0009282
7: 0.9791602, 0.9804236, 0.9792079, 0.9804681, -0.0007337, 0.0006495
8: -0.0089078, -0.0075532, -0.0088567, -0.0075055, -0.0007867, 0.0006964
9: -0.0000103, 0.0008845, -0.0000418, 0.0008507, -0.0004600, 0.0005196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004188
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004310
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0011026, -0.0006181, -0.0011091, -0.0006264, -0.0002573, 0.0002682
1: -0.0071087, -0.0058791, -0.0071250, -0.0059001, -0.0006528, 0.0006806
2: 0.0306198, 0.0313826, 0.0306096, 0.0313696, -0.0004050, 0.0004222
3: 0.0009633, 0.0023876, 0.0009876, 0.0024066, -0.0007884, 0.0007563
4: -0.0061237, -0.0048731, -0.0061404, -0.0048944, -0.0006640, 0.0006923
5: 0.0114187, 0.0118924, 0.0114124, 0.0118843, -0.0002515, 0.0002622
6: 0.0016106, 0.0034183, 0.0016415, 0.0034424, -0.0010006, 0.0009598
7: 0.9791864, 0.9804513, 0.9792079, 0.9804681, -0.0007002, 0.0006716
8: -0.0088798, -0.0075236, -0.0088567, -0.0075055, -0.0007507, 0.0007201
9: -0.0000299, 0.0008660, -0.0000418, 0.0008507, -0.0004756, 0.0004959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004184
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010873, -0.0006188, -0.0011124, -0.0006272, -0.0002375, 0.0002755
1: -0.0070698, -0.0058807, -0.0071334, -0.0059021, -0.0006027, 0.0006992
2: 0.0306439, 0.0313816, 0.0306044, 0.0313683, -0.0003739, 0.0004338
3: 0.0009651, 0.0023426, 0.0009899, 0.0024163, -0.0008100, 0.0006982
4: -0.0060842, -0.0048747, -0.0061489, -0.0048964, -0.0006131, 0.0007112
5: 0.0114337, 0.0118918, 0.0114091, 0.0118835, -0.0002322, 0.0002694
6: 0.0016130, 0.0033612, 0.0016444, 0.0034547, -0.0010280, 0.0008862
7: 0.9791880, 0.9804112, 0.9792099, 0.9804767, -0.0007193, 0.0006201
8: -0.0088780, -0.0075664, -0.0088545, -0.0074963, -0.0007712, 0.0006648
9: -0.0000015, 0.0008648, -0.0000479, 0.0008493, -0.0004392, 0.0005094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004199
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004192
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0011025, -0.0006156, -0.0011124, -0.0006272, -0.0002391, 0.0002658
1: -0.0071082, -0.0058729, -0.0071334, -0.0059021, -0.0006068, 0.0006746
2: 0.0306201, 0.0313865, 0.0306044, 0.0313683, -0.0003764, 0.0004185
3: 0.0009560, 0.0023871, 0.0009899, 0.0024163, -0.0007815, 0.0007029
4: -0.0061233, -0.0048667, -0.0061489, -0.0048964, -0.0006172, 0.0006862
5: 0.0114188, 0.0118948, 0.0114091, 0.0118835, -0.0002338, 0.0002599
6: 0.0016014, 0.0034177, 0.0016444, 0.0034547, -0.0009918, 0.0008921
7: 0.9791799, 0.9804508, 0.9792099, 0.9804767, -0.0006940, 0.0006242
8: -0.0088867, -0.0075241, -0.0088545, -0.0074963, -0.0007441, 0.0006693
9: -0.0000295, 0.0008706, -0.0000479, 0.0008493, -0.0004421, 0.0004915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004508
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004192, upper bound: 0.0004500
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010816, -0.0006092, -0.0011124, -0.0006272, -0.0002312, 0.0002843
1: -0.0070554, -0.0058564, -0.0071334, -0.0059021, -0.0005868, 0.0007214
2: 0.0306528, 0.0313967, 0.0306044, 0.0313683, -0.0003640, 0.0004476
3: 0.0009370, 0.0023259, 0.0009899, 0.0024163, -0.0008357, 0.0006797
4: -0.0060696, -0.0048500, -0.0061489, -0.0048964, -0.0005968, 0.0007338
5: 0.0114392, 0.0119011, 0.0114091, 0.0118835, -0.0002261, 0.0002779
6: 0.0015773, 0.0033400, 0.0016444, 0.0034547, -0.0010606, 0.0008627
7: 0.9791629, 0.9803964, 0.9792099, 0.9804767, -0.0007422, 0.0006037
8: -0.0089048, -0.0075823, -0.0088545, -0.0074963, -0.0007957, 0.0006472
9: 0.0000090, 0.0008825, -0.0000479, 0.0008493, -0.0004275, 0.0005256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B2_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004250
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004237
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0010967, -0.0006064, -0.0011124, -0.0006272, -0.0002335, 0.0002746
1: -0.0070935, -0.0058493, -0.0071334, -0.0059021, -0.0005926, 0.0006968
2: 0.0306292, 0.0314011, 0.0306044, 0.0313683, -0.0003677, 0.0004323
3: 0.0009287, 0.0023700, 0.0009899, 0.0024163, -0.0008072, 0.0006865
4: -0.0061083, -0.0048428, -0.0061489, -0.0048964, -0.0006028, 0.0007087
5: 0.0114245, 0.0119039, 0.0114091, 0.0118835, -0.0002283, 0.0002685
6: 0.0015668, 0.0033960, 0.0016444, 0.0034547, -0.0010244, 0.0008713
7: 0.9791556, 0.9804356, 0.9792099, 0.9804767, -0.0007168, 0.0006097
8: -0.0089127, -0.0075403, -0.0088545, -0.0074963, -0.0007686, 0.0006537
9: -0.0000188, 0.0008877, -0.0000479, 0.0008493, -0.0004318, 0.0005077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004481
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004474
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010873, -0.0006188, -0.0011243, -0.0006234, -0.0002473, 0.0002924
1: -0.0070698, -0.0058807, -0.0071635, -0.0058926, -0.0006275, 0.0007420
2: 0.0306439, 0.0313816, 0.0305857, 0.0313742, -0.0003893, 0.0004603
3: 0.0009651, 0.0023426, 0.0009789, 0.0024512, -0.0008595, 0.0007270
4: -0.0060842, -0.0048747, -0.0061795, -0.0048868, -0.0006383, 0.0007547
5: 0.0114337, 0.0118918, 0.0113975, 0.0118872, -0.0002418, 0.0002859
6: 0.0016130, 0.0033612, 0.0016304, 0.0034990, -0.0010909, 0.0009226
7: 0.9791880, 0.9804112, 0.9792001, 0.9805077, -0.0007633, 0.0006456
8: -0.0088780, -0.0075664, -0.0088649, -0.0074630, -0.0008184, 0.0006922
9: -0.0000015, 0.0008648, -0.0000698, 0.0008562, -0.0004572, 0.0005406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004056
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004055
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0011025, -0.0006156, -0.0011243, -0.0006234, -0.0002502, 0.0002831
1: -0.0071082, -0.0058729, -0.0071635, -0.0058926, -0.0006350, 0.0007185
2: 0.0306201, 0.0313865, 0.0305857, 0.0313742, -0.0003940, 0.0004457
3: 0.0009560, 0.0023871, 0.0009789, 0.0024512, -0.0008323, 0.0007356
4: -0.0061233, -0.0048667, -0.0061795, -0.0048868, -0.0006459, 0.0007308
5: 0.0114188, 0.0118948, 0.0113975, 0.0118872, -0.0002446, 0.0002768
6: 0.0016014, 0.0034177, 0.0016304, 0.0034990, -0.0010563, 0.0009336
7: 0.9791799, 0.9804508, 0.9792001, 0.9805077, -0.0007391, 0.0006533
8: -0.0088867, -0.0075241, -0.0088649, -0.0074630, -0.0007925, 0.0007004
9: -0.0000295, 0.0008706, -0.0000698, 0.0008562, -0.0004627, 0.0005235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004341
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004320
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010816, -0.0006092, -0.0011243, -0.0006234, -0.0002410, 0.0003011
1: -0.0070554, -0.0058564, -0.0071635, -0.0058926, -0.0006116, 0.0007642
2: 0.0306528, 0.0313967, 0.0305857, 0.0313742, -0.0003794, 0.0004741
3: 0.0009370, 0.0023259, 0.0009789, 0.0024512, -0.0008852, 0.0007085
4: -0.0060696, -0.0048500, -0.0061795, -0.0048868, -0.0006221, 0.0007773
5: 0.0114392, 0.0119011, 0.0113975, 0.0118872, -0.0002356, 0.0002944
6: 0.0015773, 0.0033400, 0.0016304, 0.0034990, -0.0011235, 0.0008992
7: 0.9791629, 0.9803964, 0.9792001, 0.9805077, -0.0007862, 0.0006292
8: -0.0089048, -0.0075823, -0.0088649, -0.0074630, -0.0008429, 0.0006746
9: 0.0000090, 0.0008825, -0.0000698, 0.0008562, -0.0004456, 0.0005568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004103
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004100
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0010967, -0.0006064, -0.0011243, -0.0006234, -0.0002447, 0.0002919
1: -0.0070935, -0.0058493, -0.0071635, -0.0058926, -0.0006208, 0.0007406
2: 0.0306292, 0.0314011, 0.0305857, 0.0313742, -0.0003852, 0.0004595
3: 0.0009287, 0.0023700, 0.0009789, 0.0024512, -0.0008580, 0.0007192
4: -0.0061083, -0.0048428, -0.0061795, -0.0048868, -0.0006315, 0.0007534
5: 0.0114245, 0.0119039, 0.0113975, 0.0118872, -0.0002392, 0.0002853
6: 0.0015668, 0.0033960, 0.0016304, 0.0034990, -0.0010889, 0.0009128
7: 0.9791556, 0.9804356, 0.9792001, 0.9805077, -0.0007620, 0.0006387
8: -0.0089127, -0.0075403, -0.0088649, -0.0074630, -0.0008169, 0.0006848
9: -0.0000188, 0.0008877, -0.0000698, 0.0008562, -0.0004524, 0.0005396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004318
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004303
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010978, -0.0006174, -0.0010815, -0.0006111, -0.0002553, 0.0002353
1: -0.0070964, -0.0058772, -0.0070551, -0.0058614, -0.0006478, 0.0005972
2: 0.0306274, 0.0313838, 0.0306530, 0.0313936, -0.0004019, 0.0003705
3: 0.0009611, 0.0023735, 0.0009428, 0.0023256, -0.0006918, 0.0007505
4: -0.0061113, -0.0048711, -0.0060693, -0.0048551, -0.0006589, 0.0006074
5: 0.0114234, 0.0118931, 0.0114393, 0.0118992, -0.0002496, 0.0002301
6: 0.0016078, 0.0034004, 0.0015846, 0.0033396, -0.0008780, 0.0009524
7: 0.9791844, 0.9804387, 0.9791682, 0.9803962, -0.0006144, 0.0006665
8: -0.0088819, -0.0075370, -0.0088993, -0.0075826, -0.0006587, 0.0007145
9: -0.0000210, 0.0008674, 0.0000091, 0.0008789, -0.0004720, 0.0004351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004134, upper bound: 0.0004325
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004134, upper bound: 0.0004434
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0010920, -0.0006081, -0.0010815, -0.0006111, -0.0002528, 0.0002459
1: -0.0070818, -0.0058538, -0.0070551, -0.0058614, -0.0006416, 0.0006240
2: 0.0306365, 0.0313983, 0.0306530, 0.0313936, -0.0003981, 0.0003872
3: 0.0009339, 0.0023565, 0.0009428, 0.0023256, -0.0007229, 0.0007433
4: -0.0060964, -0.0048473, -0.0060693, -0.0048551, -0.0006526, 0.0006347
5: 0.0114290, 0.0119022, 0.0114393, 0.0118992, -0.0002472, 0.0002404
6: 0.0015733, 0.0033788, 0.0015846, 0.0033396, -0.0009175, 0.0009433
7: 0.9791602, 0.9804236, 0.9791682, 0.9803962, -0.0006420, 0.0006601
8: -0.0089078, -0.0075532, -0.0088993, -0.0075826, -0.0006883, 0.0007077
9: -0.0000103, 0.0008845, 0.0000091, 0.0008789, -0.0004675, 0.0004547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004134, upper bound: 0.0004398
time: 0.55 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004134, upper bound: 0.0004506
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0011075, -0.0006273, -0.0010815, -0.0006111, -0.0002736, 0.0002288
1: -0.0071209, -0.0059023, -0.0070551, -0.0058614, -0.0006942, 0.0005806
2: 0.0306122, 0.0313682, 0.0306530, 0.0313936, -0.0004307, 0.0003602
3: 0.0009901, 0.0024018, 0.0009428, 0.0023256, -0.0006726, 0.0008042
4: -0.0061362, -0.0048967, -0.0060693, -0.0048551, -0.0007061, 0.0005906
5: 0.0114140, 0.0118834, 0.0114393, 0.0118992, -0.0002675, 0.0002237
6: 0.0016447, 0.0034363, 0.0015846, 0.0033396, -0.0008536, 0.0010206
7: 0.9792102, 0.9804638, 0.9791682, 0.9803962, -0.0005973, 0.0007142
8: -0.0088542, -0.0075101, -0.0088993, -0.0075826, -0.0006404, 0.0007657
9: -0.0000388, 0.0008491, 0.0000091, 0.0008789, -0.0005058, 0.0004230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004295
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004405
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0011026, -0.0006181, -0.0010815, -0.0006111, -0.0002708, 0.0002392
1: -0.0071087, -0.0058791, -0.0070551, -0.0058614, -0.0006871, 0.0006069
2: 0.0306198, 0.0313826, 0.0306530, 0.0313936, -0.0004263, 0.0003765
3: 0.0009633, 0.0023876, 0.0009428, 0.0023256, -0.0007030, 0.0007960
4: -0.0061237, -0.0048731, -0.0060693, -0.0048551, -0.0006989, 0.0006173
5: 0.0114187, 0.0118924, 0.0114393, 0.0118992, -0.0002647, 0.0002338
6: 0.0016106, 0.0034183, 0.0015846, 0.0033396, -0.0008923, 0.0010102
7: 0.9791864, 0.9804513, 0.9791682, 0.9803962, -0.0006244, 0.0007069
8: -0.0088798, -0.0075236, -0.0088993, -0.0075826, -0.0006694, 0.0007579
9: -0.0000299, 0.0008660, 0.0000091, 0.0008789, -0.0005006, 0.0004422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004370
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004470
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010978, -0.0006174, -0.0010934, -0.0006070, -0.0002677, 0.0002552
1: -0.0070964, -0.0058772, -0.0070852, -0.0058509, -0.0006794, 0.0006477
2: 0.0306274, 0.0313838, 0.0306343, 0.0314001, -0.0004215, 0.0004018
3: 0.0009611, 0.0023735, 0.0009306, 0.0023604, -0.0007503, 0.0007870
4: -0.0061113, -0.0048711, -0.0060998, -0.0048444, -0.0006910, 0.0006588
5: 0.0114234, 0.0118931, 0.0114277, 0.0119033, -0.0002617, 0.0002495
6: 0.0016078, 0.0034004, 0.0015692, 0.0033838, -0.0009522, 0.0009988
7: 0.9791844, 0.9804387, 0.9791573, 0.9804271, -0.0006663, 0.0006989
8: -0.0088819, -0.0075370, -0.0089109, -0.0075495, -0.0007144, 0.0007494
9: -0.0000210, 0.0008674, -0.0000127, 0.0008866, -0.0004950, 0.0004719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004241, upper bound: 0.0004168
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004241, upper bound: 0.0004288
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0010920, -0.0006081, -0.0010934, -0.0006070, -0.0002631, 0.0002632
1: -0.0070818, -0.0058538, -0.0070852, -0.0058509, -0.0006676, 0.0006679
2: 0.0306365, 0.0313983, 0.0306343, 0.0314001, -0.0004142, 0.0004144
3: 0.0009339, 0.0023565, 0.0009306, 0.0023604, -0.0007737, 0.0007733
4: -0.0060964, -0.0048473, -0.0060998, -0.0048444, -0.0006790, 0.0006794
5: 0.0114290, 0.0119022, 0.0114277, 0.0119033, -0.0002572, 0.0002573
6: 0.0015733, 0.0033788, 0.0015692, 0.0033838, -0.0009820, 0.0009815
7: 0.9791602, 0.9804236, 0.9791573, 0.9804271, -0.0006871, 0.0006868
8: -0.0089078, -0.0075532, -0.0089109, -0.0075495, -0.0007367, 0.0007363
9: -0.0000103, 0.0008845, -0.0000127, 0.0008866, -0.0004864, 0.0004866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004241, upper bound: 0.0004227
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004241, upper bound: 0.0004331
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0011075, -0.0006273, -0.0010934, -0.0006070, -0.0002860, 0.0002487
1: -0.0071209, -0.0059023, -0.0070852, -0.0058509, -0.0007258, 0.0006311
2: 0.0306122, 0.0313682, 0.0306343, 0.0314001, -0.0004503, 0.0003915
3: 0.0009901, 0.0024018, 0.0009306, 0.0023604, -0.0007311, 0.0008408
4: -0.0061362, -0.0048967, -0.0060998, -0.0048444, -0.0007382, 0.0006419
5: 0.0114140, 0.0118834, 0.0114277, 0.0119033, -0.0002796, 0.0002432
6: 0.0016447, 0.0034363, 0.0015692, 0.0033838, -0.0009279, 0.0010670
7: 0.9792102, 0.9804638, 0.9791573, 0.9804271, -0.0006493, 0.0007467
8: -0.0088542, -0.0075101, -0.0089109, -0.0075495, -0.0006961, 0.0008005
9: -0.0000388, 0.0008491, -0.0000127, 0.0008866, -0.0005288, 0.0004598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004130
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004257
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0011026, -0.0006181, -0.0010934, -0.0006070, -0.0002810, 0.0002564
1: -0.0071087, -0.0058791, -0.0070852, -0.0058509, -0.0007131, 0.0006507
2: 0.0306198, 0.0313826, 0.0306343, 0.0314001, -0.0004424, 0.0004037
3: 0.0009633, 0.0023876, 0.0009306, 0.0023604, -0.0007539, 0.0008260
4: -0.0061237, -0.0048731, -0.0060998, -0.0048444, -0.0007253, 0.0006619
5: 0.0114187, 0.0118924, 0.0114277, 0.0119033, -0.0002747, 0.0002507
6: 0.0016106, 0.0034183, 0.0015692, 0.0033838, -0.0009567, 0.0010483
7: 0.9791864, 0.9804513, 0.9791573, 0.9804271, -0.0006695, 0.0007336
8: -0.0088798, -0.0075236, -0.0089109, -0.0075495, -0.0007178, 0.0007865
9: -0.0000299, 0.0008660, -0.0000127, 0.0008866, -0.0005195, 0.0004741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004186
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010897, -0.0006197, -0.0010966, -0.0006080, -0.0002450, 0.0002592
1: -0.0070759, -0.0058832, -0.0070932, -0.0058534, -0.0006218, 0.0006577
2: 0.0306401, 0.0313801, 0.0306294, 0.0313985, -0.0003858, 0.0004080
3: 0.0009679, 0.0023497, 0.0009335, 0.0023698, -0.0007619, 0.0007203
4: -0.0060904, -0.0048772, -0.0061080, -0.0048469, -0.0006325, 0.0006690
5: 0.0114313, 0.0118908, 0.0114246, 0.0119023, -0.0002396, 0.0002534
6: 0.0016166, 0.0033702, 0.0015728, 0.0033956, -0.0009670, 0.0009142
7: 0.9791905, 0.9804175, 0.9791599, 0.9804354, -0.0006766, 0.0006397
8: -0.0088753, -0.0075597, -0.0089082, -0.0075406, -0.0007255, 0.0006859
9: -0.0000060, 0.0008631, -0.0000186, 0.0008847, -0.0004530, 0.0004792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004134, upper bound: 0.0004134
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004134, upper bound: 0.0004273
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0011049, -0.0006172, -0.0010966, -0.0006080, -0.0002498, 0.0002461
1: -0.0071143, -0.0058767, -0.0070932, -0.0058534, -0.0006339, 0.0006245
2: 0.0306162, 0.0313841, 0.0306294, 0.0313985, -0.0003933, 0.0003874
3: 0.0009605, 0.0023942, 0.0009335, 0.0023698, -0.0007234, 0.0007344
4: -0.0061295, -0.0048706, -0.0061080, -0.0048469, -0.0006448, 0.0006352
5: 0.0114165, 0.0118933, 0.0114246, 0.0119023, -0.0002442, 0.0002406
6: 0.0016071, 0.0034267, 0.0015728, 0.0033956, -0.0009181, 0.0009320
7: 0.9791839, 0.9804571, 0.9791599, 0.9804354, -0.0006425, 0.0006522
8: -0.0088824, -0.0075173, -0.0089082, -0.0075406, -0.0006888, 0.0006992
9: -0.0000340, 0.0008677, -0.0000186, 0.0008847, -0.0004619, 0.0004550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004134, upper bound: 0.0004444
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004134, upper bound: 0.0004513
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010997, -0.0006305, -0.0010966, -0.0006080, -0.0002670, 0.0002557
1: -0.0071012, -0.0059106, -0.0070932, -0.0058534, -0.0006776, 0.0006490
2: 0.0306244, 0.0313631, 0.0306294, 0.0313985, -0.0004204, 0.0004026
3: 0.0009997, 0.0023790, 0.0009335, 0.0023698, -0.0007518, 0.0007849
4: -0.0061162, -0.0049051, -0.0061080, -0.0048469, -0.0006892, 0.0006601
5: 0.0114215, 0.0118803, 0.0114246, 0.0119023, -0.0002611, 0.0002500
6: 0.0016569, 0.0034074, 0.0015728, 0.0033956, -0.0009542, 0.0009962
7: 0.9792187, 0.9804436, 0.9791599, 0.9804354, -0.0006677, 0.0006971
8: -0.0088451, -0.0075317, -0.0089082, -0.0075406, -0.0007159, 0.0007474
9: -0.0000245, 0.0008431, -0.0000186, 0.0008847, -0.0004937, 0.0004729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004113
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004238
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0011150, -0.0006271, -0.0010966, -0.0006080, -0.0002721, 0.0002424
1: -0.0071401, -0.0059020, -0.0070932, -0.0058534, -0.0006904, 0.0006152
2: 0.0306003, 0.0313684, 0.0306294, 0.0313985, -0.0004283, 0.0003817
3: 0.0009897, 0.0024240, 0.0009335, 0.0023698, -0.0007126, 0.0007998
4: -0.0061557, -0.0048963, -0.0061080, -0.0048469, -0.0007023, 0.0006257
5: 0.0114066, 0.0118836, 0.0114246, 0.0119023, -0.0002660, 0.0002370
6: 0.0016442, 0.0034646, 0.0015728, 0.0033956, -0.0009044, 0.0010151
7: 0.9792098, 0.9804836, 0.9791599, 0.9804354, -0.0006329, 0.0007103
8: -0.0088546, -0.0074889, -0.0089082, -0.0075406, -0.0006786, 0.0007616
9: -0.0000528, 0.0008494, -0.0000186, 0.0008847, -0.0005031, 0.0004482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004112
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004476
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010897, -0.0006197, -0.0011080, -0.0006035, -0.0002563, 0.0002768
1: -0.0070759, -0.0058832, -0.0071223, -0.0058420, -0.0006505, 0.0007024
2: 0.0306401, 0.0313801, 0.0306113, 0.0314056, -0.0004036, 0.0004358
3: 0.0009679, 0.0023497, 0.0009203, 0.0024035, -0.0008137, 0.0007536
4: -0.0060904, -0.0048772, -0.0061376, -0.0048353, -0.0006617, 0.0007144
5: 0.0114313, 0.0118908, 0.0114134, 0.0119067, -0.0002506, 0.0002706
6: 0.0016166, 0.0033702, 0.0015561, 0.0034385, -0.0010326, 0.0009564
7: 0.9791905, 0.9804175, 0.9791481, 0.9804654, -0.0007226, 0.0006692
8: -0.0088753, -0.0075597, -0.0089207, -0.0075085, -0.0007747, 0.0007175
9: -0.0000060, 0.0008631, -0.0000398, 0.0008930, -0.0004740, 0.0005118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004241, upper bound: 0.0003986
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004241, upper bound: 0.0004149
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0011049, -0.0006172, -0.0011080, -0.0006035, -0.0002619, 0.0002634
1: -0.0071143, -0.0058767, -0.0071223, -0.0058420, -0.0006646, 0.0006683
2: 0.0306162, 0.0313841, 0.0306113, 0.0314056, -0.0004123, 0.0004146
3: 0.0009605, 0.0023942, 0.0009203, 0.0024035, -0.0007742, 0.0007699
4: -0.0061295, -0.0048706, -0.0061376, -0.0048353, -0.0006760, 0.0006798
5: 0.0114165, 0.0118933, 0.0114134, 0.0119067, -0.0002561, 0.0002575
6: 0.0016071, 0.0034267, 0.0015561, 0.0034385, -0.0009825, 0.0009771
7: 0.9791839, 0.9804571, 0.9791481, 0.9804654, -0.0006875, 0.0006838
8: -0.0088824, -0.0075173, -0.0089207, -0.0075085, -0.0007371, 0.0007331
9: -0.0000340, 0.0008677, -0.0000398, 0.0008930, -0.0004843, 0.0004869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004241, upper bound: 0.0004302
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004241, upper bound: 0.0004342
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010997, -0.0006305, -0.0011080, -0.0006035, -0.0002783, 0.0002733
1: -0.0071012, -0.0059106, -0.0071223, -0.0058420, -0.0007063, 0.0006937
2: 0.0306244, 0.0313631, 0.0306113, 0.0314056, -0.0004382, 0.0004303
3: 0.0009997, 0.0023790, 0.0009203, 0.0024035, -0.0008036, 0.0008182
4: -0.0061162, -0.0049051, -0.0061376, -0.0048353, -0.0007184, 0.0007056
5: 0.0114215, 0.0118803, 0.0114134, 0.0119067, -0.0002721, 0.0002673
6: 0.0016569, 0.0034074, 0.0015561, 0.0034385, -0.0010198, 0.0010384
7: 0.9792187, 0.9804436, 0.9791481, 0.9804654, -0.0007136, 0.0007266
8: -0.0088451, -0.0075317, -0.0089207, -0.0075085, -0.0007651, 0.0007790
9: -0.0000245, 0.0008431, -0.0000398, 0.0008930, -0.0005146, 0.0005054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0003956
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004103
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0011150, -0.0006271, -0.0011080, -0.0006035, -0.0002842, 0.0002597
1: -0.0071401, -0.0059020, -0.0071223, -0.0058420, -0.0007212, 0.0006590
2: 0.0306003, 0.0313684, 0.0306113, 0.0314056, -0.0004474, 0.0004088
3: 0.0009897, 0.0024240, 0.0009203, 0.0024035, -0.0007634, 0.0008354
4: -0.0061557, -0.0048963, -0.0061376, -0.0048353, -0.0007335, 0.0006703
5: 0.0114066, 0.0118836, 0.0114134, 0.0119067, -0.0002778, 0.0002539
6: 0.0016442, 0.0034646, 0.0015561, 0.0034385, -0.0009689, 0.0010603
7: 0.9792098, 0.9804836, 0.9791481, 0.9804654, -0.0006780, 0.0007419
8: -0.0088546, -0.0074889, -0.0089207, -0.0075085, -0.0007269, 0.0007955
9: -0.0000528, 0.0008494, -0.0000398, 0.0008930, -0.0005254, 0.0004801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004268
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004303
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010978, -0.0006174, -0.0010918, -0.0006198, -0.0002553, 0.0002579
1: -0.0070964, -0.0058772, -0.0070813, -0.0058833, -0.0006479, 0.0006546
2: 0.0306274, 0.0313838, 0.0306368, 0.0313800, -0.0004019, 0.0004061
3: 0.0009611, 0.0023735, 0.0009681, 0.0023559, -0.0007583, 0.0007505
4: -0.0061113, -0.0048711, -0.0060959, -0.0048774, -0.0006590, 0.0006658
5: 0.0114234, 0.0118931, 0.0114292, 0.0118908, -0.0002496, 0.0002522
6: 0.0016078, 0.0034004, 0.0016168, 0.0033781, -0.0009624, 0.0009525
7: 0.9791844, 0.9804387, 0.9791906, 0.9804231, -0.0006734, 0.0006665
8: -0.0088819, -0.0075370, -0.0088751, -0.0075538, -0.0007220, 0.0007146
9: -0.0000210, 0.0008674, -0.0000099, 0.0008629, -0.0004720, 0.0004769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004236, upper bound: 0.0004284
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004236, upper bound: 0.0004404
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0011075, -0.0006273, -0.0010918, -0.0006198, -0.0002607, 0.0002396
1: -0.0071209, -0.0059023, -0.0070813, -0.0058833, -0.0006617, 0.0006080
2: 0.0306122, 0.0313682, 0.0306368, 0.0313800, -0.0004105, 0.0003772
3: 0.0009901, 0.0024018, 0.0009681, 0.0023559, -0.0007043, 0.0007665
4: -0.0061362, -0.0048967, -0.0060959, -0.0048774, -0.0006730, 0.0006184
5: 0.0114140, 0.0118834, 0.0114292, 0.0118908, -0.0002549, 0.0002342
6: 0.0016447, 0.0034363, 0.0016168, 0.0033781, -0.0008939, 0.0009728
7: 0.9792102, 0.9804638, 0.9791906, 0.9804231, -0.0006255, 0.0006807
8: -0.0088542, -0.0075101, -0.0088751, -0.0075538, -0.0006706, 0.0007298
9: -0.0000388, 0.0008491, -0.0000099, 0.0008629, -0.0004821, 0.0004430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004236, upper bound: 0.0004282
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004236, upper bound: 0.0004400
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010920, -0.0006081, -0.0010918, -0.0006198, -0.0002494, 0.0002647
1: -0.0070818, -0.0058538, -0.0070813, -0.0058833, -0.0006328, 0.0006718
2: 0.0306365, 0.0313983, 0.0306368, 0.0313800, -0.0003926, 0.0004168
3: 0.0009339, 0.0023565, 0.0009681, 0.0023559, -0.0007782, 0.0007330
4: -0.0060964, -0.0048473, -0.0060959, -0.0048774, -0.0006436, 0.0006833
5: 0.0114290, 0.0119022, 0.0114292, 0.0118908, -0.0002438, 0.0002588
6: 0.0015733, 0.0033788, 0.0016168, 0.0033781, -0.0009877, 0.0009303
7: 0.9791602, 0.9804236, 0.9791906, 0.9804231, -0.0006911, 0.0006510
8: -0.0089078, -0.0075532, -0.0088751, -0.0075538, -0.0007410, 0.0006980
9: -0.0000103, 0.0008845, -0.0000099, 0.0008629, -0.0004610, 0.0004895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004370
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004477
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0011026, -0.0006181, -0.0010918, -0.0006198, -0.0002566, 0.0002484
1: -0.0071087, -0.0058791, -0.0070813, -0.0058833, -0.0006511, 0.0006304
2: 0.0306198, 0.0313826, 0.0306368, 0.0313800, -0.0004040, 0.0003911
3: 0.0009633, 0.0023876, 0.0009681, 0.0023559, -0.0007303, 0.0007543
4: -0.0061237, -0.0048731, -0.0060959, -0.0048774, -0.0006623, 0.0006412
5: 0.0114187, 0.0118924, 0.0114292, 0.0118908, -0.0002509, 0.0002429
6: 0.0016106, 0.0034183, 0.0016168, 0.0033781, -0.0009268, 0.0009573
7: 0.9791864, 0.9804513, 0.9791906, 0.9804231, -0.0006485, 0.0006699
8: -0.0088798, -0.0075236, -0.0088751, -0.0075538, -0.0006953, 0.0007182
9: -0.0000299, 0.0008660, -0.0000099, 0.0008629, -0.0004744, 0.0004593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004363
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004469
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010978, -0.0006174, -0.0011042, -0.0006140, -0.0002670, 0.0002769
1: -0.0070964, -0.0058772, -0.0071128, -0.0058687, -0.0006775, 0.0007026
2: 0.0306274, 0.0313838, 0.0306172, 0.0313890, -0.0004203, 0.0004359
3: 0.0009611, 0.0023735, 0.0009512, 0.0023924, -0.0008139, 0.0007848
4: -0.0061113, -0.0048711, -0.0061279, -0.0048625, -0.0006891, 0.0007146
5: 0.0114234, 0.0118931, 0.0114171, 0.0118964, -0.0002610, 0.0002707
6: 0.0016078, 0.0034004, 0.0015953, 0.0034244, -0.0010329, 0.0009961
7: 0.9791844, 0.9804387, 0.9791756, 0.9804555, -0.0007228, 0.0006970
8: -0.0088819, -0.0075370, -0.0088913, -0.0075190, -0.0007750, 0.0007473
9: -0.0000210, 0.0008674, -0.0000328, 0.0008736, -0.0004936, 0.0005119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004351, upper bound: 0.0004130
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004351, upper bound: 0.0004276
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0011075, -0.0006273, -0.0011042, -0.0006140, -0.0002728, 0.0002597
1: -0.0071209, -0.0059023, -0.0071128, -0.0058687, -0.0006921, 0.0006589
2: 0.0306122, 0.0313682, 0.0306172, 0.0313890, -0.0004294, 0.0004088
3: 0.0009901, 0.0024018, 0.0009512, 0.0023924, -0.0007634, 0.0008018
4: -0.0061362, -0.0048967, -0.0061279, -0.0048625, -0.0007040, 0.0006703
5: 0.0114140, 0.0118834, 0.0114171, 0.0118964, -0.0002667, 0.0002539
6: 0.0016447, 0.0034363, 0.0015953, 0.0034244, -0.0009688, 0.0010176
7: 0.9792102, 0.9804638, 0.9791756, 0.9804555, -0.0006779, 0.0007121
8: -0.0088542, -0.0075101, -0.0088913, -0.0075190, -0.0007268, 0.0007635
9: -0.0000388, 0.0008491, -0.0000328, 0.0008736, -0.0005043, 0.0004801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004351, upper bound: 0.0004127
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004351, upper bound: 0.0004257
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0010920, -0.0006081, -0.0011042, -0.0006140, -0.0002607, 0.0002819
1: -0.0070818, -0.0058538, -0.0071128, -0.0058687, -0.0006615, 0.0007154
2: 0.0306365, 0.0313983, 0.0306172, 0.0313890, -0.0004104, 0.0004438
3: 0.0009339, 0.0023565, 0.0009512, 0.0023924, -0.0008287, 0.0007663
4: -0.0060964, -0.0048473, -0.0061279, -0.0048625, -0.0006728, 0.0007276
5: 0.0114290, 0.0119022, 0.0114171, 0.0118964, -0.0002548, 0.0002756
6: 0.0015733, 0.0033788, 0.0015953, 0.0034244, -0.0010517, 0.0009725
7: 0.9791602, 0.9804236, 0.9791756, 0.9804555, -0.0007360, 0.0006805
8: -0.0089078, -0.0075532, -0.0088913, -0.0075190, -0.0007891, 0.0007296
9: -0.0000103, 0.0008845, -0.0000328, 0.0008736, -0.0004819, 0.0005212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004188
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004310
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0011026, -0.0006181, -0.0011042, -0.0006140, -0.0002675, 0.0002667
1: -0.0071087, -0.0058791, -0.0071128, -0.0058687, -0.0006788, 0.0006768
2: 0.0306198, 0.0313826, 0.0306172, 0.0313890, -0.0004211, 0.0004199
3: 0.0009633, 0.0023876, 0.0009512, 0.0023924, -0.0007840, 0.0007864
4: -0.0061237, -0.0048731, -0.0061279, -0.0048625, -0.0006905, 0.0006884
5: 0.0114187, 0.0118924, 0.0114171, 0.0118964, -0.0002615, 0.0002607
6: 0.0016106, 0.0034183, 0.0015953, 0.0034244, -0.0009950, 0.0009980
7: 0.9791864, 0.9804513, 0.9791756, 0.9804555, -0.0006962, 0.0006984
8: -0.0088798, -0.0075236, -0.0088913, -0.0075190, -0.0007465, 0.0007488
9: -0.0000299, 0.0008660, -0.0000328, 0.0008736, -0.0004946, 0.0004931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004184
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004186, upper bound: 0.0004294
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0010816, -0.0006092, -0.0011072, -0.0006180, -0.0002418, 0.0002793
1: -0.0070554, -0.0058564, -0.0071203, -0.0058789, -0.0006135, 0.0007088
2: 0.0306528, 0.0313967, 0.0306126, 0.0313827, -0.0003806, 0.0004397
3: 0.0009370, 0.0023259, 0.0009630, 0.0024011, -0.0008211, 0.0007108
4: -0.0060696, -0.0048500, -0.0061356, -0.0048729, -0.0006241, 0.0007209
5: 0.0114392, 0.0119011, 0.0114142, 0.0118925, -0.0002364, 0.0002731
6: 0.0015773, 0.0033400, 0.0016103, 0.0034355, -0.0010420, 0.0009020
7: 0.9791629, 0.9803964, 0.9791861, 0.9804633, -0.0007292, 0.0006312
8: -0.0089048, -0.0075823, -0.0088800, -0.0075107, -0.0007818, 0.0006768
9: 0.0000090, 0.0008825, -0.0000383, 0.0008662, -0.0004470, 0.0005164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004400, upper bound: 0.0004250
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004400, upper bound: 0.0004237
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0011025, -0.0006156, -0.0011072, -0.0006180, -0.0002469, 0.0002633
1: -0.0071082, -0.0058729, -0.0071203, -0.0058789, -0.0006266, 0.0006683
2: 0.0306201, 0.0313865, 0.0306126, 0.0313827, -0.0003888, 0.0004146
3: 0.0009560, 0.0023871, 0.0009630, 0.0024011, -0.0007742, 0.0007259
4: -0.0061233, -0.0048667, -0.0061356, -0.0048729, -0.0006374, 0.0006797
5: 0.0114188, 0.0118948, 0.0114142, 0.0118925, -0.0002414, 0.0002575
6: 0.0016014, 0.0034177, 0.0016103, 0.0034355, -0.0009825, 0.0009213
7: 0.9791799, 0.9804508, 0.9791861, 0.9804633, -0.0006875, 0.0006447
8: -0.0088867, -0.0075241, -0.0088800, -0.0075107, -0.0007371, 0.0006912
9: -0.0000295, 0.0008706, -0.0000383, 0.0008662, -0.0004566, 0.0004869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B2_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004090, upper bound: 0.0004410
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004090, upper bound: 0.0004407
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0010967, -0.0006064, -0.0011072, -0.0006180, -0.0002431, 0.0002694
1: -0.0070935, -0.0058493, -0.0071203, -0.0058789, -0.0006170, 0.0006836
2: 0.0306292, 0.0314011, 0.0306126, 0.0313827, -0.0003828, 0.0004241
3: 0.0009287, 0.0023700, 0.0009630, 0.0024011, -0.0007919, 0.0007147
4: -0.0061083, -0.0048428, -0.0061356, -0.0048729, -0.0006276, 0.0006953
5: 0.0114245, 0.0119039, 0.0114142, 0.0118925, -0.0002377, 0.0002634
6: 0.0015668, 0.0033960, 0.0016103, 0.0034355, -0.0010050, 0.0009071
7: 0.9791556, 0.9804356, 0.9791861, 0.9804633, -0.0007033, 0.0006347
8: -0.0089127, -0.0075403, -0.0088800, -0.0075107, -0.0007540, 0.0006805
9: -0.0000188, 0.0008877, -0.0000383, 0.0008662, -0.0004495, 0.0004981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004090, upper bound: 0.0004481
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004090, upper bound: 0.0004474
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0010873, -0.0006188, -0.0011196, -0.0006130, -0.0002548, 0.0002890
1: -0.0070698, -0.0058807, -0.0071518, -0.0058660, -0.0006465, 0.0007333
2: 0.0306439, 0.0313816, 0.0305930, 0.0313907, -0.0004011, 0.0004549
3: 0.0009651, 0.0023426, 0.0009481, 0.0024376, -0.0008495, 0.0007489
4: -0.0060842, -0.0048747, -0.0061676, -0.0048598, -0.0006576, 0.0007459
5: 0.0114337, 0.0118918, 0.0114021, 0.0118974, -0.0002491, 0.0002825
6: 0.0016130, 0.0033612, 0.0015914, 0.0034818, -0.0010781, 0.0009505
7: 0.9791880, 0.9804112, 0.9791729, 0.9804957, -0.0007544, 0.0006651
8: -0.0088780, -0.0075664, -0.0088942, -0.0074760, -0.0008089, 0.0007131
9: -0.0000015, 0.0008648, -0.0000613, 0.0008755, -0.0004710, 0.0005343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004498, upper bound: 0.0003942
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004498, upper bound: 0.0003943
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0010816, -0.0006092, -0.0011196, -0.0006130, -0.0002515, 0.0002961
1: -0.0070554, -0.0058564, -0.0071518, -0.0058660, -0.0006383, 0.0007514
2: 0.0306528, 0.0313967, 0.0305930, 0.0313907, -0.0003960, 0.0004662
3: 0.0009370, 0.0023259, 0.0009481, 0.0024376, -0.0008705, 0.0007395
4: -0.0060696, -0.0048500, -0.0061676, -0.0048598, -0.0006493, 0.0007643
5: 0.0114392, 0.0119011, 0.0114021, 0.0118974, -0.0002459, 0.0002895
6: 0.0015773, 0.0033400, 0.0015914, 0.0034818, -0.0011048, 0.0009385
7: 0.9791629, 0.9803964, 0.9791729, 0.9804957, -0.0007731, 0.0006567
8: -0.0089048, -0.0075823, -0.0088942, -0.0074760, -0.0008288, 0.0007041
9: 0.0000090, 0.0008825, -0.0000613, 0.0008755, -0.0004651, 0.0005475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.22 + 597.60 = 600.82 seconds
