## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00104286


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0166922, 0.0175774, 0.0166922, 0.0175774, -0.0005636, 0.0005636)
1: (-0.0007587, -0.0001237, -0.0007587, -0.0001237, -0.0004140, 0.0004140)
2: (0.0037880, 0.0040742, 0.0037880, 0.0040742, -0.0001804, 0.0001804)
3: (0.0016784, 0.0022152, 0.0016784, 0.0022152, -0.0002976, 0.0002976)
4: (-0.0041406, -0.0034485, -0.0041406, -0.0034485, -0.0003716, 0.0003716)
5: (-0.0000768, 0.0003110, -0.0000768, 0.0003110, -0.0002533, 0.0002533)
6: (-0.0040527, -0.0027053, -0.0040527, -0.0027053, -0.0006711, 0.0006711)
7: (-0.0199996, -0.0160513, -0.0199996, -0.0160513, -0.0021429, 0.0021429)
8: (0.9770662, 0.9805272, 0.9770662, 0.9805272, -0.0019770, 0.0019770)
9: (0.0028510, 0.0054188, 0.0028510, 0.0054188, -0.0014091, 0.0014091)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.34 = 2.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0013719, upper bound: 0.0013719

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012937, upper bound: 0.0013154
time: 0.53 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013154, upper bound: 0.0013154
time: 0.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.16 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.16
Output dim: 8, lower bound: -0.0012937, upper bound: 0.0013154
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.16
Output dim: 8, lower bound: -0.0013154, upper bound: 0.0013154

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0167337, 0.0175555, 0.0166959, 0.0175700, -0.0005026, 0.0005349
1: -0.0007324, -0.0001343, -0.0007556, -0.0001271, -0.0003763, 0.0004001
2: 0.0037959, 0.0040606, 0.0037907, 0.0040730, -0.0001703, 0.0001600
3: 0.0016741, 0.0021946, 0.0016788, 0.0022088, -0.0002722, 0.0002659
4: -0.0041112, -0.0034575, -0.0041303, -0.0034498, -0.0003123, 0.0002960
5: -0.0000717, 0.0002956, -0.0000752, 0.0003090, -0.0002465, 0.0002320
6: -0.0040654, -0.0027599, -0.0040520, -0.0027242, -0.0005498, 0.0005532
7: -0.0198346, -0.0161087, -0.0199418, -0.0160583, -0.0018090, 0.0017125
8: 0.9772027, 0.9804342, 0.9771141, 0.9805198, -0.0017176, 0.0016164
9: 0.0028942, 0.0053139, 0.0028555, 0.0053818, -0.0011312, 0.0011961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012937, upper bound: 0.0012938
time: 0.50 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012937, upper bound: 0.0013154
time: 0.52 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0166965, 0.0175659, 0.0166935, 0.0175740, -0.0005528, 0.0005302
1: -0.0007551, -0.0001299, -0.0007576, -0.0001255, -0.0004082, 0.0003997
2: 0.0037918, 0.0040729, 0.0037892, 0.0040738, -0.0001685, 0.0001768
3: 0.0016790, 0.0022013, 0.0016786, 0.0022110, -0.0002873, 0.0002653
4: -0.0041193, -0.0034497, -0.0041343, -0.0034490, -0.0002881, 0.0003511
5: -0.0000738, 0.0003087, -0.0000759, 0.0003103, -0.0002470, 0.0002504
6: -0.0040516, -0.0027544, -0.0040522, -0.0027208, -0.0006354, 0.0005056
7: -0.0198797, -0.0160580, -0.0199642, -0.0160538, -0.0016737, 0.0020277
8: 0.9771566, 0.9805196, 0.9770935, 0.9805245, -0.0016139, 0.0018896
9: 0.0028554, 0.0053418, 0.0028526, 0.0053961, -0.0013362, 0.0011098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013154, upper bound: 0.0012938
time: 0.54 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013154, upper bound: 0.0013154
time: 0.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.39 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 8, lower bound: -0.0012937, upper bound: 0.0012938
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 8, lower bound: -0.0012937, upper bound: 0.0013154
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 8, lower bound: -0.0013154, upper bound: 0.0012938
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 8, lower bound: -0.0013154, upper bound: 0.0013154

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0167337, 0.0175555, 0.0167337, 0.0175555, -0.0004887, 0.0004887
1: -0.0007324, -0.0001343, -0.0007324, -0.0001343, -0.0003691, 0.0003691
2: 0.0037959, 0.0040606, 0.0037959, 0.0040606, -0.0001552, 0.0001552
3: 0.0016741, 0.0021946, 0.0016741, 0.0021946, -0.0002545, 0.0002545
4: -0.0041112, -0.0034575, -0.0041112, -0.0034575, -0.0002712, 0.0002712
5: -0.0000717, 0.0002956, -0.0000717, 0.0002956, -0.0002284, 0.0002284
6: -0.0040654, -0.0027599, -0.0040654, -0.0027599, -0.0004981, 0.0004981
7: -0.0198346, -0.0161087, -0.0198346, -0.0161087, -0.0015724, 0.0015724
8: 0.9772027, 0.9804342, 0.9772027, 0.9804342, -0.0015041, 0.0015041
9: 0.0028942, 0.0053139, 0.0028942, 0.0053139, -0.0010405, 0.0010405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012476
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012331
time: 0.56 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0167337, 0.0175555, 0.0166965, 0.0175659, -0.0005019, 0.0005339
1: -0.0007324, -0.0001343, -0.0007551, -0.0001299, -0.0003745, 0.0003993
2: 0.0037959, 0.0040606, 0.0037918, 0.0040729, -0.0001700, 0.0001602
3: 0.0016741, 0.0021946, 0.0016790, 0.0022013, -0.0002694, 0.0002651
4: -0.0041112, -0.0034575, -0.0041193, -0.0034497, -0.0003121, 0.0003084
5: -0.0000717, 0.0002956, -0.0000738, 0.0003087, -0.0002460, 0.0002308
6: -0.0040654, -0.0027599, -0.0040516, -0.0027544, -0.0005739, 0.0005528
7: -0.0198346, -0.0161087, -0.0198797, -0.0160580, -0.0018079, 0.0017818
8: 0.9772027, 0.9804342, 0.9771566, 0.9805196, -0.0017161, 0.0016666
9: 0.0028942, 0.0053139, 0.0028554, 0.0053418, -0.0011745, 0.0011954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012676
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012473
time: 0.55 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0166965, 0.0175659, 0.0167337, 0.0175555, -0.0005339, 0.0005019
1: -0.0007551, -0.0001299, -0.0007324, -0.0001343, -0.0003993, 0.0003745
2: 0.0037918, 0.0040729, 0.0037959, 0.0040606, -0.0001602, 0.0001700
3: 0.0016790, 0.0022013, 0.0016741, 0.0021946, -0.0002651, 0.0002694
4: -0.0041193, -0.0034497, -0.0041112, -0.0034575, -0.0003084, 0.0003121
5: -0.0000738, 0.0003087, -0.0000717, 0.0002956, -0.0002308, 0.0002460
6: -0.0040516, -0.0027544, -0.0040654, -0.0027599, -0.0005528, 0.0005739
7: -0.0198797, -0.0160580, -0.0198346, -0.0161087, -0.0017818, 0.0018079
8: 0.9771566, 0.9805196, 0.9772027, 0.9804342, -0.0016666, 0.0017161
9: 0.0028554, 0.0053418, 0.0028942, 0.0053139, -0.0011954, 0.0011745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012467
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331
time: 0.51 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0166965, 0.0175659, 0.0166965, 0.0175659, -0.0005277, 0.0005277
1: -0.0007551, -0.0001299, -0.0007551, -0.0001299, -0.0003977, 0.0003977
2: 0.0037918, 0.0040729, 0.0037918, 0.0040729, -0.0001677, 0.0001677
3: 0.0016790, 0.0022013, 0.0016790, 0.0022013, -0.0002647, 0.0002647
4: -0.0041193, -0.0034497, -0.0041193, -0.0034497, -0.0002878, 0.0002878
5: -0.0000738, 0.0003087, -0.0000738, 0.0003087, -0.0002458, 0.0002458
6: -0.0040516, -0.0027544, -0.0040516, -0.0027544, -0.0005052, 0.0005052
7: -0.0198797, -0.0160580, -0.0198797, -0.0160580, -0.0016715, 0.0016715
8: 0.9771566, 0.9805196, 0.9771566, 0.9805196, -0.0016107, 0.0016107
9: 0.0028554, 0.0053418, 0.0028554, 0.0053418, -0.0011082, 0.0011082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012467
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331
time: 0.50 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.32 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012476
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012331
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012676
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012473
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012467
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012467
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0167385, 0.0175552, 0.0167337, 0.0175555, -0.0004775, 0.0004843
1: -0.0007284, -0.0001346, -0.0007324, -0.0001343, -0.0003606, 0.0003656
2: 0.0037959, 0.0040591, 0.0037959, 0.0040606, -0.0001538, 0.0001516
3: 0.0016748, 0.0021912, 0.0016741, 0.0021946, -0.0002518, 0.0002497
4: -0.0041112, -0.0034609, -0.0041112, -0.0034575, -0.0002704, 0.0002666
5: -0.0000714, 0.0002931, -0.0000717, 0.0002956, -0.0002261, 0.0002231
6: -0.0040611, -0.0027628, -0.0040654, -0.0027599, -0.0004934, 0.0004960
7: -0.0198345, -0.0161283, -0.0198346, -0.0161087, -0.0015670, 0.0015452
8: 0.9772027, 0.9804173, 0.9772027, 0.9804342, -0.0014949, 0.0014761
9: 0.0029072, 0.0053139, 0.0028942, 0.0053139, -0.0010220, 0.0010363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012331
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012331
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0166790, 0.0175804, 0.0167371, 0.0175552, -0.0006050, 0.0004995
1: -0.0007715, -0.0001158, -0.0007300, -0.0001347, -0.0004579, 0.0003753
2: 0.0037884, 0.0040779, 0.0037960, 0.0040595, -0.0001588, 0.0001917
3: 0.0016709, 0.0022076, 0.0016750, 0.0021929, -0.0002511, 0.0002925
4: -0.0041219, -0.0034472, -0.0041112, -0.0034611, -0.0002783, 0.0003036
5: -0.0000832, 0.0003185, -0.0000714, 0.0002942, -0.0002318, 0.0002832
6: -0.0040516, -0.0027423, -0.0040600, -0.0027615, -0.0004844, 0.0005172
7: -0.0198968, -0.0160419, -0.0198345, -0.0161291, -0.0016146, 0.0017760
8: 0.9771413, 0.9805384, 0.9772027, 0.9804161, -0.0015443, 0.0017634
9: 0.0028446, 0.0053551, 0.0029074, 0.0053139, -0.0011865, 0.0010692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012331
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012331
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0167385, 0.0175552, 0.0166965, 0.0175659, -0.0004907, 0.0005295
1: -0.0007284, -0.0001346, -0.0007551, -0.0001299, -0.0003660, 0.0003957
2: 0.0037959, 0.0040591, 0.0037918, 0.0040729, -0.0001686, 0.0001566
3: 0.0016748, 0.0021912, 0.0016790, 0.0022013, -0.0002666, 0.0002603
4: -0.0041112, -0.0034609, -0.0041193, -0.0034497, -0.0003113, 0.0003038
5: -0.0000714, 0.0002931, -0.0000738, 0.0003087, -0.0002438, 0.0002255
6: -0.0040611, -0.0027628, -0.0040516, -0.0027544, -0.0005692, 0.0005506
7: -0.0198345, -0.0161283, -0.0198797, -0.0160580, -0.0018026, 0.0017546
8: 0.9772027, 0.9804173, 0.9771566, 0.9805196, -0.0017069, 0.0016385
9: 0.0029072, 0.0053139, 0.0028554, 0.0053418, -0.0011561, 0.0011912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012473
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012473
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0166790, 0.0175804, 0.0166994, 0.0175657, -0.0006185, 0.0005440
1: -0.0007715, -0.0001158, -0.0007532, -0.0001303, -0.0004634, 0.0004052
2: 0.0037884, 0.0040779, 0.0037919, 0.0040719, -0.0001735, 0.0001968
3: 0.0016709, 0.0022076, 0.0016799, 0.0021997, -0.0002662, 0.0003025
4: -0.0041219, -0.0034472, -0.0041193, -0.0034533, -0.0003182, 0.0003409
5: -0.0000832, 0.0003185, -0.0000735, 0.0003075, -0.0002492, 0.0002857
6: -0.0040516, -0.0027423, -0.0040458, -0.0027560, -0.0005602, 0.0005705
7: -0.0198968, -0.0160419, -0.0198796, -0.0160785, -0.0018444, 0.0019863
8: 0.9771413, 0.9805384, 0.9771566, 0.9805025, -0.0017508, 0.0019264
9: 0.0028446, 0.0053551, 0.0028686, 0.0053418, -0.0013210, 0.0012200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012473
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012473
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0167008, 0.0175657, 0.0167337, 0.0175555, -0.0005234, 0.0004972
1: -0.0007515, -0.0001302, -0.0007324, -0.0001343, -0.0003909, 0.0003709
2: 0.0037918, 0.0040714, 0.0037959, 0.0040606, -0.0001587, 0.0001668
3: 0.0016797, 0.0021979, 0.0016741, 0.0021946, -0.0002628, 0.0002635
4: -0.0041193, -0.0034530, -0.0041112, -0.0034575, -0.0003075, 0.0003077
5: -0.0000735, 0.0003063, -0.0000717, 0.0002956, -0.0002286, 0.0002408
6: -0.0040474, -0.0027572, -0.0040654, -0.0027599, -0.0005484, 0.0005718
7: -0.0198795, -0.0160767, -0.0198346, -0.0161087, -0.0017758, 0.0017823
8: 0.9771566, 0.9805031, 0.9772027, 0.9804342, -0.0016569, 0.0016898
9: 0.0028675, 0.0053417, 0.0028942, 0.0053139, -0.0011781, 0.0011699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012472, upper bound: 0.0012331
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012472, upper bound: 0.0012331
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0166323, 0.0175889, 0.0167371, 0.0175552, -0.0006588, 0.0005084
1: -0.0008032, -0.0001125, -0.0007300, -0.0001347, -0.0004935, 0.0003785
2: 0.0037852, 0.0040932, 0.0037960, 0.0040595, -0.0001622, 0.0002094
3: 0.0016772, 0.0022154, 0.0016750, 0.0021929, -0.0002604, 0.0003027
4: -0.0041296, -0.0034364, -0.0041112, -0.0034611, -0.0003103, 0.0003480
5: -0.0000849, 0.0003371, -0.0000714, 0.0002942, -0.0002330, 0.0003038
6: -0.0040364, -0.0027351, -0.0040600, -0.0027615, -0.0005346, 0.0005846
7: -0.0199401, -0.0159690, -0.0198345, -0.0161291, -0.0017929, 0.0020318
8: 0.9771031, 0.9806509, 0.9772027, 0.9804161, -0.0016820, 0.0019950
9: 0.0027895, 0.0053829, 0.0029074, 0.0053139, -0.0013548, 0.0011823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0167008, 0.0175657, 0.0166965, 0.0175659, -0.0005171, 0.0005232
1: -0.0007515, -0.0001302, -0.0007551, -0.0001299, -0.0003893, 0.0003941
2: 0.0037918, 0.0040714, 0.0037918, 0.0040729, -0.0001663, 0.0001644
3: 0.0016797, 0.0021979, 0.0016790, 0.0022013, -0.0002621, 0.0002602
4: -0.0041193, -0.0034530, -0.0041193, -0.0034497, -0.0002870, 0.0002834
5: -0.0000735, 0.0003063, -0.0000738, 0.0003087, -0.0002435, 0.0002406
6: -0.0040474, -0.0027572, -0.0040516, -0.0027544, -0.0005006, 0.0005029
7: -0.0198795, -0.0160767, -0.0198797, -0.0160580, -0.0016660, 0.0016456
8: 0.9771566, 0.9805031, 0.9771566, 0.9805196, -0.0016013, 0.0015833
9: 0.0028675, 0.0053417, 0.0028554, 0.0053418, -0.0010905, 0.0011037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0166323, 0.0175889, 0.0166994, 0.0175657, -0.0006502, 0.0005368
1: -0.0008032, -0.0001125, -0.0007532, -0.0001303, -0.0004914, 0.0004023
2: 0.0037852, 0.0040932, 0.0037919, 0.0040719, -0.0001708, 0.0002062
3: 0.0016772, 0.0022154, 0.0016799, 0.0021997, -0.0002593, 0.0003046
4: -0.0041296, -0.0034364, -0.0041193, -0.0034533, -0.0002937, 0.0003233
5: -0.0000849, 0.0003371, -0.0000735, 0.0003075, -0.0002481, 0.0003035
6: -0.0040364, -0.0027351, -0.0040458, -0.0027560, -0.0004892, 0.0005238
7: -0.0199401, -0.0159690, -0.0198796, -0.0160785, -0.0017066, 0.0018943
8: 0.9771031, 0.9806509, 0.9771566, 0.9805025, -0.0016458, 0.0018906
9: 0.0027895, 0.0053829, 0.0028686, 0.0053418, -0.0012672, 0.0011321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331
time: 0.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.44 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012331
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012331
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012331
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012331
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012473
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012473
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012473
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012331, upper bound: 0.0012473
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012472, upper bound: 0.0012331
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012472, upper bound: 0.0012331
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 8, lower bound: -0.0012473, upper bound: 0.0012331

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0167385, 0.0175552, 0.0167385, 0.0175552, -0.0004731, 0.0004731
1: -0.0007284, -0.0001346, -0.0007284, -0.0001346, -0.0003571, 0.0003571
2: 0.0037959, 0.0040591, 0.0037959, 0.0040591, -0.0001502, 0.0001502
3: 0.0016748, 0.0021912, 0.0016748, 0.0021912, -0.0002469, 0.0002469
4: -0.0041112, -0.0034609, -0.0041112, -0.0034609, -0.0002658, 0.0002658
5: -0.0000714, 0.0002931, -0.0000714, 0.0002931, -0.0002209, 0.0002209
6: -0.0040611, -0.0027628, -0.0040611, -0.0027628, -0.0004913, 0.0004913
7: -0.0198345, -0.0161283, -0.0198345, -0.0161283, -0.0015399, 0.0015399
8: 0.9772027, 0.9804173, 0.9772027, 0.9804173, -0.0014669, 0.0014669
9: 0.0029072, 0.0053139, 0.0029072, 0.0053139, -0.0010178, 0.0010178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009017
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0167385, 0.0175552, 0.0166790, 0.0175804, -0.0005025, 0.0005424
1: -0.0007284, -0.0001346, -0.0007715, -0.0001158, -0.0003775, 0.0004106
2: 0.0037959, 0.0040591, 0.0037884, 0.0040779, -0.0001720, 0.0001598
3: 0.0016748, 0.0021912, 0.0016709, 0.0022076, -0.0002764, 0.0002500
4: -0.0041112, -0.0034609, -0.0041219, -0.0034472, -0.0002817, 0.0002805
5: -0.0000714, 0.0002931, -0.0000832, 0.0003185, -0.0002541, 0.0002330
6: -0.0040611, -0.0027628, -0.0040516, -0.0027423, -0.0005187, 0.0004812
7: -0.0198345, -0.0161283, -0.0198968, -0.0160419, -0.0016409, 0.0016273
8: 0.9772027, 0.9804173, 0.9771413, 0.9805384, -0.0016079, 0.0015569
9: 0.0029072, 0.0053139, 0.0028446, 0.0053551, -0.0010777, 0.0010918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009017
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0166790, 0.0175804, 0.0167385, 0.0175552, -0.0005424, 0.0005025
1: -0.0007715, -0.0001158, -0.0007284, -0.0001346, -0.0004106, 0.0003775
2: 0.0037884, 0.0040779, 0.0037959, 0.0040591, -0.0001598, 0.0001720
3: 0.0016709, 0.0022076, 0.0016748, 0.0021912, -0.0002500, 0.0002764
4: -0.0041219, -0.0034472, -0.0041112, -0.0034609, -0.0002805, 0.0002817
5: -0.0000832, 0.0003185, -0.0000714, 0.0002931, -0.0002330, 0.0002541
6: -0.0040516, -0.0027423, -0.0040611, -0.0027628, -0.0004812, 0.0005187
7: -0.0198968, -0.0160419, -0.0198345, -0.0161283, -0.0016273, 0.0016409
8: 0.9771413, 0.9805384, 0.9772027, 0.9804173, -0.0015569, 0.0016079
9: 0.0028446, 0.0053551, 0.0029072, 0.0053139, -0.0010918, 0.0010777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010217, upper bound: 0.0008913
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010207, upper bound: 0.0010207
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0166790, 0.0175804, 0.0166790, 0.0175804, -0.0005986, 0.0005985
1: -0.0007715, -0.0001158, -0.0007715, -0.0001158, -0.0004527, 0.0004527
2: 0.0037884, 0.0040779, 0.0037884, 0.0040779, -0.0001897, 0.0001897
3: 0.0016709, 0.0022076, 0.0016709, 0.0022076, -0.0002883, 0.0002883
4: -0.0041219, -0.0034472, -0.0041219, -0.0034472, -0.0003029, 0.0003029
5: -0.0000832, 0.0003185, -0.0000832, 0.0003185, -0.0002800, 0.0002800
6: -0.0040516, -0.0027423, -0.0040516, -0.0027423, -0.0005069, 0.0005069
7: -0.0198968, -0.0160419, -0.0198968, -0.0160419, -0.0017706, 0.0017706
8: 0.9771413, 0.9805384, 0.9771413, 0.9805384, -0.0017517, 0.0017517
9: 0.0028446, 0.0053551, 0.0028446, 0.0053551, -0.0011817, 0.0011817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010217, upper bound: 0.0008913
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010207, upper bound: 0.0010207
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0167385, 0.0175552, 0.0167008, 0.0175657, -0.0004859, 0.0005190
1: -0.0007284, -0.0001346, -0.0007515, -0.0001302, -0.0003624, 0.0003874
2: 0.0037959, 0.0040591, 0.0037918, 0.0040714, -0.0001654, 0.0001551
3: 0.0016748, 0.0021912, 0.0016797, 0.0021979, -0.0002607, 0.0002579
4: -0.0041112, -0.0034609, -0.0041193, -0.0034530, -0.0003069, 0.0003030
5: -0.0000714, 0.0002931, -0.0000735, 0.0003063, -0.0002385, 0.0002233
6: -0.0040611, -0.0027628, -0.0040474, -0.0027572, -0.0005672, 0.0005463
7: -0.0198345, -0.0161283, -0.0198795, -0.0160767, -0.0017770, 0.0017487
8: 0.9772027, 0.9804173, 0.9771566, 0.9805031, -0.0016806, 0.0016289
9: 0.0029072, 0.0053139, 0.0028675, 0.0053417, -0.0011514, 0.0011739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009014
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0167385, 0.0175552, 0.0166323, 0.0175889, -0.0005114, 0.0005966
1: -0.0007284, -0.0001346, -0.0008032, -0.0001125, -0.0003807, 0.0004471
2: 0.0037959, 0.0040591, 0.0037852, 0.0040932, -0.0001897, 0.0001631
3: 0.0016748, 0.0021912, 0.0016772, 0.0022154, -0.0002872, 0.0002593
4: -0.0041112, -0.0034609, -0.0041296, -0.0034364, -0.0003242, 0.0003125
5: -0.0000714, 0.0002931, -0.0000849, 0.0003371, -0.0002754, 0.0002342
6: -0.0040611, -0.0027628, -0.0040364, -0.0027351, -0.0005861, 0.0005298
7: -0.0198345, -0.0161283, -0.0199401, -0.0159690, -0.0018879, 0.0018055
8: 0.9772027, 0.9804173, 0.9771031, 0.9806509, -0.0018339, 0.0016946
9: 0.0029072, 0.0053139, 0.0027895, 0.0053829, -0.0011908, 0.0012547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009014
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0166790, 0.0175804, 0.0167008, 0.0175657, -0.0005553, 0.0005485
1: -0.0007715, -0.0001158, -0.0007515, -0.0001302, -0.0004159, 0.0004077
2: 0.0037884, 0.0040779, 0.0037918, 0.0040714, -0.0001750, 0.0001769
3: 0.0016709, 0.0022076, 0.0016797, 0.0021979, -0.0002639, 0.0002874
4: -0.0041219, -0.0034472, -0.0041193, -0.0034530, -0.0003215, 0.0003188
5: -0.0000832, 0.0003185, -0.0000735, 0.0003063, -0.0002506, 0.0002565
6: -0.0040516, -0.0027423, -0.0040474, -0.0027572, -0.0005571, 0.0005737
7: -0.0198968, -0.0160419, -0.0198795, -0.0160767, -0.0018644, 0.0018497
8: 0.9771413, 0.9805384, 0.9771566, 0.9805031, -0.0017706, 0.0017699
9: 0.0028446, 0.0053551, 0.0028675, 0.0053417, -0.0012254, 0.0012337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010217, upper bound: 0.0008906
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010207, upper bound: 0.0010366
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0166790, 0.0175804, 0.0166323, 0.0175889, -0.0006117, 0.0006524
1: -0.0007715, -0.0001158, -0.0008032, -0.0001125, -0.0004576, 0.0004883
2: 0.0037884, 0.0040779, 0.0037852, 0.0040932, -0.0002074, 0.0001947
3: 0.0016709, 0.0022076, 0.0016772, 0.0022154, -0.0002986, 0.0002954
4: -0.0041219, -0.0034472, -0.0041296, -0.0034364, -0.0003473, 0.0003407
5: -0.0000832, 0.0003185, -0.0000849, 0.0003371, -0.0003006, 0.0002819
6: -0.0040516, -0.0027423, -0.0040364, -0.0027351, -0.0005760, 0.0005570
7: -0.0198968, -0.0160419, -0.0199401, -0.0159690, -0.0020263, 0.0019830
8: 0.9771413, 0.9805384, 0.9771031, 0.9806509, -0.0019833, 0.0019165
9: 0.0028446, 0.0053551, 0.0027895, 0.0053829, -0.0013173, 0.0013499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010217, upper bound: 0.0008906
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010207, upper bound: 0.0010366
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0167008, 0.0175657, 0.0167385, 0.0175552, -0.0005190, 0.0004859
1: -0.0007515, -0.0001302, -0.0007284, -0.0001346, -0.0003874, 0.0003624
2: 0.0037918, 0.0040714, 0.0037959, 0.0040591, -0.0001551, 0.0001654
3: 0.0016797, 0.0021979, 0.0016748, 0.0021912, -0.0002579, 0.0002607
4: -0.0041193, -0.0034530, -0.0041112, -0.0034609, -0.0003030, 0.0003069
5: -0.0000735, 0.0003063, -0.0000714, 0.0002931, -0.0002233, 0.0002385
6: -0.0040474, -0.0027572, -0.0040611, -0.0027628, -0.0005463, 0.0005672
7: -0.0198795, -0.0160767, -0.0198345, -0.0161283, -0.0017487, 0.0017770
8: 0.9771566, 0.9805031, 0.9772027, 0.9804173, -0.0016289, 0.0016806
9: 0.0028675, 0.0053417, 0.0029072, 0.0053139, -0.0011739, 0.0011514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010450, upper bound: 0.0009042
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0167008, 0.0175657, 0.0166790, 0.0175804, -0.0005485, 0.0005553
1: -0.0007515, -0.0001302, -0.0007715, -0.0001158, -0.0004077, 0.0004159
2: 0.0037918, 0.0040714, 0.0037884, 0.0040779, -0.0001769, 0.0001750
3: 0.0016797, 0.0021979, 0.0016709, 0.0022076, -0.0002874, 0.0002639
4: -0.0041193, -0.0034530, -0.0041219, -0.0034472, -0.0003188, 0.0003215
5: -0.0000735, 0.0003063, -0.0000832, 0.0003185, -0.0002565, 0.0002506
6: -0.0040474, -0.0027572, -0.0040516, -0.0027423, -0.0005737, 0.0005571
7: -0.0198795, -0.0160767, -0.0198968, -0.0160419, -0.0018497, 0.0018644
8: 0.9771566, 0.9805031, 0.9771413, 0.9805384, -0.0017699, 0.0017706
9: 0.0028675, 0.0053417, 0.0028446, 0.0053551, -0.0012337, 0.0012254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010450, upper bound: 0.0009042
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0166323, 0.0175889, 0.0167385, 0.0175552, -0.0005966, 0.0005114
1: -0.0008032, -0.0001125, -0.0007284, -0.0001346, -0.0004471, 0.0003807
2: 0.0037852, 0.0040932, 0.0037959, 0.0040591, -0.0001631, 0.0001897
3: 0.0016772, 0.0022154, 0.0016748, 0.0021912, -0.0002593, 0.0002872
4: -0.0041296, -0.0034364, -0.0041112, -0.0034609, -0.0003125, 0.0003242
5: -0.0000849, 0.0003371, -0.0000714, 0.0002931, -0.0002342, 0.0002754
6: -0.0040364, -0.0027351, -0.0040611, -0.0027628, -0.0005298, 0.0005861
7: -0.0199401, -0.0159690, -0.0198345, -0.0161283, -0.0018055, 0.0018879
8: 0.9771031, 0.9806509, 0.9772027, 0.9804173, -0.0016946, 0.0018339
9: 0.0027895, 0.0053829, 0.0029072, 0.0053139, -0.0012547, 0.0011908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010421, upper bound: 0.0008931
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010366, upper bound: 0.0010207
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0166323, 0.0175889, 0.0166790, 0.0175804, -0.0006524, 0.0006117
1: -0.0008032, -0.0001125, -0.0007715, -0.0001158, -0.0004883, 0.0004576
2: 0.0037852, 0.0040932, 0.0037884, 0.0040779, -0.0001947, 0.0002074
3: 0.0016772, 0.0022154, 0.0016709, 0.0022076, -0.0002954, 0.0002986
4: -0.0041296, -0.0034364, -0.0041219, -0.0034472, -0.0003407, 0.0003473
5: -0.0000849, 0.0003371, -0.0000832, 0.0003185, -0.0002819, 0.0003006
6: -0.0040364, -0.0027351, -0.0040516, -0.0027423, -0.0005570, 0.0005760
7: -0.0199401, -0.0159690, -0.0198968, -0.0160419, -0.0019830, 0.0020263
8: 0.9771031, 0.9806509, 0.9771413, 0.9805384, -0.0019165, 0.0019833
9: 0.0027895, 0.0053829, 0.0028446, 0.0053551, -0.0013499, 0.0013173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010421, upper bound: 0.0008931
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010366, upper bound: 0.0010207
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0167008, 0.0175657, 0.0167008, 0.0175657, -0.0005126, 0.0005126
1: -0.0007515, -0.0001302, -0.0007515, -0.0001302, -0.0003857, 0.0003857
2: 0.0037918, 0.0040714, 0.0037918, 0.0040714, -0.0001630, 0.0001630
3: 0.0016797, 0.0021979, 0.0016797, 0.0021979, -0.0002575, 0.0002575
4: -0.0041193, -0.0034530, -0.0041193, -0.0034530, -0.0002826, 0.0002826
5: -0.0000735, 0.0003063, -0.0000735, 0.0003063, -0.0002383, 0.0002383
6: -0.0040474, -0.0027572, -0.0040474, -0.0027572, -0.0004984, 0.0004984
7: -0.0198795, -0.0160767, -0.0198795, -0.0160767, -0.0016401, 0.0016401
8: 0.9771566, 0.9805031, 0.9771566, 0.9805031, -0.0015739, 0.0015739
9: 0.0028675, 0.0053417, 0.0028675, 0.0053417, -0.0010860, 0.0010860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010450, upper bound: 0.0009042
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0167008, 0.0175657, 0.0166323, 0.0175889, -0.0005406, 0.0005896
1: -0.0007515, -0.0001302, -0.0008032, -0.0001125, -0.0004047, 0.0004451
2: 0.0037918, 0.0040714, 0.0037852, 0.0040932, -0.0001871, 0.0001721
3: 0.0016797, 0.0021979, 0.0016772, 0.0022154, -0.0002888, 0.0002585
4: -0.0041193, -0.0034530, -0.0041296, -0.0034364, -0.0003008, 0.0002966
5: -0.0000735, 0.0003063, -0.0000849, 0.0003371, -0.0002750, 0.0002494
6: -0.0040474, -0.0027572, -0.0040364, -0.0027351, -0.0005257, 0.0004858
7: -0.0198795, -0.0160767, -0.0199401, -0.0159690, -0.0017565, 0.0017236
8: 0.9771566, 0.9805031, 0.9771031, 0.9806509, -0.0017321, 0.0016616
9: 0.0028675, 0.0053417, 0.0027895, 0.0053829, -0.0011432, 0.0011705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010450, upper bound: 0.0009042
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0166323, 0.0175889, 0.0167008, 0.0175657, -0.0005896, 0.0005406
1: -0.0008032, -0.0001125, -0.0007515, -0.0001302, -0.0004451, 0.0004047
2: 0.0037852, 0.0040932, 0.0037918, 0.0040714, -0.0001721, 0.0001871
3: 0.0016772, 0.0022154, 0.0016797, 0.0021979, -0.0002585, 0.0002888
4: -0.0041296, -0.0034364, -0.0041193, -0.0034530, -0.0002966, 0.0003008
5: -0.0000849, 0.0003371, -0.0000735, 0.0003063, -0.0002494, 0.0002750
6: -0.0040364, -0.0027351, -0.0040474, -0.0027572, -0.0004858, 0.0005257
7: -0.0199401, -0.0159690, -0.0198795, -0.0160767, -0.0017236, 0.0017565
8: 0.9771031, 0.9806509, 0.9771566, 0.9805031, -0.0016616, 0.0017321
9: 0.0027895, 0.0053829, 0.0028675, 0.0053417, -0.0011705, 0.0011432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010421, upper bound: 0.0008931
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010366, upper bound: 0.0010207
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0166323, 0.0175889, 0.0166323, 0.0175889, -0.0006431, 0.0006431
1: -0.0008032, -0.0001125, -0.0008032, -0.0001125, -0.0004854, 0.0004854
2: 0.0037852, 0.0040932, 0.0037852, 0.0040932, -0.0002040, 0.0002040
3: 0.0016772, 0.0022154, 0.0016772, 0.0022154, -0.0002986, 0.0002986
4: -0.0041296, -0.0034364, -0.0041296, -0.0034364, -0.0003227, 0.0003227
5: -0.0000849, 0.0003371, -0.0000849, 0.0003371, -0.0002997, 0.0002997
6: -0.0040364, -0.0027351, -0.0040364, -0.0027351, -0.0005115, 0.0005115
7: -0.0199401, -0.0159690, -0.0199401, -0.0159690, -0.0018895, 0.0018895
8: 0.9771031, 0.9806509, 0.9771031, 0.9806509, -0.0018793, 0.0018793
9: 0.0027895, 0.0053829, 0.0027895, 0.0053829, -0.0012629, 0.0012629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010421, upper bound: 0.0008931
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010366, upper bound: 0.0010207
time: 0.59 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.62 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009017
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010217, upper bound: 0.0008913
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010207, upper bound: 0.0010207
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010217, upper bound: 0.0008913
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010207, upper bound: 0.0010207
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009014
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010217, upper bound: 0.0008906
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010207, upper bound: 0.0010366
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010217, upper bound: 0.0008906
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010207, upper bound: 0.0010366
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010450, upper bound: 0.0009042
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010450, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010421, upper bound: 0.0008931
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010366, upper bound: 0.0010207
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010421, upper bound: 0.0008931
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010366, upper bound: 0.0010207
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010450, upper bound: 0.0009042
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010450, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010421, upper bound: 0.0008931
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010366, upper bound: 0.0010207
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010421, upper bound: 0.0008931
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 8, lower bound: -0.0010366, upper bound: 0.0010207

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167589, 0.0175552, -0.0004337, 0.0006569
1: -0.0006838, -0.0000355, -0.0007097, -0.0001346, -0.0003223, 0.0004884
2: 0.0037466, 0.0040438, 0.0037959, 0.0040529, -0.0002094, 0.0001381
3: 0.0016724, 0.0021489, 0.0016748, 0.0021761, -0.0002745, 0.0002258
4: -0.0041714, -0.0034810, -0.0041112, -0.0034687, -0.0003445, 0.0002473
5: -0.0001271, 0.0002629, -0.0000714, 0.0002803, -0.0002997, 0.0001977
6: -0.0040608, -0.0027658, -0.0040611, -0.0027639, -0.0004904, 0.0004900
7: -0.0202023, -0.0162487, -0.0198345, -0.0161767, -0.0020218, 0.0014289
8: 0.9767592, 0.9803019, 0.9772027, 0.9803704, -0.0020176, 0.0013583
9: 0.0029887, 0.0055711, 0.0029403, 0.0053139, -0.0009425, 0.0013548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 96

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 240

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005552, upper bound: 0.0009796
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006242, upper bound: 0.0006242
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166949, 0.0175804, -0.0004274, 0.0007266
1: -0.0006838, -0.0000355, -0.0007584, -0.0001158, -0.0003172, 0.0005433
2: 0.0037466, 0.0040438, 0.0037884, 0.0040730, -0.0002311, 0.0001362
3: 0.0016724, 0.0021489, 0.0016709, 0.0021952, -0.0003072, 0.0002199
4: -0.0041714, -0.0034810, -0.0041219, -0.0034544, -0.0003600, 0.0002460
5: -0.0001271, 0.0002629, -0.0000832, 0.0003094, -0.0003340, 0.0001945
6: -0.0040608, -0.0027658, -0.0040516, -0.0027432, -0.0005182, 0.0004794
7: -0.0202023, -0.0162487, -0.0198968, -0.0160854, -0.0021217, 0.0014206
8: 0.9767592, 0.9803019, 0.9771413, 0.9804986, -0.0021554, 0.0013476
9: 0.0029887, 0.0055711, 0.0028748, 0.0053551, -0.0009366, 0.0014276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167097, 0.0175657, -0.0005371, 0.0007061
1: -0.0006838, -0.0000355, -0.0007439, -0.0001302, -0.0003886, 0.0005216
2: 0.0037466, 0.0040438, 0.0037918, 0.0040686, -0.0002255, 0.0001724
3: 0.0016724, 0.0021489, 0.0016797, 0.0021910, -0.0002896, 0.0002278
4: -0.0041714, -0.0034810, -0.0041193, -0.0034566, -0.0003870, 0.0003253
5: -0.0001271, 0.0002629, -0.0000735, 0.0003011, -0.0003191, 0.0002356
6: -0.0040608, -0.0027658, -0.0040474, -0.0027577, -0.0005667, 0.0005436
7: -0.0202023, -0.0162487, -0.0198795, -0.0160962, -0.0022666, 0.0018867
8: 0.9767592, 0.9803019, 0.9771566, 0.9804847, -0.0022360, 0.0018020
9: 0.0029887, 0.0055711, 0.0028806, 0.0053417, -0.0012492, 0.0015151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166423, 0.0175889, -0.0005307, 0.0007836
1: -0.0006838, -0.0000355, -0.0007952, -0.0001125, -0.0003838, 0.0005813
2: 0.0037466, 0.0040438, 0.0037852, 0.0040901, -0.0002498, 0.0001704
3: 0.0016724, 0.0021489, 0.0016772, 0.0022071, -0.0003173, 0.0002200
4: -0.0041714, -0.0034810, -0.0041296, -0.0034407, -0.0004039, 0.0003245
5: -0.0001271, 0.0002629, -0.0000849, 0.0003317, -0.0003562, 0.0002327
6: -0.0040608, -0.0027658, -0.0040364, -0.0027357, -0.0005856, 0.0005260
7: -0.0202023, -0.0162487, -0.0199401, -0.0159946, -0.0023752, 0.0018806
8: 0.9767592, 0.9803019, 0.9771031, 0.9806291, -0.0023872, 0.0017906
9: 0.0029887, 0.0055711, 0.0028064, 0.0053829, -0.0012441, 0.0015945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0168095, 0.0175657, 0.0167385, 0.0175552, -0.0003982, 0.0004859
1: -0.0006790, -0.0001302, -0.0007284, -0.0001346, -0.0003007, 0.0003624
2: 0.0037918, 0.0040363, 0.0037959, 0.0040591, -0.0001551, 0.0001266
3: 0.0016797, 0.0021823, 0.0016748, 0.0021912, -0.0002579, 0.0002506
4: -0.0041193, -0.0034943, -0.0041112, -0.0034609, -0.0003030, 0.0002558
5: -0.0000735, 0.0002641, -0.0000714, 0.0002931, -0.0002233, 0.0001868
6: -0.0040474, -0.0027580, -0.0040611, -0.0027628, -0.0005463, 0.0005668
7: -0.0198795, -0.0163305, -0.0198345, -0.0161283, -0.0017487, 0.0014655
8: 0.9771566, 0.9802017, 0.9772027, 0.9804173, -0.0016289, 0.0013271
9: 0.0030466, 0.0053417, 0.0029072, 0.0053139, -0.0009577, 0.0011514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 96

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167441, 0.0175552, -0.0005085, 0.0007366
1: -0.0007086, 0.0000066, -0.0007230, -0.0001346, -0.0003704, 0.0005419
2: 0.0037261, 0.0040555, 0.0037959, 0.0040574, -0.0002355, 0.0001629
3: 0.0016712, 0.0021497, 0.0016748, 0.0021872, -0.0002812, 0.0002275
4: -0.0041966, -0.0034726, -0.0041112, -0.0034631, -0.0004051, 0.0003063
5: -0.0001516, 0.0002768, -0.0000714, 0.0002896, -0.0003310, 0.0002251
6: -0.0040471, -0.0027609, -0.0040611, -0.0027631, -0.0005445, 0.0005643
7: -0.0203590, -0.0161919, -0.0198345, -0.0161415, -0.0023727, 0.0017728
8: 0.9765776, 0.9803886, 0.9772027, 0.9804042, -0.0023465, 0.0016830
9: 0.0029438, 0.0056825, 0.0029163, 0.0053139, -0.0011710, 0.0015862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 96

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0168095, 0.0175657, 0.0166790, 0.0175804, -0.0004276, 0.0005553
1: -0.0006790, -0.0001302, -0.0007715, -0.0001158, -0.0003211, 0.0004159
2: 0.0037918, 0.0040363, 0.0037884, 0.0040779, -0.0001769, 0.0001362
3: 0.0016797, 0.0021823, 0.0016709, 0.0022076, -0.0002874, 0.0002537
4: -0.0041193, -0.0034943, -0.0041219, -0.0034472, -0.0003188, 0.0002704
5: -0.0000735, 0.0002641, -0.0000832, 0.0003185, -0.0002565, 0.0001989
6: -0.0040474, -0.0027580, -0.0040516, -0.0027423, -0.0005737, 0.0005567
7: -0.0198795, -0.0163305, -0.0198968, -0.0160419, -0.0018497, 0.0015529
8: 0.9771566, 0.9802017, 0.9771413, 0.9805384, -0.0017699, 0.0014172
9: 0.0030466, 0.0053417, 0.0028446, 0.0053551, -0.0010175, 0.0012254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0166813, 0.0175804, -0.0005022, 0.0008074
1: -0.0007086, 0.0000066, -0.0007695, -0.0001158, -0.0003654, 0.0005967
2: 0.0037261, 0.0040555, 0.0037884, 0.0040772, -0.0002577, 0.0001609
3: 0.0016712, 0.0021497, 0.0016709, 0.0022059, -0.0003124, 0.0002216
4: -0.0041966, -0.0034726, -0.0041219, -0.0034482, -0.0004213, 0.0003050
5: -0.0001516, 0.0002768, -0.0000832, 0.0003171, -0.0003650, 0.0002219
6: -0.0040471, -0.0027609, -0.0040516, -0.0027425, -0.0005721, 0.0005537
7: -0.0203590, -0.0161919, -0.0198968, -0.0160484, -0.0024763, 0.0017646
8: 0.9765776, 0.9803886, 0.9771413, 0.9805322, -0.0024890, 0.0016723
9: 0.0029438, 0.0056825, 0.0028491, 0.0053551, -0.0011651, 0.0016620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0168095, 0.0175657, 0.0167211, 0.0175657, -0.0003970, 0.0004893
1: -0.0006790, -0.0001302, -0.0007376, -0.0001302, -0.0003026, 0.0003686
2: 0.0037918, 0.0040363, 0.0037918, 0.0040650, -0.0001555, 0.0001259
3: 0.0016797, 0.0021823, 0.0016797, 0.0021950, -0.0002528, 0.0002359
4: -0.0041193, -0.0034943, -0.0041193, -0.0034605, -0.0002733, 0.0002329
5: -0.0000735, 0.0002641, -0.0000735, 0.0002981, -0.0002279, 0.0001883
6: -0.0040474, -0.0027580, -0.0040474, -0.0027574, -0.0004983, 0.0004977
7: -0.0198795, -0.0163305, -0.0198795, -0.0161238, -0.0015831, 0.0013382
8: 0.9771566, 0.9802017, 0.9771566, 0.9804457, -0.0015085, 0.0012350
9: 0.0030466, 0.0053417, 0.0029017, 0.0053417, -0.0008771, 0.0010465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167226, 0.0175657, -0.0004587, 0.0007517
1: -0.0007086, 0.0000066, -0.0007335, -0.0001302, -0.0003405, 0.0005579
2: 0.0037261, 0.0040555, 0.0037918, 0.0040647, -0.0002396, 0.0001461
3: 0.0016712, 0.0021497, 0.0016797, 0.0021804, -0.0002999, 0.0002294
4: -0.0041966, -0.0034726, -0.0041193, -0.0034613, -0.0003821, 0.0002574
5: -0.0001516, 0.0002768, -0.0000735, 0.0002936, -0.0003420, 0.0002088
6: -0.0040471, -0.0027609, -0.0040474, -0.0027585, -0.0004973, 0.0004969
7: -0.0203590, -0.0161919, -0.0198795, -0.0161235, -0.0022506, 0.0014914
8: 0.9765776, 0.9803886, 0.9771566, 0.9804595, -0.0022747, 0.0014267
9: 0.0029438, 0.0056825, 0.0028986, 0.0053417, -0.0009867, 0.0015128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0168095, 0.0175657, 0.0166333, 0.0175889, -0.0004250, 0.0005883
1: -0.0006790, -0.0001302, -0.0008024, -0.0001125, -0.0003216, 0.0004442
2: 0.0037918, 0.0040363, 0.0037852, 0.0040928, -0.0001867, 0.0001350
3: 0.0016797, 0.0021823, 0.0016772, 0.0022152, -0.0002885, 0.0002369
4: -0.0041193, -0.0034943, -0.0041296, -0.0034368, -0.0003003, 0.0002469
5: -0.0000735, 0.0002641, -0.0000849, 0.0003367, -0.0002745, 0.0001994
6: -0.0040474, -0.0027580, -0.0040364, -0.0027351, -0.0005256, 0.0004852
7: -0.0198795, -0.0163305, -0.0199401, -0.0159715, -0.0017536, 0.0014217
8: 0.9771566, 0.9802017, 0.9771031, 0.9806479, -0.0017287, 0.0013227
9: 0.0030466, 0.0053417, 0.0027912, 0.0053829, -0.0009343, 0.0011684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0166505, 0.0175889, -0.0004521, 0.0008296
1: -0.0007086, 0.0000066, -0.0007888, -0.0001125, -0.0003347, 0.0006185
2: 0.0037261, 0.0040555, 0.0037852, 0.0040875, -0.0002640, 0.0001441
3: 0.0016712, 0.0021497, 0.0016772, 0.0022003, -0.0003336, 0.0002217
4: -0.0041966, -0.0034726, -0.0041296, -0.0034440, -0.0003998, 0.0002563
5: -0.0001516, 0.0002768, -0.0000849, 0.0003274, -0.0003796, 0.0002050
6: -0.0040471, -0.0027609, -0.0040364, -0.0027362, -0.0005249, 0.0004838
7: -0.0203590, -0.0161919, -0.0199401, -0.0160124, -0.0023635, 0.0014839
8: 0.9765776, 0.9803886, 0.9771031, 0.9806120, -0.0024295, 0.0014183
9: 0.0029438, 0.0056825, 0.0028177, 0.0053829, -0.0009816, 0.0015949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.61 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.72 seconds
IS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0005552, upper bound: 0.0009796
IS_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0006242, upper bound: 0.0006242
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168102, 0.0175804, -0.0004911, 0.0005877
1: -0.0006838, -0.0000355, -0.0006834, -0.0001158, -0.0003678, 0.0004419
2: 0.0037466, 0.0040438, 0.0037884, 0.0040352, -0.0001867, 0.0001563
3: 0.0016724, 0.0021489, 0.0016709, 0.0021890, -0.0002790, 0.0002350
4: -0.0041714, -0.0034810, -0.0041219, -0.0035013, -0.0003042, 0.0002755
5: -0.0001271, 0.0002629, -0.0000832, 0.0002674, -0.0002728, 0.0002267
6: -0.0040608, -0.0027658, -0.0040516, -0.0027434, -0.0005175, 0.0004781
7: -0.0202023, -0.0162487, -0.0198968, -0.0163695, -0.0017808, 0.0015992
8: 0.9767592, 0.9803019, 0.9771413, 0.9801622, -0.0017633, 0.0015303
9: 0.0029887, 0.0055711, 0.0030716, 0.0053551, -0.0010594, 0.0011901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167229, 0.0176959, -0.0004274, 0.0005122
1: -0.0006838, -0.0000355, -0.0007335, -0.0000461, -0.0003172, 0.0003832
2: 0.0037466, 0.0040438, 0.0037499, 0.0040645, -0.0001626, 0.0001362
3: 0.0016724, 0.0021489, 0.0016853, 0.0021666, -0.0002589, 0.0002199
4: -0.0041714, -0.0034810, -0.0041673, -0.0034687, -0.0002631, 0.0002460
5: -0.0001271, 0.0002629, -0.0001201, 0.0002934, -0.0002356, 0.0001945
6: -0.0040608, -0.0027658, -0.0040504, -0.0027452, -0.0005179, 0.0004794
7: -0.0202023, -0.0162487, -0.0201776, -0.0161689, -0.0015321, 0.0014206
8: 0.9767592, 0.9803019, 0.9767900, 0.9804238, -0.0015081, 0.0013476
9: 0.0029887, 0.0055711, 0.0029299, 0.0055534, -0.0009366, 0.0010189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009017
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168095, 0.0175657, -0.0004746, 0.0005880
1: -0.0006838, -0.0000355, -0.0006790, -0.0001302, -0.0003528, 0.0004373
2: 0.0037466, 0.0040438, 0.0037918, 0.0040363, -0.0001875, 0.0001517
3: 0.0016724, 0.0021489, 0.0016797, 0.0021823, -0.0002859, 0.0002429
4: -0.0041714, -0.0034810, -0.0041193, -0.0034943, -0.0003369, 0.0002980
5: -0.0001271, 0.0002629, -0.0000735, 0.0002641, -0.0002691, 0.0002171
6: -0.0040608, -0.0027658, -0.0040474, -0.0027580, -0.0005670, 0.0005431
7: -0.0202023, -0.0162487, -0.0198795, -0.0163305, -0.0019615, 0.0017206
8: 0.9767592, 0.9803019, 0.9771566, 0.9802017, -0.0018891, 0.0016022
9: 0.0029887, 0.0055711, 0.0030466, 0.0053417, -0.0011332, 0.0013033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167528, 0.0177701, -0.0005371, 0.0005085
1: -0.0006838, -0.0000355, -0.0007086, 0.0000066, -0.0003886, 0.0003704
2: 0.0037466, 0.0040438, 0.0037261, 0.0040555, -0.0001629, 0.0001724
3: 0.0016724, 0.0021489, 0.0016712, 0.0021497, -0.0002275, 0.0002278
4: -0.0041714, -0.0034810, -0.0041966, -0.0034726, -0.0003063, 0.0003253
5: -0.0001271, 0.0002629, -0.0001516, 0.0002768, -0.0002251, 0.0002356
6: -0.0040608, -0.0027658, -0.0040471, -0.0027609, -0.0005643, 0.0005436
7: -0.0202023, -0.0162487, -0.0203590, -0.0161919, -0.0017729, 0.0018867
8: 0.9767592, 0.9803019, 0.9765776, 0.9803886, -0.0016830, 0.0018020
9: 0.0029887, 0.0055711, 0.0029438, 0.0056825, -0.0012492, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0009283
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0011709
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168005, 0.0175889, -0.0005000, 0.0005974
1: -0.0006838, -0.0000355, -0.0006891, -0.0001125, -0.0003710, 0.0004472
2: 0.0037466, 0.0040438, 0.0037852, 0.0040384, -0.0001902, 0.0001597
3: 0.0016724, 0.0021489, 0.0016772, 0.0021904, -0.0002961, 0.0002442
4: -0.0041714, -0.0034810, -0.0041296, -0.0035002, -0.0003286, 0.0003075
5: -0.0001271, 0.0002629, -0.0000849, 0.0002707, -0.0002757, 0.0002280
6: -0.0040608, -0.0027658, -0.0040364, -0.0027364, -0.0005854, 0.0005267
7: -0.0202023, -0.0162487, -0.0199401, -0.0163629, -0.0019175, 0.0017775
8: 0.9767592, 0.9803019, 0.9771031, 0.9801837, -0.0018666, 0.0016680
9: 0.0029887, 0.0055711, 0.0030664, 0.0053829, -0.0011726, 0.0012767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166818, 0.0177592, -0.0005307, 0.0005843
1: -0.0006838, -0.0000355, -0.0007634, -0.0000025, -0.0003838, 0.0004294
2: 0.0037466, 0.0040438, 0.0037296, 0.0040776, -0.0001866, 0.0001704
3: 0.0016724, 0.0021489, 0.0016853, 0.0021678, -0.0002610, 0.0002200
4: -0.0041714, -0.0034810, -0.0041939, -0.0034568, -0.0003227, 0.0003245
5: -0.0001271, 0.0002629, -0.0001462, 0.0003108, -0.0002619, 0.0002327
6: -0.0040608, -0.0027658, -0.0040352, -0.0027386, -0.0005836, 0.0005260
7: -0.0202023, -0.0162487, -0.0203417, -0.0160892, -0.0018786, 0.0018806
8: 0.9767592, 0.9803019, 0.9766047, 0.9805312, -0.0018315, 0.0017906
9: 0.0029887, 0.0055711, 0.0028690, 0.0056690, -0.0012441, 0.0012484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009014
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168150, 0.0175552, -0.0005099, 0.0006573
1: -0.0007086, 0.0000066, -0.0006769, -0.0001346, -0.0003792, 0.0004855
2: 0.0037261, 0.0040555, 0.0037959, 0.0040343, -0.0002100, 0.0001627
3: 0.0016712, 0.0021497, 0.0016748, 0.0021812, -0.0002694, 0.0002329
4: -0.0041966, -0.0034726, -0.0041112, -0.0034933, -0.0003691, 0.0003019
5: -0.0001516, 0.0002768, -0.0000714, 0.0002628, -0.0002973, 0.0002330
6: -0.0040471, -0.0027609, -0.0040611, -0.0027634, -0.0005442, 0.0005626
7: -0.0203590, -0.0161919, -0.0198345, -0.0163245, -0.0021554, 0.0017478
8: 0.9765776, 0.9803886, 0.9772027, 0.9801946, -0.0021048, 0.0016551
9: 0.0029438, 0.0056825, 0.0030426, 0.0053139, -0.0011547, 0.0014365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167893, 0.0177068, -0.0005085, 0.0005371
1: -0.0007086, 0.0000066, -0.0006838, -0.0000355, -0.0003704, 0.0003886
2: 0.0037261, 0.0040555, 0.0037466, 0.0040438, -0.0001724, 0.0001629
3: 0.0016712, 0.0021497, 0.0016724, 0.0021489, -0.0002278, 0.0002275
4: -0.0041966, -0.0034726, -0.0041714, -0.0034810, -0.0003253, 0.0003063
5: -0.0001516, 0.0002768, -0.0001271, 0.0002629, -0.0002356, 0.0002251
6: -0.0040471, -0.0027609, -0.0040608, -0.0027658, -0.0005436, 0.0005643
7: -0.0203590, -0.0161919, -0.0202023, -0.0162487, -0.0018867, 0.0017728
8: 0.9765776, 0.9803886, 0.9767592, 0.9803019, -0.0018020, 0.0016830
9: 0.0029438, 0.0056825, 0.0029887, 0.0055711, -0.0011710, 0.0012492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0009320
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168102, 0.0175804, -0.0005393, 0.0006637
1: -0.0007086, 0.0000066, -0.0006834, -0.0001158, -0.0003996, 0.0004921
2: 0.0037261, 0.0040555, 0.0037884, 0.0040352, -0.0002118, 0.0001722
3: 0.0016712, 0.0021497, 0.0016709, 0.0021890, -0.0002805, 0.0002360
4: -0.0041966, -0.0034726, -0.0041219, -0.0035013, -0.0003631, 0.0003166
5: -0.0001516, 0.0002768, -0.0000832, 0.0002674, -0.0003018, 0.0002451
6: -0.0040471, -0.0027609, -0.0040516, -0.0027434, -0.0005708, 0.0005526
7: -0.0203590, -0.0161919, -0.0198968, -0.0163695, -0.0021225, 0.0018352
8: 0.9765776, 0.9803886, 0.9771413, 0.9801622, -0.0020845, 0.0017452
9: 0.0029438, 0.0056825, 0.0030716, 0.0053551, -0.0012145, 0.0014162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167229, 0.0176959, -0.0005022, 0.0006156
1: -0.0007086, 0.0000066, -0.0007335, -0.0000461, -0.0003654, 0.0004495
2: 0.0037261, 0.0040555, 0.0037499, 0.0040645, -0.0001969, 0.0001609
3: 0.0016712, 0.0021497, 0.0016853, 0.0021666, -0.0002609, 0.0002216
4: -0.0041966, -0.0034726, -0.0041673, -0.0034687, -0.0003411, 0.0003050
5: -0.0001516, 0.0002768, -0.0001201, 0.0002934, -0.0002735, 0.0002219
6: -0.0040471, -0.0027609, -0.0040504, -0.0027452, -0.0005715, 0.0005537
7: -0.0203590, -0.0161919, -0.0201776, -0.0161689, -0.0019900, 0.0017646
8: 0.9765776, 0.9803886, 0.9767900, 0.9804238, -0.0019517, 0.0016723
9: 0.0029438, 0.0056825, 0.0029299, 0.0055534, -0.0011651, 0.0013256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0009042
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168095, 0.0175657, -0.0005024, 0.0006421
1: -0.0007086, 0.0000066, -0.0006790, -0.0001302, -0.0003771, 0.0004801
2: 0.0037261, 0.0040555, 0.0037918, 0.0040363, -0.0002044, 0.0001598
3: 0.0016712, 0.0021497, 0.0016797, 0.0021823, -0.0002855, 0.0002437
4: -0.0041966, -0.0034726, -0.0041193, -0.0034943, -0.0003349, 0.0002781
5: -0.0001516, 0.0002768, -0.0000735, 0.0002641, -0.0002955, 0.0002327
6: -0.0040471, -0.0027609, -0.0040474, -0.0027580, -0.0004980, 0.0004945
7: -0.0203590, -0.0161919, -0.0198795, -0.0163305, -0.0019626, 0.0016146
8: 0.9765776, 0.9803886, 0.9771566, 0.9802017, -0.0019492, 0.0015508
9: 0.0029438, 0.0056825, 0.0030466, 0.0053417, -0.0010695, 0.0013128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167528, 0.0177701, -0.0004587, 0.0004587
1: -0.0007086, 0.0000066, -0.0007086, 0.0000066, -0.0003405, 0.0003405
2: 0.0037261, 0.0040555, 0.0037261, 0.0040555, -0.0001461, 0.0001461
3: 0.0016712, 0.0021497, 0.0016712, 0.0021497, -0.0002294, 0.0002294
4: -0.0041966, -0.0034726, -0.0041966, -0.0034726, -0.0002574, 0.0002574
5: -0.0001516, 0.0002768, -0.0001516, 0.0002768, -0.0002088, 0.0002088
6: -0.0040471, -0.0027609, -0.0040471, -0.0027609, -0.0004969, 0.0004969
7: -0.0203590, -0.0161919, -0.0203590, -0.0161919, -0.0014914, 0.0014914
8: 0.9765776, 0.9803886, 0.9765776, 0.9803886, -0.0014267, 0.0014267
9: 0.0029438, 0.0056825, 0.0029438, 0.0056825, -0.0009867, 0.0009867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0009320
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168005, 0.0175889, -0.0005304, 0.0006535
1: -0.0007086, 0.0000066, -0.0006891, -0.0001125, -0.0003961, 0.0004904
2: 0.0037261, 0.0040555, 0.0037852, 0.0040384, -0.0002077, 0.0001690
3: 0.0016712, 0.0021497, 0.0016772, 0.0021904, -0.0002971, 0.0002447
4: -0.0041966, -0.0034726, -0.0041296, -0.0035002, -0.0003303, 0.0002921
5: -0.0001516, 0.0002768, -0.0000849, 0.0002707, -0.0003022, 0.0002438
6: -0.0040471, -0.0027609, -0.0040364, -0.0027364, -0.0005242, 0.0004820
7: -0.0203590, -0.0161919, -0.0199401, -0.0163629, -0.0019387, 0.0016981
8: 0.9765776, 0.9803886, 0.9771031, 0.9801837, -0.0019395, 0.0016385
9: 0.0029438, 0.0056825, 0.0030664, 0.0053829, -0.0011267, 0.0012988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0166818, 0.0177592, -0.0004521, 0.0005432
1: -0.0007086, 0.0000066, -0.0007634, -0.0000025, -0.0003347, 0.0004057
2: 0.0037261, 0.0040555, 0.0037296, 0.0040776, -0.0001726, 0.0001441
3: 0.0016712, 0.0021497, 0.0016853, 0.0021678, -0.0002630, 0.0002217
4: -0.0041966, -0.0034726, -0.0041939, -0.0034568, -0.0002753, 0.0002563
5: -0.0001516, 0.0002768, -0.0001462, 0.0003108, -0.0002492, 0.0002050
6: -0.0040471, -0.0027609, -0.0040352, -0.0027386, -0.0005244, 0.0004838
7: -0.0203590, -0.0161919, -0.0203417, -0.0160892, -0.0016077, 0.0014839
8: 0.9765776, 0.9803886, 0.9766047, 0.9805312, -0.0015935, 0.0014183
9: 0.0029438, 0.0056825, 0.0028690, 0.0056690, -0.0009816, 0.0010723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0009042
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
time: 0.57 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.67 seconds
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0011709
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.67
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168102, 0.0175804, -0.0004911, 0.0005877
1: -0.0006838, -0.0000355, -0.0006834, -0.0001158, -0.0003678, 0.0004419
2: 0.0037466, 0.0040438, 0.0037884, 0.0040352, -0.0001867, 0.0001563
3: 0.0016724, 0.0021489, 0.0016709, 0.0021890, -0.0002790, 0.0002350
4: -0.0041714, -0.0034810, -0.0041219, -0.0035013, -0.0003042, 0.0002755
5: -0.0001271, 0.0002629, -0.0000832, 0.0002674, -0.0002728, 0.0002267
6: -0.0040608, -0.0027658, -0.0040516, -0.0027434, -0.0005175, 0.0004781
7: -0.0202023, -0.0162487, -0.0198968, -0.0163695, -0.0017808, 0.0015992
8: 0.9767592, 0.9803019, 0.9771413, 0.9801622, -0.0017633, 0.0015303
9: 0.0029887, 0.0055711, 0.0030716, 0.0053551, -0.0010594, 0.0011901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167229, 0.0176959, -0.0004274, 0.0005122
1: -0.0006838, -0.0000355, -0.0007335, -0.0000461, -0.0003172, 0.0003832
2: 0.0037466, 0.0040438, 0.0037499, 0.0040645, -0.0001626, 0.0001362
3: 0.0016724, 0.0021489, 0.0016853, 0.0021666, -0.0002589, 0.0002199
4: -0.0041714, -0.0034810, -0.0041673, -0.0034687, -0.0002631, 0.0002460
5: -0.0001271, 0.0002629, -0.0001201, 0.0002934, -0.0002356, 0.0001945
6: -0.0040608, -0.0027658, -0.0040504, -0.0027452, -0.0005179, 0.0004794
7: -0.0202023, -0.0162487, -0.0201776, -0.0161689, -0.0015321, 0.0014206
8: 0.9767592, 0.9803019, 0.9767900, 0.9804238, -0.0015081, 0.0013476
9: 0.0029887, 0.0055711, 0.0029299, 0.0055534, -0.0009366, 0.0010189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168095, 0.0175657, -0.0004746, 0.0005880
1: -0.0006838, -0.0000355, -0.0006790, -0.0001302, -0.0003528, 0.0004373
2: 0.0037466, 0.0040438, 0.0037918, 0.0040363, -0.0001875, 0.0001517
3: 0.0016724, 0.0021489, 0.0016797, 0.0021823, -0.0002859, 0.0002429
4: -0.0041714, -0.0034810, -0.0041193, -0.0034943, -0.0003369, 0.0002980
5: -0.0001271, 0.0002629, -0.0000735, 0.0002641, -0.0002691, 0.0002171
6: -0.0040608, -0.0027658, -0.0040474, -0.0027580, -0.0005670, 0.0005431
7: -0.0202023, -0.0162487, -0.0198795, -0.0163305, -0.0019615, 0.0017206
8: 0.9767592, 0.9803019, 0.9771566, 0.9802017, -0.0018891, 0.0016022
9: 0.0029887, 0.0055711, 0.0030466, 0.0053417, -0.0011332, 0.0013033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0168150, 0.0175552, 0.0167528, 0.0177701, -0.0006573, 0.0005099
1: -0.0006769, -0.0001346, -0.0007086, 0.0000066, -0.0004855, 0.0003792
2: 0.0037959, 0.0040343, 0.0037261, 0.0040555, -0.0001627, 0.0002100
3: 0.0016748, 0.0021812, 0.0016712, 0.0021497, -0.0002329, 0.0002694
4: -0.0041112, -0.0034933, -0.0041966, -0.0034726, -0.0003019, 0.0003691
5: -0.0000714, 0.0002628, -0.0001516, 0.0002768, -0.0002330, 0.0002973
6: -0.0040611, -0.0027634, -0.0040471, -0.0027609, -0.0005626, 0.0005442
7: -0.0198345, -0.0163245, -0.0203590, -0.0161919, -0.0017478, 0.0021554
8: 0.9772027, 0.9801946, 0.9765776, 0.9803886, -0.0016551, 0.0021048
9: 0.0030426, 0.0053139, 0.0029438, 0.0056825, -0.0014365, 0.0011547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167528, 0.0177701, -0.0005371, 0.0005085
1: -0.0006838, -0.0000355, -0.0007086, 0.0000066, -0.0003886, 0.0003704
2: 0.0037466, 0.0040438, 0.0037261, 0.0040555, -0.0001629, 0.0001724
3: 0.0016724, 0.0021489, 0.0016712, 0.0021497, -0.0002275, 0.0002278
4: -0.0041714, -0.0034810, -0.0041966, -0.0034726, -0.0003063, 0.0003253
5: -0.0001271, 0.0002629, -0.0001516, 0.0002768, -0.0002251, 0.0002356
6: -0.0040608, -0.0027658, -0.0040471, -0.0027609, -0.0005643, 0.0005436
7: -0.0202023, -0.0162487, -0.0203590, -0.0161919, -0.0017729, 0.0018867
8: 0.9767592, 0.9803019, 0.9765776, 0.9803886, -0.0016830, 0.0018020
9: 0.0029887, 0.0055711, 0.0029438, 0.0056825, -0.0012492, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0011709
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168005, 0.0175889, -0.0005000, 0.0005974
1: -0.0006838, -0.0000355, -0.0006891, -0.0001125, -0.0003710, 0.0004472
2: 0.0037466, 0.0040438, 0.0037852, 0.0040384, -0.0001902, 0.0001597
3: 0.0016724, 0.0021489, 0.0016772, 0.0021904, -0.0002961, 0.0002442
4: -0.0041714, -0.0034810, -0.0041296, -0.0035002, -0.0003286, 0.0003075
5: -0.0001271, 0.0002629, -0.0000849, 0.0002707, -0.0002757, 0.0002280
6: -0.0040608, -0.0027658, -0.0040364, -0.0027364, -0.0005854, 0.0005267
7: -0.0202023, -0.0162487, -0.0199401, -0.0163629, -0.0019175, 0.0017775
8: 0.9767592, 0.9803019, 0.9771031, 0.9801837, -0.0018666, 0.0016680
9: 0.0029887, 0.0055711, 0.0030664, 0.0053829, -0.0011726, 0.0012767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010953
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166818, 0.0177592, -0.0005307, 0.0005843
1: -0.0006838, -0.0000355, -0.0007634, -0.0000025, -0.0003838, 0.0004294
2: 0.0037466, 0.0040438, 0.0037296, 0.0040776, -0.0001866, 0.0001704
3: 0.0016724, 0.0021489, 0.0016853, 0.0021678, -0.0002610, 0.0002200
4: -0.0041714, -0.0034810, -0.0041939, -0.0034568, -0.0003227, 0.0003245
5: -0.0001271, 0.0002629, -0.0001462, 0.0003108, -0.0002619, 0.0002327
6: -0.0040608, -0.0027658, -0.0040352, -0.0027386, -0.0005836, 0.0005260
7: -0.0202023, -0.0162487, -0.0203417, -0.0160892, -0.0018786, 0.0018806
8: 0.9767592, 0.9803019, 0.9766047, 0.9805312, -0.0018315, 0.0017906
9: 0.0029887, 0.0055711, 0.0028690, 0.0056690, -0.0012441, 0.0012484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168150, 0.0175552, -0.0005099, 0.0006573
1: -0.0007086, 0.0000066, -0.0006769, -0.0001346, -0.0003792, 0.0004855
2: 0.0037261, 0.0040555, 0.0037959, 0.0040343, -0.0002100, 0.0001627
3: 0.0016712, 0.0021497, 0.0016748, 0.0021812, -0.0002694, 0.0002329
4: -0.0041966, -0.0034726, -0.0041112, -0.0034933, -0.0003691, 0.0003019
5: -0.0001516, 0.0002768, -0.0000714, 0.0002628, -0.0002973, 0.0002330
6: -0.0040471, -0.0027609, -0.0040611, -0.0027634, -0.0005442, 0.0005626
7: -0.0203590, -0.0161919, -0.0198345, -0.0163245, -0.0021554, 0.0017478
8: 0.9765776, 0.9803886, 0.9772027, 0.9801946, -0.0021048, 0.0016551
9: 0.0029438, 0.0056825, 0.0030426, 0.0053139, -0.0011547, 0.0014365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 96

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0168095, 0.0175657, 0.0167893, 0.0177068, -0.0005880, 0.0004746
1: -0.0006790, -0.0001302, -0.0006838, -0.0000355, -0.0004373, 0.0003528
2: 0.0037918, 0.0040363, 0.0037466, 0.0040438, -0.0001517, 0.0001875
3: 0.0016797, 0.0021823, 0.0016724, 0.0021489, -0.0002429, 0.0002859
4: -0.0041193, -0.0034943, -0.0041714, -0.0034810, -0.0002980, 0.0003369
5: -0.0000735, 0.0002641, -0.0001271, 0.0002629, -0.0002171, 0.0002691
6: -0.0040474, -0.0027580, -0.0040608, -0.0027658, -0.0005431, 0.0005670
7: -0.0198795, -0.0163305, -0.0202023, -0.0162487, -0.0017206, 0.0019615
8: 0.9771566, 0.9802017, 0.9767592, 0.9803019, -0.0016022, 0.0018891
9: 0.0030466, 0.0053417, 0.0029887, 0.0055711, -0.0013033, 0.0011332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 96

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167893, 0.0177068, -0.0005085, 0.0005371
1: -0.0007086, 0.0000066, -0.0006838, -0.0000355, -0.0003704, 0.0003886
2: 0.0037261, 0.0040555, 0.0037466, 0.0040438, -0.0001724, 0.0001629
3: 0.0016712, 0.0021497, 0.0016724, 0.0021489, -0.0002278, 0.0002275
4: -0.0041966, -0.0034726, -0.0041714, -0.0034810, -0.0003253, 0.0003063
5: -0.0001516, 0.0002768, -0.0001271, 0.0002629, -0.0002356, 0.0002251
6: -0.0040471, -0.0027609, -0.0040608, -0.0027658, -0.0005436, 0.0005643
7: -0.0203590, -0.0161919, -0.0202023, -0.0162487, -0.0018867, 0.0017728
8: 0.9765776, 0.9803886, 0.9767592, 0.9803019, -0.0018020, 0.0016830
9: 0.0029438, 0.0056825, 0.0029887, 0.0055711, -0.0011710, 0.0012492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168102, 0.0175804, -0.0005393, 0.0006637
1: -0.0007086, 0.0000066, -0.0006834, -0.0001158, -0.0003996, 0.0004921
2: 0.0037261, 0.0040555, 0.0037884, 0.0040352, -0.0002118, 0.0001722
3: 0.0016712, 0.0021497, 0.0016709, 0.0021890, -0.0002805, 0.0002360
4: -0.0041966, -0.0034726, -0.0041219, -0.0035013, -0.0003631, 0.0003166
5: -0.0001516, 0.0002768, -0.0000832, 0.0002674, -0.0003018, 0.0002451
6: -0.0040471, -0.0027609, -0.0040516, -0.0027434, -0.0005708, 0.0005526
7: -0.0203590, -0.0161919, -0.0198968, -0.0163695, -0.0021225, 0.0018352
8: 0.9765776, 0.9803886, 0.9771413, 0.9801622, -0.0020845, 0.0017452
9: 0.0029438, 0.0056825, 0.0030716, 0.0053551, -0.0012145, 0.0014162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167229, 0.0176959, -0.0005022, 0.0006156
1: -0.0007086, 0.0000066, -0.0007335, -0.0000461, -0.0003654, 0.0004495
2: 0.0037261, 0.0040555, 0.0037499, 0.0040645, -0.0001969, 0.0001609
3: 0.0016712, 0.0021497, 0.0016853, 0.0021666, -0.0002609, 0.0002216
4: -0.0041966, -0.0034726, -0.0041673, -0.0034687, -0.0003411, 0.0003050
5: -0.0001516, 0.0002768, -0.0001201, 0.0002934, -0.0002735, 0.0002219
6: -0.0040471, -0.0027609, -0.0040504, -0.0027452, -0.0005715, 0.0005537
7: -0.0203590, -0.0161919, -0.0201776, -0.0161689, -0.0019900, 0.0017646
8: 0.9765776, 0.9803886, 0.9767900, 0.9804238, -0.0019517, 0.0016723
9: 0.0029438, 0.0056825, 0.0029299, 0.0055534, -0.0011651, 0.0013256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168095, 0.0175657, -0.0005024, 0.0006421
1: -0.0007086, 0.0000066, -0.0006790, -0.0001302, -0.0003771, 0.0004801
2: 0.0037261, 0.0040555, 0.0037918, 0.0040363, -0.0002044, 0.0001598
3: 0.0016712, 0.0021497, 0.0016797, 0.0021823, -0.0002855, 0.0002437
4: -0.0041966, -0.0034726, -0.0041193, -0.0034943, -0.0003349, 0.0002781
5: -0.0001516, 0.0002768, -0.0000735, 0.0002641, -0.0002955, 0.0002327
6: -0.0040471, -0.0027609, -0.0040474, -0.0027580, -0.0004980, 0.0004945
7: -0.0203590, -0.0161919, -0.0198795, -0.0163305, -0.0019626, 0.0016146
8: 0.9765776, 0.9803886, 0.9771566, 0.9802017, -0.0019492, 0.0015508
9: 0.0029438, 0.0056825, 0.0030466, 0.0053417, -0.0010695, 0.0013128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0168095, 0.0175657, 0.0167528, 0.0177701, -0.0006421, 0.0005024
1: -0.0006790, -0.0001302, -0.0007086, 0.0000066, -0.0004801, 0.0003771
2: 0.0037918, 0.0040363, 0.0037261, 0.0040555, -0.0001598, 0.0002044
3: 0.0016797, 0.0021823, 0.0016712, 0.0021497, -0.0002437, 0.0002855
4: -0.0041193, -0.0034943, -0.0041966, -0.0034726, -0.0002781, 0.0003349
5: -0.0000735, 0.0002641, -0.0001516, 0.0002768, -0.0002327, 0.0002955
6: -0.0040474, -0.0027580, -0.0040471, -0.0027609, -0.0004945, 0.0004980
7: -0.0198795, -0.0163305, -0.0203590, -0.0161919, -0.0016146, 0.0019626
8: 0.9771566, 0.9802017, 0.9765776, 0.9803886, -0.0015508, 0.0019492
9: 0.0030466, 0.0053417, 0.0029438, 0.0056825, -0.0013128, 0.0010695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167528, 0.0177701, -0.0004587, 0.0004587
1: -0.0007086, 0.0000066, -0.0007086, 0.0000066, -0.0003405, 0.0003405
2: 0.0037261, 0.0040555, 0.0037261, 0.0040555, -0.0001461, 0.0001461
3: 0.0016712, 0.0021497, 0.0016712, 0.0021497, -0.0002294, 0.0002294
4: -0.0041966, -0.0034726, -0.0041966, -0.0034726, -0.0002574, 0.0002574
5: -0.0001516, 0.0002768, -0.0001516, 0.0002768, -0.0002088, 0.0002088
6: -0.0040471, -0.0027609, -0.0040471, -0.0027609, -0.0004969, 0.0004969
7: -0.0203590, -0.0161919, -0.0203590, -0.0161919, -0.0014914, 0.0014914
8: 0.9765776, 0.9803886, 0.9765776, 0.9803886, -0.0014267, 0.0014267
9: 0.0029438, 0.0056825, 0.0029438, 0.0056825, -0.0009867, 0.0009867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168005, 0.0175889, -0.0005304, 0.0006535
1: -0.0007086, 0.0000066, -0.0006891, -0.0001125, -0.0003961, 0.0004904
2: 0.0037261, 0.0040555, 0.0037852, 0.0040384, -0.0002077, 0.0001690
3: 0.0016712, 0.0021497, 0.0016772, 0.0021904, -0.0002971, 0.0002447
4: -0.0041966, -0.0034726, -0.0041296, -0.0035002, -0.0003303, 0.0002921
5: -0.0001516, 0.0002768, -0.0000849, 0.0002707, -0.0003022, 0.0002438
6: -0.0040471, -0.0027609, -0.0040364, -0.0027364, -0.0005242, 0.0004820
7: -0.0203590, -0.0161919, -0.0199401, -0.0163629, -0.0019387, 0.0016981
8: 0.9765776, 0.9803886, 0.9771031, 0.9801837, -0.0019395, 0.0016385
9: 0.0029438, 0.0056825, 0.0030664, 0.0053829, -0.0011267, 0.0012988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0166818, 0.0177592, -0.0004521, 0.0005432
1: -0.0007086, 0.0000066, -0.0007634, -0.0000025, -0.0003347, 0.0004057
2: 0.0037261, 0.0040555, 0.0037296, 0.0040776, -0.0001726, 0.0001441
3: 0.0016712, 0.0021497, 0.0016853, 0.0021678, -0.0002630, 0.0002217
4: -0.0041966, -0.0034726, -0.0041939, -0.0034568, -0.0002753, 0.0002563
5: -0.0001516, 0.0002768, -0.0001462, 0.0003108, -0.0002492, 0.0002050
6: -0.0040471, -0.0027609, -0.0040352, -0.0027386, -0.0005244, 0.0004838
7: -0.0203590, -0.0161919, -0.0203417, -0.0160892, -0.0016077, 0.0014839
8: 0.9765776, 0.9803886, 0.9766047, 0.9805312, -0.0015935, 0.0014183
9: 0.0029438, 0.0056825, 0.0028690, 0.0056690, -0.0009816, 0.0010723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
time: 0.59 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.85 seconds
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0011709
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010953
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
IS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168102, 0.0175804, -0.0004911, 0.0005877
1: -0.0006838, -0.0000355, -0.0006834, -0.0001158, -0.0003678, 0.0004419
2: 0.0037466, 0.0040438, 0.0037884, 0.0040352, -0.0001867, 0.0001563
3: 0.0016724, 0.0021489, 0.0016709, 0.0021890, -0.0002790, 0.0002350
4: -0.0041714, -0.0034810, -0.0041219, -0.0035013, -0.0003042, 0.0002755
5: -0.0001271, 0.0002629, -0.0000832, 0.0002674, -0.0002728, 0.0002267
6: -0.0040608, -0.0027658, -0.0040516, -0.0027434, -0.0005175, 0.0004781
7: -0.0202023, -0.0162487, -0.0198968, -0.0163695, -0.0017808, 0.0015992
8: 0.9767592, 0.9803019, 0.9771413, 0.9801622, -0.0017633, 0.0015303
9: 0.0029887, 0.0055711, 0.0030716, 0.0053551, -0.0010594, 0.0011901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167229, 0.0176959, -0.0004274, 0.0005122
1: -0.0006838, -0.0000355, -0.0007335, -0.0000461, -0.0003172, 0.0003832
2: 0.0037466, 0.0040438, 0.0037499, 0.0040645, -0.0001626, 0.0001362
3: 0.0016724, 0.0021489, 0.0016853, 0.0021666, -0.0002589, 0.0002199
4: -0.0041714, -0.0034810, -0.0041673, -0.0034687, -0.0002631, 0.0002460
5: -0.0001271, 0.0002629, -0.0001201, 0.0002934, -0.0002356, 0.0001945
6: -0.0040608, -0.0027658, -0.0040504, -0.0027452, -0.0005179, 0.0004794
7: -0.0202023, -0.0162487, -0.0201776, -0.0161689, -0.0015321, 0.0014206
8: 0.9767592, 0.9803019, 0.9767900, 0.9804238, -0.0015081, 0.0013476
9: 0.0029887, 0.0055711, 0.0029299, 0.0055534, -0.0009366, 0.0010189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168102, 0.0175804, -0.0004911, 0.0005877
1: -0.0006838, -0.0000355, -0.0006834, -0.0001158, -0.0003678, 0.0004419
2: 0.0037466, 0.0040438, 0.0037884, 0.0040352, -0.0001867, 0.0001563
3: 0.0016724, 0.0021489, 0.0016709, 0.0021890, -0.0002790, 0.0002350
4: -0.0041714, -0.0034810, -0.0041219, -0.0035013, -0.0003042, 0.0002755
5: -0.0001271, 0.0002629, -0.0000832, 0.0002674, -0.0002728, 0.0002267
6: -0.0040608, -0.0027658, -0.0040516, -0.0027434, -0.0005175, 0.0004781
7: -0.0202023, -0.0162487, -0.0198968, -0.0163695, -0.0017808, 0.0015992
8: 0.9767592, 0.9803019, 0.9771413, 0.9801622, -0.0017633, 0.0015303
9: 0.0029887, 0.0055711, 0.0030716, 0.0053551, -0.0010594, 0.0011901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167229, 0.0176959, -0.0004274, 0.0005122
1: -0.0006838, -0.0000355, -0.0007335, -0.0000461, -0.0003172, 0.0003832
2: 0.0037466, 0.0040438, 0.0037499, 0.0040645, -0.0001626, 0.0001362
3: 0.0016724, 0.0021489, 0.0016853, 0.0021666, -0.0002589, 0.0002199
4: -0.0041714, -0.0034810, -0.0041673, -0.0034687, -0.0002631, 0.0002460
5: -0.0001271, 0.0002629, -0.0001201, 0.0002934, -0.0002356, 0.0001945
6: -0.0040608, -0.0027658, -0.0040504, -0.0027452, -0.0005179, 0.0004794
7: -0.0202023, -0.0162487, -0.0201776, -0.0161689, -0.0015321, 0.0014206
8: 0.9767592, 0.9803019, 0.9767900, 0.9804238, -0.0015081, 0.0013476
9: 0.0029887, 0.0055711, 0.0029299, 0.0055534, -0.0009366, 0.0010189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009017
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168095, 0.0175657, -0.0004746, 0.0005880
1: -0.0006838, -0.0000355, -0.0006790, -0.0001302, -0.0003528, 0.0004373
2: 0.0037466, 0.0040438, 0.0037918, 0.0040363, -0.0001875, 0.0001517
3: 0.0016724, 0.0021489, 0.0016797, 0.0021823, -0.0002859, 0.0002429
4: -0.0041714, -0.0034810, -0.0041193, -0.0034943, -0.0003369, 0.0002980
5: -0.0001271, 0.0002629, -0.0000735, 0.0002641, -0.0002691, 0.0002171
6: -0.0040608, -0.0027658, -0.0040474, -0.0027580, -0.0005670, 0.0005431
7: -0.0202023, -0.0162487, -0.0198795, -0.0163305, -0.0019615, 0.0017206
8: 0.9767592, 0.9803019, 0.9771566, 0.9802017, -0.0018891, 0.0016022
9: 0.0029887, 0.0055711, 0.0030466, 0.0053417, -0.0011332, 0.0013033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167528, 0.0177701, -0.0005371, 0.0005085
1: -0.0006838, -0.0000355, -0.0007086, 0.0000066, -0.0003886, 0.0003704
2: 0.0037466, 0.0040438, 0.0037261, 0.0040555, -0.0001629, 0.0001724
3: 0.0016724, 0.0021489, 0.0016712, 0.0021497, -0.0002275, 0.0002278
4: -0.0041714, -0.0034810, -0.0041966, -0.0034726, -0.0003063, 0.0003253
5: -0.0001271, 0.0002629, -0.0001516, 0.0002768, -0.0002251, 0.0002356
6: -0.0040608, -0.0027658, -0.0040471, -0.0027609, -0.0005643, 0.0005436
7: -0.0202023, -0.0162487, -0.0203590, -0.0161919, -0.0017729, 0.0018867
8: 0.9767592, 0.9803019, 0.9765776, 0.9803886, -0.0016830, 0.0018020
9: 0.0029887, 0.0055711, 0.0029438, 0.0056825, -0.0012492, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168095, 0.0175657, -0.0004746, 0.0005880
1: -0.0006838, -0.0000355, -0.0006790, -0.0001302, -0.0003528, 0.0004373
2: 0.0037466, 0.0040438, 0.0037918, 0.0040363, -0.0001875, 0.0001517
3: 0.0016724, 0.0021489, 0.0016797, 0.0021823, -0.0002859, 0.0002429
4: -0.0041714, -0.0034810, -0.0041193, -0.0034943, -0.0003369, 0.0002980
5: -0.0001271, 0.0002629, -0.0000735, 0.0002641, -0.0002691, 0.0002171
6: -0.0040608, -0.0027658, -0.0040474, -0.0027580, -0.0005670, 0.0005431
7: -0.0202023, -0.0162487, -0.0198795, -0.0163305, -0.0019615, 0.0017206
8: 0.9767592, 0.9803019, 0.9771566, 0.9802017, -0.0018891, 0.0016022
9: 0.0029887, 0.0055711, 0.0030466, 0.0053417, -0.0011332, 0.0013033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167528, 0.0177701, -0.0005371, 0.0005085
1: -0.0006838, -0.0000355, -0.0007086, 0.0000066, -0.0003886, 0.0003704
2: 0.0037466, 0.0040438, 0.0037261, 0.0040555, -0.0001629, 0.0001724
3: 0.0016724, 0.0021489, 0.0016712, 0.0021497, -0.0002275, 0.0002278
4: -0.0041714, -0.0034810, -0.0041966, -0.0034726, -0.0003063, 0.0003253
5: -0.0001271, 0.0002629, -0.0001516, 0.0002768, -0.0002251, 0.0002356
6: -0.0040608, -0.0027658, -0.0040471, -0.0027609, -0.0005643, 0.0005436
7: -0.0202023, -0.0162487, -0.0203590, -0.0161919, -0.0017729, 0.0018867
8: 0.9767592, 0.9803019, 0.9765776, 0.9803886, -0.0016830, 0.0018020
9: 0.0029887, 0.0055711, 0.0029438, 0.0056825, -0.0012492, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0009283
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0011709
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168005, 0.0175889, -0.0005000, 0.0005974
1: -0.0006838, -0.0000355, -0.0006891, -0.0001125, -0.0003710, 0.0004472
2: 0.0037466, 0.0040438, 0.0037852, 0.0040384, -0.0001902, 0.0001597
3: 0.0016724, 0.0021489, 0.0016772, 0.0021904, -0.0002961, 0.0002442
4: -0.0041714, -0.0034810, -0.0041296, -0.0035002, -0.0003286, 0.0003075
5: -0.0001271, 0.0002629, -0.0000849, 0.0002707, -0.0002757, 0.0002280
6: -0.0040608, -0.0027658, -0.0040364, -0.0027364, -0.0005854, 0.0005267
7: -0.0202023, -0.0162487, -0.0199401, -0.0163629, -0.0019175, 0.0017775
8: 0.9767592, 0.9803019, 0.9771031, 0.9801837, -0.0018666, 0.0016680
9: 0.0029887, 0.0055711, 0.0030664, 0.0053829, -0.0011726, 0.0012767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166818, 0.0177592, -0.0005307, 0.0005843
1: -0.0006838, -0.0000355, -0.0007634, -0.0000025, -0.0003838, 0.0004294
2: 0.0037466, 0.0040438, 0.0037296, 0.0040776, -0.0001866, 0.0001704
3: 0.0016724, 0.0021489, 0.0016853, 0.0021678, -0.0002610, 0.0002200
4: -0.0041714, -0.0034810, -0.0041939, -0.0034568, -0.0003227, 0.0003245
5: -0.0001271, 0.0002629, -0.0001462, 0.0003108, -0.0002619, 0.0002327
6: -0.0040608, -0.0027658, -0.0040352, -0.0027386, -0.0005836, 0.0005260
7: -0.0202023, -0.0162487, -0.0203417, -0.0160892, -0.0018786, 0.0018806
8: 0.9767592, 0.9803019, 0.9766047, 0.9805312, -0.0018315, 0.0017906
9: 0.0029887, 0.0055711, 0.0028690, 0.0056690, -0.0012441, 0.0012484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168005, 0.0175889, -0.0005000, 0.0005974
1: -0.0006838, -0.0000355, -0.0006891, -0.0001125, -0.0003710, 0.0004472
2: 0.0037466, 0.0040438, 0.0037852, 0.0040384, -0.0001902, 0.0001597
3: 0.0016724, 0.0021489, 0.0016772, 0.0021904, -0.0002961, 0.0002442
4: -0.0041714, -0.0034810, -0.0041296, -0.0035002, -0.0003286, 0.0003075
5: -0.0001271, 0.0002629, -0.0000849, 0.0002707, -0.0002757, 0.0002280
6: -0.0040608, -0.0027658, -0.0040364, -0.0027364, -0.0005854, 0.0005267
7: -0.0202023, -0.0162487, -0.0199401, -0.0163629, -0.0019175, 0.0017775
8: 0.9767592, 0.9803019, 0.9771031, 0.9801837, -0.0018666, 0.0016680
9: 0.0029887, 0.0055711, 0.0030664, 0.0053829, -0.0011726, 0.0012767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166818, 0.0177592, -0.0005307, 0.0005843
1: -0.0006838, -0.0000355, -0.0007634, -0.0000025, -0.0003838, 0.0004294
2: 0.0037466, 0.0040438, 0.0037296, 0.0040776, -0.0001866, 0.0001704
3: 0.0016724, 0.0021489, 0.0016853, 0.0021678, -0.0002610, 0.0002200
4: -0.0041714, -0.0034810, -0.0041939, -0.0034568, -0.0003227, 0.0003245
5: -0.0001271, 0.0002629, -0.0001462, 0.0003108, -0.0002619, 0.0002327
6: -0.0040608, -0.0027658, -0.0040352, -0.0027386, -0.0005836, 0.0005260
7: -0.0202023, -0.0162487, -0.0203417, -0.0160892, -0.0018786, 0.0018806
8: 0.9767592, 0.9803019, 0.9766047, 0.9805312, -0.0018315, 0.0017906
9: 0.0029887, 0.0055711, 0.0028690, 0.0056690, -0.0012441, 0.0012484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009014
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168150, 0.0175552, -0.0005099, 0.0006573
1: -0.0007086, 0.0000066, -0.0006769, -0.0001346, -0.0003792, 0.0004855
2: 0.0037261, 0.0040555, 0.0037959, 0.0040343, -0.0002100, 0.0001627
3: 0.0016712, 0.0021497, 0.0016748, 0.0021812, -0.0002694, 0.0002329
4: -0.0041966, -0.0034726, -0.0041112, -0.0034933, -0.0003691, 0.0003019
5: -0.0001516, 0.0002768, -0.0000714, 0.0002628, -0.0002973, 0.0002330
6: -0.0040471, -0.0027609, -0.0040611, -0.0027634, -0.0005442, 0.0005626
7: -0.0203590, -0.0161919, -0.0198345, -0.0163245, -0.0021554, 0.0017478
8: 0.9765776, 0.9803886, 0.9772027, 0.9801946, -0.0021048, 0.0016551
9: 0.0029438, 0.0056825, 0.0030426, 0.0053139, -0.0011547, 0.0014365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167893, 0.0177068, -0.0005085, 0.0005371
1: -0.0007086, 0.0000066, -0.0006838, -0.0000355, -0.0003704, 0.0003886
2: 0.0037261, 0.0040555, 0.0037466, 0.0040438, -0.0001724, 0.0001629
3: 0.0016712, 0.0021497, 0.0016724, 0.0021489, -0.0002278, 0.0002275
4: -0.0041966, -0.0034726, -0.0041714, -0.0034810, -0.0003253, 0.0003063
5: -0.0001516, 0.0002768, -0.0001271, 0.0002629, -0.0002356, 0.0002251
6: -0.0040471, -0.0027609, -0.0040608, -0.0027658, -0.0005436, 0.0005643
7: -0.0203590, -0.0161919, -0.0202023, -0.0162487, -0.0018867, 0.0017728
8: 0.9765776, 0.9803886, 0.9767592, 0.9803019, -0.0018020, 0.0016830
9: 0.0029438, 0.0056825, 0.0029887, 0.0055711, -0.0011710, 0.0012492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168150, 0.0175552, -0.0005099, 0.0006573
1: -0.0007086, 0.0000066, -0.0006769, -0.0001346, -0.0003792, 0.0004855
2: 0.0037261, 0.0040555, 0.0037959, 0.0040343, -0.0002100, 0.0001627
3: 0.0016712, 0.0021497, 0.0016748, 0.0021812, -0.0002694, 0.0002329
4: -0.0041966, -0.0034726, -0.0041112, -0.0034933, -0.0003691, 0.0003019
5: -0.0001516, 0.0002768, -0.0000714, 0.0002628, -0.0002973, 0.0002330
6: -0.0040471, -0.0027609, -0.0040611, -0.0027634, -0.0005442, 0.0005626
7: -0.0203590, -0.0161919, -0.0198345, -0.0163245, -0.0021554, 0.0017478
8: 0.9765776, 0.9803886, 0.9772027, 0.9801946, -0.0021048, 0.0016551
9: 0.0029438, 0.0056825, 0.0030426, 0.0053139, -0.0011547, 0.0014365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167893, 0.0177068, -0.0005085, 0.0005371
1: -0.0007086, 0.0000066, -0.0006838, -0.0000355, -0.0003704, 0.0003886
2: 0.0037261, 0.0040555, 0.0037466, 0.0040438, -0.0001724, 0.0001629
3: 0.0016712, 0.0021497, 0.0016724, 0.0021489, -0.0002278, 0.0002275
4: -0.0041966, -0.0034726, -0.0041714, -0.0034810, -0.0003253, 0.0003063
5: -0.0001516, 0.0002768, -0.0001271, 0.0002629, -0.0002356, 0.0002251
6: -0.0040471, -0.0027609, -0.0040608, -0.0027658, -0.0005436, 0.0005643
7: -0.0203590, -0.0161919, -0.0202023, -0.0162487, -0.0018867, 0.0017728
8: 0.9765776, 0.9803886, 0.9767592, 0.9803019, -0.0018020, 0.0016830
9: 0.0029438, 0.0056825, 0.0029887, 0.0055711, -0.0011710, 0.0012492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0009320
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168102, 0.0175804, -0.0005393, 0.0006637
1: -0.0007086, 0.0000066, -0.0006834, -0.0001158, -0.0003996, 0.0004921
2: 0.0037261, 0.0040555, 0.0037884, 0.0040352, -0.0002118, 0.0001722
3: 0.0016712, 0.0021497, 0.0016709, 0.0021890, -0.0002805, 0.0002360
4: -0.0041966, -0.0034726, -0.0041219, -0.0035013, -0.0003631, 0.0003166
5: -0.0001516, 0.0002768, -0.0000832, 0.0002674, -0.0003018, 0.0002451
6: -0.0040471, -0.0027609, -0.0040516, -0.0027434, -0.0005708, 0.0005526
7: -0.0203590, -0.0161919, -0.0198968, -0.0163695, -0.0021225, 0.0018352
8: 0.9765776, 0.9803886, 0.9771413, 0.9801622, -0.0020845, 0.0017452
9: 0.0029438, 0.0056825, 0.0030716, 0.0053551, -0.0012145, 0.0014162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167229, 0.0176959, -0.0005022, 0.0006156
1: -0.0007086, 0.0000066, -0.0007335, -0.0000461, -0.0003654, 0.0004495
2: 0.0037261, 0.0040555, 0.0037499, 0.0040645, -0.0001969, 0.0001609
3: 0.0016712, 0.0021497, 0.0016853, 0.0021666, -0.0002609, 0.0002216
4: -0.0041966, -0.0034726, -0.0041673, -0.0034687, -0.0003411, 0.0003050
5: -0.0001516, 0.0002768, -0.0001201, 0.0002934, -0.0002735, 0.0002219
6: -0.0040471, -0.0027609, -0.0040504, -0.0027452, -0.0005715, 0.0005537
7: -0.0203590, -0.0161919, -0.0201776, -0.0161689, -0.0019900, 0.0017646
8: 0.9765776, 0.9803886, 0.9767900, 0.9804238, -0.0019517, 0.0016723
9: 0.0029438, 0.0056825, 0.0029299, 0.0055534, -0.0011651, 0.0013256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168102, 0.0175804, -0.0005393, 0.0006637
1: -0.0007086, 0.0000066, -0.0006834, -0.0001158, -0.0003996, 0.0004921
2: 0.0037261, 0.0040555, 0.0037884, 0.0040352, -0.0002118, 0.0001722
3: 0.0016712, 0.0021497, 0.0016709, 0.0021890, -0.0002805, 0.0002360
4: -0.0041966, -0.0034726, -0.0041219, -0.0035013, -0.0003631, 0.0003166
5: -0.0001516, 0.0002768, -0.0000832, 0.0002674, -0.0003018, 0.0002451
6: -0.0040471, -0.0027609, -0.0040516, -0.0027434, -0.0005708, 0.0005526
7: -0.0203590, -0.0161919, -0.0198968, -0.0163695, -0.0021225, 0.0018352
8: 0.9765776, 0.9803886, 0.9771413, 0.9801622, -0.0020845, 0.0017452
9: 0.0029438, 0.0056825, 0.0030716, 0.0053551, -0.0012145, 0.0014162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167229, 0.0176959, -0.0005022, 0.0006156
1: -0.0007086, 0.0000066, -0.0007335, -0.0000461, -0.0003654, 0.0004495
2: 0.0037261, 0.0040555, 0.0037499, 0.0040645, -0.0001969, 0.0001609
3: 0.0016712, 0.0021497, 0.0016853, 0.0021666, -0.0002609, 0.0002216
4: -0.0041966, -0.0034726, -0.0041673, -0.0034687, -0.0003411, 0.0003050
5: -0.0001516, 0.0002768, -0.0001201, 0.0002934, -0.0002735, 0.0002219
6: -0.0040471, -0.0027609, -0.0040504, -0.0027452, -0.0005715, 0.0005537
7: -0.0203590, -0.0161919, -0.0201776, -0.0161689, -0.0019900, 0.0017646
8: 0.9765776, 0.9803886, 0.9767900, 0.9804238, -0.0019517, 0.0016723
9: 0.0029438, 0.0056825, 0.0029299, 0.0055534, -0.0011651, 0.0013256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0009042
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168095, 0.0175657, -0.0005024, 0.0006421
1: -0.0007086, 0.0000066, -0.0006790, -0.0001302, -0.0003771, 0.0004801
2: 0.0037261, 0.0040555, 0.0037918, 0.0040363, -0.0002044, 0.0001598
3: 0.0016712, 0.0021497, 0.0016797, 0.0021823, -0.0002855, 0.0002437
4: -0.0041966, -0.0034726, -0.0041193, -0.0034943, -0.0003349, 0.0002781
5: -0.0001516, 0.0002768, -0.0000735, 0.0002641, -0.0002955, 0.0002327
6: -0.0040471, -0.0027609, -0.0040474, -0.0027580, -0.0004980, 0.0004945
7: -0.0203590, -0.0161919, -0.0198795, -0.0163305, -0.0019626, 0.0016146
8: 0.9765776, 0.9803886, 0.9771566, 0.9802017, -0.0019492, 0.0015508
9: 0.0029438, 0.0056825, 0.0030466, 0.0053417, -0.0010695, 0.0013128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167528, 0.0177701, -0.0004587, 0.0004587
1: -0.0007086, 0.0000066, -0.0007086, 0.0000066, -0.0003405, 0.0003405
2: 0.0037261, 0.0040555, 0.0037261, 0.0040555, -0.0001461, 0.0001461
3: 0.0016712, 0.0021497, 0.0016712, 0.0021497, -0.0002294, 0.0002294
4: -0.0041966, -0.0034726, -0.0041966, -0.0034726, -0.0002574, 0.0002574
5: -0.0001516, 0.0002768, -0.0001516, 0.0002768, -0.0002088, 0.0002088
6: -0.0040471, -0.0027609, -0.0040471, -0.0027609, -0.0004969, 0.0004969
7: -0.0203590, -0.0161919, -0.0203590, -0.0161919, -0.0014914, 0.0014914
8: 0.9765776, 0.9803886, 0.9765776, 0.9803886, -0.0014267, 0.0014267
9: 0.0029438, 0.0056825, 0.0029438, 0.0056825, -0.0009867, 0.0009867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168095, 0.0175657, -0.0005024, 0.0006421
1: -0.0007086, 0.0000066, -0.0006790, -0.0001302, -0.0003771, 0.0004801
2: 0.0037261, 0.0040555, 0.0037918, 0.0040363, -0.0002044, 0.0001598
3: 0.0016712, 0.0021497, 0.0016797, 0.0021823, -0.0002855, 0.0002437
4: -0.0041966, -0.0034726, -0.0041193, -0.0034943, -0.0003349, 0.0002781
5: -0.0001516, 0.0002768, -0.0000735, 0.0002641, -0.0002955, 0.0002327
6: -0.0040471, -0.0027609, -0.0040474, -0.0027580, -0.0004980, 0.0004945
7: -0.0203590, -0.0161919, -0.0198795, -0.0163305, -0.0019626, 0.0016146
8: 0.9765776, 0.9803886, 0.9771566, 0.9802017, -0.0019492, 0.0015508
9: 0.0029438, 0.0056825, 0.0030466, 0.0053417, -0.0010695, 0.0013128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167528, 0.0177701, -0.0004587, 0.0004587
1: -0.0007086, 0.0000066, -0.0007086, 0.0000066, -0.0003405, 0.0003405
2: 0.0037261, 0.0040555, 0.0037261, 0.0040555, -0.0001461, 0.0001461
3: 0.0016712, 0.0021497, 0.0016712, 0.0021497, -0.0002294, 0.0002294
4: -0.0041966, -0.0034726, -0.0041966, -0.0034726, -0.0002574, 0.0002574
5: -0.0001516, 0.0002768, -0.0001516, 0.0002768, -0.0002088, 0.0002088
6: -0.0040471, -0.0027609, -0.0040471, -0.0027609, -0.0004969, 0.0004969
7: -0.0203590, -0.0161919, -0.0203590, -0.0161919, -0.0014914, 0.0014914
8: 0.9765776, 0.9803886, 0.9765776, 0.9803886, -0.0014267, 0.0014267
9: 0.0029438, 0.0056825, 0.0029438, 0.0056825, -0.0009867, 0.0009867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0009320
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168005, 0.0175889, -0.0005304, 0.0006535
1: -0.0007086, 0.0000066, -0.0006891, -0.0001125, -0.0003961, 0.0004904
2: 0.0037261, 0.0040555, 0.0037852, 0.0040384, -0.0002077, 0.0001690
3: 0.0016712, 0.0021497, 0.0016772, 0.0021904, -0.0002971, 0.0002447
4: -0.0041966, -0.0034726, -0.0041296, -0.0035002, -0.0003303, 0.0002921
5: -0.0001516, 0.0002768, -0.0000849, 0.0002707, -0.0003022, 0.0002438
6: -0.0040471, -0.0027609, -0.0040364, -0.0027364, -0.0005242, 0.0004820
7: -0.0203590, -0.0161919, -0.0199401, -0.0163629, -0.0019387, 0.0016981
8: 0.9765776, 0.9803886, 0.9771031, 0.9801837, -0.0019395, 0.0016385
9: 0.0029438, 0.0056825, 0.0030664, 0.0053829, -0.0011267, 0.0012988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0166818, 0.0177592, -0.0004521, 0.0005432
1: -0.0007086, 0.0000066, -0.0007634, -0.0000025, -0.0003347, 0.0004057
2: 0.0037261, 0.0040555, 0.0037296, 0.0040776, -0.0001726, 0.0001441
3: 0.0016712, 0.0021497, 0.0016853, 0.0021678, -0.0002630, 0.0002217
4: -0.0041966, -0.0034726, -0.0041939, -0.0034568, -0.0002753, 0.0002563
5: -0.0001516, 0.0002768, -0.0001462, 0.0003108, -0.0002492, 0.0002050
6: -0.0040471, -0.0027609, -0.0040352, -0.0027386, -0.0005244, 0.0004838
7: -0.0203590, -0.0161919, -0.0203417, -0.0160892, -0.0016077, 0.0014839
8: 0.9765776, 0.9803886, 0.9766047, 0.9805312, -0.0015935, 0.0014183
9: 0.0029438, 0.0056825, 0.0028690, 0.0056690, -0.0009816, 0.0010723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168005, 0.0175889, -0.0005304, 0.0006535
1: -0.0007086, 0.0000066, -0.0006891, -0.0001125, -0.0003961, 0.0004904
2: 0.0037261, 0.0040555, 0.0037852, 0.0040384, -0.0002077, 0.0001690
3: 0.0016712, 0.0021497, 0.0016772, 0.0021904, -0.0002971, 0.0002447
4: -0.0041966, -0.0034726, -0.0041296, -0.0035002, -0.0003303, 0.0002921
5: -0.0001516, 0.0002768, -0.0000849, 0.0002707, -0.0003022, 0.0002438
6: -0.0040471, -0.0027609, -0.0040364, -0.0027364, -0.0005242, 0.0004820
7: -0.0203590, -0.0161919, -0.0199401, -0.0163629, -0.0019387, 0.0016981
8: 0.9765776, 0.9803886, 0.9771031, 0.9801837, -0.0019395, 0.0016385
9: 0.0029438, 0.0056825, 0.0030664, 0.0053829, -0.0011267, 0.0012988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0166818, 0.0177592, -0.0004521, 0.0005432
1: -0.0007086, 0.0000066, -0.0007634, -0.0000025, -0.0003347, 0.0004057
2: 0.0037261, 0.0040555, 0.0037296, 0.0040776, -0.0001726, 0.0001441
3: 0.0016712, 0.0021497, 0.0016853, 0.0021678, -0.0002630, 0.0002217
4: -0.0041966, -0.0034726, -0.0041939, -0.0034568, -0.0002753, 0.0002563
5: -0.0001516, 0.0002768, -0.0001462, 0.0003108, -0.0002492, 0.0002050
6: -0.0040471, -0.0027609, -0.0040352, -0.0027386, -0.0005244, 0.0004838
7: -0.0203590, -0.0161919, -0.0203417, -0.0160892, -0.0016077, 0.0014839
8: 0.9765776, 0.9803886, 0.9766047, 0.9805312, -0.0015935, 0.0014183
9: 0.0029438, 0.0056825, 0.0028690, 0.0056690, -0.0009816, 0.0010723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0009042
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
time: 0.57 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.76 seconds
IS_A1_B1_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
IS_A1_B2_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0011709
IS_A1_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
IS_A2_B1_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
IS_A2_B1_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
IS_A2_B2_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
IS_A2_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.76
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168102, 0.0175804, -0.0004911, 0.0005877
1: -0.0006838, -0.0000355, -0.0006834, -0.0001158, -0.0003678, 0.0004419
2: 0.0037466, 0.0040438, 0.0037884, 0.0040352, -0.0001867, 0.0001563
3: 0.0016724, 0.0021489, 0.0016709, 0.0021890, -0.0002790, 0.0002350
4: -0.0041714, -0.0034810, -0.0041219, -0.0035013, -0.0003042, 0.0002755
5: -0.0001271, 0.0002629, -0.0000832, 0.0002674, -0.0002728, 0.0002267
6: -0.0040608, -0.0027658, -0.0040516, -0.0027434, -0.0005175, 0.0004781
7: -0.0202023, -0.0162487, -0.0198968, -0.0163695, -0.0017808, 0.0015992
8: 0.9767592, 0.9803019, 0.9771413, 0.9801622, -0.0017633, 0.0015303
9: 0.0029887, 0.0055711, 0.0030716, 0.0053551, -0.0010594, 0.0011901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167229, 0.0176959, -0.0004274, 0.0005122
1: -0.0006838, -0.0000355, -0.0007335, -0.0000461, -0.0003172, 0.0003832
2: 0.0037466, 0.0040438, 0.0037499, 0.0040645, -0.0001626, 0.0001362
3: 0.0016724, 0.0021489, 0.0016853, 0.0021666, -0.0002589, 0.0002199
4: -0.0041714, -0.0034810, -0.0041673, -0.0034687, -0.0002631, 0.0002460
5: -0.0001271, 0.0002629, -0.0001201, 0.0002934, -0.0002356, 0.0001945
6: -0.0040608, -0.0027658, -0.0040504, -0.0027452, -0.0005179, 0.0004794
7: -0.0202023, -0.0162487, -0.0201776, -0.0161689, -0.0015321, 0.0014206
8: 0.9767592, 0.9803019, 0.9767900, 0.9804238, -0.0015081, 0.0013476
9: 0.0029887, 0.0055711, 0.0029299, 0.0055534, -0.0009366, 0.0010189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168102, 0.0175804, -0.0004911, 0.0005877
1: -0.0006838, -0.0000355, -0.0006834, -0.0001158, -0.0003678, 0.0004419
2: 0.0037466, 0.0040438, 0.0037884, 0.0040352, -0.0001867, 0.0001563
3: 0.0016724, 0.0021489, 0.0016709, 0.0021890, -0.0002790, 0.0002350
4: -0.0041714, -0.0034810, -0.0041219, -0.0035013, -0.0003042, 0.0002755
5: -0.0001271, 0.0002629, -0.0000832, 0.0002674, -0.0002728, 0.0002267
6: -0.0040608, -0.0027658, -0.0040516, -0.0027434, -0.0005175, 0.0004781
7: -0.0202023, -0.0162487, -0.0198968, -0.0163695, -0.0017808, 0.0015992
8: 0.9767592, 0.9803019, 0.9771413, 0.9801622, -0.0017633, 0.0015303
9: 0.0029887, 0.0055711, 0.0030716, 0.0053551, -0.0010594, 0.0011901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167229, 0.0176959, -0.0004274, 0.0005122
1: -0.0006838, -0.0000355, -0.0007335, -0.0000461, -0.0003172, 0.0003832
2: 0.0037466, 0.0040438, 0.0037499, 0.0040645, -0.0001626, 0.0001362
3: 0.0016724, 0.0021489, 0.0016853, 0.0021666, -0.0002589, 0.0002199
4: -0.0041714, -0.0034810, -0.0041673, -0.0034687, -0.0002631, 0.0002460
5: -0.0001271, 0.0002629, -0.0001201, 0.0002934, -0.0002356, 0.0001945
6: -0.0040608, -0.0027658, -0.0040504, -0.0027452, -0.0005179, 0.0004794
7: -0.0202023, -0.0162487, -0.0201776, -0.0161689, -0.0015321, 0.0014206
8: 0.9767592, 0.9803019, 0.9767900, 0.9804238, -0.0015081, 0.0013476
9: 0.0029887, 0.0055711, 0.0029299, 0.0055534, -0.0009366, 0.0010189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168095, 0.0175657, -0.0004746, 0.0005880
1: -0.0006838, -0.0000355, -0.0006790, -0.0001302, -0.0003528, 0.0004373
2: 0.0037466, 0.0040438, 0.0037918, 0.0040363, -0.0001875, 0.0001517
3: 0.0016724, 0.0021489, 0.0016797, 0.0021823, -0.0002859, 0.0002429
4: -0.0041714, -0.0034810, -0.0041193, -0.0034943, -0.0003369, 0.0002980
5: -0.0001271, 0.0002629, -0.0000735, 0.0002641, -0.0002691, 0.0002171
6: -0.0040608, -0.0027658, -0.0040474, -0.0027580, -0.0005670, 0.0005431
7: -0.0202023, -0.0162487, -0.0198795, -0.0163305, -0.0019615, 0.0017206
8: 0.9767592, 0.9803019, 0.9771566, 0.9802017, -0.0018891, 0.0016022
9: 0.0029887, 0.0055711, 0.0030466, 0.0053417, -0.0011332, 0.0013033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167528, 0.0177701, -0.0005371, 0.0005085
1: -0.0006838, -0.0000355, -0.0007086, 0.0000066, -0.0003886, 0.0003704
2: 0.0037466, 0.0040438, 0.0037261, 0.0040555, -0.0001629, 0.0001724
3: 0.0016724, 0.0021489, 0.0016712, 0.0021497, -0.0002275, 0.0002278
4: -0.0041714, -0.0034810, -0.0041966, -0.0034726, -0.0003063, 0.0003253
5: -0.0001271, 0.0002629, -0.0001516, 0.0002768, -0.0002251, 0.0002356
6: -0.0040608, -0.0027658, -0.0040471, -0.0027609, -0.0005643, 0.0005436
7: -0.0202023, -0.0162487, -0.0203590, -0.0161919, -0.0017729, 0.0018867
8: 0.9767592, 0.9803019, 0.9765776, 0.9803886, -0.0016830, 0.0018020
9: 0.0029887, 0.0055711, 0.0029438, 0.0056825, -0.0012492, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168095, 0.0175657, -0.0004746, 0.0005880
1: -0.0006838, -0.0000355, -0.0006790, -0.0001302, -0.0003528, 0.0004373
2: 0.0037466, 0.0040438, 0.0037918, 0.0040363, -0.0001875, 0.0001517
3: 0.0016724, 0.0021489, 0.0016797, 0.0021823, -0.0002859, 0.0002429
4: -0.0041714, -0.0034810, -0.0041193, -0.0034943, -0.0003369, 0.0002980
5: -0.0001271, 0.0002629, -0.0000735, 0.0002641, -0.0002691, 0.0002171
6: -0.0040608, -0.0027658, -0.0040474, -0.0027580, -0.0005670, 0.0005431
7: -0.0202023, -0.0162487, -0.0198795, -0.0163305, -0.0019615, 0.0017206
8: 0.9767592, 0.9803019, 0.9771566, 0.9802017, -0.0018891, 0.0016022
9: 0.0029887, 0.0055711, 0.0030466, 0.0053417, -0.0011332, 0.0013033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0168150, 0.0175552, 0.0167528, 0.0177701, -0.0006573, 0.0005099
1: -0.0006769, -0.0001346, -0.0007086, 0.0000066, -0.0004855, 0.0003792
2: 0.0037959, 0.0040343, 0.0037261, 0.0040555, -0.0001627, 0.0002100
3: 0.0016748, 0.0021812, 0.0016712, 0.0021497, -0.0002329, 0.0002694
4: -0.0041112, -0.0034933, -0.0041966, -0.0034726, -0.0003019, 0.0003691
5: -0.0000714, 0.0002628, -0.0001516, 0.0002768, -0.0002330, 0.0002973
6: -0.0040611, -0.0027634, -0.0040471, -0.0027609, -0.0005626, 0.0005442
7: -0.0198345, -0.0163245, -0.0203590, -0.0161919, -0.0017478, 0.0021554
8: 0.9772027, 0.9801946, 0.9765776, 0.9803886, -0.0016551, 0.0021048
9: 0.0030426, 0.0053139, 0.0029438, 0.0056825, -0.0014365, 0.0011547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167528, 0.0177701, -0.0005371, 0.0005085
1: -0.0006838, -0.0000355, -0.0007086, 0.0000066, -0.0003886, 0.0003704
2: 0.0037466, 0.0040438, 0.0037261, 0.0040555, -0.0001629, 0.0001724
3: 0.0016724, 0.0021489, 0.0016712, 0.0021497, -0.0002275, 0.0002278
4: -0.0041714, -0.0034810, -0.0041966, -0.0034726, -0.0003063, 0.0003253
5: -0.0001271, 0.0002629, -0.0001516, 0.0002768, -0.0002251, 0.0002356
6: -0.0040608, -0.0027658, -0.0040471, -0.0027609, -0.0005643, 0.0005436
7: -0.0202023, -0.0162487, -0.0203590, -0.0161919, -0.0017729, 0.0018867
8: 0.9767592, 0.9803019, 0.9765776, 0.9803886, -0.0016830, 0.0018020
9: 0.0029887, 0.0055711, 0.0029438, 0.0056825, -0.0012492, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0011709
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168005, 0.0175889, -0.0005000, 0.0005974
1: -0.0006838, -0.0000355, -0.0006891, -0.0001125, -0.0003710, 0.0004472
2: 0.0037466, 0.0040438, 0.0037852, 0.0040384, -0.0001902, 0.0001597
3: 0.0016724, 0.0021489, 0.0016772, 0.0021904, -0.0002961, 0.0002442
4: -0.0041714, -0.0034810, -0.0041296, -0.0035002, -0.0003286, 0.0003075
5: -0.0001271, 0.0002629, -0.0000849, 0.0002707, -0.0002757, 0.0002280
6: -0.0040608, -0.0027658, -0.0040364, -0.0027364, -0.0005854, 0.0005267
7: -0.0202023, -0.0162487, -0.0199401, -0.0163629, -0.0019175, 0.0017775
8: 0.9767592, 0.9803019, 0.9771031, 0.9801837, -0.0018666, 0.0016680
9: 0.0029887, 0.0055711, 0.0030664, 0.0053829, -0.0011726, 0.0012767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010953
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166818, 0.0177592, -0.0005307, 0.0005843
1: -0.0006838, -0.0000355, -0.0007634, -0.0000025, -0.0003838, 0.0004294
2: 0.0037466, 0.0040438, 0.0037296, 0.0040776, -0.0001866, 0.0001704
3: 0.0016724, 0.0021489, 0.0016853, 0.0021678, -0.0002610, 0.0002200
4: -0.0041714, -0.0034810, -0.0041939, -0.0034568, -0.0003227, 0.0003245
5: -0.0001271, 0.0002629, -0.0001462, 0.0003108, -0.0002619, 0.0002327
6: -0.0040608, -0.0027658, -0.0040352, -0.0027386, -0.0005836, 0.0005260
7: -0.0202023, -0.0162487, -0.0203417, -0.0160892, -0.0018786, 0.0018806
8: 0.9767592, 0.9803019, 0.9766047, 0.9805312, -0.0018315, 0.0017906
9: 0.0029887, 0.0055711, 0.0028690, 0.0056690, -0.0012441, 0.0012484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168005, 0.0175889, -0.0005000, 0.0005974
1: -0.0006838, -0.0000355, -0.0006891, -0.0001125, -0.0003710, 0.0004472
2: 0.0037466, 0.0040438, 0.0037852, 0.0040384, -0.0001902, 0.0001597
3: 0.0016724, 0.0021489, 0.0016772, 0.0021904, -0.0002961, 0.0002442
4: -0.0041714, -0.0034810, -0.0041296, -0.0035002, -0.0003286, 0.0003075
5: -0.0001271, 0.0002629, -0.0000849, 0.0002707, -0.0002757, 0.0002280
6: -0.0040608, -0.0027658, -0.0040364, -0.0027364, -0.0005854, 0.0005267
7: -0.0202023, -0.0162487, -0.0199401, -0.0163629, -0.0019175, 0.0017775
8: 0.9767592, 0.9803019, 0.9771031, 0.9801837, -0.0018666, 0.0016680
9: 0.0029887, 0.0055711, 0.0030664, 0.0053829, -0.0011726, 0.0012767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010953
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166818, 0.0177592, -0.0005307, 0.0005843
1: -0.0006838, -0.0000355, -0.0007634, -0.0000025, -0.0003838, 0.0004294
2: 0.0037466, 0.0040438, 0.0037296, 0.0040776, -0.0001866, 0.0001704
3: 0.0016724, 0.0021489, 0.0016853, 0.0021678, -0.0002610, 0.0002200
4: -0.0041714, -0.0034810, -0.0041939, -0.0034568, -0.0003227, 0.0003245
5: -0.0001271, 0.0002629, -0.0001462, 0.0003108, -0.0002619, 0.0002327
6: -0.0040608, -0.0027658, -0.0040352, -0.0027386, -0.0005836, 0.0005260
7: -0.0202023, -0.0162487, -0.0203417, -0.0160892, -0.0018786, 0.0018806
8: 0.9767592, 0.9803019, 0.9766047, 0.9805312, -0.0018315, 0.0017906
9: 0.0029887, 0.0055711, 0.0028690, 0.0056690, -0.0012441, 0.0012484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168150, 0.0175552, -0.0005099, 0.0006573
1: -0.0007086, 0.0000066, -0.0006769, -0.0001346, -0.0003792, 0.0004855
2: 0.0037261, 0.0040555, 0.0037959, 0.0040343, -0.0002100, 0.0001627
3: 0.0016712, 0.0021497, 0.0016748, 0.0021812, -0.0002694, 0.0002329
4: -0.0041966, -0.0034726, -0.0041112, -0.0034933, -0.0003691, 0.0003019
5: -0.0001516, 0.0002768, -0.0000714, 0.0002628, -0.0002973, 0.0002330
6: -0.0040471, -0.0027609, -0.0040611, -0.0027634, -0.0005442, 0.0005626
7: -0.0203590, -0.0161919, -0.0198345, -0.0163245, -0.0021554, 0.0017478
8: 0.9765776, 0.9803886, 0.9772027, 0.9801946, -0.0021048, 0.0016551
9: 0.0029438, 0.0056825, 0.0030426, 0.0053139, -0.0011547, 0.0014365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167893, 0.0177068, -0.0005085, 0.0005371
1: -0.0007086, 0.0000066, -0.0006838, -0.0000355, -0.0003704, 0.0003886
2: 0.0037261, 0.0040555, 0.0037466, 0.0040438, -0.0001724, 0.0001629
3: 0.0016712, 0.0021497, 0.0016724, 0.0021489, -0.0002278, 0.0002275
4: -0.0041966, -0.0034726, -0.0041714, -0.0034810, -0.0003253, 0.0003063
5: -0.0001516, 0.0002768, -0.0001271, 0.0002629, -0.0002356, 0.0002251
6: -0.0040471, -0.0027609, -0.0040608, -0.0027658, -0.0005436, 0.0005643
7: -0.0203590, -0.0161919, -0.0202023, -0.0162487, -0.0018867, 0.0017728
8: 0.9765776, 0.9803886, 0.9767592, 0.9803019, -0.0018020, 0.0016830
9: 0.0029438, 0.0056825, 0.0029887, 0.0055711, -0.0011710, 0.0012492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168150, 0.0175552, -0.0005099, 0.0006573
1: -0.0007086, 0.0000066, -0.0006769, -0.0001346, -0.0003792, 0.0004855
2: 0.0037261, 0.0040555, 0.0037959, 0.0040343, -0.0002100, 0.0001627
3: 0.0016712, 0.0021497, 0.0016748, 0.0021812, -0.0002694, 0.0002329
4: -0.0041966, -0.0034726, -0.0041112, -0.0034933, -0.0003691, 0.0003019
5: -0.0001516, 0.0002768, -0.0000714, 0.0002628, -0.0002973, 0.0002330
6: -0.0040471, -0.0027609, -0.0040611, -0.0027634, -0.0005442, 0.0005626
7: -0.0203590, -0.0161919, -0.0198345, -0.0163245, -0.0021554, 0.0017478
8: 0.9765776, 0.9803886, 0.9772027, 0.9801946, -0.0021048, 0.0016551
9: 0.0029438, 0.0056825, 0.0030426, 0.0053139, -0.0011547, 0.0014365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 96

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0168095, 0.0175657, 0.0167893, 0.0177068, -0.0005880, 0.0004746
1: -0.0006790, -0.0001302, -0.0006838, -0.0000355, -0.0004373, 0.0003528
2: 0.0037918, 0.0040363, 0.0037466, 0.0040438, -0.0001517, 0.0001875
3: 0.0016797, 0.0021823, 0.0016724, 0.0021489, -0.0002429, 0.0002859
4: -0.0041193, -0.0034943, -0.0041714, -0.0034810, -0.0002980, 0.0003369
5: -0.0000735, 0.0002641, -0.0001271, 0.0002629, -0.0002171, 0.0002691
6: -0.0040474, -0.0027580, -0.0040608, -0.0027658, -0.0005431, 0.0005670
7: -0.0198795, -0.0163305, -0.0202023, -0.0162487, -0.0017206, 0.0019615
8: 0.9771566, 0.9802017, 0.9767592, 0.9803019, -0.0016022, 0.0018891
9: 0.0030466, 0.0053417, 0.0029887, 0.0055711, -0.0013033, 0.0011332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 96

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167893, 0.0177068, -0.0005085, 0.0005371
1: -0.0007086, 0.0000066, -0.0006838, -0.0000355, -0.0003704, 0.0003886
2: 0.0037261, 0.0040555, 0.0037466, 0.0040438, -0.0001724, 0.0001629
3: 0.0016712, 0.0021497, 0.0016724, 0.0021489, -0.0002278, 0.0002275
4: -0.0041966, -0.0034726, -0.0041714, -0.0034810, -0.0003253, 0.0003063
5: -0.0001516, 0.0002768, -0.0001271, 0.0002629, -0.0002356, 0.0002251
6: -0.0040471, -0.0027609, -0.0040608, -0.0027658, -0.0005436, 0.0005643
7: -0.0203590, -0.0161919, -0.0202023, -0.0162487, -0.0018867, 0.0017728
8: 0.9765776, 0.9803886, 0.9767592, 0.9803019, -0.0018020, 0.0016830
9: 0.0029438, 0.0056825, 0.0029887, 0.0055711, -0.0011710, 0.0012492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 96

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168102, 0.0175804, -0.0005393, 0.0006637
1: -0.0007086, 0.0000066, -0.0006834, -0.0001158, -0.0003996, 0.0004921
2: 0.0037261, 0.0040555, 0.0037884, 0.0040352, -0.0002118, 0.0001722
3: 0.0016712, 0.0021497, 0.0016709, 0.0021890, -0.0002805, 0.0002360
4: -0.0041966, -0.0034726, -0.0041219, -0.0035013, -0.0003631, 0.0003166
5: -0.0001516, 0.0002768, -0.0000832, 0.0002674, -0.0003018, 0.0002451
6: -0.0040471, -0.0027609, -0.0040516, -0.0027434, -0.0005708, 0.0005526
7: -0.0203590, -0.0161919, -0.0198968, -0.0163695, -0.0021225, 0.0018352
8: 0.9765776, 0.9803886, 0.9771413, 0.9801622, -0.0020845, 0.0017452
9: 0.0029438, 0.0056825, 0.0030716, 0.0053551, -0.0012145, 0.0014162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167229, 0.0176959, -0.0005022, 0.0006156
1: -0.0007086, 0.0000066, -0.0007335, -0.0000461, -0.0003654, 0.0004495
2: 0.0037261, 0.0040555, 0.0037499, 0.0040645, -0.0001969, 0.0001609
3: 0.0016712, 0.0021497, 0.0016853, 0.0021666, -0.0002609, 0.0002216
4: -0.0041966, -0.0034726, -0.0041673, -0.0034687, -0.0003411, 0.0003050
5: -0.0001516, 0.0002768, -0.0001201, 0.0002934, -0.0002735, 0.0002219
6: -0.0040471, -0.0027609, -0.0040504, -0.0027452, -0.0005715, 0.0005537
7: -0.0203590, -0.0161919, -0.0201776, -0.0161689, -0.0019900, 0.0017646
8: 0.9765776, 0.9803886, 0.9767900, 0.9804238, -0.0019517, 0.0016723
9: 0.0029438, 0.0056825, 0.0029299, 0.0055534, -0.0011651, 0.0013256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168102, 0.0175804, -0.0005393, 0.0006637
1: -0.0007086, 0.0000066, -0.0006834, -0.0001158, -0.0003996, 0.0004921
2: 0.0037261, 0.0040555, 0.0037884, 0.0040352, -0.0002118, 0.0001722
3: 0.0016712, 0.0021497, 0.0016709, 0.0021890, -0.0002805, 0.0002360
4: -0.0041966, -0.0034726, -0.0041219, -0.0035013, -0.0003631, 0.0003166
5: -0.0001516, 0.0002768, -0.0000832, 0.0002674, -0.0003018, 0.0002451
6: -0.0040471, -0.0027609, -0.0040516, -0.0027434, -0.0005708, 0.0005526
7: -0.0203590, -0.0161919, -0.0198968, -0.0163695, -0.0021225, 0.0018352
8: 0.9765776, 0.9803886, 0.9771413, 0.9801622, -0.0020845, 0.0017452
9: 0.0029438, 0.0056825, 0.0030716, 0.0053551, -0.0012145, 0.0014162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167229, 0.0176959, -0.0005022, 0.0006156
1: -0.0007086, 0.0000066, -0.0007335, -0.0000461, -0.0003654, 0.0004495
2: 0.0037261, 0.0040555, 0.0037499, 0.0040645, -0.0001969, 0.0001609
3: 0.0016712, 0.0021497, 0.0016853, 0.0021666, -0.0002609, 0.0002216
4: -0.0041966, -0.0034726, -0.0041673, -0.0034687, -0.0003411, 0.0003050
5: -0.0001516, 0.0002768, -0.0001201, 0.0002934, -0.0002735, 0.0002219
6: -0.0040471, -0.0027609, -0.0040504, -0.0027452, -0.0005715, 0.0005537
7: -0.0203590, -0.0161919, -0.0201776, -0.0161689, -0.0019900, 0.0017646
8: 0.9765776, 0.9803886, 0.9767900, 0.9804238, -0.0019517, 0.0016723
9: 0.0029438, 0.0056825, 0.0029299, 0.0055534, -0.0011651, 0.0013256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168095, 0.0175657, -0.0005024, 0.0006421
1: -0.0007086, 0.0000066, -0.0006790, -0.0001302, -0.0003771, 0.0004801
2: 0.0037261, 0.0040555, 0.0037918, 0.0040363, -0.0002044, 0.0001598
3: 0.0016712, 0.0021497, 0.0016797, 0.0021823, -0.0002855, 0.0002437
4: -0.0041966, -0.0034726, -0.0041193, -0.0034943, -0.0003349, 0.0002781
5: -0.0001516, 0.0002768, -0.0000735, 0.0002641, -0.0002955, 0.0002327
6: -0.0040471, -0.0027609, -0.0040474, -0.0027580, -0.0004980, 0.0004945
7: -0.0203590, -0.0161919, -0.0198795, -0.0163305, -0.0019626, 0.0016146
8: 0.9765776, 0.9803886, 0.9771566, 0.9802017, -0.0019492, 0.0015508
9: 0.0029438, 0.0056825, 0.0030466, 0.0053417, -0.0010695, 0.0013128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167528, 0.0177701, -0.0004587, 0.0004587
1: -0.0007086, 0.0000066, -0.0007086, 0.0000066, -0.0003405, 0.0003405
2: 0.0037261, 0.0040555, 0.0037261, 0.0040555, -0.0001461, 0.0001461
3: 0.0016712, 0.0021497, 0.0016712, 0.0021497, -0.0002294, 0.0002294
4: -0.0041966, -0.0034726, -0.0041966, -0.0034726, -0.0002574, 0.0002574
5: -0.0001516, 0.0002768, -0.0001516, 0.0002768, -0.0002088, 0.0002088
6: -0.0040471, -0.0027609, -0.0040471, -0.0027609, -0.0004969, 0.0004969
7: -0.0203590, -0.0161919, -0.0203590, -0.0161919, -0.0014914, 0.0014914
8: 0.9765776, 0.9803886, 0.9765776, 0.9803886, -0.0014267, 0.0014267
9: 0.0029438, 0.0056825, 0.0029438, 0.0056825, -0.0009867, 0.0009867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168095, 0.0175657, -0.0005024, 0.0006421
1: -0.0007086, 0.0000066, -0.0006790, -0.0001302, -0.0003771, 0.0004801
2: 0.0037261, 0.0040555, 0.0037918, 0.0040363, -0.0002044, 0.0001598
3: 0.0016712, 0.0021497, 0.0016797, 0.0021823, -0.0002855, 0.0002437
4: -0.0041966, -0.0034726, -0.0041193, -0.0034943, -0.0003349, 0.0002781
5: -0.0001516, 0.0002768, -0.0000735, 0.0002641, -0.0002955, 0.0002327
6: -0.0040471, -0.0027609, -0.0040474, -0.0027580, -0.0004980, 0.0004945
7: -0.0203590, -0.0161919, -0.0198795, -0.0163305, -0.0019626, 0.0016146
8: 0.9765776, 0.9803886, 0.9771566, 0.9802017, -0.0019492, 0.0015508
9: 0.0029438, 0.0056825, 0.0030466, 0.0053417, -0.0010695, 0.0013128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0168095, 0.0175657, 0.0167528, 0.0177701, -0.0006421, 0.0005024
1: -0.0006790, -0.0001302, -0.0007086, 0.0000066, -0.0004801, 0.0003771
2: 0.0037918, 0.0040363, 0.0037261, 0.0040555, -0.0001598, 0.0002044
3: 0.0016797, 0.0021823, 0.0016712, 0.0021497, -0.0002437, 0.0002855
4: -0.0041193, -0.0034943, -0.0041966, -0.0034726, -0.0002781, 0.0003349
5: -0.0000735, 0.0002641, -0.0001516, 0.0002768, -0.0002327, 0.0002955
6: -0.0040474, -0.0027580, -0.0040471, -0.0027609, -0.0004945, 0.0004980
7: -0.0198795, -0.0163305, -0.0203590, -0.0161919, -0.0016146, 0.0019626
8: 0.9771566, 0.9802017, 0.9765776, 0.9803886, -0.0015508, 0.0019492
9: 0.0030466, 0.0053417, 0.0029438, 0.0056825, -0.0013128, 0.0010695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167528, 0.0177701, -0.0004587, 0.0004587
1: -0.0007086, 0.0000066, -0.0007086, 0.0000066, -0.0003405, 0.0003405
2: 0.0037261, 0.0040555, 0.0037261, 0.0040555, -0.0001461, 0.0001461
3: 0.0016712, 0.0021497, 0.0016712, 0.0021497, -0.0002294, 0.0002294
4: -0.0041966, -0.0034726, -0.0041966, -0.0034726, -0.0002574, 0.0002574
5: -0.0001516, 0.0002768, -0.0001516, 0.0002768, -0.0002088, 0.0002088
6: -0.0040471, -0.0027609, -0.0040471, -0.0027609, -0.0004969, 0.0004969
7: -0.0203590, -0.0161919, -0.0203590, -0.0161919, -0.0014914, 0.0014914
8: 0.9765776, 0.9803886, 0.9765776, 0.9803886, -0.0014267, 0.0014267
9: 0.0029438, 0.0056825, 0.0029438, 0.0056825, -0.0009867, 0.0009867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168005, 0.0175889, -0.0005304, 0.0006535
1: -0.0007086, 0.0000066, -0.0006891, -0.0001125, -0.0003961, 0.0004904
2: 0.0037261, 0.0040555, 0.0037852, 0.0040384, -0.0002077, 0.0001690
3: 0.0016712, 0.0021497, 0.0016772, 0.0021904, -0.0002971, 0.0002447
4: -0.0041966, -0.0034726, -0.0041296, -0.0035002, -0.0003303, 0.0002921
5: -0.0001516, 0.0002768, -0.0000849, 0.0002707, -0.0003022, 0.0002438
6: -0.0040471, -0.0027609, -0.0040364, -0.0027364, -0.0005242, 0.0004820
7: -0.0203590, -0.0161919, -0.0199401, -0.0163629, -0.0019387, 0.0016981
8: 0.9765776, 0.9803886, 0.9771031, 0.9801837, -0.0019395, 0.0016385
9: 0.0029438, 0.0056825, 0.0030664, 0.0053829, -0.0011267, 0.0012988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0166818, 0.0177592, -0.0004521, 0.0005432
1: -0.0007086, 0.0000066, -0.0007634, -0.0000025, -0.0003347, 0.0004057
2: 0.0037261, 0.0040555, 0.0037296, 0.0040776, -0.0001726, 0.0001441
3: 0.0016712, 0.0021497, 0.0016853, 0.0021678, -0.0002630, 0.0002217
4: -0.0041966, -0.0034726, -0.0041939, -0.0034568, -0.0002753, 0.0002563
5: -0.0001516, 0.0002768, -0.0001462, 0.0003108, -0.0002492, 0.0002050
6: -0.0040471, -0.0027609, -0.0040352, -0.0027386, -0.0005244, 0.0004838
7: -0.0203590, -0.0161919, -0.0203417, -0.0160892, -0.0016077, 0.0014839
8: 0.9765776, 0.9803886, 0.9766047, 0.9805312, -0.0015935, 0.0014183
9: 0.0029438, 0.0056825, 0.0028690, 0.0056690, -0.0009816, 0.0010723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168005, 0.0175889, -0.0005304, 0.0006535
1: -0.0007086, 0.0000066, -0.0006891, -0.0001125, -0.0003961, 0.0004904
2: 0.0037261, 0.0040555, 0.0037852, 0.0040384, -0.0002077, 0.0001690
3: 0.0016712, 0.0021497, 0.0016772, 0.0021904, -0.0002971, 0.0002447
4: -0.0041966, -0.0034726, -0.0041296, -0.0035002, -0.0003303, 0.0002921
5: -0.0001516, 0.0002768, -0.0000849, 0.0002707, -0.0003022, 0.0002438
6: -0.0040471, -0.0027609, -0.0040364, -0.0027364, -0.0005242, 0.0004820
7: -0.0203590, -0.0161919, -0.0199401, -0.0163629, -0.0019387, 0.0016981
8: 0.9765776, 0.9803886, 0.9771031, 0.9801837, -0.0019395, 0.0016385
9: 0.0029438, 0.0056825, 0.0030664, 0.0053829, -0.0011267, 0.0012988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0166818, 0.0177592, -0.0004521, 0.0005432
1: -0.0007086, 0.0000066, -0.0007634, -0.0000025, -0.0003347, 0.0004057
2: 0.0037261, 0.0040555, 0.0037296, 0.0040776, -0.0001726, 0.0001441
3: 0.0016712, 0.0021497, 0.0016853, 0.0021678, -0.0002630, 0.0002217
4: -0.0041966, -0.0034726, -0.0041939, -0.0034568, -0.0002753, 0.0002563
5: -0.0001516, 0.0002768, -0.0001462, 0.0003108, -0.0002492, 0.0002050
6: -0.0040471, -0.0027609, -0.0040352, -0.0027386, -0.0005244, 0.0004838
7: -0.0203590, -0.0161919, -0.0203417, -0.0160892, -0.0016077, 0.0014839
8: 0.9765776, 0.9803886, 0.9766047, 0.9805312, -0.0015935, 0.0014183
9: 0.0029438, 0.0056825, 0.0028690, 0.0056690, -0.0009816, 0.0010723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
time: 0.61 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 2.91 seconds
IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0011709
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010953
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010953
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.91
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168102, 0.0175804, -0.0004911, 0.0005877
1: -0.0006838, -0.0000355, -0.0006834, -0.0001158, -0.0003678, 0.0004419
2: 0.0037466, 0.0040438, 0.0037884, 0.0040352, -0.0001867, 0.0001563
3: 0.0016724, 0.0021489, 0.0016709, 0.0021890, -0.0002790, 0.0002350
4: -0.0041714, -0.0034810, -0.0041219, -0.0035013, -0.0003042, 0.0002755
5: -0.0001271, 0.0002629, -0.0000832, 0.0002674, -0.0002728, 0.0002267
6: -0.0040608, -0.0027658, -0.0040516, -0.0027434, -0.0005175, 0.0004781
7: -0.0202023, -0.0162487, -0.0198968, -0.0163695, -0.0017808, 0.0015992
8: 0.9767592, 0.9803019, 0.9771413, 0.9801622, -0.0017633, 0.0015303
9: 0.0029887, 0.0055711, 0.0030716, 0.0053551, -0.0010594, 0.0011901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167229, 0.0176959, -0.0004274, 0.0005122
1: -0.0006838, -0.0000355, -0.0007335, -0.0000461, -0.0003172, 0.0003832
2: 0.0037466, 0.0040438, 0.0037499, 0.0040645, -0.0001626, 0.0001362
3: 0.0016724, 0.0021489, 0.0016853, 0.0021666, -0.0002589, 0.0002199
4: -0.0041714, -0.0034810, -0.0041673, -0.0034687, -0.0002631, 0.0002460
5: -0.0001271, 0.0002629, -0.0001201, 0.0002934, -0.0002356, 0.0001945
6: -0.0040608, -0.0027658, -0.0040504, -0.0027452, -0.0005179, 0.0004794
7: -0.0202023, -0.0162487, -0.0201776, -0.0161689, -0.0015321, 0.0014206
8: 0.9767592, 0.9803019, 0.9767900, 0.9804238, -0.0015081, 0.0013476
9: 0.0029887, 0.0055711, 0.0029299, 0.0055534, -0.0009366, 0.0010189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168102, 0.0175804, -0.0004911, 0.0005877
1: -0.0006838, -0.0000355, -0.0006834, -0.0001158, -0.0003678, 0.0004419
2: 0.0037466, 0.0040438, 0.0037884, 0.0040352, -0.0001867, 0.0001563
3: 0.0016724, 0.0021489, 0.0016709, 0.0021890, -0.0002790, 0.0002350
4: -0.0041714, -0.0034810, -0.0041219, -0.0035013, -0.0003042, 0.0002755
5: -0.0001271, 0.0002629, -0.0000832, 0.0002674, -0.0002728, 0.0002267
6: -0.0040608, -0.0027658, -0.0040516, -0.0027434, -0.0005175, 0.0004781
7: -0.0202023, -0.0162487, -0.0198968, -0.0163695, -0.0017808, 0.0015992
8: 0.9767592, 0.9803019, 0.9771413, 0.9801622, -0.0017633, 0.0015303
9: 0.0029887, 0.0055711, 0.0030716, 0.0053551, -0.0010594, 0.0011901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167229, 0.0176959, -0.0004274, 0.0005122
1: -0.0006838, -0.0000355, -0.0007335, -0.0000461, -0.0003172, 0.0003832
2: 0.0037466, 0.0040438, 0.0037499, 0.0040645, -0.0001626, 0.0001362
3: 0.0016724, 0.0021489, 0.0016853, 0.0021666, -0.0002589, 0.0002199
4: -0.0041714, -0.0034810, -0.0041673, -0.0034687, -0.0002631, 0.0002460
5: -0.0001271, 0.0002629, -0.0001201, 0.0002934, -0.0002356, 0.0001945
6: -0.0040608, -0.0027658, -0.0040504, -0.0027452, -0.0005179, 0.0004794
7: -0.0202023, -0.0162487, -0.0201776, -0.0161689, -0.0015321, 0.0014206
8: 0.9767592, 0.9803019, 0.9767900, 0.9804238, -0.0015081, 0.0013476
9: 0.0029887, 0.0055711, 0.0029299, 0.0055534, -0.0009366, 0.0010189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168102, 0.0175804, -0.0004911, 0.0005877
1: -0.0006838, -0.0000355, -0.0006834, -0.0001158, -0.0003678, 0.0004419
2: 0.0037466, 0.0040438, 0.0037884, 0.0040352, -0.0001867, 0.0001563
3: 0.0016724, 0.0021489, 0.0016709, 0.0021890, -0.0002790, 0.0002350
4: -0.0041714, -0.0034810, -0.0041219, -0.0035013, -0.0003042, 0.0002755
5: -0.0001271, 0.0002629, -0.0000832, 0.0002674, -0.0002728, 0.0002267
6: -0.0040608, -0.0027658, -0.0040516, -0.0027434, -0.0005175, 0.0004781
7: -0.0202023, -0.0162487, -0.0198968, -0.0163695, -0.0017808, 0.0015992
8: 0.9767592, 0.9803019, 0.9771413, 0.9801622, -0.0017633, 0.0015303
9: 0.0029887, 0.0055711, 0.0030716, 0.0053551, -0.0010594, 0.0011901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167229, 0.0176959, -0.0004274, 0.0005122
1: -0.0006838, -0.0000355, -0.0007335, -0.0000461, -0.0003172, 0.0003832
2: 0.0037466, 0.0040438, 0.0037499, 0.0040645, -0.0001626, 0.0001362
3: 0.0016724, 0.0021489, 0.0016853, 0.0021666, -0.0002589, 0.0002199
4: -0.0041714, -0.0034810, -0.0041673, -0.0034687, -0.0002631, 0.0002460
5: -0.0001271, 0.0002629, -0.0001201, 0.0002934, -0.0002356, 0.0001945
6: -0.0040608, -0.0027658, -0.0040504, -0.0027452, -0.0005179, 0.0004794
7: -0.0202023, -0.0162487, -0.0201776, -0.0161689, -0.0015321, 0.0014206
8: 0.9767592, 0.9803019, 0.9767900, 0.9804238, -0.0015081, 0.0013476
9: 0.0029887, 0.0055711, 0.0029299, 0.0055534, -0.0009366, 0.0010189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168102, 0.0175804, -0.0004911, 0.0005877
1: -0.0006838, -0.0000355, -0.0006834, -0.0001158, -0.0003678, 0.0004419
2: 0.0037466, 0.0040438, 0.0037884, 0.0040352, -0.0001867, 0.0001563
3: 0.0016724, 0.0021489, 0.0016709, 0.0021890, -0.0002790, 0.0002350
4: -0.0041714, -0.0034810, -0.0041219, -0.0035013, -0.0003042, 0.0002755
5: -0.0001271, 0.0002629, -0.0000832, 0.0002674, -0.0002728, 0.0002267
6: -0.0040608, -0.0027658, -0.0040516, -0.0027434, -0.0005175, 0.0004781
7: -0.0202023, -0.0162487, -0.0198968, -0.0163695, -0.0017808, 0.0015992
8: 0.9767592, 0.9803019, 0.9771413, 0.9801622, -0.0017633, 0.0015303
9: 0.0029887, 0.0055711, 0.0030716, 0.0053551, -0.0010594, 0.0011901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167229, 0.0176959, -0.0004274, 0.0005122
1: -0.0006838, -0.0000355, -0.0007335, -0.0000461, -0.0003172, 0.0003832
2: 0.0037466, 0.0040438, 0.0037499, 0.0040645, -0.0001626, 0.0001362
3: 0.0016724, 0.0021489, 0.0016853, 0.0021666, -0.0002589, 0.0002199
4: -0.0041714, -0.0034810, -0.0041673, -0.0034687, -0.0002631, 0.0002460
5: -0.0001271, 0.0002629, -0.0001201, 0.0002934, -0.0002356, 0.0001945
6: -0.0040608, -0.0027658, -0.0040504, -0.0027452, -0.0005179, 0.0004794
7: -0.0202023, -0.0162487, -0.0201776, -0.0161689, -0.0015321, 0.0014206
8: 0.9767592, 0.9803019, 0.9767900, 0.9804238, -0.0015081, 0.0013476
9: 0.0029887, 0.0055711, 0.0029299, 0.0055534, -0.0009366, 0.0010189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009017
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168095, 0.0175657, -0.0004746, 0.0005880
1: -0.0006838, -0.0000355, -0.0006790, -0.0001302, -0.0003528, 0.0004373
2: 0.0037466, 0.0040438, 0.0037918, 0.0040363, -0.0001875, 0.0001517
3: 0.0016724, 0.0021489, 0.0016797, 0.0021823, -0.0002859, 0.0002429
4: -0.0041714, -0.0034810, -0.0041193, -0.0034943, -0.0003369, 0.0002980
5: -0.0001271, 0.0002629, -0.0000735, 0.0002641, -0.0002691, 0.0002171
6: -0.0040608, -0.0027658, -0.0040474, -0.0027580, -0.0005670, 0.0005431
7: -0.0202023, -0.0162487, -0.0198795, -0.0163305, -0.0019615, 0.0017206
8: 0.9767592, 0.9803019, 0.9771566, 0.9802017, -0.0018891, 0.0016022
9: 0.0029887, 0.0055711, 0.0030466, 0.0053417, -0.0011332, 0.0013033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167528, 0.0177701, -0.0005371, 0.0005085
1: -0.0006838, -0.0000355, -0.0007086, 0.0000066, -0.0003886, 0.0003704
2: 0.0037466, 0.0040438, 0.0037261, 0.0040555, -0.0001629, 0.0001724
3: 0.0016724, 0.0021489, 0.0016712, 0.0021497, -0.0002275, 0.0002278
4: -0.0041714, -0.0034810, -0.0041966, -0.0034726, -0.0003063, 0.0003253
5: -0.0001271, 0.0002629, -0.0001516, 0.0002768, -0.0002251, 0.0002356
6: -0.0040608, -0.0027658, -0.0040471, -0.0027609, -0.0005643, 0.0005436
7: -0.0202023, -0.0162487, -0.0203590, -0.0161919, -0.0017729, 0.0018867
8: 0.9767592, 0.9803019, 0.9765776, 0.9803886, -0.0016830, 0.0018020
9: 0.0029887, 0.0055711, 0.0029438, 0.0056825, -0.0012492, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168095, 0.0175657, -0.0004746, 0.0005880
1: -0.0006838, -0.0000355, -0.0006790, -0.0001302, -0.0003528, 0.0004373
2: 0.0037466, 0.0040438, 0.0037918, 0.0040363, -0.0001875, 0.0001517
3: 0.0016724, 0.0021489, 0.0016797, 0.0021823, -0.0002859, 0.0002429
4: -0.0041714, -0.0034810, -0.0041193, -0.0034943, -0.0003369, 0.0002980
5: -0.0001271, 0.0002629, -0.0000735, 0.0002641, -0.0002691, 0.0002171
6: -0.0040608, -0.0027658, -0.0040474, -0.0027580, -0.0005670, 0.0005431
7: -0.0202023, -0.0162487, -0.0198795, -0.0163305, -0.0019615, 0.0017206
8: 0.9767592, 0.9803019, 0.9771566, 0.9802017, -0.0018891, 0.0016022
9: 0.0029887, 0.0055711, 0.0030466, 0.0053417, -0.0011332, 0.0013033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167528, 0.0177701, -0.0005371, 0.0005085
1: -0.0006838, -0.0000355, -0.0007086, 0.0000066, -0.0003886, 0.0003704
2: 0.0037466, 0.0040438, 0.0037261, 0.0040555, -0.0001629, 0.0001724
3: 0.0016724, 0.0021489, 0.0016712, 0.0021497, -0.0002275, 0.0002278
4: -0.0041714, -0.0034810, -0.0041966, -0.0034726, -0.0003063, 0.0003253
5: -0.0001271, 0.0002629, -0.0001516, 0.0002768, -0.0002251, 0.0002356
6: -0.0040608, -0.0027658, -0.0040471, -0.0027609, -0.0005643, 0.0005436
7: -0.0202023, -0.0162487, -0.0203590, -0.0161919, -0.0017729, 0.0018867
8: 0.9767592, 0.9803019, 0.9765776, 0.9803886, -0.0016830, 0.0018020
9: 0.0029887, 0.0055711, 0.0029438, 0.0056825, -0.0012492, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168095, 0.0175657, -0.0004746, 0.0005880
1: -0.0006838, -0.0000355, -0.0006790, -0.0001302, -0.0003528, 0.0004373
2: 0.0037466, 0.0040438, 0.0037918, 0.0040363, -0.0001875, 0.0001517
3: 0.0016724, 0.0021489, 0.0016797, 0.0021823, -0.0002859, 0.0002429
4: -0.0041714, -0.0034810, -0.0041193, -0.0034943, -0.0003369, 0.0002980
5: -0.0001271, 0.0002629, -0.0000735, 0.0002641, -0.0002691, 0.0002171
6: -0.0040608, -0.0027658, -0.0040474, -0.0027580, -0.0005670, 0.0005431
7: -0.0202023, -0.0162487, -0.0198795, -0.0163305, -0.0019615, 0.0017206
8: 0.9767592, 0.9803019, 0.9771566, 0.9802017, -0.0018891, 0.0016022
9: 0.0029887, 0.0055711, 0.0030466, 0.0053417, -0.0011332, 0.0013033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167528, 0.0177701, -0.0005371, 0.0005085
1: -0.0006838, -0.0000355, -0.0007086, 0.0000066, -0.0003886, 0.0003704
2: 0.0037466, 0.0040438, 0.0037261, 0.0040555, -0.0001629, 0.0001724
3: 0.0016724, 0.0021489, 0.0016712, 0.0021497, -0.0002275, 0.0002278
4: -0.0041714, -0.0034810, -0.0041966, -0.0034726, -0.0003063, 0.0003253
5: -0.0001271, 0.0002629, -0.0001516, 0.0002768, -0.0002251, 0.0002356
6: -0.0040608, -0.0027658, -0.0040471, -0.0027609, -0.0005643, 0.0005436
7: -0.0202023, -0.0162487, -0.0203590, -0.0161919, -0.0017729, 0.0018867
8: 0.9767592, 0.9803019, 0.9765776, 0.9803886, -0.0016830, 0.0018020
9: 0.0029887, 0.0055711, 0.0029438, 0.0056825, -0.0012492, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168095, 0.0175657, -0.0004746, 0.0005880
1: -0.0006838, -0.0000355, -0.0006790, -0.0001302, -0.0003528, 0.0004373
2: 0.0037466, 0.0040438, 0.0037918, 0.0040363, -0.0001875, 0.0001517
3: 0.0016724, 0.0021489, 0.0016797, 0.0021823, -0.0002859, 0.0002429
4: -0.0041714, -0.0034810, -0.0041193, -0.0034943, -0.0003369, 0.0002980
5: -0.0001271, 0.0002629, -0.0000735, 0.0002641, -0.0002691, 0.0002171
6: -0.0040608, -0.0027658, -0.0040474, -0.0027580, -0.0005670, 0.0005431
7: -0.0202023, -0.0162487, -0.0198795, -0.0163305, -0.0019615, 0.0017206
8: 0.9767592, 0.9803019, 0.9771566, 0.9802017, -0.0018891, 0.0016022
9: 0.0029887, 0.0055711, 0.0030466, 0.0053417, -0.0011332, 0.0013033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167528, 0.0177701, -0.0005371, 0.0005085
1: -0.0006838, -0.0000355, -0.0007086, 0.0000066, -0.0003886, 0.0003704
2: 0.0037466, 0.0040438, 0.0037261, 0.0040555, -0.0001629, 0.0001724
3: 0.0016724, 0.0021489, 0.0016712, 0.0021497, -0.0002275, 0.0002278
4: -0.0041714, -0.0034810, -0.0041966, -0.0034726, -0.0003063, 0.0003253
5: -0.0001271, 0.0002629, -0.0001516, 0.0002768, -0.0002251, 0.0002356
6: -0.0040608, -0.0027658, -0.0040471, -0.0027609, -0.0005643, 0.0005436
7: -0.0202023, -0.0162487, -0.0203590, -0.0161919, -0.0017729, 0.0018867
8: 0.9767592, 0.9803019, 0.9765776, 0.9803886, -0.0016830, 0.0018020
9: 0.0029887, 0.0055711, 0.0029438, 0.0056825, -0.0012492, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0009283
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0011709
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168005, 0.0175889, -0.0005000, 0.0005974
1: -0.0006838, -0.0000355, -0.0006891, -0.0001125, -0.0003710, 0.0004472
2: 0.0037466, 0.0040438, 0.0037852, 0.0040384, -0.0001902, 0.0001597
3: 0.0016724, 0.0021489, 0.0016772, 0.0021904, -0.0002961, 0.0002442
4: -0.0041714, -0.0034810, -0.0041296, -0.0035002, -0.0003286, 0.0003075
5: -0.0001271, 0.0002629, -0.0000849, 0.0002707, -0.0002757, 0.0002280
6: -0.0040608, -0.0027658, -0.0040364, -0.0027364, -0.0005854, 0.0005267
7: -0.0202023, -0.0162487, -0.0199401, -0.0163629, -0.0019175, 0.0017775
8: 0.9767592, 0.9803019, 0.9771031, 0.9801837, -0.0018666, 0.0016680
9: 0.0029887, 0.0055711, 0.0030664, 0.0053829, -0.0011726, 0.0012767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166818, 0.0177592, -0.0005307, 0.0005843
1: -0.0006838, -0.0000355, -0.0007634, -0.0000025, -0.0003838, 0.0004294
2: 0.0037466, 0.0040438, 0.0037296, 0.0040776, -0.0001866, 0.0001704
3: 0.0016724, 0.0021489, 0.0016853, 0.0021678, -0.0002610, 0.0002200
4: -0.0041714, -0.0034810, -0.0041939, -0.0034568, -0.0003227, 0.0003245
5: -0.0001271, 0.0002629, -0.0001462, 0.0003108, -0.0002619, 0.0002327
6: -0.0040608, -0.0027658, -0.0040352, -0.0027386, -0.0005836, 0.0005260
7: -0.0202023, -0.0162487, -0.0203417, -0.0160892, -0.0018786, 0.0018806
8: 0.9767592, 0.9803019, 0.9766047, 0.9805312, -0.0018315, 0.0017906
9: 0.0029887, 0.0055711, 0.0028690, 0.0056690, -0.0012441, 0.0012484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168005, 0.0175889, -0.0005000, 0.0005974
1: -0.0006838, -0.0000355, -0.0006891, -0.0001125, -0.0003710, 0.0004472
2: 0.0037466, 0.0040438, 0.0037852, 0.0040384, -0.0001902, 0.0001597
3: 0.0016724, 0.0021489, 0.0016772, 0.0021904, -0.0002961, 0.0002442
4: -0.0041714, -0.0034810, -0.0041296, -0.0035002, -0.0003286, 0.0003075
5: -0.0001271, 0.0002629, -0.0000849, 0.0002707, -0.0002757, 0.0002280
6: -0.0040608, -0.0027658, -0.0040364, -0.0027364, -0.0005854, 0.0005267
7: -0.0202023, -0.0162487, -0.0199401, -0.0163629, -0.0019175, 0.0017775
8: 0.9767592, 0.9803019, 0.9771031, 0.9801837, -0.0018666, 0.0016680
9: 0.0029887, 0.0055711, 0.0030664, 0.0053829, -0.0011726, 0.0012767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166818, 0.0177592, -0.0005307, 0.0005843
1: -0.0006838, -0.0000355, -0.0007634, -0.0000025, -0.0003838, 0.0004294
2: 0.0037466, 0.0040438, 0.0037296, 0.0040776, -0.0001866, 0.0001704
3: 0.0016724, 0.0021489, 0.0016853, 0.0021678, -0.0002610, 0.0002200
4: -0.0041714, -0.0034810, -0.0041939, -0.0034568, -0.0003227, 0.0003245
5: -0.0001271, 0.0002629, -0.0001462, 0.0003108, -0.0002619, 0.0002327
6: -0.0040608, -0.0027658, -0.0040352, -0.0027386, -0.0005836, 0.0005260
7: -0.0202023, -0.0162487, -0.0203417, -0.0160892, -0.0018786, 0.0018806
8: 0.9767592, 0.9803019, 0.9766047, 0.9805312, -0.0018315, 0.0017906
9: 0.0029887, 0.0055711, 0.0028690, 0.0056690, -0.0012441, 0.0012484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168005, 0.0175889, -0.0005000, 0.0005974
1: -0.0006838, -0.0000355, -0.0006891, -0.0001125, -0.0003710, 0.0004472
2: 0.0037466, 0.0040438, 0.0037852, 0.0040384, -0.0001902, 0.0001597
3: 0.0016724, 0.0021489, 0.0016772, 0.0021904, -0.0002961, 0.0002442
4: -0.0041714, -0.0034810, -0.0041296, -0.0035002, -0.0003286, 0.0003075
5: -0.0001271, 0.0002629, -0.0000849, 0.0002707, -0.0002757, 0.0002280
6: -0.0040608, -0.0027658, -0.0040364, -0.0027364, -0.0005854, 0.0005267
7: -0.0202023, -0.0162487, -0.0199401, -0.0163629, -0.0019175, 0.0017775
8: 0.9767592, 0.9803019, 0.9771031, 0.9801837, -0.0018666, 0.0016680
9: 0.0029887, 0.0055711, 0.0030664, 0.0053829, -0.0011726, 0.0012767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166818, 0.0177592, -0.0005307, 0.0005843
1: -0.0006838, -0.0000355, -0.0007634, -0.0000025, -0.0003838, 0.0004294
2: 0.0037466, 0.0040438, 0.0037296, 0.0040776, -0.0001866, 0.0001704
3: 0.0016724, 0.0021489, 0.0016853, 0.0021678, -0.0002610, 0.0002200
4: -0.0041714, -0.0034810, -0.0041939, -0.0034568, -0.0003227, 0.0003245
5: -0.0001271, 0.0002629, -0.0001462, 0.0003108, -0.0002619, 0.0002327
6: -0.0040608, -0.0027658, -0.0040352, -0.0027386, -0.0005836, 0.0005260
7: -0.0202023, -0.0162487, -0.0203417, -0.0160892, -0.0018786, 0.0018806
8: 0.9767592, 0.9803019, 0.9766047, 0.9805312, -0.0018315, 0.0017906
9: 0.0029887, 0.0055711, 0.0028690, 0.0056690, -0.0012441, 0.0012484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168005, 0.0175889, -0.0005000, 0.0005974
1: -0.0006838, -0.0000355, -0.0006891, -0.0001125, -0.0003710, 0.0004472
2: 0.0037466, 0.0040438, 0.0037852, 0.0040384, -0.0001902, 0.0001597
3: 0.0016724, 0.0021489, 0.0016772, 0.0021904, -0.0002961, 0.0002442
4: -0.0041714, -0.0034810, -0.0041296, -0.0035002, -0.0003286, 0.0003075
5: -0.0001271, 0.0002629, -0.0000849, 0.0002707, -0.0002757, 0.0002280
6: -0.0040608, -0.0027658, -0.0040364, -0.0027364, -0.0005854, 0.0005267
7: -0.0202023, -0.0162487, -0.0199401, -0.0163629, -0.0019175, 0.0017775
8: 0.9767592, 0.9803019, 0.9771031, 0.9801837, -0.0018666, 0.0016680
9: 0.0029887, 0.0055711, 0.0030664, 0.0053829, -0.0011726, 0.0012767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166818, 0.0177592, -0.0005307, 0.0005843
1: -0.0006838, -0.0000355, -0.0007634, -0.0000025, -0.0003838, 0.0004294
2: 0.0037466, 0.0040438, 0.0037296, 0.0040776, -0.0001866, 0.0001704
3: 0.0016724, 0.0021489, 0.0016853, 0.0021678, -0.0002610, 0.0002200
4: -0.0041714, -0.0034810, -0.0041939, -0.0034568, -0.0003227, 0.0003245
5: -0.0001271, 0.0002629, -0.0001462, 0.0003108, -0.0002619, 0.0002327
6: -0.0040608, -0.0027658, -0.0040352, -0.0027386, -0.0005836, 0.0005260
7: -0.0202023, -0.0162487, -0.0203417, -0.0160892, -0.0018786, 0.0018806
8: 0.9767592, 0.9803019, 0.9766047, 0.9805312, -0.0018315, 0.0017906
9: 0.0029887, 0.0055711, 0.0028690, 0.0056690, -0.0012441, 0.0012484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 96

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009014
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168150, 0.0175552, -0.0005099, 0.0006573
1: -0.0007086, 0.0000066, -0.0006769, -0.0001346, -0.0003792, 0.0004855
2: 0.0037261, 0.0040555, 0.0037959, 0.0040343, -0.0002100, 0.0001627
3: 0.0016712, 0.0021497, 0.0016748, 0.0021812, -0.0002694, 0.0002329
4: -0.0041966, -0.0034726, -0.0041112, -0.0034933, -0.0003691, 0.0003019
5: -0.0001516, 0.0002768, -0.0000714, 0.0002628, -0.0002973, 0.0002330
6: -0.0040471, -0.0027609, -0.0040611, -0.0027634, -0.0005442, 0.0005626
7: -0.0203590, -0.0161919, -0.0198345, -0.0163245, -0.0021554, 0.0017478
8: 0.9765776, 0.9803886, 0.9772027, 0.9801946, -0.0021048, 0.0016551
9: 0.0029438, 0.0056825, 0.0030426, 0.0053139, -0.0011547, 0.0014365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167893, 0.0177068, -0.0005085, 0.0005371
1: -0.0007086, 0.0000066, -0.0006838, -0.0000355, -0.0003704, 0.0003886
2: 0.0037261, 0.0040555, 0.0037466, 0.0040438, -0.0001724, 0.0001629
3: 0.0016712, 0.0021497, 0.0016724, 0.0021489, -0.0002278, 0.0002275
4: -0.0041966, -0.0034726, -0.0041714, -0.0034810, -0.0003253, 0.0003063
5: -0.0001516, 0.0002768, -0.0001271, 0.0002629, -0.0002356, 0.0002251
6: -0.0040471, -0.0027609, -0.0040608, -0.0027658, -0.0005436, 0.0005643
7: -0.0203590, -0.0161919, -0.0202023, -0.0162487, -0.0018867, 0.0017728
8: 0.9765776, 0.9803886, 0.9767592, 0.9803019, -0.0018020, 0.0016830
9: 0.0029438, 0.0056825, 0.0029887, 0.0055711, -0.0011710, 0.0012492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168150, 0.0175552, -0.0005099, 0.0006573
1: -0.0007086, 0.0000066, -0.0006769, -0.0001346, -0.0003792, 0.0004855
2: 0.0037261, 0.0040555, 0.0037959, 0.0040343, -0.0002100, 0.0001627
3: 0.0016712, 0.0021497, 0.0016748, 0.0021812, -0.0002694, 0.0002329
4: -0.0041966, -0.0034726, -0.0041112, -0.0034933, -0.0003691, 0.0003019
5: -0.0001516, 0.0002768, -0.0000714, 0.0002628, -0.0002973, 0.0002330
6: -0.0040471, -0.0027609, -0.0040611, -0.0027634, -0.0005442, 0.0005626
7: -0.0203590, -0.0161919, -0.0198345, -0.0163245, -0.0021554, 0.0017478
8: 0.9765776, 0.9803886, 0.9772027, 0.9801946, -0.0021048, 0.0016551
9: 0.0029438, 0.0056825, 0.0030426, 0.0053139, -0.0011547, 0.0014365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167893, 0.0177068, -0.0005085, 0.0005371
1: -0.0007086, 0.0000066, -0.0006838, -0.0000355, -0.0003704, 0.0003886
2: 0.0037261, 0.0040555, 0.0037466, 0.0040438, -0.0001724, 0.0001629
3: 0.0016712, 0.0021497, 0.0016724, 0.0021489, -0.0002278, 0.0002275
4: -0.0041966, -0.0034726, -0.0041714, -0.0034810, -0.0003253, 0.0003063
5: -0.0001516, 0.0002768, -0.0001271, 0.0002629, -0.0002356, 0.0002251
6: -0.0040471, -0.0027609, -0.0040608, -0.0027658, -0.0005436, 0.0005643
7: -0.0203590, -0.0161919, -0.0202023, -0.0162487, -0.0018867, 0.0017728
8: 0.9765776, 0.9803886, 0.9767592, 0.9803019, -0.0018020, 0.0016830
9: 0.0029438, 0.0056825, 0.0029887, 0.0055711, -0.0011710, 0.0012492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168150, 0.0175552, -0.0005099, 0.0006573
1: -0.0007086, 0.0000066, -0.0006769, -0.0001346, -0.0003792, 0.0004855
2: 0.0037261, 0.0040555, 0.0037959, 0.0040343, -0.0002100, 0.0001627
3: 0.0016712, 0.0021497, 0.0016748, 0.0021812, -0.0002694, 0.0002329
4: -0.0041966, -0.0034726, -0.0041112, -0.0034933, -0.0003691, 0.0003019
5: -0.0001516, 0.0002768, -0.0000714, 0.0002628, -0.0002973, 0.0002330
6: -0.0040471, -0.0027609, -0.0040611, -0.0027634, -0.0005442, 0.0005626
7: -0.0203590, -0.0161919, -0.0198345, -0.0163245, -0.0021554, 0.0017478
8: 0.9765776, 0.9803886, 0.9772027, 0.9801946, -0.0021048, 0.0016551
9: 0.0029438, 0.0056825, 0.0030426, 0.0053139, -0.0011547, 0.0014365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167893, 0.0177068, -0.0005085, 0.0005371
1: -0.0007086, 0.0000066, -0.0006838, -0.0000355, -0.0003704, 0.0003886
2: 0.0037261, 0.0040555, 0.0037466, 0.0040438, -0.0001724, 0.0001629
3: 0.0016712, 0.0021497, 0.0016724, 0.0021489, -0.0002278, 0.0002275
4: -0.0041966, -0.0034726, -0.0041714, -0.0034810, -0.0003253, 0.0003063
5: -0.0001516, 0.0002768, -0.0001271, 0.0002629, -0.0002356, 0.0002251
6: -0.0040471, -0.0027609, -0.0040608, -0.0027658, -0.0005436, 0.0005643
7: -0.0203590, -0.0161919, -0.0202023, -0.0162487, -0.0018867, 0.0017728
8: 0.9765776, 0.9803886, 0.9767592, 0.9803019, -0.0018020, 0.0016830
9: 0.0029438, 0.0056825, 0.0029887, 0.0055711, -0.0011710, 0.0012492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168150, 0.0175552, -0.0005099, 0.0006573
1: -0.0007086, 0.0000066, -0.0006769, -0.0001346, -0.0003792, 0.0004855
2: 0.0037261, 0.0040555, 0.0037959, 0.0040343, -0.0002100, 0.0001627
3: 0.0016712, 0.0021497, 0.0016748, 0.0021812, -0.0002694, 0.0002329
4: -0.0041966, -0.0034726, -0.0041112, -0.0034933, -0.0003691, 0.0003019
5: -0.0001516, 0.0002768, -0.0000714, 0.0002628, -0.0002973, 0.0002330
6: -0.0040471, -0.0027609, -0.0040611, -0.0027634, -0.0005442, 0.0005626
7: -0.0203590, -0.0161919, -0.0198345, -0.0163245, -0.0021554, 0.0017478
8: 0.9765776, 0.9803886, 0.9772027, 0.9801946, -0.0021048, 0.0016551
9: 0.0029438, 0.0056825, 0.0030426, 0.0053139, -0.0011547, 0.0014365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167893, 0.0177068, -0.0005085, 0.0005371
1: -0.0007086, 0.0000066, -0.0006838, -0.0000355, -0.0003704, 0.0003886
2: 0.0037261, 0.0040555, 0.0037466, 0.0040438, -0.0001724, 0.0001629
3: 0.0016712, 0.0021497, 0.0016724, 0.0021489, -0.0002278, 0.0002275
4: -0.0041966, -0.0034726, -0.0041714, -0.0034810, -0.0003253, 0.0003063
5: -0.0001516, 0.0002768, -0.0001271, 0.0002629, -0.0002356, 0.0002251
6: -0.0040471, -0.0027609, -0.0040608, -0.0027658, -0.0005436, 0.0005643
7: -0.0203590, -0.0161919, -0.0202023, -0.0162487, -0.0018867, 0.0017728
8: 0.9765776, 0.9803886, 0.9767592, 0.9803019, -0.0018020, 0.0016830
9: 0.0029438, 0.0056825, 0.0029887, 0.0055711, -0.0011710, 0.0012492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0009320
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168102, 0.0175804, -0.0005393, 0.0006637
1: -0.0007086, 0.0000066, -0.0006834, -0.0001158, -0.0003996, 0.0004921
2: 0.0037261, 0.0040555, 0.0037884, 0.0040352, -0.0002118, 0.0001722
3: 0.0016712, 0.0021497, 0.0016709, 0.0021890, -0.0002805, 0.0002360
4: -0.0041966, -0.0034726, -0.0041219, -0.0035013, -0.0003631, 0.0003166
5: -0.0001516, 0.0002768, -0.0000832, 0.0002674, -0.0003018, 0.0002451
6: -0.0040471, -0.0027609, -0.0040516, -0.0027434, -0.0005708, 0.0005526
7: -0.0203590, -0.0161919, -0.0198968, -0.0163695, -0.0021225, 0.0018352
8: 0.9765776, 0.9803886, 0.9771413, 0.9801622, -0.0020845, 0.0017452
9: 0.0029438, 0.0056825, 0.0030716, 0.0053551, -0.0012145, 0.0014162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167229, 0.0176959, -0.0005022, 0.0006156
1: -0.0007086, 0.0000066, -0.0007335, -0.0000461, -0.0003654, 0.0004495
2: 0.0037261, 0.0040555, 0.0037499, 0.0040645, -0.0001969, 0.0001609
3: 0.0016712, 0.0021497, 0.0016853, 0.0021666, -0.0002609, 0.0002216
4: -0.0041966, -0.0034726, -0.0041673, -0.0034687, -0.0003411, 0.0003050
5: -0.0001516, 0.0002768, -0.0001201, 0.0002934, -0.0002735, 0.0002219
6: -0.0040471, -0.0027609, -0.0040504, -0.0027452, -0.0005715, 0.0005537
7: -0.0203590, -0.0161919, -0.0201776, -0.0161689, -0.0019900, 0.0017646
8: 0.9765776, 0.9803886, 0.9767900, 0.9804238, -0.0019517, 0.0016723
9: 0.0029438, 0.0056825, 0.0029299, 0.0055534, -0.0011651, 0.0013256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168102, 0.0175804, -0.0005393, 0.0006637
1: -0.0007086, 0.0000066, -0.0006834, -0.0001158, -0.0003996, 0.0004921
2: 0.0037261, 0.0040555, 0.0037884, 0.0040352, -0.0002118, 0.0001722
3: 0.0016712, 0.0021497, 0.0016709, 0.0021890, -0.0002805, 0.0002360
4: -0.0041966, -0.0034726, -0.0041219, -0.0035013, -0.0003631, 0.0003166
5: -0.0001516, 0.0002768, -0.0000832, 0.0002674, -0.0003018, 0.0002451
6: -0.0040471, -0.0027609, -0.0040516, -0.0027434, -0.0005708, 0.0005526
7: -0.0203590, -0.0161919, -0.0198968, -0.0163695, -0.0021225, 0.0018352
8: 0.9765776, 0.9803886, 0.9771413, 0.9801622, -0.0020845, 0.0017452
9: 0.0029438, 0.0056825, 0.0030716, 0.0053551, -0.0012145, 0.0014162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167229, 0.0176959, -0.0005022, 0.0006156
1: -0.0007086, 0.0000066, -0.0007335, -0.0000461, -0.0003654, 0.0004495
2: 0.0037261, 0.0040555, 0.0037499, 0.0040645, -0.0001969, 0.0001609
3: 0.0016712, 0.0021497, 0.0016853, 0.0021666, -0.0002609, 0.0002216
4: -0.0041966, -0.0034726, -0.0041673, -0.0034687, -0.0003411, 0.0003050
5: -0.0001516, 0.0002768, -0.0001201, 0.0002934, -0.0002735, 0.0002219
6: -0.0040471, -0.0027609, -0.0040504, -0.0027452, -0.0005715, 0.0005537
7: -0.0203590, -0.0161919, -0.0201776, -0.0161689, -0.0019900, 0.0017646
8: 0.9765776, 0.9803886, 0.9767900, 0.9804238, -0.0019517, 0.0016723
9: 0.0029438, 0.0056825, 0.0029299, 0.0055534, -0.0011651, 0.0013256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168102, 0.0175804, -0.0005393, 0.0006637
1: -0.0007086, 0.0000066, -0.0006834, -0.0001158, -0.0003996, 0.0004921
2: 0.0037261, 0.0040555, 0.0037884, 0.0040352, -0.0002118, 0.0001722
3: 0.0016712, 0.0021497, 0.0016709, 0.0021890, -0.0002805, 0.0002360
4: -0.0041966, -0.0034726, -0.0041219, -0.0035013, -0.0003631, 0.0003166
5: -0.0001516, 0.0002768, -0.0000832, 0.0002674, -0.0003018, 0.0002451
6: -0.0040471, -0.0027609, -0.0040516, -0.0027434, -0.0005708, 0.0005526
7: -0.0203590, -0.0161919, -0.0198968, -0.0163695, -0.0021225, 0.0018352
8: 0.9765776, 0.9803886, 0.9771413, 0.9801622, -0.0020845, 0.0017452
9: 0.0029438, 0.0056825, 0.0030716, 0.0053551, -0.0012145, 0.0014162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167229, 0.0176959, -0.0005022, 0.0006156
1: -0.0007086, 0.0000066, -0.0007335, -0.0000461, -0.0003654, 0.0004495
2: 0.0037261, 0.0040555, 0.0037499, 0.0040645, -0.0001969, 0.0001609
3: 0.0016712, 0.0021497, 0.0016853, 0.0021666, -0.0002609, 0.0002216
4: -0.0041966, -0.0034726, -0.0041673, -0.0034687, -0.0003411, 0.0003050
5: -0.0001516, 0.0002768, -0.0001201, 0.0002934, -0.0002735, 0.0002219
6: -0.0040471, -0.0027609, -0.0040504, -0.0027452, -0.0005715, 0.0005537
7: -0.0203590, -0.0161919, -0.0201776, -0.0161689, -0.0019900, 0.0017646
8: 0.9765776, 0.9803886, 0.9767900, 0.9804238, -0.0019517, 0.0016723
9: 0.0029438, 0.0056825, 0.0029299, 0.0055534, -0.0011651, 0.0013256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168102, 0.0175804, -0.0005393, 0.0006637
1: -0.0007086, 0.0000066, -0.0006834, -0.0001158, -0.0003996, 0.0004921
2: 0.0037261, 0.0040555, 0.0037884, 0.0040352, -0.0002118, 0.0001722
3: 0.0016712, 0.0021497, 0.0016709, 0.0021890, -0.0002805, 0.0002360
4: -0.0041966, -0.0034726, -0.0041219, -0.0035013, -0.0003631, 0.0003166
5: -0.0001516, 0.0002768, -0.0000832, 0.0002674, -0.0003018, 0.0002451
6: -0.0040471, -0.0027609, -0.0040516, -0.0027434, -0.0005708, 0.0005526
7: -0.0203590, -0.0161919, -0.0198968, -0.0163695, -0.0021225, 0.0018352
8: 0.9765776, 0.9803886, 0.9771413, 0.9801622, -0.0020845, 0.0017452
9: 0.0029438, 0.0056825, 0.0030716, 0.0053551, -0.0012145, 0.0014162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167229, 0.0176959, -0.0005022, 0.0006156
1: -0.0007086, 0.0000066, -0.0007335, -0.0000461, -0.0003654, 0.0004495
2: 0.0037261, 0.0040555, 0.0037499, 0.0040645, -0.0001969, 0.0001609
3: 0.0016712, 0.0021497, 0.0016853, 0.0021666, -0.0002609, 0.0002216
4: -0.0041966, -0.0034726, -0.0041673, -0.0034687, -0.0003411, 0.0003050
5: -0.0001516, 0.0002768, -0.0001201, 0.0002934, -0.0002735, 0.0002219
6: -0.0040471, -0.0027609, -0.0040504, -0.0027452, -0.0005715, 0.0005537
7: -0.0203590, -0.0161919, -0.0201776, -0.0161689, -0.0019900, 0.0017646
8: 0.9765776, 0.9803886, 0.9767900, 0.9804238, -0.0019517, 0.0016723
9: 0.0029438, 0.0056825, 0.0029299, 0.0055534, -0.0011651, 0.0013256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0009042
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168095, 0.0175657, -0.0005024, 0.0006421
1: -0.0007086, 0.0000066, -0.0006790, -0.0001302, -0.0003771, 0.0004801
2: 0.0037261, 0.0040555, 0.0037918, 0.0040363, -0.0002044, 0.0001598
3: 0.0016712, 0.0021497, 0.0016797, 0.0021823, -0.0002855, 0.0002437
4: -0.0041966, -0.0034726, -0.0041193, -0.0034943, -0.0003349, 0.0002781
5: -0.0001516, 0.0002768, -0.0000735, 0.0002641, -0.0002955, 0.0002327
6: -0.0040471, -0.0027609, -0.0040474, -0.0027580, -0.0004980, 0.0004945
7: -0.0203590, -0.0161919, -0.0198795, -0.0163305, -0.0019626, 0.0016146
8: 0.9765776, 0.9803886, 0.9771566, 0.9802017, -0.0019492, 0.0015508
9: 0.0029438, 0.0056825, 0.0030466, 0.0053417, -0.0010695, 0.0013128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167528, 0.0177701, -0.0004587, 0.0004587
1: -0.0007086, 0.0000066, -0.0007086, 0.0000066, -0.0003405, 0.0003405
2: 0.0037261, 0.0040555, 0.0037261, 0.0040555, -0.0001461, 0.0001461
3: 0.0016712, 0.0021497, 0.0016712, 0.0021497, -0.0002294, 0.0002294
4: -0.0041966, -0.0034726, -0.0041966, -0.0034726, -0.0002574, 0.0002574
5: -0.0001516, 0.0002768, -0.0001516, 0.0002768, -0.0002088, 0.0002088
6: -0.0040471, -0.0027609, -0.0040471, -0.0027609, -0.0004969, 0.0004969
7: -0.0203590, -0.0161919, -0.0203590, -0.0161919, -0.0014914, 0.0014914
8: 0.9765776, 0.9803886, 0.9765776, 0.9803886, -0.0014267, 0.0014267
9: 0.0029438, 0.0056825, 0.0029438, 0.0056825, -0.0009867, 0.0009867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168095, 0.0175657, -0.0005024, 0.0006421
1: -0.0007086, 0.0000066, -0.0006790, -0.0001302, -0.0003771, 0.0004801
2: 0.0037261, 0.0040555, 0.0037918, 0.0040363, -0.0002044, 0.0001598
3: 0.0016712, 0.0021497, 0.0016797, 0.0021823, -0.0002855, 0.0002437
4: -0.0041966, -0.0034726, -0.0041193, -0.0034943, -0.0003349, 0.0002781
5: -0.0001516, 0.0002768, -0.0000735, 0.0002641, -0.0002955, 0.0002327
6: -0.0040471, -0.0027609, -0.0040474, -0.0027580, -0.0004980, 0.0004945
7: -0.0203590, -0.0161919, -0.0198795, -0.0163305, -0.0019626, 0.0016146
8: 0.9765776, 0.9803886, 0.9771566, 0.9802017, -0.0019492, 0.0015508
9: 0.0029438, 0.0056825, 0.0030466, 0.0053417, -0.0010695, 0.0013128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167528, 0.0177701, -0.0004587, 0.0004587
1: -0.0007086, 0.0000066, -0.0007086, 0.0000066, -0.0003405, 0.0003405
2: 0.0037261, 0.0040555, 0.0037261, 0.0040555, -0.0001461, 0.0001461
3: 0.0016712, 0.0021497, 0.0016712, 0.0021497, -0.0002294, 0.0002294
4: -0.0041966, -0.0034726, -0.0041966, -0.0034726, -0.0002574, 0.0002574
5: -0.0001516, 0.0002768, -0.0001516, 0.0002768, -0.0002088, 0.0002088
6: -0.0040471, -0.0027609, -0.0040471, -0.0027609, -0.0004969, 0.0004969
7: -0.0203590, -0.0161919, -0.0203590, -0.0161919, -0.0014914, 0.0014914
8: 0.9765776, 0.9803886, 0.9765776, 0.9803886, -0.0014267, 0.0014267
9: 0.0029438, 0.0056825, 0.0029438, 0.0056825, -0.0009867, 0.0009867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168095, 0.0175657, -0.0005024, 0.0006421
1: -0.0007086, 0.0000066, -0.0006790, -0.0001302, -0.0003771, 0.0004801
2: 0.0037261, 0.0040555, 0.0037918, 0.0040363, -0.0002044, 0.0001598
3: 0.0016712, 0.0021497, 0.0016797, 0.0021823, -0.0002855, 0.0002437
4: -0.0041966, -0.0034726, -0.0041193, -0.0034943, -0.0003349, 0.0002781
5: -0.0001516, 0.0002768, -0.0000735, 0.0002641, -0.0002955, 0.0002327
6: -0.0040471, -0.0027609, -0.0040474, -0.0027580, -0.0004980, 0.0004945
7: -0.0203590, -0.0161919, -0.0198795, -0.0163305, -0.0019626, 0.0016146
8: 0.9765776, 0.9803886, 0.9771566, 0.9802017, -0.0019492, 0.0015508
9: 0.0029438, 0.0056825, 0.0030466, 0.0053417, -0.0010695, 0.0013128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167528, 0.0177701, -0.0004587, 0.0004587
1: -0.0007086, 0.0000066, -0.0007086, 0.0000066, -0.0003405, 0.0003405
2: 0.0037261, 0.0040555, 0.0037261, 0.0040555, -0.0001461, 0.0001461
3: 0.0016712, 0.0021497, 0.0016712, 0.0021497, -0.0002294, 0.0002294
4: -0.0041966, -0.0034726, -0.0041966, -0.0034726, -0.0002574, 0.0002574
5: -0.0001516, 0.0002768, -0.0001516, 0.0002768, -0.0002088, 0.0002088
6: -0.0040471, -0.0027609, -0.0040471, -0.0027609, -0.0004969, 0.0004969
7: -0.0203590, -0.0161919, -0.0203590, -0.0161919, -0.0014914, 0.0014914
8: 0.9765776, 0.9803886, 0.9765776, 0.9803886, -0.0014267, 0.0014267
9: 0.0029438, 0.0056825, 0.0029438, 0.0056825, -0.0009867, 0.0009867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168095, 0.0175657, -0.0005024, 0.0006421
1: -0.0007086, 0.0000066, -0.0006790, -0.0001302, -0.0003771, 0.0004801
2: 0.0037261, 0.0040555, 0.0037918, 0.0040363, -0.0002044, 0.0001598
3: 0.0016712, 0.0021497, 0.0016797, 0.0021823, -0.0002855, 0.0002437
4: -0.0041966, -0.0034726, -0.0041193, -0.0034943, -0.0003349, 0.0002781
5: -0.0001516, 0.0002768, -0.0000735, 0.0002641, -0.0002955, 0.0002327
6: -0.0040471, -0.0027609, -0.0040474, -0.0027580, -0.0004980, 0.0004945
7: -0.0203590, -0.0161919, -0.0198795, -0.0163305, -0.0019626, 0.0016146
8: 0.9765776, 0.9803886, 0.9771566, 0.9802017, -0.0019492, 0.0015508
9: 0.0029438, 0.0056825, 0.0030466, 0.0053417, -0.0010695, 0.0013128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0167528, 0.0177701, -0.0004587, 0.0004587
1: -0.0007086, 0.0000066, -0.0007086, 0.0000066, -0.0003405, 0.0003405
2: 0.0037261, 0.0040555, 0.0037261, 0.0040555, -0.0001461, 0.0001461
3: 0.0016712, 0.0021497, 0.0016712, 0.0021497, -0.0002294, 0.0002294
4: -0.0041966, -0.0034726, -0.0041966, -0.0034726, -0.0002574, 0.0002574
5: -0.0001516, 0.0002768, -0.0001516, 0.0002768, -0.0002088, 0.0002088
6: -0.0040471, -0.0027609, -0.0040471, -0.0027609, -0.0004969, 0.0004969
7: -0.0203590, -0.0161919, -0.0203590, -0.0161919, -0.0014914, 0.0014914
8: 0.9765776, 0.9803886, 0.9765776, 0.9803886, -0.0014267, 0.0014267
9: 0.0029438, 0.0056825, 0.0029438, 0.0056825, -0.0009867, 0.0009867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0009320
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168005, 0.0175889, -0.0005304, 0.0006535
1: -0.0007086, 0.0000066, -0.0006891, -0.0001125, -0.0003961, 0.0004904
2: 0.0037261, 0.0040555, 0.0037852, 0.0040384, -0.0002077, 0.0001690
3: 0.0016712, 0.0021497, 0.0016772, 0.0021904, -0.0002971, 0.0002447
4: -0.0041966, -0.0034726, -0.0041296, -0.0035002, -0.0003303, 0.0002921
5: -0.0001516, 0.0002768, -0.0000849, 0.0002707, -0.0003022, 0.0002438
6: -0.0040471, -0.0027609, -0.0040364, -0.0027364, -0.0005242, 0.0004820
7: -0.0203590, -0.0161919, -0.0199401, -0.0163629, -0.0019387, 0.0016981
8: 0.9765776, 0.9803886, 0.9771031, 0.9801837, -0.0019395, 0.0016385
9: 0.0029438, 0.0056825, 0.0030664, 0.0053829, -0.0011267, 0.0012988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0166818, 0.0177592, -0.0004521, 0.0005432
1: -0.0007086, 0.0000066, -0.0007634, -0.0000025, -0.0003347, 0.0004057
2: 0.0037261, 0.0040555, 0.0037296, 0.0040776, -0.0001726, 0.0001441
3: 0.0016712, 0.0021497, 0.0016853, 0.0021678, -0.0002630, 0.0002217
4: -0.0041966, -0.0034726, -0.0041939, -0.0034568, -0.0002753, 0.0002563
5: -0.0001516, 0.0002768, -0.0001462, 0.0003108, -0.0002492, 0.0002050
6: -0.0040471, -0.0027609, -0.0040352, -0.0027386, -0.0005244, 0.0004838
7: -0.0203590, -0.0161919, -0.0203417, -0.0160892, -0.0016077, 0.0014839
8: 0.9765776, 0.9803886, 0.9766047, 0.9805312, -0.0015935, 0.0014183
9: 0.0029438, 0.0056825, 0.0028690, 0.0056690, -0.0009816, 0.0010723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168005, 0.0175889, -0.0005304, 0.0006535
1: -0.0007086, 0.0000066, -0.0006891, -0.0001125, -0.0003961, 0.0004904
2: 0.0037261, 0.0040555, 0.0037852, 0.0040384, -0.0002077, 0.0001690
3: 0.0016712, 0.0021497, 0.0016772, 0.0021904, -0.0002971, 0.0002447
4: -0.0041966, -0.0034726, -0.0041296, -0.0035002, -0.0003303, 0.0002921
5: -0.0001516, 0.0002768, -0.0000849, 0.0002707, -0.0003022, 0.0002438
6: -0.0040471, -0.0027609, -0.0040364, -0.0027364, -0.0005242, 0.0004820
7: -0.0203590, -0.0161919, -0.0199401, -0.0163629, -0.0019387, 0.0016981
8: 0.9765776, 0.9803886, 0.9771031, 0.9801837, -0.0019395, 0.0016385
9: 0.0029438, 0.0056825, 0.0030664, 0.0053829, -0.0011267, 0.0012988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0166818, 0.0177592, -0.0004521, 0.0005432
1: -0.0007086, 0.0000066, -0.0007634, -0.0000025, -0.0003347, 0.0004057
2: 0.0037261, 0.0040555, 0.0037296, 0.0040776, -0.0001726, 0.0001441
3: 0.0016712, 0.0021497, 0.0016853, 0.0021678, -0.0002630, 0.0002217
4: -0.0041966, -0.0034726, -0.0041939, -0.0034568, -0.0002753, 0.0002563
5: -0.0001516, 0.0002768, -0.0001462, 0.0003108, -0.0002492, 0.0002050
6: -0.0040471, -0.0027609, -0.0040352, -0.0027386, -0.0005244, 0.0004838
7: -0.0203590, -0.0161919, -0.0203417, -0.0160892, -0.0016077, 0.0014839
8: 0.9765776, 0.9803886, 0.9766047, 0.9805312, -0.0015935, 0.0014183
9: 0.0029438, 0.0056825, 0.0028690, 0.0056690, -0.0009816, 0.0010723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168005, 0.0175889, -0.0005304, 0.0006535
1: -0.0007086, 0.0000066, -0.0006891, -0.0001125, -0.0003961, 0.0004904
2: 0.0037261, 0.0040555, 0.0037852, 0.0040384, -0.0002077, 0.0001690
3: 0.0016712, 0.0021497, 0.0016772, 0.0021904, -0.0002971, 0.0002447
4: -0.0041966, -0.0034726, -0.0041296, -0.0035002, -0.0003303, 0.0002921
5: -0.0001516, 0.0002768, -0.0000849, 0.0002707, -0.0003022, 0.0002438
6: -0.0040471, -0.0027609, -0.0040364, -0.0027364, -0.0005242, 0.0004820
7: -0.0203590, -0.0161919, -0.0199401, -0.0163629, -0.0019387, 0.0016981
8: 0.9765776, 0.9803886, 0.9771031, 0.9801837, -0.0019395, 0.0016385
9: 0.0029438, 0.0056825, 0.0030664, 0.0053829, -0.0011267, 0.0012988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0166818, 0.0177592, -0.0004521, 0.0005432
1: -0.0007086, 0.0000066, -0.0007634, -0.0000025, -0.0003347, 0.0004057
2: 0.0037261, 0.0040555, 0.0037296, 0.0040776, -0.0001726, 0.0001441
3: 0.0016712, 0.0021497, 0.0016853, 0.0021678, -0.0002630, 0.0002217
4: -0.0041966, -0.0034726, -0.0041939, -0.0034568, -0.0002753, 0.0002563
5: -0.0001516, 0.0002768, -0.0001462, 0.0003108, -0.0002492, 0.0002050
6: -0.0040471, -0.0027609, -0.0040352, -0.0027386, -0.0005244, 0.0004838
7: -0.0203590, -0.0161919, -0.0203417, -0.0160892, -0.0016077, 0.0014839
8: 0.9765776, 0.9803886, 0.9766047, 0.9805312, -0.0015935, 0.0014183
9: 0.0029438, 0.0056825, 0.0028690, 0.0056690, -0.0009816, 0.0010723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0168005, 0.0175889, -0.0005304, 0.0006535
1: -0.0007086, 0.0000066, -0.0006891, -0.0001125, -0.0003961, 0.0004904
2: 0.0037261, 0.0040555, 0.0037852, 0.0040384, -0.0002077, 0.0001690
3: 0.0016712, 0.0021497, 0.0016772, 0.0021904, -0.0002971, 0.0002447
4: -0.0041966, -0.0034726, -0.0041296, -0.0035002, -0.0003303, 0.0002921
5: -0.0001516, 0.0002768, -0.0000849, 0.0002707, -0.0003022, 0.0002438
6: -0.0040471, -0.0027609, -0.0040364, -0.0027364, -0.0005242, 0.0004820
7: -0.0203590, -0.0161919, -0.0199401, -0.0163629, -0.0019387, 0.0016981
8: 0.9765776, 0.9803886, 0.9771031, 0.9801837, -0.0019395, 0.0016385
9: 0.0029438, 0.0056825, 0.0030664, 0.0053829, -0.0011267, 0.0012988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0167528, 0.0177701, 0.0166818, 0.0177592, -0.0004521, 0.0005432
1: -0.0007086, 0.0000066, -0.0007634, -0.0000025, -0.0003347, 0.0004057
2: 0.0037261, 0.0040555, 0.0037296, 0.0040776, -0.0001726, 0.0001441
3: 0.0016712, 0.0021497, 0.0016853, 0.0021678, -0.0002630, 0.0002217
4: -0.0041966, -0.0034726, -0.0041939, -0.0034568, -0.0002753, 0.0002563
5: -0.0001516, 0.0002768, -0.0001462, 0.0003108, -0.0002492, 0.0002050
6: -0.0040471, -0.0027609, -0.0040352, -0.0027386, -0.0005244, 0.0004838
7: -0.0203590, -0.0161919, -0.0203417, -0.0160892, -0.0016077, 0.0014839
8: 0.9765776, 0.9803886, 0.9766047, 0.9805312, -0.0015935, 0.0014183
9: 0.0029438, 0.0056825, 0.0028690, 0.0056690, -0.0009816, 0.0010723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 156

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0009042
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
time: 0.64 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 3.15 seconds
IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009017
IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0011709
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0009014
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0009042
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0009042
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.15
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168102, 0.0175804, -0.0004911, 0.0005877
1: -0.0006838, -0.0000355, -0.0006834, -0.0001158, -0.0003678, 0.0004419
2: 0.0037466, 0.0040438, 0.0037884, 0.0040352, -0.0001867, 0.0001563
3: 0.0016724, 0.0021489, 0.0016709, 0.0021890, -0.0002790, 0.0002350
4: -0.0041714, -0.0034810, -0.0041219, -0.0035013, -0.0003042, 0.0002755
5: -0.0001271, 0.0002629, -0.0000832, 0.0002674, -0.0002728, 0.0002267
6: -0.0040608, -0.0027658, -0.0040516, -0.0027434, -0.0005175, 0.0004781
7: -0.0202023, -0.0162487, -0.0198968, -0.0163695, -0.0017808, 0.0015992
8: 0.9767592, 0.9803019, 0.9771413, 0.9801622, -0.0017633, 0.0015303
9: 0.0029887, 0.0055711, 0.0030716, 0.0053551, -0.0010594, 0.0011901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167229, 0.0176959, -0.0004274, 0.0005122
1: -0.0006838, -0.0000355, -0.0007335, -0.0000461, -0.0003172, 0.0003832
2: 0.0037466, 0.0040438, 0.0037499, 0.0040645, -0.0001626, 0.0001362
3: 0.0016724, 0.0021489, 0.0016853, 0.0021666, -0.0002589, 0.0002199
4: -0.0041714, -0.0034810, -0.0041673, -0.0034687, -0.0002631, 0.0002460
5: -0.0001271, 0.0002629, -0.0001201, 0.0002934, -0.0002356, 0.0001945
6: -0.0040608, -0.0027658, -0.0040504, -0.0027452, -0.0005179, 0.0004794
7: -0.0202023, -0.0162487, -0.0201776, -0.0161689, -0.0015321, 0.0014206
8: 0.9767592, 0.9803019, 0.9767900, 0.9804238, -0.0015081, 0.0013476
9: 0.0029887, 0.0055711, 0.0029299, 0.0055534, -0.0009366, 0.0010189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168102, 0.0175804, -0.0004911, 0.0005877
1: -0.0006838, -0.0000355, -0.0006834, -0.0001158, -0.0003678, 0.0004419
2: 0.0037466, 0.0040438, 0.0037884, 0.0040352, -0.0001867, 0.0001563
3: 0.0016724, 0.0021489, 0.0016709, 0.0021890, -0.0002790, 0.0002350
4: -0.0041714, -0.0034810, -0.0041219, -0.0035013, -0.0003042, 0.0002755
5: -0.0001271, 0.0002629, -0.0000832, 0.0002674, -0.0002728, 0.0002267
6: -0.0040608, -0.0027658, -0.0040516, -0.0027434, -0.0005175, 0.0004781
7: -0.0202023, -0.0162487, -0.0198968, -0.0163695, -0.0017808, 0.0015992
8: 0.9767592, 0.9803019, 0.9771413, 0.9801622, -0.0017633, 0.0015303
9: 0.0029887, 0.0055711, 0.0030716, 0.0053551, -0.0010594, 0.0011901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167229, 0.0176959, -0.0004274, 0.0005122
1: -0.0006838, -0.0000355, -0.0007335, -0.0000461, -0.0003172, 0.0003832
2: 0.0037466, 0.0040438, 0.0037499, 0.0040645, -0.0001626, 0.0001362
3: 0.0016724, 0.0021489, 0.0016853, 0.0021666, -0.0002589, 0.0002199
4: -0.0041714, -0.0034810, -0.0041673, -0.0034687, -0.0002631, 0.0002460
5: -0.0001271, 0.0002629, -0.0001201, 0.0002934, -0.0002356, 0.0001945
6: -0.0040608, -0.0027658, -0.0040504, -0.0027452, -0.0005179, 0.0004794
7: -0.0202023, -0.0162487, -0.0201776, -0.0161689, -0.0015321, 0.0014206
8: 0.9767592, 0.9803019, 0.9767900, 0.9804238, -0.0015081, 0.0013476
9: 0.0029887, 0.0055711, 0.0029299, 0.0055534, -0.0009366, 0.0010189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168102, 0.0175804, -0.0004911, 0.0005877
1: -0.0006838, -0.0000355, -0.0006834, -0.0001158, -0.0003678, 0.0004419
2: 0.0037466, 0.0040438, 0.0037884, 0.0040352, -0.0001867, 0.0001563
3: 0.0016724, 0.0021489, 0.0016709, 0.0021890, -0.0002790, 0.0002350
4: -0.0041714, -0.0034810, -0.0041219, -0.0035013, -0.0003042, 0.0002755
5: -0.0001271, 0.0002629, -0.0000832, 0.0002674, -0.0002728, 0.0002267
6: -0.0040608, -0.0027658, -0.0040516, -0.0027434, -0.0005175, 0.0004781
7: -0.0202023, -0.0162487, -0.0198968, -0.0163695, -0.0017808, 0.0015992
8: 0.9767592, 0.9803019, 0.9771413, 0.9801622, -0.0017633, 0.0015303
9: 0.0029887, 0.0055711, 0.0030716, 0.0053551, -0.0010594, 0.0011901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167229, 0.0176959, -0.0004274, 0.0005122
1: -0.0006838, -0.0000355, -0.0007335, -0.0000461, -0.0003172, 0.0003832
2: 0.0037466, 0.0040438, 0.0037499, 0.0040645, -0.0001626, 0.0001362
3: 0.0016724, 0.0021489, 0.0016853, 0.0021666, -0.0002589, 0.0002199
4: -0.0041714, -0.0034810, -0.0041673, -0.0034687, -0.0002631, 0.0002460
5: -0.0001271, 0.0002629, -0.0001201, 0.0002934, -0.0002356, 0.0001945
6: -0.0040608, -0.0027658, -0.0040504, -0.0027452, -0.0005179, 0.0004794
7: -0.0202023, -0.0162487, -0.0201776, -0.0161689, -0.0015321, 0.0014206
8: 0.9767592, 0.9803019, 0.9767900, 0.9804238, -0.0015081, 0.0013476
9: 0.0029887, 0.0055711, 0.0029299, 0.0055534, -0.0009366, 0.0010189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168102, 0.0175804, -0.0004911, 0.0005877
1: -0.0006838, -0.0000355, -0.0006834, -0.0001158, -0.0003678, 0.0004419
2: 0.0037466, 0.0040438, 0.0037884, 0.0040352, -0.0001867, 0.0001563
3: 0.0016724, 0.0021489, 0.0016709, 0.0021890, -0.0002790, 0.0002350
4: -0.0041714, -0.0034810, -0.0041219, -0.0035013, -0.0003042, 0.0002755
5: -0.0001271, 0.0002629, -0.0000832, 0.0002674, -0.0002728, 0.0002267
6: -0.0040608, -0.0027658, -0.0040516, -0.0027434, -0.0005175, 0.0004781
7: -0.0202023, -0.0162487, -0.0198968, -0.0163695, -0.0017808, 0.0015992
8: 0.9767592, 0.9803019, 0.9771413, 0.9801622, -0.0017633, 0.0015303
9: 0.0029887, 0.0055711, 0.0030716, 0.0053551, -0.0010594, 0.0011901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167229, 0.0176959, -0.0004274, 0.0005122
1: -0.0006838, -0.0000355, -0.0007335, -0.0000461, -0.0003172, 0.0003832
2: 0.0037466, 0.0040438, 0.0037499, 0.0040645, -0.0001626, 0.0001362
3: 0.0016724, 0.0021489, 0.0016853, 0.0021666, -0.0002589, 0.0002199
4: -0.0041714, -0.0034810, -0.0041673, -0.0034687, -0.0002631, 0.0002460
5: -0.0001271, 0.0002629, -0.0001201, 0.0002934, -0.0002356, 0.0001945
6: -0.0040608, -0.0027658, -0.0040504, -0.0027452, -0.0005179, 0.0004794
7: -0.0202023, -0.0162487, -0.0201776, -0.0161689, -0.0015321, 0.0014206
8: 0.9767592, 0.9803019, 0.9767900, 0.9804238, -0.0015081, 0.0013476
9: 0.0029887, 0.0055711, 0.0029299, 0.0055534, -0.0009366, 0.0010189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168095, 0.0175657, -0.0004746, 0.0005880
1: -0.0006838, -0.0000355, -0.0006790, -0.0001302, -0.0003528, 0.0004373
2: 0.0037466, 0.0040438, 0.0037918, 0.0040363, -0.0001875, 0.0001517
3: 0.0016724, 0.0021489, 0.0016797, 0.0021823, -0.0002859, 0.0002429
4: -0.0041714, -0.0034810, -0.0041193, -0.0034943, -0.0003369, 0.0002980
5: -0.0001271, 0.0002629, -0.0000735, 0.0002641, -0.0002691, 0.0002171
6: -0.0040608, -0.0027658, -0.0040474, -0.0027580, -0.0005670, 0.0005431
7: -0.0202023, -0.0162487, -0.0198795, -0.0163305, -0.0019615, 0.0017206
8: 0.9767592, 0.9803019, 0.9771566, 0.9802017, -0.0018891, 0.0016022
9: 0.0029887, 0.0055711, 0.0030466, 0.0053417, -0.0011332, 0.0013033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167528, 0.0177701, -0.0005371, 0.0005085
1: -0.0006838, -0.0000355, -0.0007086, 0.0000066, -0.0003886, 0.0003704
2: 0.0037466, 0.0040438, 0.0037261, 0.0040555, -0.0001629, 0.0001724
3: 0.0016724, 0.0021489, 0.0016712, 0.0021497, -0.0002275, 0.0002278
4: -0.0041714, -0.0034810, -0.0041966, -0.0034726, -0.0003063, 0.0003253
5: -0.0001271, 0.0002629, -0.0001516, 0.0002768, -0.0002251, 0.0002356
6: -0.0040608, -0.0027658, -0.0040471, -0.0027609, -0.0005643, 0.0005436
7: -0.0202023, -0.0162487, -0.0203590, -0.0161919, -0.0017729, 0.0018867
8: 0.9767592, 0.9803019, 0.9765776, 0.9803886, -0.0016830, 0.0018020
9: 0.0029887, 0.0055711, 0.0029438, 0.0056825, -0.0012492, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168095, 0.0175657, -0.0004746, 0.0005880
1: -0.0006838, -0.0000355, -0.0006790, -0.0001302, -0.0003528, 0.0004373
2: 0.0037466, 0.0040438, 0.0037918, 0.0040363, -0.0001875, 0.0001517
3: 0.0016724, 0.0021489, 0.0016797, 0.0021823, -0.0002859, 0.0002429
4: -0.0041714, -0.0034810, -0.0041193, -0.0034943, -0.0003369, 0.0002980
5: -0.0001271, 0.0002629, -0.0000735, 0.0002641, -0.0002691, 0.0002171
6: -0.0040608, -0.0027658, -0.0040474, -0.0027580, -0.0005670, 0.0005431
7: -0.0202023, -0.0162487, -0.0198795, -0.0163305, -0.0019615, 0.0017206
8: 0.9767592, 0.9803019, 0.9771566, 0.9802017, -0.0018891, 0.0016022
9: 0.0029887, 0.0055711, 0.0030466, 0.0053417, -0.0011332, 0.0013033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167528, 0.0177701, -0.0005371, 0.0005085
1: -0.0006838, -0.0000355, -0.0007086, 0.0000066, -0.0003886, 0.0003704
2: 0.0037466, 0.0040438, 0.0037261, 0.0040555, -0.0001629, 0.0001724
3: 0.0016724, 0.0021489, 0.0016712, 0.0021497, -0.0002275, 0.0002278
4: -0.0041714, -0.0034810, -0.0041966, -0.0034726, -0.0003063, 0.0003253
5: -0.0001271, 0.0002629, -0.0001516, 0.0002768, -0.0002251, 0.0002356
6: -0.0040608, -0.0027658, -0.0040471, -0.0027609, -0.0005643, 0.0005436
7: -0.0202023, -0.0162487, -0.0203590, -0.0161919, -0.0017729, 0.0018867
8: 0.9767592, 0.9803019, 0.9765776, 0.9803886, -0.0016830, 0.0018020
9: 0.0029887, 0.0055711, 0.0029438, 0.0056825, -0.0012492, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168095, 0.0175657, -0.0004746, 0.0005880
1: -0.0006838, -0.0000355, -0.0006790, -0.0001302, -0.0003528, 0.0004373
2: 0.0037466, 0.0040438, 0.0037918, 0.0040363, -0.0001875, 0.0001517
3: 0.0016724, 0.0021489, 0.0016797, 0.0021823, -0.0002859, 0.0002429
4: -0.0041714, -0.0034810, -0.0041193, -0.0034943, -0.0003369, 0.0002980
5: -0.0001271, 0.0002629, -0.0000735, 0.0002641, -0.0002691, 0.0002171
6: -0.0040608, -0.0027658, -0.0040474, -0.0027580, -0.0005670, 0.0005431
7: -0.0202023, -0.0162487, -0.0198795, -0.0163305, -0.0019615, 0.0017206
8: 0.9767592, 0.9803019, 0.9771566, 0.9802017, -0.0018891, 0.0016022
9: 0.0029887, 0.0055711, 0.0030466, 0.0053417, -0.0011332, 0.0013033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167528, 0.0177701, -0.0005371, 0.0005085
1: -0.0006838, -0.0000355, -0.0007086, 0.0000066, -0.0003886, 0.0003704
2: 0.0037466, 0.0040438, 0.0037261, 0.0040555, -0.0001629, 0.0001724
3: 0.0016724, 0.0021489, 0.0016712, 0.0021497, -0.0002275, 0.0002278
4: -0.0041714, -0.0034810, -0.0041966, -0.0034726, -0.0003063, 0.0003253
5: -0.0001271, 0.0002629, -0.0001516, 0.0002768, -0.0002251, 0.0002356
6: -0.0040608, -0.0027658, -0.0040471, -0.0027609, -0.0005643, 0.0005436
7: -0.0202023, -0.0162487, -0.0203590, -0.0161919, -0.0017729, 0.0018867
8: 0.9767592, 0.9803019, 0.9765776, 0.9803886, -0.0016830, 0.0018020
9: 0.0029887, 0.0055711, 0.0029438, 0.0056825, -0.0012492, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168095, 0.0175657, -0.0004746, 0.0005880
1: -0.0006838, -0.0000355, -0.0006790, -0.0001302, -0.0003528, 0.0004373
2: 0.0037466, 0.0040438, 0.0037918, 0.0040363, -0.0001875, 0.0001517
3: 0.0016724, 0.0021489, 0.0016797, 0.0021823, -0.0002859, 0.0002429
4: -0.0041714, -0.0034810, -0.0041193, -0.0034943, -0.0003369, 0.0002980
5: -0.0001271, 0.0002629, -0.0000735, 0.0002641, -0.0002691, 0.0002171
6: -0.0040608, -0.0027658, -0.0040474, -0.0027580, -0.0005670, 0.0005431
7: -0.0202023, -0.0162487, -0.0198795, -0.0163305, -0.0019615, 0.0017206
8: 0.9767592, 0.9803019, 0.9771566, 0.9802017, -0.0018891, 0.0016022
9: 0.0029887, 0.0055711, 0.0030466, 0.0053417, -0.0011332, 0.0013033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0168150, 0.0175552, 0.0167528, 0.0177701, -0.0006573, 0.0005099
1: -0.0006769, -0.0001346, -0.0007086, 0.0000066, -0.0004855, 0.0003792
2: 0.0037959, 0.0040343, 0.0037261, 0.0040555, -0.0001627, 0.0002100
3: 0.0016748, 0.0021812, 0.0016712, 0.0021497, -0.0002329, 0.0002694
4: -0.0041112, -0.0034933, -0.0041966, -0.0034726, -0.0003019, 0.0003691
5: -0.0000714, 0.0002628, -0.0001516, 0.0002768, -0.0002330, 0.0002973
6: -0.0040611, -0.0027634, -0.0040471, -0.0027609, -0.0005626, 0.0005442
7: -0.0198345, -0.0163245, -0.0203590, -0.0161919, -0.0017478, 0.0021554
8: 0.9772027, 0.9801946, 0.9765776, 0.9803886, -0.0016551, 0.0021048
9: 0.0030426, 0.0053139, 0.0029438, 0.0056825, -0.0014365, 0.0011547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0167528, 0.0177701, -0.0005371, 0.0005085
1: -0.0006838, -0.0000355, -0.0007086, 0.0000066, -0.0003886, 0.0003704
2: 0.0037466, 0.0040438, 0.0037261, 0.0040555, -0.0001629, 0.0001724
3: 0.0016724, 0.0021489, 0.0016712, 0.0021497, -0.0002275, 0.0002278
4: -0.0041714, -0.0034810, -0.0041966, -0.0034726, -0.0003063, 0.0003253
5: -0.0001271, 0.0002629, -0.0001516, 0.0002768, -0.0002251, 0.0002356
6: -0.0040608, -0.0027658, -0.0040471, -0.0027609, -0.0005643, 0.0005436
7: -0.0202023, -0.0162487, -0.0203590, -0.0161919, -0.0017729, 0.0018867
8: 0.9767592, 0.9803019, 0.9765776, 0.9803886, -0.0016830, 0.0018020
9: 0.0029887, 0.0055711, 0.0029438, 0.0056825, -0.0012492, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 156

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0011709
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168005, 0.0175889, -0.0005000, 0.0005974
1: -0.0006838, -0.0000355, -0.0006891, -0.0001125, -0.0003710, 0.0004472
2: 0.0037466, 0.0040438, 0.0037852, 0.0040384, -0.0001902, 0.0001597
3: 0.0016724, 0.0021489, 0.0016772, 0.0021904, -0.0002961, 0.0002442
4: -0.0041714, -0.0034810, -0.0041296, -0.0035002, -0.0003286, 0.0003075
5: -0.0001271, 0.0002629, -0.0000849, 0.0002707, -0.0002757, 0.0002280
6: -0.0040608, -0.0027658, -0.0040364, -0.0027364, -0.0005854, 0.0005267
7: -0.0202023, -0.0162487, -0.0199401, -0.0163629, -0.0019175, 0.0017775
8: 0.9767592, 0.9803019, 0.9771031, 0.9801837, -0.0018666, 0.0016680
9: 0.0029887, 0.0055711, 0.0030664, 0.0053829, -0.0011726, 0.0012767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010953
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166818, 0.0177592, -0.0005307, 0.0005843
1: -0.0006838, -0.0000355, -0.0007634, -0.0000025, -0.0003838, 0.0004294
2: 0.0037466, 0.0040438, 0.0037296, 0.0040776, -0.0001866, 0.0001704
3: 0.0016724, 0.0021489, 0.0016853, 0.0021678, -0.0002610, 0.0002200
4: -0.0041714, -0.0034810, -0.0041939, -0.0034568, -0.0003227, 0.0003245
5: -0.0001271, 0.0002629, -0.0001462, 0.0003108, -0.0002619, 0.0002327
6: -0.0040608, -0.0027658, -0.0040352, -0.0027386, -0.0005836, 0.0005260
7: -0.0202023, -0.0162487, -0.0203417, -0.0160892, -0.0018786, 0.0018806
8: 0.9767592, 0.9803019, 0.9766047, 0.9805312, -0.0018315, 0.0017906
9: 0.0029887, 0.0055711, 0.0028690, 0.0056690, -0.0012441, 0.0012484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168005, 0.0175889, -0.0005000, 0.0005974
1: -0.0006838, -0.0000355, -0.0006891, -0.0001125, -0.0003710, 0.0004472
2: 0.0037466, 0.0040438, 0.0037852, 0.0040384, -0.0001902, 0.0001597
3: 0.0016724, 0.0021489, 0.0016772, 0.0021904, -0.0002961, 0.0002442
4: -0.0041714, -0.0034810, -0.0041296, -0.0035002, -0.0003286, 0.0003075
5: -0.0001271, 0.0002629, -0.0000849, 0.0002707, -0.0002757, 0.0002280
6: -0.0040608, -0.0027658, -0.0040364, -0.0027364, -0.0005854, 0.0005267
7: -0.0202023, -0.0162487, -0.0199401, -0.0163629, -0.0019175, 0.0017775
8: 0.9767592, 0.9803019, 0.9771031, 0.9801837, -0.0018666, 0.0016680
9: 0.0029887, 0.0055711, 0.0030664, 0.0053829, -0.0011726, 0.0012767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010953
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166818, 0.0177592, -0.0005307, 0.0005843
1: -0.0006838, -0.0000355, -0.0007634, -0.0000025, -0.0003838, 0.0004294
2: 0.0037466, 0.0040438, 0.0037296, 0.0040776, -0.0001866, 0.0001704
3: 0.0016724, 0.0021489, 0.0016853, 0.0021678, -0.0002610, 0.0002200
4: -0.0041714, -0.0034810, -0.0041939, -0.0034568, -0.0003227, 0.0003245
5: -0.0001271, 0.0002629, -0.0001462, 0.0003108, -0.0002619, 0.0002327
6: -0.0040608, -0.0027658, -0.0040352, -0.0027386, -0.0005836, 0.0005260
7: -0.0202023, -0.0162487, -0.0203417, -0.0160892, -0.0018786, 0.0018806
8: 0.9767592, 0.9803019, 0.9766047, 0.9805312, -0.0018315, 0.0017906
9: 0.0029887, 0.0055711, 0.0028690, 0.0056690, -0.0012441, 0.0012484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168005, 0.0175889, -0.0005000, 0.0005974
1: -0.0006838, -0.0000355, -0.0006891, -0.0001125, -0.0003710, 0.0004472
2: 0.0037466, 0.0040438, 0.0037852, 0.0040384, -0.0001902, 0.0001597
3: 0.0016724, 0.0021489, 0.0016772, 0.0021904, -0.0002961, 0.0002442
4: -0.0041714, -0.0034810, -0.0041296, -0.0035002, -0.0003286, 0.0003075
5: -0.0001271, 0.0002629, -0.0000849, 0.0002707, -0.0002757, 0.0002280
6: -0.0040608, -0.0027658, -0.0040364, -0.0027364, -0.0005854, 0.0005267
7: -0.0202023, -0.0162487, -0.0199401, -0.0163629, -0.0019175, 0.0017775
8: 0.9767592, 0.9803019, 0.9771031, 0.9801837, -0.0018666, 0.0016680
9: 0.0029887, 0.0055711, 0.0030664, 0.0053829, -0.0011726, 0.0012767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010953
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166818, 0.0177592, -0.0005307, 0.0005843
1: -0.0006838, -0.0000355, -0.0007634, -0.0000025, -0.0003838, 0.0004294
2: 0.0037466, 0.0040438, 0.0037296, 0.0040776, -0.0001866, 0.0001704
3: 0.0016724, 0.0021489, 0.0016853, 0.0021678, -0.0002610, 0.0002200
4: -0.0041714, -0.0034810, -0.0041939, -0.0034568, -0.0003227, 0.0003245
5: -0.0001271, 0.0002629, -0.0001462, 0.0003108, -0.0002619, 0.0002327
6: -0.0040608, -0.0027658, -0.0040352, -0.0027386, -0.0005836, 0.0005260
7: -0.0202023, -0.0162487, -0.0203417, -0.0160892, -0.0018786, 0.0018806
8: 0.9767592, 0.9803019, 0.9766047, 0.9805312, -0.0018315, 0.0017906
9: 0.0029887, 0.0055711, 0.0028690, 0.0056690, -0.0012441, 0.0012484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0168005, 0.0175889, -0.0005000, 0.0005974
1: -0.0006838, -0.0000355, -0.0006891, -0.0001125, -0.0003710, 0.0004472
2: 0.0037466, 0.0040438, 0.0037852, 0.0040384, -0.0001902, 0.0001597
3: 0.0016724, 0.0021489, 0.0016772, 0.0021904, -0.0002961, 0.0002442
4: -0.0041714, -0.0034810, -0.0041296, -0.0035002, -0.0003286, 0.0003075
5: -0.0001271, 0.0002629, -0.0000849, 0.0002707, -0.0002757, 0.0002280
6: -0.0040608, -0.0027658, -0.0040364, -0.0027364, -0.0005854, 0.0005267
7: -0.0202023, -0.0162487, -0.0199401, -0.0163629, -0.0019175, 0.0017775
8: 0.9767592, 0.9803019, 0.9771031, 0.9801837, -0.0018666, 0.0016680
9: 0.0029887, 0.0055711, 0.0030664, 0.0053829, -0.0011726, 0.0012767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010953
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0167893, 0.0177068, 0.0166818, 0.0177592, -0.0005307, 0.0005843
1: -0.0006838, -0.0000355, -0.0007634, -0.0000025, -0.0003838, 0.0004294
2: 0.0037466, 0.0040438, 0.0037296, 0.0040776, -0.0001866, 0.0001704
3: 0.0016724, 0.0021489, 0.0016853, 0.0021678, -0.0002610, 0.0002200
4: -0.0041714, -0.0034810, -0.0041939, -0.0034568, -0.0003227, 0.0003245
5: -0.0001271, 0.0002629, -0.0001462, 0.0003108, -0.0002619, 0.0002327
6: -0.0040608, -0.0027658, -0.0040352, -0.0027386, -0.0005836, 0.0005260
7: -0.0202023, -0.0162487, -0.0203417, -0.0160892, -0.0018786, 0.0018806
8: 0.9767592, 0.9803019, 0.9766047, 0.9805312, -0.0018315, 0.0017906
9: 0.0029887, 0.0055711, 0.0028690, 0.0056690, -0.0012441, 0.0012484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 204

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
time: 0.63 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 3.04 seconds
IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008916, upper bound: 0.0010606
IS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010606
IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0009283
IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0009319, upper bound: 0.0011709
IS_A1_B2_A1_B1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0011118, upper bound: 0.0011709
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010953
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010953
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010953
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010953
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0010949
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.04
Output dim: 8, lower bound: -0.0010282, upper bound: 0.0010949
IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0009320
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584
IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0009283, upper bound: 0.0011118
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0009320
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0011709, upper bound: 0.0011118
IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0008909, upper bound: 0.0010584
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.04
Output dim: 8, lower bound: -0.0010423, upper bound: 0.0010584

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.67 + 598.05 = 600.71 seconds
